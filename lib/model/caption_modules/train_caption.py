from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
from processCaption import Vocabulary
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from model.utils.net_utils import adjust_learning_rate
from torch.nn.utils.rnn import pack_padded_sequence
from model.caption_modules.utils import AverageMeter, fine_tune, clip_gradient, accuracy, save_model
from model.utils.net_utils import save_checkpoint
import wandb


def train_caption(args,
                  dataloader,
                  lstm_criterion,
                  faster_rcnn,
                  faster_rcnn_optimizer,
                  lstm,
                  lstm_optimizer,
                  iter_num=0):
    grad_clip = 5.  # clip gradients at an absolute value of
    fine_tune(faster_rcnn.RCNN_base
              if not args.mGPUs else faster_rcnn.module.RCNN_base,
              freeze=True)
    faster_rcnn.train()
    lstm.train()
    iter_num = iter_num
    has_finetuned = False
    has_adjusted_lr = False
    decay_schedule = 0
    print(f"cap_sampling {args.cap_sampling}")
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        if iter_num >= args.caption_total_iter:
            print(f"caption training stopped at iteration {iter_num}")
            break

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        save_model(args, iter_num, epoch, faster_rcnn, faster_rcnn_optimizer,
                   lstm, lstm_optimizer)

        for i, data in enumerate(dataloader):
            args.cap_curr_iter = i
            if iter_num >= args.caption_total_iter:
                break
            p = iter_num / args.caption_total_iter
            if args.cap_sampling:
                decay_schedule = (2.0 /
                                  (1. + np.exp(-args.decay_schedule * p)) - 1)
                if decay_schedule >= 0.5:
                    decay_schedule = 0.5
                use_sampling = np.random.random() < decay_schedule
            else:
                use_sampling = False

            iter_num += 1
            if iter_num >= args.caption_ft_begin_iter and not has_finetuned:
                print("caption: begin finetuning")
                fine_tune(faster_rcnn.RCNN_base
                          if not args.mGPUs else faster_rcnn.module.RCNN_base,
                          freeze=False)
                has_finetuned = True
            if (iter_num >=
                (args.caption_total_iter * 0.75)) and not has_adjusted_lr:
                print("caption: adjust learning rate")
                adjust_learning_rate(lstm_optimizer, 0.1)
                if iter_num >= args.caption_ft_begin_iter:
                    adjust_learning_rate(faster_rcnn_optimizer, 0.1)
                has_adjusted_lr = True
            imgs = data[0].cuda()
            all_captions = data[4].cuda()
            all_caplen = data[5].cuda()
            if not args.mGPUs:
                encoded_imgs = faster_rcnn.RCNN_base(imgs)
            else:
                encoded_imgs = faster_rcnn.module.RCNN_base(imgs)
            total_loss = 0
            data_time.update(time.time() - start)
            for j in range(all_captions.size(1)):
                captions = all_captions[:, j, :]
                caplen = all_caplen[:, j]
                if not args.mGPUs:
                    scores, caps_sorted, decode_lengths, alphas, sort_ind = lstm(
                        encoded_imgs, captions, caplen, use_sampling)
                    targets = caps_sorted[:, 1:]
                    if i % 50 == 0:
                        scores_copy = scores.clone()
                        targets_copy = targets.clone()
                    scores, *_ = pack_padded_sequence(scores,
                                                      decode_lengths,
                                                      batch_first=True)
                    targets, *_ = pack_padded_sequence(targets,
                                                       decode_lengths,
                                                       batch_first=True)
                    loss = lstm_criterion(scores, targets)
                    loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
                else:
                    loss = lstm(encoded_imgs,
                                captions,
                                caplen,
                                use_sampling,
                                data_parallel=args.mGPUs,
                                lstm_criterion=lstm_criterion,
                                args=args)
                    if i % 50 == 0:
                        scores_copy = args.scores_copy
                        targets_copy = args.targets_copy
                        decode_lengths = args.decode_lengths
                    loss = loss.mean()

                total_loss += loss
            loss = total_loss / all_captions.size(1)
            lstm_optimizer.zero_grad()
            if iter_num >= args.caption_ft_begin_iter:
                faster_rcnn_optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                clip_gradient(lstm_optimizer, grad_clip)
                if iter_num >= args.caption_ft_begin_iter:
                    clip_gradient(faster_rcnn_optimizer, grad_clip)

            lstm_optimizer.step()
            if iter_num >= args.caption_ft_begin_iter:
                faster_rcnn_optimizer.step()
            top5 = accuracy(scores, targets, 5) if not args.mGPUs else 0
            if not args.mGPUs:
                losses.update(loss.item(), sum(decode_lengths))
                top5accs.update(top5, sum(decode_lengths))
            else:
                losses.update(loss.item())
                top5accs.update(top5)
            if args.wandb is not None:
                wandb.log({
                    "loss": total_loss.item() / all_captions.size(1),
                    "top5acc": top5,
                    "decay_schedule": decay_schedule
                })
            batch_time.update(time.time() - start)

            if i % 50 == 0:
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                targets_copy = targets_copy.tolist()
                temp_preds = list()
                temp_targets = list()
                for j, p in enumerate(preds):
                    temp_preds.append(preds[j][:decode_lengths[j] -
                                               1])  # remove pads
                preds = temp_preds
                for j, p in enumerate(targets_copy):
                    temp_targets.append(targets_copy[j][:decode_lengths[j] -
                                                        1])  # remove pads
                targets_copy = temp_targets
                for w in range(len(preds)):
                    pred_words = [args.vocab.idx2word[ind] for ind in preds[w]]
                    target_words = [
                        args.vocab.idx2word[ind] for ind in targets_copy[w]
                    ]
                    print('pred_words:')
                    for w in pred_words:
                        print(w + ' ', end='')
                    print('')
                    print('target_words:')
                    for w in target_words:
                        print(w + ' ', end='')
                    print('')
            start = time.time()
            if i % 100 == 0:
                print(f'current iterations: {iter_num}')
                print(
                    'Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch,
                        i,
                        len(dataloader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        top5=top5accs))


#####################save model
    save_model(args, iter_num, epoch, faster_rcnn, faster_rcnn_optimizer, lstm,
               lstm_optimizer)
