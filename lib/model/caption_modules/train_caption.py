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
from model.caption_modules.utils import AverageMeter, fine_tune, clip_gradient, accuracy
from model.utils.net_utils import save_checkpoint
from model.utils.config import cfg


def train_caption(args, dataloader, lstm_criterion, faster_rcnn,
                  faster_rcnn_optimizer, lstm, lstm_optimizer):
    grad_clip = 5.  # clip gradients at an absolute value of
    fine_tune(faster_rcnn.RCNN_base, freeze=True)
    faster_rcnn.train()
    lstm.train()
    iter_num = 0
    has_finetuned = False
    has_adjusted_lr = False
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        if iter_num >= args.caption_total_iter:
            print(f"caption training stopped at iteration {iter_num}")
            break

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()
        if epoch % 4 == 0:
            save_name = os.path.join(
                args.output_dir,
                'cap_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch,
                                                      iter_num))
            save_checkpoint(
                {
                    'session':
                    args.session,
                    'epoch':
                    epoch + 1,
                    'model':
                    faster_rcnn.module.state_dict()
                    if args.mGPUs else faster_rcnn.state_dict(),
                    'optimizer':
                    faster_rcnn_optimizer.state_dict(),
                    'pooling_mode':
                    cfg.POOLING_MODE,
                    'class_agnostic':
                    args.class_agnostic,
                }, save_name)
            print('save model: {}'.format(save_name))
            save_name = os.path.join(
                args.output_dir,
                'cap_lstm_{}_{}_{}.pth'.format(args.session, epoch, iter_num))
            save_checkpoint(
                {
                    'session':
                    args.session,
                    'epoch':
                    epoch + 1,
                    'model':
                    lstm.module.state_dict()
                    if args.mGPUs else lstm.state_dict(),
                    'optimizer':
                    lstm_optimizer.state_dict(),
                    'pooling_mode':
                    cfg.POOLING_MODE,
                    'class_agnostic':
                    args.class_agnostic,
                }, save_name)
            print('save model: {}'.format(save_name))
        for i, data in enumerate(dataloader):
            if iter_num >= args.caption_total_iter:
                break
            iter_num += 1
            if iter_num >= args.caption_ft_begin_iter and not has_finetuned:
                print("caption: begin finetuning")
                fine_tune(faster_rcnn.RCNN_base, freeze=False)
                has_finetuned = True
            if (iter_num >=
                (args.caption_total_iter * 0.75)) and not has_adjusted_lr:
                print("caption: adjust learning rate")
                adjust_learning_rate(lstm_optimizer, 0.1)
                if iter_num >= args.caption_ft_begin_iter:
                    adjust_learning_rate(faster_rcnn_optimizer, 0.1)
                has_adjusted_lr = True
            imgs = data[0].cuda()
            captions = data[4].cuda()
            caplen = data[5].cuda()

            data_time.update(time.time() - start)
            imgs = faster_rcnn.RCNN_base(imgs)
            caption_lengths, _ = caplen.squeeze(1).sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
            # args.decode_lengths = decode_lengths
            scores, caps_sorted, decode_lengths, alphas, sort_ind = lstm(
                imgs, captions, caplen)
            targets = caps_sorted[:, 1:]
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
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            if i % 50 == 0:
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                targets_copy = targets_copy.tolist()
                temp_preds = list()
                temp_targets = list()
                for j, p in enumerate(preds):
                    temp_preds.append(
                        preds[j][:decode_lengths[j]])  # remove pads
                preds = temp_preds
                for j, p in enumerate(targets_copy):
                    temp_targets.append(
                        targets_copy[j][:decode_lengths[j]])  # remove pads
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
    save_name = os.path.join(
        args.output_dir,
        'cap_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, iter_num))
    save_checkpoint(
        {
            'session':
            args.session,
            'epoch':
            epoch + 1,
            'model':
            faster_rcnn.module.state_dict()
            if args.mGPUs else faster_rcnn.state_dict(),
            'optimizer':
            faster_rcnn_optimizer.state_dict(),
            'pooling_mode':
            cfg.POOLING_MODE,
            'class_agnostic':
            args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))
    save_name = os.path.join(
        args.output_dir,
        'cap_lstm_{}_{}_{}.pth'.format(args.session, epoch, iter_num))
    save_checkpoint(
        {
            'session': args.session,
            'epoch': epoch + 1,
            'model':
            lstm.module.state_dict() if args.mGPUs else lstm.state_dict(),
            'optimizer': lstm_optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))