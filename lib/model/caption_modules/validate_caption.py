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
from nltk.translate.bleu_score import corpus_bleu


def validate_caption(args, val_dataloader, lstm_criterion, faster_rcnn, lstm):
    iter_num = 0
    lstm.eval()
    faster_rcnn.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list(
    )  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            if iter_num >= args.cap_val_iter:
                break
            iter_num += 1
            imgs = data[0].cuda()
            captions = data[4].cuda()
            caplen = data[5].cuda()
            imgs = faster_rcnn.RCNN_base(imgs)
            caption_lengths, _ = caplen.squeeze(1).sort(dim=0, descending=True)
            decode_lengths = (caption_lengths - 1).tolist()
            args.decode_lengths = decode_lengths
            scores, caps_sorted, alphas, sort_ind = lstm(
                imgs, captions, caplen)
            targets = caps_sorted[:, 1:]
            scores_copy = scores.clone()

            scores, *_ = pack_padded_sequence(scores,
                                              decode_lengths,
                                              batch_first=True)
            targets, *_ = pack_padded_sequence(targets,
                                               decode_lengths,
                                               batch_first=True)
            loss = lstm_criterion(scores, targets)
            loss += args.alpha_c * ((1. - alphas.sum(dim=1))**2).mean()
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()
            if i % 100 == 0:
                print(
                    'Validation: [{0}/{1}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                        i,
                        len(val_dataloader),
                        batch_time=batch_time,
                        loss=losses,
                        top5=top5accs))
            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds

            hypotheses.extend(preds)

            temp_targets = targets.tolist()
            targets = list()
            targets.append(temp_targets)
            temp_targets = list()
            for j, p in enumerate(targets):
                temp_targets.append(targets[j][:decode_lengths[j]])
            targets = temp_targets
            temp_list = list()
            temp_list.append(targets)
            references.extend(temp_list)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'
            .format(loss=losses, top5=top5accs, bleu=bleu4))

    return bleu4