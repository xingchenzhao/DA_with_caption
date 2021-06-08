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
from torchvision.utils import save_image


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
    greedy_hypotheses = list()
    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            if iter_num >= args.cap_val_iter:
                break
            iter_num += 1
            imgs = data[0].cuda()
            all_captions = data[4].cuda()
            all_caplen = data[5].cuda()
            imgs = faster_rcnn.RCNN_base(imgs)

            captions = all_captions[:, 0, :]
            caplen = all_caplen[:, 0]
            scores, caps_sorted, decode_lengths, alphas, sort_ind = lstm(
                imgs, captions, caplen)
            sentence, greedy_alphas, greedy_scores, _ = lstm.greedy_search(
                imgs)

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
            # Greed Hypotheses

            sentence = sentence.tolist()
            for j in range(len(sentence)):
                current_sentence = sentence[j]
                temp_sentence = []
                for p, word in enumerate(current_sentence):
                    if word == args.vocab.word2idx['<end>']:
                        break
                    temp_sentence.append(word)
                    if iter_num % 50 == 0:
                        # orig_imgs = data[0]
                        # for img_idx in range(orig_imgs.size(0)):
                        #     current_img = orig_imgs[img_idx]
                        #     save_image(
                        #         current_img,
                        #         f'{args.output_dir}/img_{iter_num}_{img_idx}.png',
                        #         normalize=True)
                        str_word = args.vocab.idx2word[word]
                        print(str_word + ' ', end='')
                    if word == args.vocab.word2idx['.']:
                        break
                sentence[j] = temp_sentence
                if iter_num % 50 == 0:
                    print('')
            greedy_hypotheses.extend(sentence)
            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j] -
                                           1])  # remove pads
            preds = temp_preds

            hypotheses.extend(preds)

            #References
            if all_captions.size(0) != 1:
                all_captions = all_captions[
                    sort_ind]  # because images were sorted in the decoder
            for j in range(all_captions.shape[0]):
                img_caps = all_captions[j].tolist()
                img_captions = list(
                    map(
                        lambda c: [
                            w for w in c if w not in {
                                args.vocab.word2idx['<start>'], args.vocab.
                                word2idx['<end>'], args.vocab.word2idx['<pad>']
                            }
                        ], img_caps))  # remove <start> and pads
                references.append(img_captions)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        # bleu4 = corpus_bleu(references, hypotheses)
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references,
                            hypotheses,
                            weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0))
        bleu4 = corpus_bleu(references,
                            hypotheses,
                            weights=(0.25, 0.25, 0.25, 0.25))
        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}\n'
            .format(loss=losses,
                    top5=top5accs,
                    bleu1=bleu1,
                    bleu2=bleu2,
                    bleu3=bleu3,
                    bleu4=bleu4))
        greedy_bleu1 = corpus_bleu(references,
                                   greedy_hypotheses,
                                   weights=(1.0, 0, 0, 0))
        greedy_bleu2 = corpus_bleu(references,
                                   greedy_hypotheses,
                                   weights=(0.5, 0.5, 0, 0))
        greedy_bleu3 = corpus_bleu(references,
                                   greedy_hypotheses,
                                   weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0,
                                            0))
        greedy_bleu4 = corpus_bleu(references,
                                   greedy_hypotheses,
                                   weights=(0.25, 0.25, 0.25, 0.25))
        print(
            f'greedy_BLEU-1 {greedy_bleu1}, greedy_BLEU-2 {greedy_bleu2},greedy_BLEU-3 {greedy_bleu3},greedy_BLEU-4 {greedy_bleu4}'
        )

    return bleu4