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
from model.utils.net_utils import save_checkpoint
from model.utils.config import cfg


def save_model(args, iter_num, epoch, faster_rcnn, faster_rcnn_optimizer, lstm,
               lstm_optimizer):
    save_name = os.path.join(
        args.output_dir,
        'cap_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, iter_num))
    save_checkpoint(
        {
            'session':
            args.session,
            'epoch':
            epoch + 1,
            'iter_num':
            0,
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
            'iter_num': iter_num + 1,
            'model':
            lstm.module.state_dict() if args.mGPUs else lstm.state_dict(),
            'optimizer': lstm_optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fine_tune(model, freeze=False):
    if freeze:
        for layer in model.parameters():
            layer.requires_grad = False
    else:
        for layer in range(10):
            for p in model[layer].parameters():
                p.requires_grad = False
        for layer in range(10, len(model)):
            for p in model[layer].parameters():
                p.requires_grad = True


def clip_gradient(optimizer, grad_clip=5):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
