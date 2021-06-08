from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.caption_modules.train_caption import train_caption
from model.caption_modules.validate_caption import validate_caption
import torch
import torch.nn as nn
import torch.optim as optim
import os


def caption_main(args,
                 dataloader,
                 val_dataloader,
                 lstm_criterion,
                 faster_rcnn,
                 faster_rcnn_optimizer,
                 lstm,
                 lstm_optimizer,
                 iter_num=0):
    iter_num = iter_num

    if args.mGPUs:
        lstm_dp = nn.DataParallel(lstm)

    train_caption(args,
                  dataloader,
                  lstm_criterion,
                  faster_rcnn,
                  faster_rcnn_optimizer,
                  lstm if not args.mGPUs else lstm_dp,
                  lstm_optimizer,
                  iter_num=iter_num)
    if args.mGPUs:
        del lstm_dp
        torch.cuda.empty_cache()
    bleu4 = validate_caption(args, val_dataloader, lstm_criterion, faster_rcnn,
                             lstm)
