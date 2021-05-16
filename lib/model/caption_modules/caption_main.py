from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.caption_modules.train_caption import train_caption
from model.caption_modules.validate_caption import validate_caption
import torch
import torch.nn as nn
import torch.optim as optim
import os


def caption_main(args, dataloader, val_dataloader, lstm_criterion, faster_rcnn,
                 faster_rcnn_optimizer, lstm, lstm_optimizer):
    iter_num = 0
    if args.cap_resume:
        load_name = os.path.join(
            args.output_dir,
            'cap_lstm_{}_{}_{}.pth'.format(args.checksession, args.checkepoch,
                                           args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
        lstm.load_state_dict(checkpoint['model'])
        lstm_optimizer.load_state_dict(checkpoint['optimizer'])
        lr = lstm_optimizer.param_groups[0]['lr']

        print("loaded checkpoint %s" % (load_name))
    if args.mGPUs:
        lstm_decoder = nn.DataParallel(lstm_decoder)
    train_caption(args,
                  dataloader,
                  lstm_criterion,
                  faster_rcnn,
                  faster_rcnn_optimizer,
                  lstm,
                  lstm_optimizer,
                  iter_num=iter_num)
    bleu4 = validate_caption(args, val_dataloader, lstm_criterion, faster_rcnn,
                             lstm)
