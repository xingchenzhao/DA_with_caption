from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model.caption_modules.train_caption import train_caption
from model.caption_modules.validate_caption import validate_caption


def caption_main(args, dataloader, val_dataloader, lstm_criterion, faster_rcnn,
                 faster_rcnn_optimizer, lstm, lstm_optimizer):
    train_caption(args, dataloader, lstm_criterion, faster_rcnn,
                  faster_rcnn_optimizer, lstm, lstm_optimizer)
    bleu4 = validate_caption(args, val_dataloader, lstm_criterion, faster_rcnn,lstm)
