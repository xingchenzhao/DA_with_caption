# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
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

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient, get_lr_at_iter
from model.utils.set_random_seed import set_random_seed
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.caption_modules.model import DecoderWithAttention
from torch.nn.utils.rnn import pack_padded_sequence
from model.caption_modules.caption_main import caption_main
from model.discriminator.dann import DANN, Discriminator
import copy
from model.caption_modules.utils import fine_tune
from model.discriminator.distance import CORAL_loss, msda_regulizer, mmd_rbf_noaccelerate
import math


def parse_args():
    """
  Parse input arguments
  """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset',
                        dest='dataset',
                        help='training dataset',
                        default='coco',
                        type=str)
    parser.add_argument('--tgt_dataset',
                        dest='target_dataset',
                        help='target training dataset',
                        default='pascal_voc',
                        type=str)
    parser.add_argument('--net',
                        dest='net',
                        help='vgg16, res101',
                        default='vgg16',
                        type=str)
    parser.add_argument('--start_epoch',
                        dest='start_epoch',
                        help='starting epoch',
                        default=1,
                        type=int)
    parser.add_argument('--epochs',
                        dest='max_epochs',
                        help='number of epochs to train',
                        default=200,
                        type=int)
    parser.add_argument('--max_iter',
                        help='number of iterations to train',
                        default=20000,
                        type=int)
    parser.add_argument('--disp_interval',
                        dest='disp_interval',
                        help='number of iterations to display',
                        default=100,
                        type=int)
    parser.add_argument('--checkpoint_interval',
                        dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000,
                        type=int)

    parser.add_argument('--save_dir',
                        dest='save_dir',
                        help='directory to save models',
                        default="results",
                        type=str)
    parser.add_argument('--nw',
                        dest='num_workers',
                        help='number of worker to load data',
                        default=0,
                        type=int)
    parser.add_argument('--cuda',
                        dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True)
    parser.add_argument('--ls',
                        dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs',
                        dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs',
                        dest='batch_size',
                        help='batch_size',
                        default=1,
                        type=int)
    parser.add_argument('--cag',
                        dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o',
                        dest='optimizer',
                        help='training optimizer',
                        default="sgd",
                        type=str)
    parser.add_argument('--lr',
                        dest='lr',
                        help='starting learning rate',
                        default=0.001,
                        type=float)
    parser.add_argument(
        '--lr_decay_iter',
        dest='lr_decay_iter',
        help='iteration to do learning rate decay, unit is epoch',
        default=14000,
        type=int)

    parser.add_argument('--lr_decay_gamma',
                        dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1,
                        type=float)

    # set training session
    parser.add_argument('--s',
                        dest='session',
                        help='training session',
                        default=1,
                        type=int)

    # resume trained model
    parser.add_argument('--r',
                        dest='resume',
                        help='resume checkpoint or not',
                        default=False,
                        type=bool)
    parser.add_argument('--checksession',
                        dest='checksession',
                        help='checksession to load model',
                        default=1,
                        type=int)
    parser.add_argument('--checkepoch',
                        dest='checkepoch',
                        help='checkepoch to load model',
                        default=1,
                        type=int)
    parser.add_argument('--checkpoint',
                        dest='checkpoint',
                        help='checkpoint to load model',
                        default=0,
                        type=int)
    # log and diaplay
    parser.add_argument('--use_tfb',
                        dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    parser.add_argument('--warm_up',
                        dest='warm_up',
                        help='warm_up iters',
                        default=200,
                        type=int)
    parser.add_argument('--save_model_dir',
                        dest='save_model_dir',
                        help='save_model_dir',
                        default="cityscape_multi",
                        type=str)
    parser.add_argument('--wandb',
                        type=str,
                        help='Plot on wandb ',
                        default=None)

    parser.add_argument('--wandb_id',
                        type=str,
                        help='id for the current run',
                        default=None)
    #caption hyerparameter
    parser.add_argument('--caption_for_da',
                        help='use caption for domain adaptation',
                        action='store_true',
                        default=False)
    parser.add_argument('--caption_ft_begin_iter',
                        help='caption fine tune begin iterations',
                        default=11000,
                        type=int)
    parser.add_argument('--caption_total_iter',
                        help='caption fine tune begin iterations',
                        default=15000,
                        type=int)
    parser.add_argument(
        '--cap_no_ft_bs',
        help=
        'batch size when training the caption model without finetuning the cnn',
        default=8)
    parser.add_argument(
        '--cap_ft_bs',
        help=
        'batch size when training the caption model with finetuning the cnn',
        default=4)
    parser.add_argument('--lstm_lr', default=4e-4, type=float)
    parser.add_argument('--attention_dim',
                        help='dimension of attention linear layers',
                        default=512,
                        type=int)
    parser.add_argument('--embed_dim',
                        help='dimension of word embeddings',
                        default=512,
                        type=int)
    parser.add_argument('--decoder_dim',
                        help='dimension of decoder RNN',
                        default=512,
                        type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--alpha_c',
                        help='alpha rate for attention',
                        default=1,
                        type=float)
    parser.add_argument('--cap_val_iter',
                        help='validate iterations for caption model',
                        default=5000,
                        type=int)
    parser.add_argument('--use_glove',
                        help='Use glove pretrained embedding',
                        default=True,
                        type=bool)
    parser.add_argument('--cap_resume',
                        help='resume the training',
                        default=False,
                        type=bool)
    parser.add_argument('--cap_sampling',
                        help='scheduled sampling',
                        action='store_true',
                        default=False)
    parser.add_argument('--cap_pretraining',
                        help='cap_pretraining',
                        action='store_true',
                        default=False)
    parser.add_argument('--decay_schedule',
                        help='scheduled decay',
                        default=3,
                        type=float)
    parser.add_argument('--dann_weight',
                        help='dann weight',
                        default=0.1,
                        type=float)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='Weight Decay',
                        default=5e-4)

    args = parser.parse_args()
    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size,
                                         train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,
                                                           1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch,
                                        self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),
                                           0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    print('Called with args:')
    print(args)
    print(f"cap_sampling {args.cap_sampling}")

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '50'
        ]
    elif args.dataset == "coco_pascal_voc":
        args.imdb_name = "coco_2014_train"
        args.imdbval_name = "coco_2014_val"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '50'
        ]
        args.tgt_imdb_name = "voc_2007_trainval"
        args.tgt_imdbval_name = "voc_2007_test"
        args.tgt_set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    elif args.dataset == "coco_clipart":
        args.imdb_name = "coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '50'
        ]
        args.tgt_imdb_name = "clipart1k_train"
        args.tgt_imdbval_name = "clipart1k_test"
        args.tgt_set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    elif args.dataset == "clipart":
        args.imdb_name = "clipart1k_train"
        args.imdbval_name = "clipart1k_test"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]

    elif args.dataset == "cityscape_multi":
        args.imdb_name = "cityscape_multi_train"
        args.imdbval_name = "cityscape_multi_val"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    elif args.dataset == "foggy_cityscape_multi":
        args.imdb_name = "foggy_cityscape_multi_train"
        args.imdbval_name = "foggy_cityscape_multi_val"
        args.set_cfgs = [
            'ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]',
            'MAX_NUM_GT_BOXES', '20'
        ]
    args.cfg_file = "cfgs/{}_ls.yml".format(
        args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.wandb_id)
    else:
        wandb = None
    print('Using config:')
    pprint.pprint(cfg)
    set_random_seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print(
            "WARNING: You have a CUDA device, so you should probably run with --cuda"
        )

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    #source data
    imdb, roidb, ratio_list, ratio_index, vocab = combined_roidb(
        args.imdb_name)
    _, _, _, _, vocab = combined_roidb("coco_2014_train")
    train_size = len(roidb)

    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.save_model_dir
    args.output_dir = output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                             imdb.num_classes, training=True,domain=0)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             sampler=sampler_batch,
                                             num_workers=args.num_workers,
                                             drop_last=True,
                                             pin_memory=True)

    #target data
    tgt_imdb, tgt_roidb, tgt_ratio_list, tgt_ratio_index, _ = combined_roidb(
        args.tgt_imdb_name)
    tgt_train_size = len(tgt_roidb)
    tgt_sampler_batch = sampler(tgt_train_size, args.batch_size)
    tgt_dataset = roibatchLoader(tgt_roidb,
                                 tgt_ratio_list,
                                 tgt_ratio_index,
                                 args.batch_size,
                                 tgt_imdb.num_classes,
                                 training=True,
                                 domain=1)
    tgt_dataloader = torch.utils.data.DataLoader(tgt_dataset,
                                                 batch_size=args.batch_size,
                                                 sampler=tgt_sampler_batch,
                                                 num_workers=args.num_workers,
                                                 drop_last=True,
                                                 pin_memory=True)

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes,
                           pretrained=True,
                           class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes,
                            101,
                            pretrained=True,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes,
                            50,
                            pretrained=True,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes,
                            152,
                            pretrained=True,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    iter_num = 0

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                        'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{
                    'params': [value],
                    'lr': lr,
                    'weight_decay': cfg.TRAIN.WEIGHT_DECAY
                }]

    if args.cuda:
        fasterRCNN.cuda()

    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.resume:
        load_name = os.path.join(
            output_dir,
            'cap_faster_rcnn_{}_{}_{}.pth'.format(args.checksession,
                                                  args.checkepoch,
                                                  args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    iters_per_epoch = int(train_size / args.batch_size)

    if args.caption_for_da:
        dann = Discriminator(DANN(2))
        dann_optimizer = torch.optim.SGD(dann.parameters(),
                                         args.lr,
                                         momentum=cfg.TRAIN.MOMENTUM,
                                         nesterov=True,
                                         weight_decay=args.weight_decay)
        if args.cuda:
            dann.cuda()
        if args.use_glove:
            glove_vectors = pickle.load(open('glove.6B/glove_words.pkl', 'rb'))
            glove_vectors = torch.FloatTensor(glove_vectors)
            args.embed_dim = 300
            print('use glove embedding')
        lstm = DecoderWithAttention(attention_dim=args.attention_dim,
                                    embed_dim=args.embed_dim,
                                    decoder_dim=args.decoder_dim,
                                    vocab_size=len(vocab.idx2word),
                                    dropout=args.dropout,
                                    args=args)
        lstm.load_pretrained_embeddings(glove_vectors)
        lstm.fine_tune_embeddings(True)
        lstm_optimizer = torch.optim.Adam(params=filter(
            lambda p: p.requires_grad, lstm.parameters()),
                                          lr=args.lstm_lr)
        lstm_criterion = nn.CrossEntropyLoss()
        dann_criterion = nn.CrossEntropyLoss()

        if args.cuda:
            lstm.cuda()

        # train and validate the captioning model
        if args.cap_pretraining:
            val_imdb, val_roidb, val_ratio_list, val_ratio_index, _ = combined_roidb(
                args.imdbval_name)

            val_dataset = roibatchLoader(val_roidb,
                                         val_ratio_list,
                                         val_ratio_index,
                                         1,
                                         val_imdb.num_classes,
                                         training=False,
                                         normalize=False)

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True)
        args.vocab = vocab
        cap_iter_num = 0
        if args.cap_resume:
            load_name = os.path.join(
                args.output_dir,
                'cap_lstm_{}_{}_{}.pth'.format(args.checksession,
                                               args.checkepoch,
                                               args.checkpoint))
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name)
            args.session = checkpoint['session']
            args.start_epoch = checkpoint['epoch']
            cap_iter_num = checkpoint['iter_num']
            lstm.load_state_dict(checkpoint['model'])
            lstm_optimizer.load_state_dict(checkpoint['optimizer'])
            lr = lstm_optimizer.param_groups[0]['lr']

            print("loaded checkpoint %s" % (load_name))
        if args.cap_pretraining:
            caption_main(args, dataloader,
                         val_dataloader, lstm_criterion, fasterRCNN,
                         copy.deepcopy(optimizer), lstm, lstm_optimizer,
                         cap_iter_num)
        else:
            fine_tune(lstm, freeze=True)
            fine_tune(fasterRCNN.RCNN_base, freeze=False)
    torch.autograd.set_detect_anomaly(True)

    iter_num = 0
    args.start_epoch = 0
    has_adjusted_lr = False
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        if iter_num >= args.max_iter:
            print(f"stopped at iterations {iter_num}")
            break

        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        dann_loss_temp = 0
        start = time.time()

        data_iter = iter(dataloader)
        tgt_data_iter = iter(tgt_dataloader)
        for step in range(iters_per_epoch):
            if iter_num >= args.max_iter:
                break
            iter_num += 1
            p = iter_num / args.max_iter
            if args.caption_for_da:
                dann_suppression = (2.0 /
                                    (1. + np.exp(-args.decay_schedule * p)) -
                                    1)
                beta = 1 * args.dann_weight
                beta *= dann_suppression
                # dann.set_beta(beta)
            if iter_num >= (args.max_iter * 0.75) and not has_adjusted_lr:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                if args.caption_for_da:
                    adjust_learning_rate(dann_optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma
                has_adjusted_lr = True

            data = next(data_iter)
            if args.caption_for_da:
                try:
                    tgt_data = next(tgt_data_iter)
                except StopIteration:
                    tgt_data_iter = iter(tgt_dataloader)
                    tgt_data = next(tgt_data_iter)
            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])
                if args.imdb_name == "coco_train":
                    domain = data[6].cuda()
                else:
                    domain = data[4].cuda()
                if args.caption_for_da:
                    tgt_im_data = tgt_data[0].cuda()
                    tgt_domain = tgt_data[4].cuda()
            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label, base_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
            if args.caption_for_da:
                tgt_base_feat = fasterRCNN.RCNN_base(tgt_im_data)
                _, _, pred, cap_h_pred, cap_c_pred = lstm.greedy_search(
                    base_feat)
                _, _, tgt_pred, tgt_cap_h_pred, tgt_cap_c_pred = lstm.greedy_search(
                    tgt_base_feat)

                # cap_pred = torch.cat([cap_h_pred, cap_c_pred], dim=2)
                # tgt_cap_pred = torch.cat([tgt_cap_h_pred, tgt_cap_c_pred],
                #                          dim=2)
                # cap_dann_pred = dann(cap_h_pred)
                # tgt_cap_dann_pred = dann(tgt_cap_h_pred)
                mmd_loss = mmd_rbf_noaccelerate(pred, tgt_pred)
                # cap_dann_loss = dann_criterion(cap_dann_pred, domain)
                # tgt_cap_dann_loss = dann_criterion(tgt_cap_dann_pred,
                #                                    tgt_domain)
                # dann_loss = cap_dann_loss.mean() + tgt_cap_dann_loss.mean()
                dann_loss = mmd_loss

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()
            if args.caption_for_da:
                loss += dann_loss * beta
                dann_loss_temp += dann_loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()
            # if args.caption_for_da:
            #     clip_gradient(dann, 5.)
            #     dann_optimizer.zero_grad()
            #     dann_optimizer.step()
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                    if args.caption_for_da:
                        dann_loss_temp /= (args.disp_interval + 1)
                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt


                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, dann_loss: %.4f, lr: %.2e" \
                                        % (args.session, epoch, step, iters_per_epoch, loss_temp, dann_loss_temp if args.caption_for_da else 0, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                              % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                if args.use_tfboard:
                    info = {
                        'loss': loss_temp,
                        'loss_rpn_cls': loss_rpn_cls,
                        'loss_rpn_box': loss_rpn_box,
                        'loss_rcnn_cls': loss_rcnn_cls,
                        'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars("logs_s_{}/losses".format(args.session),
                                       info,
                                       (epoch - 1) * iters_per_epoch + step)
                if args.wandb is not None:
                    wandb.log({
                        'fr_loss':
                        loss_temp,
                        'fr_loss_rpn_cls':
                        loss_rpn_cls,
                        'fr_loss_rpn_box':
                        loss_rpn_box,
                        'fr_loss_rcnn_cls':
                        loss_rcnn_cls,
                        'fr_loss_rcnn_box':
                        loss_rcnn_box,
                        'fr_loss_dann':
                        dann_loss_temp if args.caption_for_da else None,
                        'dann_suppression':
                        dann_suppression if args.caption_for_da else None
                    })

                loss_temp = 0
                start = time.time()
        save_name = os.path.join(
            output_dir,
            'cap_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch,
                                                  iter_num))
        save_checkpoint(
            {
                'session':
                args.session,
                'epoch':
                epoch + 1,
                'iter_num':
                iter_num + 1,
                'model':
                fasterRCNN.module.state_dict()
                if args.mGPUs else fasterRCNN.state_dict(),
                'optimizer':
                optimizer.state_dict(),
                'pooling_mode':
                cfg.POOLING_MODE,
                'class_agnostic':
                args.class_agnostic
            }, save_name)
        print('save model: {}'.format(save_name))
################################################saving the final model #############################################################
    save_name = os.path.join(
        output_dir,
        'cap_faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, iter_num))
    save_checkpoint(
        {
            'session':
            args.session,
            'epoch':
            epoch + 1,
            'iter_num':
            iter_num + 1,
            'model':
            fasterRCNN.module.state_dict()
            if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer':
            optimizer.state_dict(),
            'pooling_mode':
            cfg.POOLING_MODE,
            'class_agnostic':
            args.class_agnostic,
        }, save_name)
    print('save model: {}'.format(save_name))
