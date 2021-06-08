# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco_caption import coco
from datasets.coco_original import coco as coco_original
from datasets.imagenet import imagenet
from datasets.vg import vg
from datasets.cityscape_multi import cityscape_multi
from datasets.foggy_cityscape_multi import foggy_cityscape_multi
from datasets.clipart1k import clipart1k

import numpy as np

num_shot = 10

# Set up voc_<year>_<split>
for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

for year in ['2014']:
    for split in ['minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (
            lambda split=split, year=year: coco_original(split, year))
# Set up coco_2014_cap_<split>
for year in ['2014']:
    for split in ['train', 'val', 'capval', 'valminuscapval', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in [
        '150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450',
        '1600-400-20'
]:
    for split in [
            'minitrain', 'smalltrain', 'train', 'minival', 'smallval', 'val',
            'test'
    ]:
        name = 'vg_{}_{}'.format(version, split)
        __sets[name] = (
            lambda split=split, version=version: vg(version, split))

# set up image net.
for split in ['train', 'val', 'val1', 'val2', 'test']:
    name = 'imagenet_{}'.format(split)
    devkit_path = 'data/imagenet/ILSVRC/devkit'
    data_path = 'data/imagenet/ILSVRC'
    __sets[name] = (lambda split=split, devkit_path=devkit_path, data_path=
                    data_path: imagenet(split, devkit_path, data_path))

for split in ['train', 'val']:
    name = 'cityscape_multi_{}'.format(split)
    __sets[name] = (lambda split=split: cityscape_multi(split))
    tgt_name = 'cityscape_multi_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: cityscape_multi(
        split, num_shot))

for split in ['train', 'val']:
    name = 'foggy_cityscape_multi_{}'.format(split)
    __sets[name] = (lambda split=split: foggy_cityscape_multi(split))
    tgt_name = 'foggy_cityscape_multi_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: cityscape_multi(
        split, num_shot))

for split in ['train', 'test']:
    name = 'clipart1k_{}'.format(split)
    __sets[name] = (lambda split=split, year=year: clipart1k(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
