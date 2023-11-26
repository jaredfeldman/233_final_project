# Import the libraries we'll use below.
import numpy as np
import argparse
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import torch
from PIL import Image
import os
from IPython.display import display, HTML
import cv2
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn.init as init

from PIL import Image, ImageEnhance

import tensorflow as tf
from tensorflow import keras
from keras import metrics
tf.get_logger().setLevel('INFO')
from util import ImageDataset, NoisyAccumulation, IRSEModel, AllGather, ArcFace, AverageMeter, DistCrossEntropy
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchjpeg import dct
from sklearn import preprocessing


"""
This file contains the majority of logic that handles:
- image/data loading
- dct transformation
- recognition model
- validation
"""

"""
Define the required inputs
"""
def arg_parse():
    parser = argparse.ArgumentParser(description='Train IRSE model with perturbed images based on DCT')
    parser.add_argument('--index-file', help='index file containing labels and image paths', required=True)

    args = parser.parse_args()
    return args

"""
Used to setup ImageDataset, load in data, and create an iterable
"""
def prepare_data(index_file):
    rgb_mean = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    rgb_std = [0.5, 0.5, 0.5]

    # series of transformations
    transform = transforms.Compose([
        transforms.ToPILImage(), # convert image to PIL for easier matplotlib reading
        transforms.RandomHorizontalFlip(), # apply some random flip for data augmentation
        transforms.ToTensor(), # convert the flipped image back to a pytorch tensor
        transforms.Normalize(mean=rgb_mean, std=rgb_std) # normalize the tensor using mean and std from above
    ])
    # setup the dataset
    image_dataset = ImageDataset.ImageDataset(index_file, transform)
    # read in the data
    image_dataset.build_inputs()

    # Use DataLoader to create an iteratable object
    print('Read in a total of ' + str(image_dataset.sample_nums) + ' samples.')
    image_data_loader = DataLoader(image_dataset, image_dataset.sample_nums, drop_last=False)
    return image_data_loader, image_dataset.sample_nums

"""
This is where the DCT magic happens
"""
def images_to_batch(x):
    # input has range -1 to 1, this changes to range from 0 to 255
    x = (x + 1) / 2 * 255

    # scale_factor=8 does the blockify magic
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

    # raise error if # of channels does not equal 3
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return

    # convert to ycbcr
    x = dct.to_ycbcr(x)  # convert RGB to YCBCR

    # DCT is designed to work on values ranging from -128 to 127
    # Subtracting 128 from values 0-255 will change range to be -128 to 127
    # https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
    x -= 128

    # assign variables batch size, channel, height, weight based on x.shape
    bs, ch, h, w = x.shape

    # set the number of blocks
    block_num = h // 8
    # gives you insight of the stack that is fed into the "upsampling" piece
    x = x.view(bs * ch, 1, h, w)

    # 8 fold upsampling
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                 stride=(8, 8))

    # transposed to be able to feed into dct
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)

    # do dct
    dct_block = dct.block_dct(x)

    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)

    # remove DC as its important for visualization, but not recognition
    dct_block = dct_block[:, :, 1:, :, :]

    # gather
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)
    return dct_block

def init_process_group():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)


def get_class_split(num_classes, num_gpus):
    """ split the num of classes by num of gpus
    """
    class_split = []
    for i in range(num_gpus):
        _class_num = num_classes // num_gpus
        if i < (num_classes % num_gpus):
            _class_num += 1
        class_split.append(_class_num)
    return class_split

def setup_arc_face(sample_nums):
    metric = ArcFace.ArcFace
    class_num = sample_nums
    class_shard = get_class_split(class_num, 1)
    embedding_size = 512
    init_value = torch.FloatTensor(embedding_size, class_num)
    init.normal_(init_value, std=0.01)
    head = metric(in_features=embedding_size,
                          gpu_index=0,
                          weight_init=init_value,
                          class_split=class_shard,
                          scale=64,
                          margin=0.4)

    return head, class_shard

def setup_loss():
    loss = DistCrossEntropy.DistCrossEntropy()
    return loss

def accuracy_dist(outputs, labels, class_split, topk=(1,5)):
    """ Computes the precision@k for the specified values of k in parallel
    """
    assert 1 == len(class_split), \
        "world size should equal to the number of class split"
    base = sum(class_split[:1])
    maxk = max(topk)

    # add each gpu part max index by base
    scores, preds = outputs.topk(maxk, 0, True, True)
    preds += base

    batch_size = labels.size(0)

    # all_gather
    scores_gather = [torch.zeros_like(scores)
                     for _ in range(1)]
    dist.all_gather(scores_gather, scores)
    preds_gather = [torch.zeros_like(preds) for _ in range(1)]
    dist.all_gather(preds_gather, preds)
    # stack
    _scores = torch.cat(scores_gather, dim=0)
    _preds = torch.cat(preds_gather, dim=0)

    _, idx = _scores.topk(maxk, 0, True, True)
    pred = torch.gather(_preds, dim=0, index=idx)
    pred = pred.t()
    print('in hereeee')
    print(labels.shape)
    print(pred.shape)
    correct = pred.eq(labels.expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def main():
    # setup the arguments required for this script
    args = arg_parse()
    init_process_group()
    index_file = args.index_file
    input_size = [112, 112]

    training_data, num_samples = prepare_data(index_file)
    batch_sizes = [num_samples]

    # setup noise object
    noise_accumulation = NoisyAccumulation.NoisyAccumulation(budget_mean=4)
    noise_accumulation.train()

    # setup IR model
    irse_model = IRSEModel.IRSEModel(input_size, 50, 'ir')

    # setup arcface
    head, class_shard = setup_arc_face(num_samples)
    head = [head]

    # loop through each sample in the dataloader object
    for step, samples in enumerate(training_data):
        inputs = samples[0]
        labels_arr = samples[1]
        le = preprocessing.LabelEncoder()
        targets = le.fit_transform(labels_arr)
        targets = torch.as_tensor(targets)
        labels = targets

        inputs = images_to_batch(inputs)
        inputs = noise_accumulation(inputs)

        features = irse_model(inputs)

        features_gather = AllGather.AllGather(features, 1)
        features_gather = [torch.split(x, batch_sizes) for x in features_gather] # set to the number of images incoming
        all_features = []
        for i in range(len(batch_sizes)):
            all_features.append(torch.cat([x[i] for x in features_gather], dim=0))

        with torch.no_grad():
            labels_gather = AllGather.AllGather(labels, 1)

        labels_gather = [torch.split(x, batch_sizes) for x in labels_gather]
        all_labels = []
        for i in range(len(batch_sizes)):
            all_labels.append(torch.cat([x[i] for x in labels_gather], dim=0))

        losses = []

        am_losses = [AverageMeter.AverageMeter() for _ in batch_sizes]
        am_top1s = [AverageMeter.AverageMeter() for _ in batch_sizes]
        am_top5s = [AverageMeter.AverageMeter() for _ in batch_sizes]

        for i in range(len(batch_sizes)):
            outputs, labels, original_outputs = head[i](all_features[i], all_labels[i])

            loss = setup_loss()
            loss = loss(outputs, labels)
            losses.append(loss)
            prec1 = accuracy_dist(original_outputs.data, all_labels[i], class_shard, topk=(1,num_samples))
            am_losses[i].update(loss.data.item(), all_features[i].size(0))

        # update summary and log_buffer
        scalars = {
            'train/loss': am_losses
        }

        total_loss = sum(losses)
        print(total_loss)


if __name__ == "__main__":
    main()
