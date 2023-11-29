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
import torch.optim as optim
import torch.cpu.amp as amp
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from ignite.metrics import Accuracy, Precision, Recall
import torch.nn as nn
from PIL import Image, ImageEnhance

import tensorflow as tf
from tensorflow import keras
from keras import metrics
tf.get_logger().setLevel('INFO')
from util import ImageDataset, NoisyAccumulation, IRSEModel, AllGather, ArcFace, AverageMeter, DistCrossEntropy, ImageCNN
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchjpeg import dct
from sklearn import preprocessing
from ignite.utils import to_onehot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


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
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train the model based on')
    parser.add_argument('--run-perturbed', default=False, action='store_true', help='Run training model with perturbed images')
    parser.add_argument('--run-clean', default=False, action='store_true', help='Run the training model with normal images')

    args = parser.parse_args()
    return args

"""
Used to setup ImageDataset, load in data, and create an iterable
"""
def prepare_data(index_file, test_size=0.2):
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

    # train/test split
    train_idx, val_idx = train_test_split(list(range(len(image_dataset))), test_size=test_size)
    train_dataset = Subset(image_dataset, train_idx)
    validation_dataset = Subset(image_dataset, val_idx)

    # Use DataLoader to create an iteratable object
    print('Read in a total of ' + str(len(train_idx) + len(val_idx)) + ' samples.')
    train_data_loader = DataLoader(train_dataset, len(train_idx), drop_last=True)
    test_data_loader = DataLoader(train_dataset, len(val_idx), drop_last=True)
    return train_data_loader, test_data_loader, len(train_idx)

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

def expand_tensor(x):
    # scale_factor=8 does the blockify magic
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

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

    x = x.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)

    # gather
    x = x.reshape(bs, -1, block_num, block_num)
    return x

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

    del init_value

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
    correct = pred.eq(labels.expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def separate_resnet_bn_paras(modules):
    """ sepeated bn params and wo-bn params
    """
    all_parameters = modules.parameters()
    paras_only_bn = []

    for pname, param in modules.named_parameters():
        if pname.find('bn') >= 0:
            paras_only_bn.append(param)

    paras_only_bn_id = list(map(id, paras_only_bn))
    paras_wo_bn = list(filter(lambda p: id(p) not in paras_only_bn_id,
                              all_parameters))

    return paras_only_bn, paras_wo_bn

def get_optimizer(backbone, heads):
    """ build optimizers
    """
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
    learning_rates = [0.1, 0.01, 0.001, 0.0001] # divide by 10
    init_lr = learning_rates[0]
    weight_decay = 0.0005
    momentum = 0.9
    backbone_opt = optim.SGD([
        {'params': backbone_paras_wo_bn, 'weight_decay': weight_decay},
        {'params': backbone_paras_only_bn}], lr=init_lr, momentum=momentum)

    head_opts = OrderedDict()
    for name, head in heads.items():
        opt = optim.SGD([{'params': head.parameters()}], lr=init_lr, momentum=momentum, weight_decay=weight_decay)
        head_opts[name] = opt

    optimizer = {
        'backbone': backbone_opt,
        'heads': head_opts,
    }
    return optimizer

def get_noise_opt(noise_model):
    lrs_noise = [0.1, 0.01, 0.001, 0.0001]
    optimizer = optim.Adam(list(noise_model.parameters()), lr=lrs_noise[0])
    return optimizer

def main(irse_model, noise_accumulation, training_data, num_samples, noise_opt, run_perturbed=False):
    # setup the arguments required for this script
    #args = arg_parse()
    #init_process_group()
    #index_file = args.index_file
    #input_size = [112, 112]

    #training_data, num_samples = prepare_data(index_file)
    batch_sizes = [num_samples]

    # setup noise object
    #noise_accumulation = NoisyAccumulation.NoisyAccumulation(budget_mean=4)
    noise_accumulation.train()

    # setup IR model
    #irse_model = IRSEModel.IRSEModel(input_size, 50, 'ir')

    # setup arcface
    heads = OrderedDict()
    head, class_shard = setup_arc_face(num_samples)
    heads[0] = head


    optimizer = get_optimizer(irse_model, heads)
    noise_opt = get_noise_opt(noise_accumulation)

    backbone_opt, head_opts, noise_opt = optimizer['backbone'], list(optimizer['heads'].values()), noise_opt

    label_encoder = LabelEncoder()

    if not run_perturbed:
        cnnModel = ImageCNN.ImageCNN(normal_image=True)
    else:
        cnnModel = ImageCNN.ImageCNN()
    print('Printing out the model skel')
    print(cnnModel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnnModel.parameters(), lr=0.001)
    train_losses = []
    val_losses = []

    # loop through each sample in the dataloader object
    for step, samples in enumerate(training_data):
        inputs = samples[0]
        labels_arr = samples[1]

        # change to one hot encoding
        label_encoder.fit(labels_arr)
        labels_encoded = label_encoder.transform(labels_arr)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = labels_encoded.reshape(len(labels_encoded), 1)
        labels = onehot_encoder.fit_transform(integer_encoded)

        labels = torch.from_numpy(labels)

        if(run_perturbed):
            inputs = images_to_batch(inputs)
            inputs = noise_accumulation(inputs)

        # zero grad
        optimizer.zero_grad()

        output = cnnModel(inputs)

        loss = criterion(output, labels)

        running_loss += loss.item() * images.size(0)

        loss.backward()

        optimizer.step()

    epoch_train_loss = running_loss / len(train_loader.dataset)
    print('Epoch {}, train loss : {}'.format(e, epoch_train_loss))

    train_losses.append(epoch_train_loss)
    cnnModel.eval()



    return total_loss, cnnModel

def set_optimizer_lr(optimizer, lr):
    if isinstance(optimizer, dict):
        backbone_opt, head_opts = optimizer['backbone'], optimizer['heads']
        for param_group in backbone_opt.param_groups:
            param_group['lr'] = lr
        for _, head_opt in head_opts.items():
            for param_group in head_opt.param_groups:
                param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def adjust_lr(epoch, learning_rates, stages, optimizer):
    """ Decay the learning rate based on schedule
    """

    pos = bisect(stages, epoch)
    lr = learning_rates[pos]
    logging.info("Current epoch {}, learning rate {}".format(epoch + 1, lr))

    set_optimizer_lr(optimizer, lr)

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    print('in thresholdddd')
    print(y_pred)
    y_ohe = to_onehot(y, num_classes=y_pred.shape[1])
    return y_pred, y_ohe

def compute_validation_score(model , test_dat):
    label_encoder = LabelEncoder()
    with torch.no_grad():
        binary_accuracy = Accuracy(output_transform=thresholded_output_transform)
        precision = Precision(output_transform=thresholded_output_transform)
        recall = Recall(output_transform=thresholded_output_transform)
        for step, samples in enumerate(test_dat):
            inputs = samples[0]
            # grab the y_true label
            labels = samples[1]
            label_encoder.fit(labels)
            labels = label_encoder.transform(labels)

            # determine the y_pred
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            labels = torch.from_numpy(labels)
            print('about to get the accuracies')
            print(predicted)
            print(labels)
            binary_accuracy.update((predicted, labels))
            precision.update((predicted, labels))
            recall.update((predicted, labels))
        print('Model accuracy : ', binary_accuracy.compute())
        print('Model Precision : ', precision.compute().item())
        print('Model Recall : ', recall.compute().item())



def epoch_controller():
    # setup models and activation models required for model training and image generation
    args = arg_parse()
    init_process_group()
    index_file = args.index_file
    num_epochs = args.epochs
    run_perturbed = args.run_perturbed
    run_clean = args.run_clean

    # decide if we want to run with perturbed images
    if run_perturbed:
        perturbed = True
    else:
        perturbed = False

    training_data, test_data, num_samples = prepare_data(index_file)
    batch_sizes = [num_samples]
    input_size = [112, 112]

    # setup noise object
    noise_accumulation = NoisyAccumulation.NoisyAccumulation(budget_mean=4)

    # setup IR model
    if run_clean:
        irse_model = IRSEModel.IRSEModel(input_size, 50, 'ir', normal_image=True)
    else:
        irse_model = IRSEModel.IRSEModel(input_size, 50, 'ir')
    save_epochs = [10, 18, 22, 24]
    losses = OrderedDict()

    noise_opt = get_noise_opt(noise_accumulation)
    lr_noise = [0.1, 0.01, 0.001, 0.0001]
    stages = [10, 18, 22]
    for epoch in range(num_epochs):
        # update noise opt if epoch matches expected
        #epoch_index = save_epochs.index(epoch)
        if (epoch in save_epochs):
            # update the learning rate for the noise_opt
            epoch_index = save_epochs.index(epoch)
            adjust_lr(epoch, lr_noise, stages, noise_opt)

        losses[epoch], model = main(irse_model, noise_accumulation, training_data, num_samples, noise_opt, run_perturbed=perturbed)


    print(losses)
    compute_validation_score(model, test_data)



if __name__ == "__main__":
    epoch_controller()
