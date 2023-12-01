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
from torchvision import transforms, datasets
import torch.distributed as dist
from typing import List
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
import scipy
from scipy.fftpack import idct


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

def target_to_oh(target):
    NUM_CLASS = 12  # hard code here, can do partial
    one_hot = torch.eye(NUM_CLASS)[target]
    return one_hot

"""
Used to setup ImageDataset, load in data, and create an iterable
"""
def prepare_data(index_file, test_size=0.2):
    rgb_mean = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    rgb_std = [0.5, 0.5, 0.5]

    # series of transformations
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(), # apply some random flip for data augmentation
        transforms.ToTensor(), # convert the flipped image back to a pytorch tensor
        transforms.Normalize(mean=rgb_mean, std=rgb_std) # normalize the tensor using mean and std from above
    ])
    # setup the dataset
    data_dir = '../Data/filtered'
    # train/test split
    train_data = datasets.ImageFolder(data_dir, transform, target_transform=target_to_oh)
    # Use DataLoader to create an iteratable object
    train_data_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    return train_data_loader, len(train_data)

"""
This is where the DCT magic happens
"""
def block_idct(dct_block):
    """
    Apply Inverse Discrete Cosine Transform (IDCT) to each 8x8 block.
    Args:
    dct_block (numpy.ndarray): An array of DCT coefficients.
    Returns:
    numpy.ndarray: The reconstructed image blocks after IDCT.
    """
    # Define a function to apply IDCT to a single block
    def idct_2d(block):
        # Apply IDCT in both dimensions
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    # Assuming the input is of shape (bs, ch, h, w)
    bs, ch, h, w = dct_block.shape
    # Initialize an empty array for the output
    idct_image = np.zeros_like(dct_block, dtype=np.float32)
    # Apply IDCT to each block
    for b in range(bs):
        for c in range(ch):
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    # Extract the block
                    block = dct_block[b, c, i:i+8, j:j+8]
                    # Perform IDCT
                    idct_image[b, c, i:i+8, j:j+8] = idct_2d(block)
    return idct_image

def images_to_batch(x):

    # input has range -1 to 1, this changes to range from 0 to 255
    x = (x + 1) / 2 * 255
    print('Step 1')
    print(x.shape)

    # scale_factor=8 does the blockify magic
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    print('Step 2')
    print(x.shape)

    # raise error if # of channels does not equal 3
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return

    # convert to ycbcr
    x = dct.to_ycbcr(x)  # convert RGB to YCBCR
    print('Step 3')
    print(x.shape)

    # DCT is designed to work on values ranging from -128 to 127
    # Subtracting 128 from values 0-255 will change range to be -128 to 127
    # https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
    x -= 128
    print('Step 4')

    # assign variables batch size, channel, height, weight based on x.shape
    bs, ch, h, w = x.shape
    print('Step 5')
    print(x.shape)

    # set the number of blocks
    block_num = h // 8
    # gives you insight of the stack that is fed into the "upsampling" piece
    x = x.view(bs * ch, 1, h, w)
    print('Step 6')
    print(x.shape)

    # 8 fold upsampling
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                 stride=(8, 8))
    print('Step 7')
    print(x.shape)

    # transposed to be able to feed into dct
    x = x.transpose(1, 2)
    print('Step 8')
    print(x.shape)
    x = x.view(bs, ch, -1, 8, 8)
    print('Step 9')
    print(x.shape)

    # do dct
    dct_block = dct.block_dct(x)
    print('Step 10')
    print(dct_block.shape)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    print('Step 11')
    print(dct_block.shape)
    # remove DC as its important for visualization, but not recognition
    dct_block = dct_block[:, :, 1:, :, :]
    print('Step 12')
    print(dct_block.shape)
    # gather
    print('Before reshape')
    print(dct_block.shape)
    #dct_block = dct_block.reshape(bs, -1, block_num, block_num)

    dc_coefficient = torch.zeros(bs, ch, 1, h // 8, w // 8, device=dct_block.device)
    dct_block = torch.cat((dc_coefficient, dct_block), dim=2)
    # Reshape to the format suitable for inverse DCT
    dct_block = dct_block.permute(0, 1, 3, 4, 2).reshape(bs, ch, h, w)
    # Apply inverse DCT
    x = dct.block_idct(dct_block)  # Assuming your dct module has a block_idct function
    # Add 128 to each pixel
    x += 128
    # Convert from YCbCr to RGB
    x = dct.to_rgb(x)  # Convert YCbCr back to RGB
    # Normalize the image back to the range -1 to 1
    x = F.interpolate(x, scale_factor=1/8, mode='bilinear', align_corners=True)
    x = (x / 255) * 2 - 1

    print('Final idcted')
    print(x.shape)
    return x

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

def main(noise_accumulation, training_data, num_samples, noise_opt, running_loss, e, run_perturbed=False):
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




    #if not run_perturbed:
        #cnnModel = ImageCNN.ImageCNN(normal_image=True)
    #else:
    cnnModel = ImageCNN.ImageCNN()
    print('Printing out the model skel')
    print(cnnModel)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(cnnModel.parameters(), lr=0.001)
    train_losses = []
    val_losses = []


    # TODO:
    # Use the transforms to resize -- DONE
    # Update how we're labelling to be the one-hot encoded version -- DONE
    # Update the training portion of the model and replace it with the unperturbed images -- DONE
    # Investigate why the perturbed image has different number of channels
    # Run predictions against perturbed images
    # measure the accuracy, precision and recall


    # loop through each sample in the dataloader object
    for step, samples in enumerate(training_data):
        inputs = samples[0]
        labels = samples[1]
        print('Labels')
        print(labels.shape)

        # change to one hot encoding
        inputs = images_to_batch(inputs)
        #print(inputs.shape)

        """
        if(run_perturbed):
            inputs = images_to_batch(inputs)
            inputs = noise_accumulation(inputs)
        """

        # zero grad
        optimizer.zero_grad()
        print('Start training')
        output = cnnModel(inputs)

        loss = criterion(output, labels)

        running_loss += loss.item() * inputs.size(0)

        loss.backward()

        optimizer.step()


    epoch_train_loss = running_loss / len(training_data.dataset)
    print('Epoch {}, train loss : {}'.format(e, epoch_train_loss))
    return cnnModel

    """
    epoch_train_loss = running_loss / len(train_loader.dataset)
    print('Epoch {}, train loss : {}'.format(e, epoch_train_loss))

    train_losses.append(epoch_train_loss)
    cnnModel.eval()
    """


    #return epoch_train_loss


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
    # tirm down the
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

    training_data, num_samples = prepare_data(index_file)
    batch_sizes = [num_samples]
    input_size = [112, 112]

    print("Got the images back from the data loader")
    print(len(training_data.dataset))

    # setup noise object
    noise_accumulation = NoisyAccumulation.NoisyAccumulation(budget_mean=4)

    save_epochs = [10, 18, 22, 24]
    losses = OrderedDict()

    noise_opt = get_noise_opt(noise_accumulation)
    lr_noise = [0.1, 0.01, 0.001, 0.0001]
    stages = [10, 18, 22]
    running_loss = 0
    train_losses = []
    for epoch in range(num_epochs):
        # update noise opt if epoch matches expected
        #epoch_index = save_epochs.index(epoch)
        #if (epoch in save_epochs):
            # update the learning rate for the noise_opt
            #epoch_index = save_epochs.index(epoch)
            #adjust_lr(epoch, lr_noise, stages, noise_opt)

        #losses[epoch], model = main(noise_accumulation, training_data, num_samples, noise_opt, run_perturbed=perturbed)
        train_losses.append(main(noise_accumulation, training_data, num_samples, noise_opt, running_loss, epoch, run_perturbed=perturbed))

    print(train_losses)


    #print(losses)
    #compute_validation_score(model, training_data)



if __name__ == "__main__":
    epoch_controller()
