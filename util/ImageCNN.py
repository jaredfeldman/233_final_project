
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np


class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()

        # input image size 3, 224, 224
        self.conv1 = nn.Conv2d(189, 64, 3, (3, 3), 1, 1)
        self.batch1 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        # image size 8, 112, 112
        self.conv2 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.batch2 = nn.BatchNorm2d(64, affine=True, track_running_stats=True)

        # image size 16, 56, 56
        self.conv3 = nn.Conv2d(64, 128, 4, stride = 1, padding = 1)
        self.batch3 = nn.BatchNorm2d(128, affine=True, track_running_stats=True)

        self.conv4 = nn.Conv2d(128, 256, 6, stride = 1, padding = 1)
        self.batch4 = nn.BatchNorm2d(256, affine=True, track_running_stats=True)

        self.pool = nn.MaxPool2d(2, 2)

        # input size 32, 28, 28
        self.fc1 = nn.Linear(64 * 14 * 14, 4096)
        self.fc2 = nn.Linear(4096, 512)
        self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 2)

        self.batch5 = nn.BatchNorm1d(512)
        self.batch6 = nn.BatchNorm1d(64)

        self.drop = nn.Dropout(p=.3)

    def forward(self, x):
        print('In forward')
        print('Getting image size')
        print(x.shape)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool(self.batch1(x))

        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool(self.batch2(x))
        x = self.drop(x)

        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool(self.batch3(x))
        x = self.drop(x)
        print('Before conv4')
        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool(self.batch4(x))
        print('After conv4')
        x = x.view(-1, 64 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.batch5(x)
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.batch6(x)
        x = self.drop(x)
        x = self.fc4(x)

        return x
