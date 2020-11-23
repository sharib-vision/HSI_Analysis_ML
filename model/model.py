#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 16:54:50 2020

@author: sharib
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MultilabelCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultilabelCrossEntropyLoss, self).__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source = source.sigmoid()
        score = -1. * target * source.log() - (1 - target) * torch.log(1-source)
        return score.sum()
    
    
class model(nn.Module):
    def __init__(self, nlabel):
        super(model, self).__init__()
        self.conv1 = nn.Conv1d(1, 18, kernel_size=3) #1 input channels, 18 output channels
        self.conv2 = nn.Conv1d(18, 36, kernel_size=3) #18 input channels from previous Conv. layer, 36 out
        self.conv2_drop = nn.Dropout2d() #dropout
        self.conv3 = nn.Conv1d(36, 72, kernel_size=3) #18 input channels from previous Conv. layer, 36 out
        self.conv3_drop = nn.Dropout2d() #dropout
        # new data 433 ()
        self.fc1 = nn.Linear(3744, 72)
        self.fc2 = nn.Linear(72, nlabel) #Fully-connected classifier layer
    
    def forward(self, x):
        x = F.relu(F.max_pool1d(self.conv1(x), 2))
        x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)),2))
        x = F.relu(F.max_pool1d(self.conv3_drop(self.conv3(x)),2))
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict(self, x):
        with torch.no_grad():
            outp = self.forward(x)
            return F.log_softmax(outp)