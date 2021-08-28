# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import csv
import numpy as np
import torch.nn as nn
import torch
import os.path as osp

def make_dir(dataset):
    if dataset == 'NTU':
        output_dir = os.path.join('./results/NTU/')
    elif dataset == 'NTU120':
        output_dir = os.path.join('./results/NTU120/')
    elif dataset == 'UAV':
        output_dir = os.path.join('./results/UAV/')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir

def get_num_classes(dataset):
    if dataset == 'NTU':
        return 60
    elif dataset == 'NTU120':
        return 120
    elif dataset == 'UAV':
        return 155

    
