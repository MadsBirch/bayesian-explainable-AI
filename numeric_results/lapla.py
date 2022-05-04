import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.distributions as dists
import numpy as np
from laplace import Laplace

import time
import copy
import pandas as pd
import torch
from torch.autograd import Variable
from cnn import Net, small_cnn, small_cnn_reg
from utils import plot_training, n_p, get_count
from train_cnn import train_model, get_metrics
from load_data_cnn import get_data, get_dataloaders

# #### load study level dict data
data = get_data()

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(data, batch_size=128)
dataset_sizes = {x: len(data[x]) for x in data_cat}

# deifine data loaders
train_loader = dataloaders['train']
val_loader = dataloaders['valid']

# #### Build model
model = small_cnn_reg(in_features=1, out_features=7)
model.load_state_dict(torch.load('./models/model.pth'))
model.cuda().eval()

# Laplace
la = Laplace(model, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='kron')
la.fit(train_loader)
print(la)
#la.optimize_prior_precision(method='marglik')
#print(la(testloader))