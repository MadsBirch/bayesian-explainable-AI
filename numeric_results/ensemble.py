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
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from cnn import Net, big_daddy, small_cnn, small_cnn_reg
from utils import plot_training, n_p, get_count
from train_cnn import train_model, get_metrics
from load_data_cnn import get_data, get_dataloaders
import itertools
from netcal.metrics import ECE
import pickle 
from sklearn.metrics import f1_score
import gc

#### load study level dict data
data = get_data()

batch_size = 128

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(data, batch_size=batch_size, shuffle=False)
dataset_sizes = {x: len(data[x]) for x in data_cat}

# deifine data loaders
train_loader = dataloaders['train']
val_loader = dataloaders['valid']
targets = torch.cat([y.argmax(-1) for x, y in val_loader], dim=0).cpu()

# #### Build model
#MAP_model = small_cnn_reg(in_features=1, out_features=7)
model = big_daddy(in_channels=1, n_cats=7)



def ensemble(n_models):

    #ensemble_predictions = np.zeros((n_models, len(val_loader), 7))
    #ensemble_predictions = torch.zeros(n_models, len(val_loader), 7)
    ensemble_predictions = []

    for m in range(n_models):

        # LOAD pretrained MAP model  
        PATH = './models_lap/model_' + str(m) + '.pth'
        model.load_state_dict(torch.load(PATH))
        model.cuda().eval()

        model_predictions = []

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                outputs = model(inputs)

                model_predictions.extend(F.softmax(outputs, dim = 1).detach().cpu())
                
                del inputs, labels, outputs
                gc.collect()

        # stack predeictions
        #model_predictions = torch.Tensor(model_predictions)
        #model_predictions = np.asarray(model_predictions)
        #print(model_predictions.shape)
        #ensemble_predictions = torch.cat((ensemble_predictions, model_predictions),dim = 0)
        ensemble_predictions.append(model_predictions)
    #print(model_predictions[0,:])
    print(model_predictions[:2])
    mean_predictions = np.mean(ensemble_predictions, axis=1)
    
    predictions = np.argmax(mean_predictions, axis=0)

    return predictions


def test_ensemble(n_models=2):
    predictions = ensemble(n_models)
    corrects = 0
    for idx, (image, label) in enumerate(val_loader):
        lab_idx = torch.argmax(label, dim=1)

        corrects += torch.sum(lab_idx == predictions[idx*batch_size:(idx+1)*batch_size])
    
    accuracy = corrects/len(predictions)
    print(accuracy)
    return accuracy


test_ensemble(n_models=2)