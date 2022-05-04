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
#model = small_cnn_reg(in_features=1, out_features=7)
model = big_daddy(in_channels=1, n_cats=7)



def ensemble(n_models):

    ensemble_predictions = []

    model_idxs = np.random.choice(range(15), size=n_models, replace=False) #range(17)
    print(model_idxs)

    for m in model_idxs:
        
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

                model_predictions.extend(F.softmax(outputs, dim=1).detach().cpu())
                
                del inputs, labels, outputs
                gc.collect()

        ensemble_predictions.append(model_predictions)

    pred = np.asarray(ensemble_predictions)
    mean_predictions = np.mean(pred, axis=0)

    predictions = []
    for p in mean_predictions:
        predictions.append(torch.argmax(p, dim=0).item())
    

    mean_preds = np.zeros((len(predictions),7))
    for i, p in enumerate(mean_predictions):
        mean_preds[i,:] = p.detach().numpy()

    predictions = torch.Tensor(predictions)
    print(mean_preds.shape)
    print(type(mean_preds[0,0]))

    return predictions, mean_preds


def test_ensemble(n_models=2):
    predictions, mean_preds = ensemble(n_models)
    corrects = 0
    for idx, (image, label) in enumerate(val_loader):
        lab_idx = torch.argmax(label, dim=1)

        corrects += torch.sum(lab_idx == predictions[idx*batch_size:(idx+1)*batch_size])
    

    ece_ = ECE(bins=5).measure(mean_preds, targets.numpy())
    print(f'ece: {ece_}')
    accuracy = corrects/len(predictions)
    print(f'acc: {accuracy}')
    return (accuracy, ece_)
    #return accuracy

accs = []
eces = []
for i in range(15):
    results = test_ensemble(n_models=2)
    accs.append(results[0])
    eces.append(results[1])
    print(f'iteration: {i}')
#arr = np.array(accs)

stdev = np.std(accs)
mean = np.mean(accs)
print("-----ACCURACY-----")
print(f'stdev: {stdev}')
print(f'mean: {mean}')

ece_stdev = np.std(eces)
ece_mean = np.mean(eces)
print('-----ECE-----')
print(f'stdev: {ece_stdev}')
print(f'mean: {ece_mean}')