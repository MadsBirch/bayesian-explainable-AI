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

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(data, batch_size=1)
dataset_sizes = {x: len(data[x]) for x in data_cat}

# deifine data loaders
train_loader = dataloaders['train']
val_loader = dataloaders['valid']
targets = torch.cat([y.argmax(-1) for x, y in val_loader], dim=0).cpu()

# #### Build model
#MAP_model = small_cnn_reg(in_features=1, out_features=7)
MAP_model = big_daddy(in_channels=1, n_cats=7)

def get_model_metrics(n_iter = 15, hessian_structure = 'full'):
    print(f'Hessian structure: {hessian_structure}')
    METRICS_DICT={
    'MAP' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    'LA' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    } 
    
    LA_time = []
    # criterion
    ACC_LIST = np.zeros([2,len(np.arange(n_iter))])
    ECE_LIST = np.zeros([2,len(np.arange(n_iter))])
    F1_LIST = np.zeros([2,len(np.arange(n_iter))])

    for i in range(n_iter):
        print(f'ITER: {i+1}')
        print('-' * 10)

        # LOAD pretrained MAP model  
        PATH = './models_lap/model_' + str(i) + '.pth'
        MAP_model.load_state_dict(torch.load(PATH))
        MAP_model.cuda().eval()

        # LA with full Hessian 
        la_start = time.time()
        LA_model = Laplace(MAP_model, 'classification',
                subset_of_weights='last_layer',
                hessian_structure=hessian_structure)
        LA_model.fit(train_loader)
        LA_model.optimize_prior_precision(method='marglik')
        la_compute = time.time() - la_start
        LA_time.append(la_compute)
        print(f'Finished LA in: {la_compute}')

        model_list = [MAP_model, LA_model]
    
        for j, m in enumerate(model_list):
            print(f'Test approx. #: {j+1}')
            running_corrects = 0
            predictions = []
            F1s = []

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, total=len(val_loader)):
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                    outputs = m(inputs)
                    predictions.append(F.softmax(outputs, dim = 1))
                    lab_idx = torch.argmax(labels, dim=1).cuda()
                    #loss = criterion(outputs, lab_idx)

                    preds = torch.argmax(F.softmax(outputs, dim = 1),dim=1)
                    running_corrects += torch.sum(preds == lab_idx)
                    F1s.append(f1_score(lab_idx.detach().cpu().numpy(), preds.detach().cpu().numpy(), average = 'weighted'))
                    del inputs, labels, outputs
                    #gc.collect()

            predictions = torch.cat(predictions).cpu()
            ACC_LIST[j,i] = (running_corrects/len(val_loader)).detach().cpu().numpy()
            ECE_LIST[j,i] = ECE(bins=15).measure(predictions.detach().numpy(), targets.numpy())
            F1_LIST[j,i] = np.mean(F1s)

    ### WRITE MEAN ACC, ECE & TIME TO DICTIONARY ###
    METRICS_DICT['MAP']['mean_ACC'] = np.mean(ACC_LIST[0,:])
    METRICS_DICT['LA']['mean_ACC'] = np.mean(ACC_LIST[1,:])

    METRICS_DICT['MAP']['mean_F1'] = np.mean(F1_LIST[0,:])
    METRICS_DICT['LA']['mean_F1'] = np.mean(F1_LIST[1,:])

    METRICS_DICT['MAP']['mean_ECE'] = np.mean(ECE_LIST[0,:])
    METRICS_DICT['LA']['mean_ECE'] = np.mean(ECE_LIST[1,:])

    METRICS_DICT['MAP']['mean_time'] = 0
    METRICS_DICT['LA']['mean_time'] = np.mean(LA_time)

    ### WRITE STD OF ACC, ECE & TIME TO DICTIONARY ###
    METRICS_DICT['MAP']['sd_ACC'] = np.std(ACC_LIST[0,:])
    METRICS_DICT['LA']['sd_ACC'] = np.std(ACC_LIST[1,:])

    METRICS_DICT['MAP']['sd_F1'] = np.std(F1_LIST[0,:])
    METRICS_DICT['LA']['sd_F1'] = np.std(F1_LIST[1,:])

    METRICS_DICT['MAP']['sd_ECE'] = np.std(ECE_LIST[0,:])
    METRICS_DICT['LA']['sd_ECE'] = np.std(ECE_LIST[1,:])

    METRICS_DICT['MAP']['sd_time'] = 0
    METRICS_DICT['LA']['sd_time'] = np.std(LA_time)

    return METRICS_DICT

hessian_structures = ['full']

for h in hessian_structures:
    METRICS_DICT = get_model_metrics(n_iter = 15, hessian_structure = h)

    # SAVE
    with open('./dicts/' + h +'_BIG_DICT.pkl', 'wb') as pkl_handle:
	    pickle.dump(METRICS_DICT, pkl_handle)

