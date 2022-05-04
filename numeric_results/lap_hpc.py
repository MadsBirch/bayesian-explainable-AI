import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dists
import numpy as np
from laplace import Laplace

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from cnn import Net, small_cnn, small_cnn_reg
from utils import plot_training, n_p, get_count
from train_cnn import train_model, get_metrics
from load_data_cnn import get_data, get_dataloaders
import itertools
from netcal.metrics import ECE
import pickle 

import copy
from torchnet import meter
from utils import plot_training

from sklearn.metrics import f1_score

#### load study level dict data
data = get_data()

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(data, batch_size=128)
dataset_sizes = {x: len(data[x]) for x in data_cat}

# deifine data loaders
train_loader = dataloaders['train']
val_loader = dataloaders['valid']
targets = torch.cat([y.argmax(-1) for x, y in val_loader], dim=0).cpu()

# #### Build model
model = small_cnn_reg(in_channels=1, n_cats=7)
model = model.cuda()

def get_model_metrics(n_iter = 10):

    # criterion
    ACC_LIST = np.zeros([4,len(np.arange(n_iter))])
    ECE_LIST = np.zeros([4,len(np.arange(n_iter))])
    F1_LIST = np.zeros([4,len(np.arange(n_iter))])

    # time lists
    train_times = []
    full_times = []
    kron_times = []
    diag_times = []

    # dictioary
    METRICS_DICT={
    'MAP_model' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    'LA_full' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    'LA_kron' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    'LA_diag' : {'mean_ACC': [], 'sd_ACC': [], 'mean_F1': [], 'sd_F1': [], 'mean_ECE': [],'sd_ECE': [],'mean_time': [], 'sd_time': []},
    }   
    

    for i in range(n_iter):
        print(f'ITER: {i+1}')
        print('-' * 10)

        # LOAD pretrained MAP model  
        MAP_model = small_cnn_reg(in_channels=1, n_cats = 7)
        PATH = './ensemble/model_' + str(i) + '.pth'
        MAP_model.load_state_dict(torch.load(PATH))
        MAP_model.cuda().eval()
        
        # LA with full Hessian 
        full_start = time.time()
        LA_full = Laplace(MAP_model, 'classification',
                subset_of_weights='last_layer',
                hessian_structure='full')
        LA_full.fit(train_loader)
        LA_full.optimize_prior_precision(method='marglik')
        full_time = time.time() - full_start
        full_times.append(full_time)
        print(f'Finished LA with full Hessian in: {full_time}')

        # LA with kron Hessian
        kron_start = time.time()
        LA_kron = Laplace(MAP_model, 'classification',
                    subset_of_weights='last_layer',
                    hessian_structure='kron')
        LA_kron.fit(train_loader)
        LA_kron.optimize_prior_precision(method='marglik')
        kron_time = time.time() - kron_start
        kron_times.append(kron_time)
        print(f'Finished LA with kron Hessian in: {kron_time}')

        # LA with diagonal Hessian
        diag_start = time.time()
        LA_diag = Laplace(MAP_model, 'classification',
                    subset_of_weights='last_layer',
                    hessian_structure='diag')
        LA_diag.fit(train_loader)
        LA_diag.optimize_prior_precision(method='marglik')
        diag_time = time.time() - diag_start
        diag_times.append(diag_time)
        print(f'Finished LA with diag Hessian in: {diag_time}')

        model_list = [MAP_model, LA_full, LA_kron, LA_diag]
        model_names = ['MAP_model', 'LA_full', 'LA_kron', 'LA_diag']
    
        print(f'Testing all approximations...')
        for j, (m, n) in enumerate(zip(model_list, model_names)):
            print(f'Test approx. #: {j+1}')
            running_corrects = 0
            predictions = []
            F1s = []
            for inputs, labels in tqdm(val_loader, total=len(val_loader)):

                outputs = m(inputs.cuda())
                predictions.append(F.softmax(outputs, dim = 1))
                lab_idx = torch.argmax(labels, dim=1).cuda()
                #loss = criterion(outputs, lab_idx)

                preds = torch.argmax(F.softmax(outputs, dim = 1),dim=1)
                running_corrects += torch.sum(preds == lab_idx)
                F1s.append(f1_score(lab_idx.detach().cpu().numpy(), preds.detach().cpu().numpy(), average = 'weighted'))

            predictions = torch.cat(predictions).cpu()
            ACC_LIST[j,i] = (running_corrects/len(val_loader)).detach().cpu().numpy()
            ECE_LIST[j,i] = ECE(bins=15).measure(predictions.detach().numpy(), targets.numpy())
            F1_LIST[j,i] = F1s

    ### WRITE MEAN ACC, ECE & TIME TO DICTIONARY ###
    METRICS_DICT['MAP_model']['mean_ACC'] = np.mean(ACC_LIST[0,:])
    METRICS_DICT['LA_full']['mean_ACC'] = np.mean(ACC_LIST[1,:])
    METRICS_DICT['LA_kron']['mean_ACC'] = np.mean(ACC_LIST[2,:])
    METRICS_DICT['LA_diag']['mean_ACC'] = np.mean(ACC_LIST[3,:])

    METRICS_DICT['MAP_model']['mean_F1'] = np.mean(F1_LIST[0,:])
    METRICS_DICT['LA_full']['mean_F1'] = np.mean(F1_LIST[1,:])
    METRICS_DICT['LA_kron']['mean_F1'] = np.mean(F1_LIST[2,:])
    METRICS_DICT['LA_diag']['mean_F1'] = np.mean(F1_LIST[3,:])


    METRICS_DICT['MAP_model']['mean_ECE'] = np.mean(ECE_LIST[0,:])
    METRICS_DICT['LA_full']['mean_ECE'] = np.mean(ECE_LIST[1,:])
    METRICS_DICT['LA_kron']['mean_ECE'] = np.mean(ECE_LIST[2,:])
    METRICS_DICT['LA_diag']['mean_ECE'] = np.mean(ECE_LIST[3,:])

    METRICS_DICT['MAP_model']['mean_time'] = np.mean(train_times)
    METRICS_DICT['LA_full']['mean_time'] = np.mean(full_times)
    METRICS_DICT['LA_kron']['mean_time'] = np.mean(kron_times)
    METRICS_DICT['LA_diag']['mean_time'] = np.mean(diag_times)

    ### WRITE STD OF ACC, ECE & TIME TO DICTIONARY ###
    METRICS_DICT['MAP_model']['sd_ACC'] = np.std(ACC_LIST[0,:])
    METRICS_DICT['LA_full']['sd_ACC'] = np.std(ACC_LIST[1,:])
    METRICS_DICT['LA_kron']['sd_ACC'] = np.std(ACC_LIST[2,:])
    METRICS_DICT['LA_diag']['sd_ACC'] = np.std(ACC_LIST[3,:])

    METRICS_DICT['MAP_model']['sd_F1'] = np.std(F1_LIST[0,:])
    METRICS_DICT['LA_full']['sd_F1'] = np.std(F1_LIST[1,:])
    METRICS_DICT['LA_kron']['sd_F1'] = np.std(F1_LIST[2,:])
    METRICS_DICT['LA_diag']['sd_F1'] = np.std(F1_LIST[3,:])

    METRICS_DICT['MAP_model']['sd_ECE'] = np.std(ECE_LIST[0,:])
    METRICS_DICT['LA_full']['sd_ECE'] = np.std(ECE_LIST[1,:])
    METRICS_DICT['LA_kron']['sd_ECE'] = np.std(ECE_LIST[2,:])
    METRICS_DICT['LA_diag']['sd_ECE'] = np.std(ECE_LIST[3,:])

    METRICS_DICT['MAP_model']['sd_time'] = np.std(train_times)
    METRICS_DICT['LA_full']['sd_time'] = np.std(full_times)
    METRICS_DICT['LA_kron']['sd_time'] = np.std(kron_times)
    METRICS_DICT['LA_diag']['sd_time'] = np.std(diag_times)

    return METRICS_DICT

METRICS_DICT = get_model_metrics(n_iter = 10)

# SAVE
with open("METRICS_DICT.pkl", "wb") as pkl_handle:
	pickle.dump(METRICS_DICT, pkl_handle)