import time
import copy
import pandas as pd
import torch
from torch.autograd import Variable
from cnn import Net, small_cnn, small_cnn_reg, big_daddy
from utils import plot_training, n_p, get_count
from train_cnn import train_model, get_metrics
from load_data_cnn import get_data, get_dataloaders

# #### load study level dict data
data = get_data()

# #### Create dataloaders pipeline
data_cat = ['train', 'valid'] # data categories
dataloaders = get_dataloaders(data, batch_size=128)
dataset_sizes = {x: len(data[x]) for x in data_cat}

# #### Build model
model = small_cnn_reg(in_features=1, out_features=7)
model = model.cuda()

n_iter = 10
print(f'Training small_cnn_reg...')
for i in range(n_iter):
    print(f'ITER: {i+1}')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay= 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    # #### Train model
    model = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=25)
    PATH = './ensemble/small_model_2_' + str(i) + '.pth'
    torch.save(model.state_dict(), PATH)