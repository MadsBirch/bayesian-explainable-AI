import time
import copy
import torch
import torch.nn.functional as F
from torchnet import meter
from torch.autograd import Variable
from utils import plot_training
from sklearn.metrics import f1_score
import numpy as np

data_cat = ['train', 'valid'] # data categories

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    F1s =  {x:[] for x in data_cat}
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(7, normalized=True) 
                            for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            F1 = []
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):

                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)

                #outputs = torch.mean(outputs,dim=0).unsqueeze(0)
                lab_idx = torch.argmax(labels, dim=1).cuda()
                loss = criterion(outputs, lab_idx)
                running_loss += loss.data
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                #preds = outputs.type(torch.cuda.HalfTensor)#Float
                #preds = preds.view(1)
                preds = torch.argmax(F.softmax(outputs, dim=1),dim=1).cuda() #
                running_corrects += torch.sum(preds == lab_idx)
                F1.append(f1_score(lab_idx.detach().cpu().numpy(), preds.detach().cpu().numpy(), average = 'weighted'))
            
                #confusion_matrix[phase].add(preds.detach(), lab_idx.detach())
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            epoch_F1 = np.mean(F1)
            costs[phase].append(epoch_loss.detach().cpu())
            accs[phase].append(epoch_acc.detach().cpu())
            F1s[phase].append(epoch_F1)
            print(f'{phase} Loss: {epoch_loss:.2} Acc: {epoch_acc:.2} F1: {epoch_F1:.2}')
            #print('Confusion Meter:\n', confusion_matrix[phase].value())
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    
    
    plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_metrics(model, criterion, dataloaders, dataset_sizes, phase='valid'):
    '''
    Loops over phase (train or valid) set to determine acc, loss and 
    confusion meter of the model.
    '''
    confusion_matrix = meter.ConfusionMeter(2, normalized=True)
    running_loss = 0.0
    running_corrects = 0
    preds = []
    for i, data in enumerate(dataloaders[phase]):
        print(i, end='\r')
        inputs, labels = data
        # forward
        outputs = model(inputs)
        lab_idx = torch.argmax(labels, dim=1).cuda()
        loss = criterion(outputs, lab_idx)
        # statistics
        running_loss += loss * inputs.size(0) #.data[0]
        #preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        #preds = outputs.type(torch.cuda.HalfTensor)#Float
        #preds = preds.view(1)
        running_corrects += torch.sum(outputs == lab_idx)
        confusion_matrix.add(outputs, lab_idx)


        outputs = model(inputs.cuda())
        preds.append(F.softmax(outputs, dim = 1))
        lab_idx = torch.argmax(labels, dim=1).cuda()
        #loss = criterion(outputs, lab_idx)

        preds = torch.argmax(F.softmax(outputs, dim = 1),dim=1)
        running_corrects += torch.sum(preds == lab_idx)

    loss = running_loss / dataset_sizes[phase]
    acc = running_corrects / dataset_sizes[phase]
    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, loss, acc))
    #print('Confusion Meter:\n', confusion_matrix.value())