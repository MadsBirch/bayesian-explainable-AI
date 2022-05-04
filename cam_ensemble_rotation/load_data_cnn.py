import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories
labels = ['XR_ELBOW','XR_FINGER','XR_FOREARM','XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']

def get_data():
    data = {}
    for phase in data_cat:
        BASE_DIR = './MURA-v1.1/%s/' % (phase)
        data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for bone in labels:
            patients = list(os.walk(BASE_DIR+bone))[0][1] # list of patient folder names
            #sub_patients = patients[:100]
            for patient in tqdm(patients): # for each patient folder
                for study in os.listdir(BASE_DIR + bone + '/' + patient): # for each study in that patient folder
                    label = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
                    label[labels.index(bone)] = 1.0 # one hot encoded label
                    #label = labels.index(bone)
                    path = BASE_DIR + bone + '/' + patient + '/' + study + '/' # path to this study
                    data[phase].loc[i] = [path, len(os.listdir(path)), label] # add new row
                    i+=1
    return data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        image = pil_loader(study_path + 'image1.png')
        image = self.transform(image)
        label = self.df.iloc[idx, 2]
        return image, label

def get_dataloaders(data, batch_size=1, shuffle = True):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((256, 256)),#224 ran out of mem
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=2) for x in data_cat}
    return dataloaders

if __name__=='main':
    pass
