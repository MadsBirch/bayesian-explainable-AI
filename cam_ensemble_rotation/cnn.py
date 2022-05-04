import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.conv1 = nn.Conv2d(in_features, in_features*2, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_features*2, in_features*4, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_features*4, in_features*8, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_features*8, in_features*4, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_features*4, in_features*2, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_features*2, in_features, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features*128*128, out_features) #224*224
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = torch.flatten(out, 1) # Flatten all dimensions except batch
        #out = out.view(-1,out.size(1))
        #out = out.view(out.size(0),-1)
        #out = F.relu(self.fc1(out))
        out = self.fc1(out)

        # Get probabilities
        #out = F.softmax(out,0)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class small_cnn(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*4, in_features*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*8, in_features*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*4, in_features*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features*256*256, out_features), 
            #nn.Linear(in_features*128*128, out_features), 
            #nn.Linear(in_features*128*128, out_features)
        )
        
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1) 
        out = self.linear(out)

        #out = F.softmax(out,0)
        return out

class big_daddy(nn.Module):
    def __init__(self, in_channels, n_cats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_channels*32*32, 2048),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_cats)
        )
        
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1) 
        out = self.linear(out)

        #out = F.softmax(out,0)
        return out

class small_cnn_reg(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*2),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*4),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*4, in_features*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*8),
            nn.MaxPool2d(2,2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*8, in_features*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*4, in_features*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features*2, in_features, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

        self.linear = nn.Sequential(
            nn.Linear(in_features*32*32, 512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_features)
        )
        
    
    def forward(self, x):
        out = self.conv(x)
        out = torch.flatten(out, 1) 
        out = self.linear(out)

        #out = F.softmax(out,0)
        return out