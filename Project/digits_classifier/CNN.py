import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Pre-processing with a convolutional filter
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        
        # MLP 1 layer, 100 units : The output of the convolutional layer is reduce into a single vector of size 256 -> 100 -> 10 through the MLP
        self.fc1 = nn.Linear(256, 100)
        self.fc2 = nn.Linear(100, 10) # output 10 classes
        
        #Dropout method drop connections between nodes in the MLP during training to reduce training time and avoid over-fitting
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        
        # Pre-processing with a convolutional filter
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=4))
       
        # MLP 1 layer, 100 units
        x = self.dropout(F.relu(self.fc1(x.view(-1, 256))))
        x = self.fc2(x)

        return x