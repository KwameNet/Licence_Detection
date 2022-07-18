## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn=nn.BatchNorm2d(32)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool_2 = nn.MaxPool2d(2, 1)
        self.conv3 = nn.Conv2d(64,128, 4)
        self.conv3_bn=nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,64, 5)
        self.conv4_bn=nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64,32, 5)
        self.conv2_drop = nn.Dropout(p=0.4)
        
        """"self.fc1 = nn.Linear(32*39*39, 512)
        self.fc1_bn=nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_bn=nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 136)"""
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        self.conv_1x = self.conv1(x)
        self.conv_1bnx = self.conv1_bn(self.conv_1x)
        self.activated_1x = F.tanh(self.conv_1bnx)
        self.pooled_1x = self.pool_1(self.activated_1x)
        #x = self.pool_1(F.tanh(self.conv1_bn(self.conv1(x))))
        
        self.conv_2x = self.conv2(self.pooled_1x)
        self.activated_2x = F.relu(self.conv_2x)
        self.drop_2x = self.conv2_drop(self.activated_2x)
        self.pooled_2x = self.pool_1(self.drop_2x)
        #x = self.pool_1(F.relu(self.conv2(x)))
        
        self.conv_3x = self.conv3(self.pooled_2x)
        self.conv_3bnx = self.conv3_bn(self.conv_3x)
        self.activated_3x = F.relu(self.conv_3bnx)
        self.drop_3x = self.conv2_drop(self.activated_3x)
        self.pooled_3x = self.pool_2(self.drop_3x)
        #x = self.pool_2(self.conv2_drop(F.relu(self.conv3_bn(self.conv3(x)))))
        
        self.conv_4x = self.conv4(self.pooled_3x)
        self.conv_4bnx = self.conv4_bn(self.conv_4x)
        self.activated_4x = F.relu(self.conv_4bnx)
        self.drop_4x = self.conv2_drop(self.activated_4x)
        self.pooled_4x = self.pool_2(self.drop_4x)
        
        #x = self.pool_2(F.relu(self.conv4_bn(self.conv4(x))))
        
        self.conv_5x = self.conv5(self.pooled_4x)
        self.activated_5x = F.relu(self.conv_5x)
        #self.drop_5x = self.conv2_drop(self.activated_5x)
        #self.pooled_5x = self.pool_2(self.drop_5x)
        #x = self.pool_2(self.conv2_drop(F.relu(self.conv5(x))))
        x = self.activated_5x
        
        #x = self.conv2_drop(F.relu(self.conv2(x)))
        #x = self.pool_2(x)
        #x = F.relu(self.conv3(x))
        """"x = self.pooled_5x.view(self.pooled_5x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.tanh(self.fc2_bn(self.fc2(x)))
        x = F.tanh(self.fc3(x))"""
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x