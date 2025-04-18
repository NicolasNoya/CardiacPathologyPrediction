# The denseNet model implementation in PyTorch is based on the paper:
# Densely Connected Fully Convolutional Network for Short-Axis Cardiac Cine MR Image Segmentation and Heart Diagnosis Using Random Forest 
# https://link.springer.com/chapter/10.1007/978-3-319-75541-0_15#Tab3
import torch
import torch.nn as nn
import sys

sys.path.append("./densenet")  # Add the parent directory to the path
from dense_block import DenseBlock, InceptionX
from transitions import TransitionDown, TransitionUp


class DenseNet(nn.Module):
    def __init__(self):
        """
        This is the DenseNet model for image segmentation based on the paper:
        "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
        and the reference paper of this project.
        The layers are organized as follows:
        - Inception_X
        - Dense Block (3 layers)
        - Transition Down
        - Dense Block (4 layers)
        - Transition Down
        - Dense Block (5 layers)
        - Transition Down
        - Bottleneck
        - Transition Up
        - Dense Block (5 layers)
        - Transition Up
        - Dense Block (4 layers)
        - Transition Up
        - Dense Block (3 layers)
        - 1x1 convolution
        - softmax activation
        """
        super(DenseNet, self).__init__()
        growth_rate = 8

        self.inception=InceptionX(1) # output channels = 24
        self.downdense1=DenseBlock(24, 3, growth_rate=growth_rate) # output channels = 24
        self.td1=TransitionDown(48, 48)
        self.downdense2=DenseBlock(48, 4, growth_rate=growth_rate) #output channels = 32
        self.td2=TransitionDown(80, 80)
        self.downdense3=DenseBlock(80, 5, growth_rate=growth_rate) # output channels = 40
        self.td3=TransitionDown(120, 120)
        self.bottleneck=DenseBlock(120, 8, growth_rate=7) # Bottleneck output channels = 56
        self.tu1=TransitionUp(56, 56)
        self.updense1=DenseBlock(176, 5, growth_rate=growth_rate) # output channels = 40
        self.tu2=TransitionUp(40, 40)
        self.updense2=DenseBlock(120, 4, growth_rate=growth_rate) # output channels = 32
        self.tu3=TransitionUp(32, 32)
        self.updense3=DenseBlock(80, 3, growth_rate=growth_rate) # output channels = 24
        self.finalconv=nn.Conv2d(24, out_channels=4, kernel_size=1) # output channels = 4
        # softmax activation
        self.softmax = nn.Softmax(dim=1) # output channels = 4
    
    def forward(self, x):
        x = self.inception(x) # size 128x128
        x1 = self.downdense1(x)
        x11 = torch.cat([x, x1], dim=1) # channels = 48
        x12 = self.td1(x11) 
        x2 = self.downdense2(x12)
        x21 = torch.cat([x12, x2], dim=1) # channels = 56
        x22 = self.td2(x21)
        x3 = self.downdense3(x22)
        x31 = torch.cat([x22, x3], dim=1) # channels = 120
        x32 = self.td3(x31)
        x4 = self.bottleneck(x32)
        x42 = self.tu1(x4)
        x43 = torch.cat([x31, x42], dim=1) 
        x44 = self.updense1(x43)
        x45 = self.tu2(x44)
        x46 = torch.cat([x21, x45], dim=1)
        x47 = self.updense2(x46)
        x48 = self.tu3(x47)
        x49 = torch.cat([x11, x48], dim=1)
        x5 = self.updense3(x49)
        x51 = self.finalconv(x5)
        x52 = self.softmax(x51)
        return x52

    def load_model(self, model_path):
        """
        Load the model weights from a file.
        Args:
            model_path (str): Path to the model weights file.
        """
        self.load_state_dict(torch.load(model_path))
    


   
