import os
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

import numpy as np
import torch
import torch.nn.functional as F

# I used the year as it was used by the researchers in the paper
np.random.seed(2025) 

###################
## Configuration ##
###################
INPUT_HEIGHT = 220 # Change this value from 128
INPUT_WIDTH = 220 # Change this value from 128
OUTPUTS = 4
LR = 0.001
EPOCHS = 20001
BATCH_SIZE = 32

class LeftCavityLocalizationModel(torch.nn.Module):
    def __init__(self):
        """
        This model is a torch implementation of the one did by the 
        authors (in tensorflow) of the article:
        https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/iet-cvi.2016.0482
        """

        # Torch model
        
        # Activation functions
        self.relu = torch.nn.ReLU()

        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling and Dropout layers
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout2 = torch.nn.Dropout(p=0.2)
        self.dropout3 = torch.nn.Dropout(p=0.3)
        self.dropoutfc = torch.nn.Dropout(p=0.5)

        # Feedforward layers
        self.fc1 = torch.nn.Linear(128*(INPUT_WIDTH//8)*(INPUT_HEIGHT//8), 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, OUTPUTS)
        

    def forward(self, x):
        """
        Forward pass of the model.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        
        x = torch.flatten(x, start_dim=1)  # Flatten for FC layer
        
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        
        x = self.fc3(x)  # No activation (for regression)
        
        return x