import torch.nn as nn

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        DenseNet layer with Batch Normalization, ELU activation, 
        Convolution, and Dropout.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels. This is the growth rate.
        """
        super(Layer, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True), # Exponential ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.2)
        )
    
    def forward(self, x):
        x = self.block(x)
        return x

