import torch
import torch.nn as nn
from layer import Layer



class DenseBlock(nn.Module):
    """
    Dense block for DenseNet.
    This block consists of multiple layers where each layer's output is concatenated
    to the input of the next layer.
    This class was developed following the paper:
    "The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation"
    and the reference paper of this project.
    Args:
        in_channels (int): Number of input channels.
        num_layers (int): Number of layers in the dense block.
        growth_rate (int): Growth rate for the dense block.
    """
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Layer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        outputs = []
        for layer in self.block:
            output = layer(x)
            outputs.append(output)
            x = torch.cat([x, output], dim=1)  # Concatenate along channel axis
        
        # implementation from the model found in the paper:  https://arxiv.org/pdf/1611.09326
        output = torch.cat(outputs, dim=1)  # Concatenate all outputs
        return output


class InceptionX(nn.Module):
    """
    InceptionX block with three branches of different kernel sizes.
    This is the first block of the DenseNet model.
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(InceptionX, self).__init__()
        # Each branch with a different padding to keep the size of the output
        self.branch_3x3 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.branch_5x5 = nn.Conv2d(in_channels, 4, kernel_size=5, padding=2, bias=False)
        self.branch_7x7 = nn.Conv2d(in_channels, 4, kernel_size=7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(24)

    def forward(self, x):
        out_3x3 = self.branch_3x3(x)
        out_5x5 = self.branch_5x5(x)
        out_7x7 = self.branch_7x7(x)
        out = torch.cat([out_3x3, out_5x5, out_7x7], dim=1)  # concatenate along channel axis
        return self.bn(out)