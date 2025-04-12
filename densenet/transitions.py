import torch.nn as nn

class TransitionDown(nn.Module):
    """
    Transition down block for DenseNet.
    This is the downsampling used in the first half of the network.
    The block downsamples the input by a factor of 2 using MaxPooling.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionDown, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True), # Exponential ReLU
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsamples by 2
        )
    
    def forward(self, x):
        return self.block(x)


class TransitionUp(nn.Module):
    """
    Transition up block for DenseNet.
    This is the upsampling used in the second half of the network.
    The block upsamples the input by a factor of 2 using ConvTranspose.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.convtrans = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2,
            padding=1,
            # not extremely happy with this output padding
            # but it has to be there because otherwise the 
            # output size will be necesarily a odd number
            # according to the formula
            # Hout​=(Hin​−1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1
            # source: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html#torch.nn.ConvTranspose2d
            output_padding=1, 
            )  # Upsamples by 2
    
    def forward(self, x):
        return self.convtrans(x)

