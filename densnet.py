#%%
# The denseNet model implementation in PyTorch is based on the paper:
# Densely Connected Fully Convolutional Network for Short-Axis Cardiac Cine MR Image Segmentation and Heart Diagnosis Using Random Forest 
# https://link.springer.com/chapter/10.1007/978-3-319-75541-0_15#Tab3
from tqdm import tqdm
import torch
import torch.nn as nn
from roi import ROI
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from niidataloader import NiftiDataset
import matplotlib.pyplot as plt


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


class TransitionDown(nn.Module):
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


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        """
        Args:
            in_channels (int): Number of input channels.
            num_layers (int): Number of layers in the dense block.
            growth_rate (int): Growth rate for the dense block.
        """
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
        
        # implementation from https://arxiv.org/pdf/1611.09326
        output = torch.cat(outputs, dim=1)  # Concatenate all outputs
        return output


class InceptionX(nn.Module):
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


class DenseNet(nn.Module):
    def __init__(self):
        """
        This is the DenseNet model for image segmentation based on the paper:
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


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-8):  # alpha balances the two losses
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        loss_ce = self.ce(preds, targets)
        loss_dice = self.dice(preds, targets)
        return self.alpha * loss_ce + (1 - self.alpha) * loss_dice

    def dice(self, preds, targets):
        total_dice = 0
        for i in range(preds.shape[0]):
            intersection = (preds[i] == targets[i]).sum().item()
            union = preds.sum().item() + targets.sum().item()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_dice += dice
        dice = total_dice / preds.shape[0]  # Average over batch
        return 1 - dice  # Dice Loss = 1 - Dice coefficient


class DenseNetTrainer:
    def __init__(self, path_to_images, epochs=1, alpha=0.5, train_fraction=0.8, check_val_every=10):
        """
        Args:
            path_to_images (str): Path to Test or Train folders.
        """
        self.path_to_images = path_to_images
        self.dataset = NiftiDataset(path_to_images, augment=True, roi=True)
        self.train_size = int(train_fraction * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.criterion = CombinedLoss(alpha=alpha)
        self.generator = torch.Generator().manual_seed(2001)
        train_dataset, val_dataset = random_split(self.dataset, [self.train_size, self.val_size], generator=self.generator)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=1)
        self.epochs = epochs
        self.best_dice_model_val = 0
        self.best_loss_model_val = 1000
        self.check_val_every = check_val_every
    
    def train(self, criterion = None, train_loader = None, epochs = None):
        model = DenseNet()
        # Let the user tune the hyperparameters
        criterion = (self.criterion if criterion is None else criterion)
        # generator = (self.generator if generator is None else generator)
        train_loader = (self.train_loader if train_loader is None else train_loader)
        epochs = (self.epochs if epochs is None else epochs)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # TODO: Check the hyperparameters in the paper
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        total_loss = 0
        losses_train = []
        dices_train = []
        losses_val = []
        dices_val = []
        total_dice = 0
        print("starts")
        for epoch in range(epochs):
            for images in tqdm(train_loader,  desc=f"Training{epoch+1}/{epochs}"):
                x_diastole = images[0][0].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                x_systole = images[0][2].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                print("The shape of the images is: ", x_diastole.shape)
                # Channels of the true values: 4,128,128,7 
                # 4 for every element to segment, and 7 for the 7 layers of the image
                y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64), num_classes=4).permute(3,0,1,2).to(torch.float32)
                y_true_systole = F.one_hot((images[0][3]*3).to(torch.int64), num_classes=4).permute(3,0,1,2).to(torch.float32)
                # To device
                x_diastole = x_diastole.to(device)
                x_systole = x_systole.to(device)
                y_true_diastole = y_true_diastole.to(device)
                y_true_systole = y_true_systole.to(device)

                optimizer.zero_grad()
                y_pred_diastole = model(x_diastole).permute(1,2,3,0)
                y_pred_systole = model(x_systole).permute(1,2,3,0)
                loss = criterion(y_pred_diastole, y_true_diastole) + criterion(y_pred_systole, y_true_systole)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                dice = 1 - criterion.dice(y_pred_diastole, y_true_diastole)
                total_dice += dice
            print("The average dice is: ", total_dice / len(train_loader))
            print("The average loss is: ", total_loss / len(train_loader))
            losses_train.append(total_loss / len(train_loader))
            dices_train.append(total_dice / len(train_loader))
            if epochs%self.check_val_every == 0:
                self.validate(model, self.val_loader, criterion, device)
    
    def validate(self, model, dataloader, criterion, device):
        model.eval()
        total_loss = 0
        total_dice = 0
        with torch.no_grad():
            for images in tqdm(dataloader,  desc=f"Validation"):
                x_diastole = images[0][0].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                x_systole = images[0][2].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                print("The shape of the images is: ", x_diastole.shape)
                # Channels of the true values: 4,128,128,7 
                # 4 for every element to segment, and 7 for the 7 layers of the image
                # y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64)).permute(3,0,1,2).to(torch.float32)
                # y_true_systole = F.one_hot((images[0][3]*3).to(torch.int64)).permute(3,0,1,2).to(torch.float32)
                y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64), num_classes=4).permute(3,0,1,2).to(torch.float32)
                y_true_systole = F.one_hot((images[0][3]*3).to(torch.int64), num_classes=4).permute(3,0,1,2).to(torch.float32)
 
                # To device
                x_diastole = x_diastole.to(device)
                x_systole = x_systole.to(device)
                y_true_diastole = y_true_diastole.to(device)
                y_true_systole = y_true_systole.to(device)

                y_pred_diastole = model(x_diastole).permute(1,2,3,0)
                y_pred_systole = model(x_systole).permute(1,2,3,0)
                loss = criterion(y_pred_diastole, y_true_diastole) + criterion(y_pred_systole, y_true_systole)
                total_loss += loss.item()
                dice = 1 - criterion.dice(y_pred_diastole, y_true_diastole)
                total_dice += dice
        dice_val = total_dice / len(dataloader)
        loss_val = total_loss / len(dataloader)
        print("The average dice is: ", dice_val)
        print("The average loss is: ", loss_val)

        if dice_val > self.best_dice_model_val:
            self.best_dice_model_val = dice_val
            print("Best dice model saved")
            torch.save(model.state_dict(), 'model_weights_best_dice_val.pth')
        if loss_val < self.best_loss_model_val:
            self.best_loss_model_val = loss_val
            print("Best loss model saved")
            torch.save(model.state_dict(), 'model_weights_best_dice_los.pth')

#%%
if __name__ == "__main__":
    path_to_images = "./data/Train"
    trainer = DenseNetTrainer(path_to_images, epochs=1, alpha=0.5, train_fraction=0.8, check_val_every=10)
    trainer.train()
    # trainer.validate(trainer.model, trainer.val_loader, trainer.criterion, device)