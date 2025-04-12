#%%
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from niidataloader import NiftiDataset
import matplotlib.pyplot as plt

from densenet.densenet import DenseNet
from densenet.custom_loss import CombinedLoss
from profiler import Profiler


class DenseNetTrainer:
    def __init__(self, path_to_images, epochs=1, alpha=0.25, train_fraction=0.8, check_val_every=10, profiling_dir="runs"):
        """
        This class is used to train the DenseNet model.
        It takes care of splitting the dataset into training and validation sets,
        and training the model using the specified loss function.
        It also handles the validation of the model and saves the best model
        based on the validation loss and dice score.
        The model is trained using the Adam optimizer.
        It uses the CombinedLoss function which is a combination of
        CrossEntropyLoss and Dice Loss.
        All the hyperparameters are set in the constructor and were tuned like in 
        the paper of reference of this project.

        Args:
            path_to_images (str): Path to Test or Train folders.
            epochs (int): Number of epochs to train the model.
            alpha (float): Weight for CrossEntropyLoss. The weight for Dice Loss is (1 - alpha).
            train_fraction (float): Fraction of the dataset to use for training.
            check_val_every (int): How often to check the validation set.
            profiling_dir (str): Directory for TensorBoard logs.
        """
        self.path_to_images = path_to_images
        self.dataset = NiftiDataset(path_to_images, augment=True, roi=True)
        self.train_size = int(train_fraction * len(self.dataset))
        self.val_size = len(self.dataset) - self.train_size
        self.criterion = CombinedLoss(alpha=alpha)
        self.generator = torch.Generator().manual_seed(2001)
        train_dataset, val_dataset = random_split(self.dataset, [self.train_size, self.val_size], generator=self.generator)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=16) #This is a batchsize of 1 because the images are 3D
        self.val_loader = DataLoader(val_dataset, batch_size=1, num_workers=16)                   # Which is more or less 10 so is equal to the papers implementation
        self.epochs = epochs
        self.best_dice_model_val = 0
        self.best_loss_model_val = 1000
        self.check_val_every = check_val_every
        self.profiler = Profiler(log_dir=profiling_dir)
    

    def train(self, criterion = None, train_loader = None, epochs = None):
        model = DenseNet()
        # Let the user tune the hyperparameters
        criterion = (self.criterion if criterion is None else criterion)
        train_loader = (self.train_loader if train_loader is None else train_loader)
        epochs = (self.epochs if epochs is None else epochs)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        losses_train = []
        dices_train = []
        losses_val = []
        dices_val = []
        for epoch in range(epochs):
            total_loss = 0
            total_dice = 0
            for images in tqdm(train_loader,  desc=f"Training: {epoch+1}/{epochs} Average Loss: {total_loss/len(train_loader):.4f} Average Dice: {total_dice/len(train_loader):.4f}"):
                x_diastole = images[0][0].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                x_systole = images[0][2].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                # Channels of the true values: 4,128,128, X
                # 4 for every element to segment, and X for the X layers of the image
                y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64), num_classes=4).permute(2,3,0,1).to(torch.float32)
                y_true_systole = F.one_hot((images[0][3]*3).to(torch.int64), num_classes=4).permute(2,3,0,1).to(torch.float32)

                # To device
                x_diastole = x_diastole.to(device)
                x_systole = x_systole.to(device)
                y_true_diastole = y_true_diastole.to(device)
                y_true_systole = y_true_systole.to(device)

                optimizer.zero_grad()
                y_pred_diastole = model(x_diastole)
                y_pred_systole = model(x_systole)

                loss = criterion(y_pred_diastole, y_true_diastole) + criterion(y_pred_systole, y_true_systole)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                dice = criterion.dice(y_pred_diastole, y_true_diastole)
                total_dice += dice
            losses_train.append(total_loss / len(train_loader))
            dices_train.append(total_dice / len(train_loader))

            if epoch%self.check_val_every == 0:
                # Profile the training
                self.profiler.profile_segmentation_triplets(x_diastole, y_true_diastole, y_pred_diastole, tag="Segmentation Diastole Validation", max_images=2)
                self.profiler.log_metric(losses_train[-1], metric_name="Train Loss", step=epoch)
                self.profiler.log_metric(dices_train[-1], metric_name="Train Dice", step=epoch)

                dice_val, los_val = self.validate(model, self.val_loader, criterion, device, epoch)
                losses_val.append(los_val)
                dices_val.append(dice_val)
                
    
    def validate(self, model, dataloader=None, criterion=None, device=None, epoch=0):
        # Let the user tune the hyperparameters
        criterion = (self.criterion if criterion is None else criterion)
        dataloader = (self.val_loader if dataloader is None else dataloader)
        device = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device)

        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_dice = 0
            for images in tqdm(dataloader,  desc=f"Validation"):
                x_diastole = images[0][0].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                x_systole = images[0][2].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
                y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64), num_classes=4).permute(2,3,0,1).to(torch.float32)
                y_true_systole = F.one_hot((images[0][3]*3).to(torch.int64), num_classes=4).permute(2,3,0,1).to(torch.float32)
 
                # To device
                x_diastole = x_diastole.to(device)
                x_systole = x_systole.to(device)
                y_true_diastole = y_true_diastole.to(device)
                y_true_systole = y_true_systole.to(device)

                y_pred_diastole = model(x_diastole)
                y_pred_systole = model(x_systole)
                loss = criterion(y_pred_diastole, y_true_diastole) + criterion(y_pred_systole, y_true_systole)
                total_loss += loss.item()
                dice = criterion.dice(y_pred_diastole, y_true_diastole)
                total_dice += dice
            

        dice_val = total_dice / len(dataloader)
        loss_val = total_loss / len(dataloader)
        print("The average dice in validation is: ", dice_val)
        print("The average loss in validation is: ", loss_val)
        # Profile the validation
        self.profiler.profile_segmentation_triplets(x_diastole, y_true_diastole, y_pred_diastole, tag="Segmentation Diastole Validation", max_images=2)
        self.profiler.log_metric(loss_val, metric_name="Val Loss", step=epoch)
        self.profiler.log_metric(dice_val, metric_name="Val Dice", step=epoch)

        if dice_val > self.best_dice_model_val:
            self.best_dice_model_val = dice_val
            print("Best dice model saved")
            torch.save(model.state_dict(), 'model_weights_best_dice_val.pth')
        if loss_val < self.best_loss_model_val:
            self.best_loss_model_val = loss_val
            print("Best loss model saved")
            torch.save(model.state_dict(), 'model_weights_best_dice_los.pth')
        
        return dice_val, loss_val
#%%
if __name__ == "__main__":
    path_to_images = "./data/Train"
    trainer = DenseNetTrainer(path_to_images, epochs=1, alpha=0.25, train_fraction=0.8, check_val_every=10)
    model = DenseNet()
    trainer.validate(model)