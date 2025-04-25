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
from densnet_trainer import DenseNetTrainer

device = torch.device("cuda")
model = DenseNet()
model.load_state_dict(torch.load("/home/onyxia/work/project/CardiacPathologyPrediction/model_weights_best_dice_val0.7920951843261719.pth", weights_only=True)["model_state_dict"])
model.to(device)
model.eval()

path_to_images = "./data/Train"
trainer = DenseNetTrainer(path_to_images, epochs=200, alpha=0.25, train_fraction=0.8, check_val_every=10)
#%%
for images in trainer.train_loader:
    image = images[0][0].unsqueeze(0).permute(3, 0, 1, 2).to(torch.float32)
    print(image.shape)
    image = image.to(device)
    y_pred = model(image)
    # print(y_pred.size())
    # plt.imshow(image[0][0], cmap='grey')
    # plt.show()
    # y_true_diastole = F.one_hot((images[0][1]*3).to(torch.int64), num_classes=4).permute(2,3,0,1).to(torch.float32)
    # plt.imshow(y_true_diastole[0][0], cmap='grey')
    # plt.imshow(y_pred.detach().numpy()[0][1], cmap='grey')
    plt.imshow(y_pred.detach().cpu().numpy()[0][1], cmap='gray')
    # plt.show()
    break
    # plt.imshow(y_true_diastole[0][1].cpu(), cmap='grey')
    # plt.show()
#%%
plt.imshow(image[0][0].cpu(), cmap='grey')