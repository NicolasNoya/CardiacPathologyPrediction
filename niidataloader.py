#%%
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from typing import Tuple

class NiftiDataset(Dataset):
    def __init__(self, image_path: str, csv_path: str, transform=None, augment=False):
        """
        Args:
            image_path (str): Path to Test or Train folders.
            csv_path (str): Path to CSV file containing metadata.
            transform (callable, optional): Optional transform to apply to the images.
            augment (bool): Whether to apply data augmentation.
        """
        self.image_path = image_path
        self.image_list = os.listdir(image_path)
        self.transform = transform
        self.augment = augment
        self.csv_path = csv_path
    
    def __len__(self):
        return len(self.image_list)
    
    def preprocess(self, image_data):
        """
        Apply preprocessing steps such as normalization.
        Args:
            image_data (np.ndarray): Raw image data.
        Returns:
            np.ndarray: Preprocessed image data.
        """
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data) + 1e-8)
        return image_data
    
    def augment_image(self, image_tensor):
        """
        Apply data augmentation transformations.
        Args:
            image_tensor (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Augmented image tensor.
        """
        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
        return augmentations(image_tensor)
    
    def __getitem__(self, idx):
        folder=os.path.join(self.image_path, self.image_list[idx])
        image_folder = os.listdir(folder)
        # Sort the list, the order will be ED, ED-seg, ES, ES-seg
        image_folder.sort()
        nii_image=[]

        for image in image_folder:
            image_path = os.path.join(folder, image)

            nii_img= nib.load(image_path)  # Load NIfTI image
            image_data = nii_img.get_fdata()  # Convert to numpy array
            image_data = np.asarray(image_data, dtype=np.float32)  # Ensure correct dtype
            
            # Apply preprocessing
            image_data = self.preprocess(image_data)
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image_data).unsqueeze(0)  # Add channel dimension
            
            # Apply augmentation if enabled
            if self.augment:
                image_tensor = self.augment_image(image_tensor)
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            nii_image.append(image_tensor)

        # convert nii_image to tensor
        nii_image_tensor = torch.cat(nii_image, dim=0)
        
        return nii_image_tensor
    
    def __getitem_class__(self, idx)->Tuple[torch.Tensor, str]:
        image_tensor = self.__getitem__(idx)
        image_class = pd.read_csv(self.csv_path)['Category'][idx]
        return image_tensor, image_class

if __name__=="__main__":
    # Test the NiftiDataset class
    image_paths = 'data/Train'
    csv_path = './data/metaDataTrain.csv'
    dataset = NiftiDataset(image_paths, csv_path)
    image_tensor = dataset[10]
    print(image_tensor.shape)
    image_tensor, image_class = dataset.__getitem_class__(10)
    print(image_tensor.shape, image_class)
    plt.imshow(image_tensor[3,:,:,0], cmap='gray')
    plt.show()
    