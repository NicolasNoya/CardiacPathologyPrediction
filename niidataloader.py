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
    def __init__(self, image_path: str, csv_path: str=None, transform=None, augment=False):
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
        Apply preprocessing step, normalization.
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
            
            # Change the size of each element of the tensor to make it faster 
            image_tensor = image_tensor.to(torch.float16)

            
            # Apply augmentation if enabled
            if self.augment:
                image_tensor = self.augment_image(image_tensor)
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            nii_image.append(image_tensor)

        # convert nii_image to tensor
        nii_image_tensor = torch.cat(nii_image, dim=0)

        # Padd the image to make it the same size for all images
        nii_image_tensor = torch.nn.functional.pad(nii_image_tensor, (0, 0, 0, 180, 0, 180), "constant", 0)
        # Trim the image to make it the same size for all images
        nii_image_tensor = nii_image_tensor[:, :220, :220]
        
        return nii_image_tensor
    
    def __getitem_class__(self, idx)->Tuple[torch.Tensor, str]:
        if self.csv_path is None:
            raise ValueError("CSV path is not provided.")
        else:
            image_tensor = self.__getitem__(idx)
            image_class = pd.read_csv(self.csv_path)['Category'][idx]
            return image_tensor, image_class

if __name__=="__main__":
    # Test the NiftiDataset class
    image_paths = 'data/imagesTr'
    csv_path = './data/metaDataTrain.csv'
    dataset = NiftiDataset(image_paths, csv_path)
    # image_tensor = dataset[0]
    height = []
    width = []
    for three_d_img in dataset:
        for index, img in enumerate(three_d_img):
            height.append(img.shape[1])
            width.append(img.shape[0])
    
    height = list(set(height))
    width = list(set(width))
    print("The width of the image is: ", width)
    print("The height of the image is: ", height)

    #%%
    for image in dataset:
        plt.imshow(image[0,:,:,5], cmap='gray')
        plt.show()
    img = dataset[0]
    print("The number of images in the dataset is: ", img[0,:,:,1].shape)
    plt.imshow(img[2,:,:,5], cmap='gray')
    plt.show()