#%%
import matplotlib.pyplot as plt
import torch
import nibabel as nib
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
from typing import Tuple

class NiftiDataset(Dataset):
    def __init__(self, image_path: str, csv_path: str=None, augment=False):
        """
        Args:
            image_path (str): Path to Test or Train folders.
            csv_path (str): Path to CSV file containing metadata.
            transform (callable, optional): Optional transform to apply to the images.
            augment (bool): Whether to apply data augmentation.
        """
        self.image_path = image_path
        self.image_list = os.listdir(image_path)
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
    
    def augment_image(self, image_tensor, angle, hflip, sheare_angle, shifts):
        """
        Apply data augmentation transformations.
        Args:
            image_tensor (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Augmented image tensor.
        """
        image_tensor = image_tensor.permute(0, 3, 1, 2)
        image_tensor = image_tensor.to(torch.float32)
        image_tensor = TF.hflip(image_tensor) if hflip else image_tensor
        image_tensor = TF.affine(image_tensor, 
                                 angle=angle, 
                                 translate=(shifts[0], shifts[1]), 
                                 scale=1, 
                                 shear=sheare_angle
                                 )
        image_tensor = image_tensor.permute(0, 2, 3, 1)
        image_tensor = image_tensor.to(torch.float16)
        return image_tensor
    
    def __getitem__(self, idx):
        folder=os.path.join(self.image_path, self.image_list[idx])
        image_folder = os.listdir(folder)
        # Sort the list, the order will be ED, ED-seg, ES, ES-seg
        image_folder.sort()
        nii_image=[]
        if self.augment:
            #define random augmentations for the whole image
            angle = torch.randint(0, 360, (1,)).item()  # Random rotation
            hflip = torch.rand(1).item() > 0.5  # Random horizontal flip
            sheare_angle = torch.randint(-20, 20, (1,)).item()  # Random shear angle
            shift = (torch.randint(-30, 30, (1,)).item() , torch.randint(-10, 10, (1,)).item())  # Random shift


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
                image_tensor = self.augment_image(
                        image_tensor, 
                        angle, 
                        hflip, 
                        sheare_angle, 
                        shift,
                    )
            
            
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
    image_paths = 'data/Train'
    csv_path = './data/metaDataTrain.csv'
    dataset = NiftiDataset(image_paths, csv_path, augment=True)
    for i in range(10):
        image_tensor = dataset[0]
        # plt.imshow(image_tensor[0, :, :,0]+image_tensor[1,:,:,0], cmap='gray')
        # plt.show()