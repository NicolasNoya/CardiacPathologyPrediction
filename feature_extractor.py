# This class will extract the features presented in the reference paper of this project.
# Also some other features might be ingcluded from other sources.
import torch
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, diastole_image: torch.Tensor, systole_image: torch.Tensor, height, weight, voxel_spacing) -> torch.Tensor:
        """
        This function takes as an input two 4D (B, C, H, W) tensor corresponding to 
        the diastolic and systolic phases of the heart cycle. And extracts 
        the characteristics of the heart. The output will be a tensor of shape the 
        number of features to be extracted. 

        Args:
            image (torch.Tensor): The input image tensor of shape (B, C, H, W), the permited values 
            are 0 or 1.
                B is the slices of the heart.
                C is the number of masks (background, left ventricle, right ventricle, myocardium).
                
            height float: is the height of the patiend.
            weight float: is the weight of the patiend. 
            
        Returns:
            features (torch.Tensor): The output tensor of shape (N) where N is the number of 
            features to be extracted.
        """
        if not (torch.all((diastole_image == 0) | (diastole_image == 1))): 
                # or not torch.all((systole_image == 0) | (systole_image == 1))):
            raise ValueError("The input image tensor must be binary (0 or 1).")
        
        spacing_x, spacing_y, spacing_z = voxel_spacing
        voxel_volume = spacing_x * spacing_y * spacing_z

        lv_diastole = diastole_image[:, 1, :, :]
        lr_diastole = diastole_image[:, 2, :, :]
        myocardium_diastole = diastole_image[:, 3, :, :]

        lv_systole = systole_image[:, 1, :, :]
        lr_systole = systole_image[:, 2, :, :]
        myocardium_systole = systole_image[:, 3, :, :]

        # Extract the features from the diastolic and systolic images
        lv_ejection_fraction = self.compute_ejection_fraction(lv_diastole, lv_systole, voxel_volume)
        lr_ejection_fraction = self.compute_ejection_fraction(lr_diastole, lr_systole, voxel_volume)
        
        lv_vol_diastole = (lv_diastole==1).sum() * voxel_volume
        lv_vol_systole = (lv_systole==1).sum() * voxel_volume

        lr_vol_diastole = (lr_diastole==1).sum() * voxel_volume
        lr_vol_systole = (lr_systole==1).sum() * voxel_volume

        ratio_lv_lr_ed = lr_vol_diastole/(lv_vol_diastole+1e-6)
        ratio_lv_lr_es = lr_vol_systole/(lv_vol_systole+1e-6)

        # The paper uses the myocardium mass but in
        # this case is just the volume times the density 
        # which will be constant for every patient so 
        # is the same as using only the volume.
        myo_vol_diastole = (myocardium_diastole==1).sum() * voxel_volume
        myo_vol_systole = (myocardium_systole==1).sum() * voxel_volume

        ratio_myo_lv_ed = myo_vol_diastole/(lv_vol_diastole+1e-6)
        ratio_myo_lv_es = myo_vol_systole/(lv_vol_systole+1e-6)

        output_tensor = np.array([
            # lv_vol_diastole,
            # lv_vol_systole,
            lv_ejection_fraction,
            # lr_vol_diastole,
            # lr_vol_systole,
            lr_ejection_fraction,
            myo_vol_diastole,
            myo_vol_systole,
            ratio_lv_lr_ed,
            ratio_lv_lr_es,
            ratio_myo_lv_ed,
            ratio_myo_lv_es,
            # height,
            # weight,
        ])  
        return output_tensor
    
    def compute_ejection_fraction(self, diastole_masks, systole_masks, voxel_volume)-> torch.Tensor:
        """
        This function computes the ejection fraction of the heart, it can be used 
        for the left ventricle as well as for the right ventricle.
        The formula to compute the ejection fraction was found: https://my.clevelandclinic.org/health/articles/16950-ejection-fraction
        
        Args:
            diastole_masks (torch.Tensor): The input image tensor of shape (B, H, W). 
            systole_masks (torch.Tensor): The input image tensor of shape (B, H, W). 
        
        Returns:
            ejection_fraction (torch.Tensor): The output tensor of shape (1).
        
        Note: The formula used is: EF = (SV/EDV) x 100 = ((EDV - ESV)/EDV) x 100
        Where:
            SV = Stroke Volume
            EDV = End Diastolic Volume
            ESV = End Systolic Volume
        """
        sv = (diastole_masks==1).sum() * voxel_volume - (systole_masks==1).sum() * voxel_volume
        edv = (diastole_masks==1).sum() * voxel_volume
        return (sv/edv)*100