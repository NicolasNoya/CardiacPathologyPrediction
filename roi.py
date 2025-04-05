#%%
from niidataloader import NiftiDataset

import numpy as np
from scipy.fft import fft
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt


class ROI:
    def __init__(self, path_to_images):
        """
        Args:
            path_to_images (str): Path to Test or Train folders.
        """
        self.path_to_images = path_to_images
        self.dataset = NiftiDataset(path_to_images, augment=False)

    def get_roi(self, image_idx):
        """
        Get the region of interest (ROI) from the image at the specified index.
        Args:
            image_idx (int): Index of the image in the dataset.
        Returns:
            np.ndarray: ROI extracted from the image.
        """
        # Load the image
        image = self.dataset[image_idx][0]
        mask = self.dataset[image_idx][1]
        volume = self.get_temoral_slices(image_idx)
        first_harmonic = np.transpose(self.first_harmonic_image(volume), (2,0,1))

        # apply canny edge detection
        edges = np.array([canny(fh, sigma=1.0) for fh in first_harmonic])
        # apply hough transform to find circles
        cx,cy,radius = self.circular_hough_transform(edges, (15, 30))
        roi_image = []
        roi_mask = []

        for i in range(cx.shape[0]):
            roi_image.append(self.crop_roi(image[:,:,i], (cx[i], cy[i]), 128))
            roi_mask.append(self.crop_roi(mask[:,:,i], (cx[i], cy[i]), 128))
            #Pad if the cropped image is not 128x128
            if roi_image[-1].shape != (128,128):
                pad_h = max(0, 128 - roi_image[-1].shape[0])
                pad_w = max(0, 128 - roi_image[-1].shape[1])
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                roi_image[-1] = np.pad(roi_image[-1], ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
                roi_mask[-1] = np.pad(roi_mask[-1], ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

        roi_image = np.array(roi_image)
        roi_mask = np.array(roi_mask)
        return roi_image, roi_mask


    def crop_roi(self, image, center, size=98):
        x, y = center
        half = int(size // 2)
        x1, y1 = int(max(x - half, 0)), int(max(y - half, 0))
        x2, y2 = int(x1 + size), int(y1 + size)
        return image[y1:y2, x1:x2]


    def circular_hough_transform(self, edges, radius):
        hough_radii = np.arange(radius[0], radius[1], 1)
        hough_res = np.array([hough_circle(edge, hough_radii) for edge in edges])
        cx = np.zeros(hough_res.shape[0])
        cy = np.zeros(hough_res.shape[0])
        radii = np.zeros(hough_res.shape[0])
        for i in range(hough_res.shape[0]):
            _, cx[i], cy[i], radii[i] = hough_circle_peaks(hough_res[i], hough_radii, total_num_peaks=1)
        return cx, cy, radii
 

    def get_temoral_slices(self, image_idx):
        """
        Get the temporal slices from the image data.
        Args:
        Returns:
        """
        image = self.dataset[image_idx]
        diastole_image = image[0]
        systole_image = image[2]
        volume = np.stack([diastole_image, systole_image], axis=0)
        return np.moveaxis(volume, 0, -1)




    def first_harmonic_image(self, temporal_slices):
        fft_result = fft(temporal_slices, axis=-1)
        first_harmonic = np.abs(fft_result[..., 1])
        return first_harmonic
        

        

if __name__ == "__main__":
    path = "./data/Train"
    roi_extractor = ROI(path)
    roi, mask = roi_extractor.get_roi(0)