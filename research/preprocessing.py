#%%https://pdf.sciencedirectassets.com/271322/1-s2.0-S0169260722X00072/1-s2.0-S0169260722002978/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK3%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIBZFiin5aBIv6thGFV5fTVrbuWONfyE9nQJaZvXK47ElAiEAupnX4FYggcZRjHR4IVuEDw37lDnIxkR121X3a25EzBIqsgUIFRAFGgwwNTkwMDM1NDY4NjUiDJTxg8i9F%2BKOIwGdyyqPBcGMBCxWVdSswPVYHTJE2H1Wp7Y6urw1SLBNTNudm%2Fmj5qTSHpAVu%2BdgC3lxbxNLXAmlB9CDSYe3aecvMKOffgolwpBUP5%2Fhd5gVUY66xmW%2BrG0orB0yLEu9CyeBSC%2FerShzaRjV5T1jHyuSZGQddQoH512uvXZBgRkkI1LxrAifUoMYJ7V2BUsQevjDQmM5wfwV%2FBEuQAIXTfLUEd8OlM4Fgd3zieCsVyTV9lGbYRM3mCuIqAxX2xRFxe79P6bTbCCOcZqjdofokZICen6fHHeWU5KQnQyoYObadiwCm2BE4gekX2CCiLYnWUZ4e5sNcPPjTvkt2vezvqjeCAlsD%2B9p5B0U8miFxpgIM%2BHWcftD6sJRi7HJIvAYK1Bu6nW7cRNan8LfdqqrvBJn74nrezLypzFi7voimQvrk8v1AjUXdcujieSGKP1ry%2BjxASD6aQPmAu7IwjBRQcNSPiKIDrOkJeWmqbOtSEMgID0WNYx0wudfrKtmmY27JrmmsClGVLMzaCzom0MsdUI6k9djyUSmbooJRAtvS9Hl53opVc2vS4WzkM5e0%2BASFKwCs%2Fy3O0lpSCKhStHHpfS864zHcTQdXFXpyurQ0Bbqm9ygHWPel%2BvfFgE55n%2BV3LhAnDhambwAwfWlzexJidcWfCAwlNZJZqE7HJad%2BfD7AwpXtXzv8CAEFLaySP2P0Tk4z3d6kwvt6FrvtSr5kkPmCd5iyJXoteyUuQxe5A2sH5BhjImpaHQnhsvjaDoP28Qyb2CCATGueploke14%2BEYrii3fwlXdQqAOAuKyxQ4lqMOPWzIzAKRFpmKWWNHRUBxbPq4n9wTH6s73Sc0zgDu%2FBEgJ7hzO8e%2FGC8SXdH6ev4gPmGcwgLiKvwY6sQGeastj2CSqYEpHMFrqGmM%2FeJikSPGizwkEaXxRhdJMwsZf%2FmL9uZpeelVvapwH3wOMVrgBHw3vXRvY%2BJ8by0UtsuH%2FqaLDzLHrN8ACm%2FLB3D2OZojjO1cFTpGmNSV1MJqUy9oXTwxk3jlL4gaYkksi7BS40nw929ym%2FDfcm9TEd%2BO24OtNY1%2FVMsSXIc8QZ86TMb5Z0XDifE2nhuemtDGVhVnF8tS%2BGpRjJKBCa%2BcvyXQ%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250325T130247Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWDBDO3KR%2F20250325%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=1156784a68ac5d9d1245ffd8ea58d9dec83a3ad23ccd487a6f1c8557bbe44b9b&hash=4ee470563a03d3352a185ad5b5d49db029773ba9bb7e6ff3dab0b32cb33490c3&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0169260722002978&tid=spdf-450a9723-b012-4b17-96e2-da913f7cfbf1&sid=9db4a0ae58d5f848ba3b0e769a2dd20d7580gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0014565354065603040c&rr=925e9acb1d6fd15e&cc=fr
import torchvision.transforms.functional as F1
from niidataloader import NiftiDataset
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
from sklearn.cluster import KMeans
S = 0.1
THETA_SIGMA = 0.1
ALPHA = 0.1
NUMBER_OF_CLUSTERS = 5
IMG_WIDTH = 220
IMG_HEIGHT = 220

class Preprocessing:
    def __init__(self, clusters: int =20):
        """
        This class is used to extract features from the images. 
        The information needed to make this class was extracted from the article:
        https://doi.org/10.1016/j.compbiomed.2005.01.005
        """
        self.clusters = clusters
        self.alpha = ALPHA
        self.theta_sigma = THETA_SIGMA
        self.S = S
        self.number_of_clusters = NUMBER_OF_CLUSTERS


    def preprocess(self, img: torch.Tensor):
        """
        This method is used to preprocess one image of the dataset. 
        The preprocessing is based in the article with some modifications.
        
        Args:
            index (int): Index of the image in the dataset.
            alpha (float): Parameter used to control the exponential decay of the spatial adaptivity weights.
            theta_sigma (float): Parameter used to normalize the variance image.
            S (float): Parameter used to normalize the edge preservation weights
        
        Returns:
            torch.Tensor: Preprocessed image.
        """
        def __update_image__(I_updated, eta, gamma):
            """
            Iteratively updates the image according to the given formula.

            Args:
                I_t (torch.Tensor): Input image (HxW)
                eta (torch.Tensor): Spatial adaptivity weights (HxW)
                gamma (torch.Tensor): Edge preservation weights (HxW)
                num_iters (int): Number of iterations to apply the update.

            Returns:
                torch.Tensor: Updated image.
            """

            # Define a 3x3 averaging filter (excluding center pixel)
            kernel = torch.tensor([[1, 1, 1], 
                                [1, 1, 1], 
                                [1, 1, 1]], dtype=torch.float16) / 8.0  # Normalize

            kernel = kernel.view(1, 1, 3, 3)  # Shape needed for conv2d

            # Compute weighted sum of neighboring pixels
            weighted_I = F.conv2d(I_updated, kernel, padding=1).squeeze()

            # Compute sum of weights (η_ij * γ_ij)
            weight_sum = F.conv2d((eta * gamma), kernel, padding=1).squeeze()

            # Compute update term
            weight_sum = weight_sum.reshape(1,1,220,220)
            weighted_I= weighted_I.reshape(1,1,220,220)
            update_term = (eta * (weighted_I - I_updated) * gamma) / (weight_sum + 1e-8)  # Avoid division by zero

            # Update the image
            I_updated = I_updated + update_term

            return I_updated

        kernelH = torch.tensor([[0,0,0],
                               [1,0,-1],
                               [0,0,0]
                               ], dtype=torch.float16).reshape(1,1,3,3)
        kernelV = torch.tensor([[0,1,0],
                               [0,0,0],
                               [0,-1,0]
                               ], dtype=torch.float16) .reshape(1,1,3,3)
        kernelD = torch.tensor([[1,0,0],
                               [0,0,0],
                               [0,0,-1]
                               ], dtype=torch.float16).reshape(1,1,3,3)
        kernelC = torch.tensor([[0,0,1],
                                 [0,0,0],
                                 [-1,0,0]
                                 ], dtype=torch.float16).reshape(1,1,3,3)
        kernelmu = torch.tensor([[[1/25,1/25,1/25,1/25,1/25]] * 5], dtype=torch.float16).reshape(1,1,5,5)

        images = img
        output_image = []

        image = images.reshape(1,1,images.shape[0],images.shape[1])
        image_old = torch.tensor(np.zeros_like(image))
        while torch.sum(torch.abs(image_old-image)) > 1:
            image = image.reshape(1,1,images.shape[0],images.shape[1])
            image_old = copy.deepcopy(image)
            filtered_imageH = torch.abs(F.conv2d(image, kernelH, padding=1))
            filtered_imageV = torch.abs(F.conv2d(image, kernelV, padding=1))
            filtered_imageD = torch.abs(F.conv2d(image, kernelD, padding=1))
            filtered_imageC = torch.abs(F.conv2d(image, kernelC, padding=1))
            filtered_imageE = (filtered_imageC + filtered_imageD + filtered_imageH + filtered_imageV)/4

            filtered_imagemu = F.conv2d(image, kernelmu, padding=2)

            variance_image = (image-filtered_imagemu)**2
            variance_image = F.conv2d(variance_image, kernelmu, padding=2)
            variance_image_normalized = (variance_image-torch.min(variance_image))/torch.max(variance_image)
            variance_image_normalized_alleviate = variance_image_normalized/(variance_image_normalized + self.theta_sigma)

            etta = torch.exp(-self.alpha*variance_image_normalized_alleviate)
            lamda = torch.exp(-filtered_imageE/self.S)

            image=image+torch.mul(__update_image__(image, etta, lamda), etta)
        return image
    
    def create_the_clusters(self, img: torch.Tensor):
        """
        This function creates clusters in the image with the objective to segments the objects.
        """
        def reorder_labels_numpy(labels):
            # Count unique labels and their occurrences
            unique_labels, counts = np.unique(labels, return_counts=True)

            # Sort labels by frequency (descending order)
            sorted_indices = np.argsort(-counts)  # Negative for descending sort
            sorted_labels = unique_labels[sorted_indices]  # Sorted old labels

            # Create mapping: {old_label → new_label}
            label_mapping = {old: new for new, old in enumerate(sorted_labels)}

            # Apply mapping to relabel the image
            new_labels = np.vectorize(label_mapping.get)(labels)

            return torch.tensor(new_labels)  # Return relabeled image and mapping


        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=42, n_init=15)
        filtered_image = self.preprocess(img)
        filtered_image=filtered_image.reshape(-1,1)

        #Get rid of NaN values
        filtered_image[filtered_image != filtered_image] = 0
        filtered_image[filtered_image > 255] = 255
        filtered_image[filtered_image < 0] = 0
        print("The max is: ",torch.max(filtered_image))

        image_clustered = []
        cluster_list=[]
        # print("The max is: ",torch.max(filtered_image))
        kmeans.fit(filtered_image)
        cluster_list.append(kmeans.labels_)
        image_clustered.append(cluster_list)
        image_clustered = torch.tensor(image_clustered).reshape(220,220)
        # NOTE: the label value in each cluster is random, now we need to 
        # make that each label value in each image is the same for the same class
        # in ED and ES images
        # Biggest class 0, second biggest class 1, and so on
        return torch.tensor(image_clustered)
    
    def get_the_lv_cavity(self, img: torch.Tensor):
        """
        This method is used to get the left ventricular cavity.
        """
        clusters=self.create_the_clusters(img)
        # Now we have to create A
        kernel = torch.tensor([[1, 1, 1],
                                [1, -8, 1],
                                [1, 1, 1]], dtype=torch.float16) / 8.0  # Normalize
        kernel = kernel.view(1, 1, 3, 3)  # Shape needed for conv2d
        # Compute weighted sum of neighboring pixels
        for clust in clusters:
            clust=clust.to(torch.float16)
            cluster_borders = F.conv2d(clust.unsqueeze(1), kernel, padding=1)

            # Now we have to create A
            # Here I have to use float32 because the inverse function does not work with float16 and 
            # to avoid problems with inf and -inf
            A_mat = torch.nonzero(cluster_borders>0, as_tuple=False)[:,2:].to(torch.float32)
            b_vect = (A_mat*A_mat)@(torch.ones(2).to(torch.float32))
            A_mat = torch.cat((A_mat, torch.ones(A_mat.shape[0]).unsqueeze(1)), dim=1)
            optimization_output = torch.inverse(A_mat.T @ A_mat)
            A_mat = A_mat
            optimization_output= optimization_output@A_mat.T@b_vect
            optimization_output=optimization_output.to(torch.int)



#%%
if __name__=="__main__":
    preproc = Preprocessing()
    dataloader = NiftiDataset("./data/Train")
    image = dataloader[34][2,:,:,0]
    print(image.shape)
    print(torch.max(image))
    plt.imshow(image, cmap='gray')
    plt.show() 
    preprocess_image = preproc.create_the_clusters(image)
    print(torch.max(preprocess_image))
    plt.imshow(preprocess_image, cmap='gray')
    plt.show() 
    #%%
    # preprocess_image[preprocess_image != 0] = 1
    preprocess_image = 1-preprocess_image
    plt.imshow(preprocess_image, cmap='gray')
    plt.show()

    #%%
    plt.imshow(image + preprocess_image, cmap='gray')
    plt.show()
    plt.imshow(image, cmap='gray')
    plt.show()
    # filtered_image = feature_extractor.preprocessing()
    # cluster_labels = feature_extractor.create_the_clusters(1)
    #%%
    print(dataloader[0].shape[-1])

    # cluster_labels = feature_extractor.create_the_clusters(0)
    # plt.imshow(cluster_labels[0][0], cmap='gray')
    # plt.show()
    # #%%
    # feature_extractor.get_the_lv_cavity(1)
# %%
for images in range(len(dataloader)):
    img = dataloader[images]
    for j in [0,2]:
        for k in range(img.shape[-1]):
            image = img[j,:,:,k]
            image = preproc.create_the_clusters(image)
            mask = img[j+1,:,:,k]
            # Save the images
            plt.imsave(f"./process_data/image{images}.{j}.{k}.png", image.numpy(), cmap='gray')
            plt.imsave(f"./process_data/mask{images}.{j}.{k}.png", mask.numpy(), cmap='gray')