#%%
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import os
from niidataloader import NiftiDataset
from sklearn.cluster import KMeans

S = 0.1
THETA_SIGMA = 0.1
ALPHA = 0.1
NUMBER_OF_CLUSTERS = 4
IMAGES_IN_VOLUMEN = 8

class FeatureExtractor:
    def __init__(self, images_path:str, csv_path:str=None, clusters: int =20):
        """
        This class is used to extract features from the images. 
        The information needed to make this class was extracted from the article:
        https://doi.org/10.1016/j.compbiomed.2005.01.005
        """
        self.images_path = images_path
        self.csv_path = csv_path    
        self.dataloader = NiftiDataset(image_path=images_path, csv_path=csv_path)
        self.clusters = clusters
        self.alpha = ALPHA
        self.theta_sigma = THETA_SIGMA
        self.S = S
        self.number_of_clusters = NUMBER_OF_CLUSTERS

    def preprocessing(self, index:int):
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
            weight_sum = weight_sum.unsqueeze(1)
            weighted_I= weighted_I.unsqueeze(1)
            update_term = (eta * (weighted_I - I_updated) * gamma) / (weight_sum + 1e-8)  # Avoid division by zero

            # Update the image
            I_updated = I_updated + update_term

            return I_updated

        kernelH = torch.tensor([[0,0,0],
                               [1,0,-1],
                               [0,0,0]
                               ], dtype=torch.float16)
        kernelV = torch.tensor([[0,1,0],
                               [0,0,0],
                               [0,-1,0]
                               ], dtype=torch.float16) 
        kernelD = torch.tensor([[1,0,0],
                               [0,0,0],
                               [0,0,-1]
                               ], dtype=torch.float16)
        kernelC = torch.tensor([[0,0,1],
                                 [0,0,0],
                                 [-1,0,0]
                                 ], dtype=torch.float16)
        kernelmu = torch.tensor([[[1/25,1/25,1/25,1/25,1/25]] * 5], dtype=torch.float16)

        images = self.dataloader[index]
        output_image = []

        for image in [images[0],images[2]]: 
            image = image.permute(2, 0, 1).unsqueeze(1)
            image_old = torch.tensor(np.zeros_like(image))
            while torch.sum(torch.abs(image_old-image)) > 1:
                image_old = copy.deepcopy(image)
                filtered_imageH = torch.abs(F.conv2d(image, kernelH.unsqueeze(0).unsqueeze(0), padding=1))
                filtered_imageV = torch.abs(F.conv2d(image, kernelV.unsqueeze(0).unsqueeze(0), padding=1))
                filtered_imageD = torch.abs(F.conv2d(image, kernelD.unsqueeze(0).unsqueeze(0), padding=1))
                filtered_imageC = torch.abs(F.conv2d(image, kernelC.unsqueeze(0).unsqueeze(0), padding=1))
                filtered_imageE = (filtered_imageC + filtered_imageD + filtered_imageH + filtered_imageV)/4

                filtered_imagemu = F.conv2d(image, kernelmu.unsqueeze(0), padding=2)

                variance_image = (image-filtered_imagemu)**2
                variance_image = F.conv2d(variance_image, kernelmu.unsqueeze(0), padding=2)
                variance_image_normalized = (variance_image-torch.min(variance_image))/torch.max(variance_image)
                variance_image_normalized_alleviate = variance_image_normalized/(variance_image_normalized + self.theta_sigma)

                etta = torch.exp(-self.alpha*variance_image_normalized_alleviate)
                lamda = torch.exp(-filtered_imageE/self.S)

                image=image+torch.mul(__update_image__(image, etta, lamda), etta)
            output_image.append(image)
        return torch.stack(output_image)
    
    def create_the_clusters(self, index:int):
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
        filtered_image = self.preprocessing(index)
        print("The shape of the image is: ", filtered_image.shape)

        #Get rid of NaN values
        filtered_image[filtered_image != filtered_image] = 0
        image_clustered = []
        for img in range(2):
            cluster_list=[]
            for i in range(IMAGES_IN_VOLUMEN):
                filtered_image[img,i,0].unsqueeze(2)
                N,M=filtered_image[img,i,0].shape
                kmeans.fit(filtered_image[img,i,0].reshape(-1,1))
                cluster_list.append(kmeans.labels_.reshape(N,M))
            image_clustered.append(cluster_list)
        image_clustered = torch.tensor(image_clustered)
        print("The shape of the image clustered is: ", image_clustered.shape)
        # NOTE: the label value in each cluster is random, now we need to 
        # make that each label value in each image is the same for the same class
        # in ED and ES images
        # Biggest class 0, second biggest class 1, and so on
        for img in range(2):
            for i in range(IMAGES_IN_VOLUMEN):
                image_clustered[img][i] = reorder_labels_numpy(image_clustered[img][i])
        return torch.tensor(image_clustered)
    
    def get_the_lv_cavity(self, index:int):
        """
        This method is used to get the left ventricular cavity.
        """
        clusters=self.create_the_clusters(index)
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
            print(optimization_output)



#%%
if __name__=="__main__":
    feature_extractor = FeatureExtractor(images_path="data/Train", csv_path="./data/metaDataTest.csv")
    filtered_image = feature_extractor.preprocessing(1)
    cluster_labels = feature_extractor.create_the_clusters(1)
    #%%
    cluster_labels = feature_extractor.create_the_clusters(0)
    plt.imshow(cluster_labels[0][0], cmap='gray')
    plt.show()
    #%%
    feature_extractor.get_the_lv_cavity(1)