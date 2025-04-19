import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF
import pandas as pd

CLASSES = [0, 1, 2, 3, 4]

class Profiler:
    def __init__(self, path_to_csv:str="data/metaDataTrain.csv", log_dir:str="runs"):
        """
        Class for visualizing data distribution, confusion matrix, etc., using TensorBoard.
        Args:
            log_dir (str): Path for TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir)
        self.path_to_csv = path_to_csv

    
    def check_labels_distribution(self):
        """Checks and logs class distribution as a histogram."""
        df = pd.read_csv(self.path_to_csv)
        fig, ax = plt.subplots()
        ax.hist(df["Category"], bins=len(CLASSES), edgecolor='black', alpha=0.75)
        ax.set_xticks(CLASSES)
        ax.set_xticklabels(CLASSES, rotation=45)
        ax.set_title("Class Distribution Histogram")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Count")
        
        # Log to TensorBoard
        self.writer.add_figure("Class Distribution Histogram", fig)
        plt.close(fig)
    
    def check_confusion_matrix(self, y_true, y_pred):
        """
        Computes and logs the confusion matrix as a heatmap.
        
        Args:
            y_true (array-like): True class labels.
            y_pred (array-like): Predicted class labels.
        """
        cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
        
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        # Log to TensorBoard
        self.writer.add_figure("Confusion Matrix", fig)
        plt.close(fig)
    
    def profile_segmentation_triplets(self, images, gt_masks, pred_masks, tag="Segmentation Triplets", max_images=2):
        """
        Logs a grid of triplets (image, ground truth mask, predicted mask) to TensorBoard.
        It logs three images, one for the original image, the next for the ground thruth mask,
        and the last for the predicted mask. It will choose a random layer of the 3D tensor
        to display for each mask.

        Args:
            images (Tensor): Tensor of shape (N, 1, H, W) — input images.
            gt_masks (Tensor): Tensor of shape (N, C, H, W) — ground truth masks.
            pred_masks (Tensor): Tensor of shape (N, C, H, W) — predicted masks.
            tag (str): TensorBoard tag.
            max_images (int): Number of triplets to log.
        """

        images = images[:max_images]
        gt_masks = gt_masks[:max_images]
        pred_masks = pred_masks[:max_images]

        num_samples = images.size(0)
        fig, axs = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))

        if num_samples == 1:
            axs = axs.unsqueeze(0)  # make it iterable if only one sample

        titles = ["Original Image", "Ground Truth Mask", "Predicted Mask"]

        for i in range(num_samples):
            img = TF.to_pil_image(images[i].cpu())
            # Get a random mask between 0 and 3
            mask_index = torch.randint(0, gt_masks[i].shape[0], (1,)).item()
            gt = gt_masks[i][mask_index].cpu().numpy()
            # pred = pred_masks[i][mask_index].cpu().numpy()
            pred = pred_masks[i][mask_index]

            axs[i, 0].imshow(img, cmap="gray")
            axs[i, 0].set_title(titles[0])

            axs[i, 1].imshow(gt, cmap="gray")
            axs[i, 1].set_title(titles[1])

            axs[i, 2].imshow(pred, cmap="gray")
            axs[i, 2].set_title(titles[2])

            for ax in axs[i]:
                ax.axis("off")

        plt.tight_layout()
        self.writer.add_figure(tag, fig)
        plt.close(fig)

    
    def log_metric(self, value, metric_name="Val Loss", step=0):
        """
        Logs a single metric value to TensorBoard.

        Args:
            value (float): The current value of the metric.
            metric_name (str): The name of the metric (e.g., "Loss", "Dice Score").
            step (int): The training step or epoch number.
        """
        self.writer.add_scalar(metric_name, value, step)

