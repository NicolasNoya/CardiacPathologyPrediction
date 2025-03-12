#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import pandas as pd

CLASSES = [0, 1, 2, 3, 4]

class DataViewer:
    def __init__(self, path_to_csv:str, log_dir:str="runs"):
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
        # labels = df["Category"]
        # amount = np.histogram(labels, bins=len(CLASSES))
        
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

if __name__=="__main__":
    # Test the DataViewer class
    path_to_csv = "data/metaDataTrain.csv"
    log_dir = "runs"
    data_viewer = DataViewer(path_to_csv, log_dir)
    data_viewer.check_labels_distribution()