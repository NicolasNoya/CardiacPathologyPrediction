#%% Importing the classes
import pandas as pd
import torch
import random
from densenet.densenet import DenseNet
from roi import ROI
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from densnet_trainer import DenseNetTrainer
from niidataloader import NiftiDataset
from feature_extractor import FeatureExtractor
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data_path = "./data/Train"
niftiidataset = NiftiDataset(image_path = data_path, roi=False, augment=False)

import torch.nn.functional as F
df = pd.read_csv('/home/onyxia/work/project/CardiacPathologyPrediction/data/metaDataTrain.csv')  # replace with your actual file path
# Show column names (to confirm names like 'Height' and 'Weight')
Heights = df['Height']
Weights = df['Weight']
gt_disease = df['Category']
dataloader = DataLoader(niftiidataset, batch_size=1, shuffle=False, num_workers=1) #This is a batchsize of 1 because the images are 3D
feature_extractor = FeatureExtractor()
features_tensors = []
for idx, img in tqdm(enumerate(dataloader), desc=f"Creating the dataset"):
    predicted_diastole = F.one_hot((img[0,1,:,:,:]*3).long(), num_classes=4).permute(2,3,0,1)
    predicted_systole = F.one_hot((img[0,3,:,:,:]*3).long(), num_classes=4).permute(2,3,0,1)
    # print(predicted_systole.shape)
    # predicted_systole = predicted_systole(0.5)
    # predicted_systole = (predicted_systole > 0.5).astype(int)
    # predicted_diastole = (predicted_diastole > 0.5).astype(int)
    # print(predicted_systole.shape)

    # Reordering the features to match the feature extractor needs
    predicted_systole_tensor = torch.cat((
        (predicted_systole[:,0]).unsqueeze(0),
        (predicted_systole[:,3]).unsqueeze(0),
        (predicted_systole[:,1]).unsqueeze(0),
        (predicted_systole[:,2]).unsqueeze(0),
    ), dim=0).permute(1,0,2,3)

    predicted_diastole_tensor = torch.cat((
        (predicted_diastole[:,0]).unsqueeze(0),
        (predicted_diastole[:,3]).unsqueeze(0),
        (predicted_diastole[:,1]).unsqueeze(0),
        (predicted_diastole[:,2]).unsqueeze(0),
    ), dim=0).permute(1,0,2,3)

    voxel_spacing = niftiidataset.__get_spacing__(idx)
    print(voxel_spacing)

    features_tensor = feature_extractor.extract_features(
        predicted_diastole_tensor, 
        predicted_systole_tensor, 
        Heights[idx], 
        Weights[idx],
        voxel_spacing=voxel_spacing
    )

    # plt.imshow(predicted_diastole_tensor[0,1], cmap='grey')
    # print("The feature tensor is: ", features_tensor)
    # print("The actual disease is: ", gt_disease[idx])
    features_tensors.append(features_tensor)

    # if idx==50:
    #     break
gt_disease = [x for idx, x in enumerate(gt_disease) if features_tensors[idx][0]>1]
features_tensors = [x for x in features_tensors if x[0]>1]
#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# features_tensors = np.array(features_tensors)
# gt_disease = [x for idx, x in enumerate(gt_disease) if features_tensors[idx][0]<1]
feat_train, feat_test, label_train, label_test = train_test_split(
    features_tensors, 
    gt_disease, 
    test_size=0.5, 
    random_state=2025
)

# feat_train = features_tensors
# label_train = gt_disease
#%%
print(feat_train[0])

#%% Normalize the data
normalization = MinMaxScaler()
normalization.fit(feat_train)
feat_train = normalization.transform(feat_train)
feat_test = normalization.transform(feat_test)
print(np.unique(label_test))
print(np.unique(label_train))
#%%


print("Start")

RF=RandomForestClassifier(n_estimators=100)
p_grid_RF = {'n_estimators':[100], }

grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, scoring="accuracy", cv=10)
grid_RF.fit(feat_train, label_train)

print("Best Validation Score: {}".format(grid_RF.best_score_))
print("Best params: {}".format(grid_RF.best_params_))
print("Random Forest test score :",grid_RF.score(feat_test, label_test))
print("Score in train: ", grid_RF.score(feat_train, label_train))
#%%
np.unique(label_train, return_counts=True)

# np.unique(feat_test)
#%%
from sklearn.tree import DecisionTreeClassifier
Tree = DecisionTreeClassifier()
Tree.fit(feat_train, label_train)
Tree.score(feat_test, label_test)
#%%
labels_tests = grid_RF.predict(feat_test)
print(labels_tests)

#%%
labels_trains = grid_RF.predict(feat_train)
print(labels_trains)
#%%
print(list(zip(label_test, labels_tests)))
#%%
#%%
Tree.score(feat_train, label_train)
#%%
print(Tree.tree_.max_depth)
print(Tree.tree_.node_count)

#%%
feat_train[1]
#%%
gt_disease[19]
