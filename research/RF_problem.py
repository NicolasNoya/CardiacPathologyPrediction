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
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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
features_list = []

for idx, img in tqdm(enumerate(dataloader), desc=f"Creating the dataset"):
    mask_diastole = (img[0,1,:,:,:]*3)
    mask_systole = (img[0,3,:,:,:]*3)
    plt.imshow(mask_diastole[:,:,2], cmap='grey')
    # print(np.unique(mask_diastole))
    mask_lr_ed = (mask_diastole==1)
    mask_lr_es = (mask_systole==1)
    # plt.imshow(mask_lr[:,:,1])
    # plt.show()
    mask_myo_ed = (mask_diastole==2)
    mask_myo_es = (mask_systole==2)
    # plt.imshow(mask_myo[:,:,1])
    # plt.show()
    mask_lv_ed = (mask_diastole==3)
    mask_lv_es = (mask_systole==3)
    # plt.imshow(mask_lv[:,:,1])
    # plt.show()
    vol_lv_ed = mask_lv_ed.sum()
    vol_lr_ed = mask_lr_ed.sum()
    vol_myo_ed = mask_myo_ed.sum()
    vol_lv_es = mask_lv_es.sum()
    vol_lr_es = mask_lr_es.sum()
    vol_myo_es = mask_myo_es.sum()
    

    ratio_myo_lv_ed = vol_myo_ed/(vol_lv_ed+1e-6)
    ratio_myo_lv_es = vol_myo_es/(vol_lv_es+1e-6)

    ratio_lv_lr_ed = vol_lr_ed/(vol_lv_ed+1e-6)
    ratio_lv_lr_es = vol_lr_es/(vol_lv_es+1e-6)

    ejec_frac_lv = (vol_lv_ed - vol_lv_es) / vol_lv_ed
    ejec_frac_lr = (vol_lr_ed - vol_lr_es) / vol_lr_ed 

    features = [
        vol_lv_ed, 
        vol_lr_ed, 
        vol_myo_ed,
        vol_lv_es,
        vol_lr_es,
        vol_myo_es,
        ejec_frac_lv,
        ejec_frac_lr,
        ratio_lv_lr_ed,
        ratio_lv_lr_es,
        ratio_myo_lv_ed,
        ratio_myo_lv_es,
        Heights[idx], 
        Weights[idx],
        ]
    features_list.append(features)
    # print("Hola")
    # break
#%%
len(features_list)
#%%
    # predicted_diastole = F.one_hot((img[0,1,:,:,:]*3).long(), num_classes=4).permute(2,3,0,1)
    # predicted_systole = F.one_hot((img[0,3,:,:,:]*3).long(), num_classes=4).permute(2,3,0,1)
    # print(predicted_systole.shape)
    # predicted_systole = predicted_systole(0.5)
    # predicted_systole = (predicted_systole > 0.5).astype(int)
    # predicted_diastole = (predicted_diastole > 0.5).astype(int)
    # print(predicted_systole.shape)

    # Reordering the features to match the feature extractor needs
    # predicted_systole_tensor = torch.cat((
    #     (predicted_systole[:,0]).unsqueeze(0),
    #     (predicted_systole[:,3]).unsqueeze(0),
    # #     (predicted_systole[:,1]).unsqueeze(0),
    # #     (predicted_systole[:,2]).unsqueeze(0),
    # # ), dim=0).permute(1,0,2,3)


    # # predicted_diastole_tensor = torch.cat((
    # #     (predicted_diastole[:,0]).unsqueeze(0),
    # #     (predicted_diastole[:,3]).unsqueeze(0),
    # #     (predicted_diastole[:,1]).unsqueeze(0),
    # #     (predicted_diastole[:,2]).unsqueeze(0),
    # # ), dim=0).permute(1,0,2,3)
    # break
    # predicted_diastole_tensor = predicted_diastole
    
    # predicted_systole_tensor = predicted_systole

    # voxel_spacing = niftiidataset.__get_spacing__(idx)
    # # print(voxel_spacing)

    # features_tensor = feature_extractor.extract_features(
    #     predicted_diastole_tensor, 
    #     predicted_systole_tensor, 
    #     Heights[idx], 
    #     Weights[idx],
    #     voxel_spacing=voxel_spacing
    # )

    # # plt.imshow(predicted_diastole_tensor[0,1], cmap='grey')
    # # print("The feature tensor is: ", features_tensor)
    # # print("The actual disease is: ", gt_disease[idx])
    # features_tensors.append(features_tensor)

    # if idx==50:
    #     break
gt_disease = [x for idx, x in enumerate(gt_disease) if features_list[idx][0]>1]
features_list = [x for x in features_list if x[0]>1]
#%%
print(predicted_diastole.shape)
plt.imshow(predicted_diastole[2,3], cmap='grey')

#%%
print(features_tensors[0])
#%%
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#%%


# features_tensors = np.array(features_tensors)
# gt_disease = [x for idx, x in enumerate(gt_disease) if features_tensors[idx][0]<1]
feat_train, feat_test, label_train, label_test = train_test_split(
    features_list, 
    gt_disease, 
    test_size=0.2, 
    random_state=2025
)

# for i in range(len(list_w_h)):
    # print(list_w_h[i], gt_disease[i])
# feat_train = features_tensors
# label_train = gt_disease
#%%
print(feat_train[0])
print(features_tensors[0])

#%% Normalize the data
normalization = MinMaxScaler()
normalization.fit(feat_train)
feat_train = normalization.transform(feat_train)
feat_test = normalization.transform(feat_test)
print(np.unique(label_test))
print(np.unique(label_train))
#%%
print("Start")
RF=RandomForestClassifier()
p_grid_RF = {'n_estimators': [50,100, 1000], 'min_samples_leaf': [6,7,9], 'max_depth': [2,3,4,5,6,7]}
# p_grid_RF = {'n_estimators':[50,100,150,200,250,300,350,400]}

grid_RF = GridSearchCV(estimator=RF, param_grid=p_grid_RF, scoring="accuracy", cv=10)
grid_RF.fit(feat_train, label_train)

print("Best Validation Score: {}".format(grid_RF.best_score_))
print("Best params: {}".format(grid_RF.best_params_))
print("Random Forest test score :",grid_RF.score(feat_test, label_test))
print("Score in train: ", grid_RF.score(feat_train, label_train))
#%%
RF2 = RandomForestClassifier(n_estimators=100, max_depth=5)
RF2.fit(feat_train, label_train)
RF2.score(feat_train, label_train)
RF2.feature_importances_
RF2.score(feat_test, label_test)
    