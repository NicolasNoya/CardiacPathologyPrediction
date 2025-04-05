#%%
import torch
import tensorflow as tf
import copy
import os
# import niidataloader
import sys
# sys.path.append('../niidataloader')
from niidataloader import NiftiDataset
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np

# Configuration
data = NiftiDataset("./data/Train", augment=False)
UNFRONTZED_LAYERS = 25
DATASET_SIZE = len(data)
TRAIN_SIZE = 0.8
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 1e-4
model_path = "best_model.h5"


os.environ["KERAS_BACKEND"] = "jax"

# ✅ Register and define missing functions
@register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

@register_keras_serializable()
def gl_sl(*args, **kwargs):
    pass  # Placeholder function (update if needed)

# ✅ Load the model with registered custom objects
unet = load_model(model_path, custom_objects={"dice_coef": dice_coef, "gl_sl": gl_sl}, compile=False)
# unet = load_model(model_path)
unet.summary()

# ✅ Recompile with fresh optimizer and correct loss function
unet.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy", dice_coef])

print("✅ Model loaded and recompiled successfully!")

# Dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# freeze the model
for layer in unet.layers:
    layer.trainable = False

#%%
# predict with the model
from research.feature_extraction_model import FeatureExtractor
feature_extractor = FeatureExtractor(images_path="./data/Train")

data = NiftiDataset("./data/Train", augment=False)

# idx = int(len(data)*0.8+2)
idx = 8
image = data[idx][0,:,:,0]
print(image.shape)
# for idx in range(int(len(data)*0.8), len(data)):
# image = feature_extractor.preprocessing(idx)
plt.imshow(image.reshape(220,220), cmap='grey')
plt.show()
# image = image[0,0,:,:,:].reshape(220,220)

image = F.interpolate(image.reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
mask = F.interpolate(data[idx][1,:,:,0].reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
# # predcition = unet.predict(data[0][0,:,:,0].reshape(1, 256, 256, 1))
# plt.imshow(image.reshape(128,128), cmap='gray')
# plt.show()
plt.imshow(mask.reshape(128,128), cmap='gray')
plt.show()

prediction = unet.predict(image.reshape(1,128,128,1))
image = np.array(image)
plt.imshow(prediction.reshape(128,128)*image.reshape(128,128), cmap='gray')
plt.show()

#%%
mask = mask.to(torch.float32)
dice_score = dice_coef(np.array(mask), prediction)
print(dice_score)


