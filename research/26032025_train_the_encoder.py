#%%
# Given the results from this paper: https://arxiv.org/pdf/2002.08438
# I will train the encoder to learn the features of the images
# Howver, I will also train the last layers of the decoder
import torch
import tensorflow as tf
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
from research.feature_extraction_model import FeatureExtractor
from research.preprocessing import Preprocessing

# Configuration
data = NiftiDataset("./data/Train", augment=True)
UNFROZEN_LAYERS_ENCODER = 8
UNFROZEN_LAYERS_DECODER = 16
DATASET_SIZE = len(data)
TRAIN_SIZE = 0.8
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-3
model_path = "unet_trained_model_2_dice0.26813364028930664.h5"


os.environ["KERAS_BACKEND"] = "jax"
#%%

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

# ✅ Recompile with fresh optimizer and correct loss function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

unet.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=dice_loss, metrics=["accuracy", dice_coef])

print("✅ Model loaded and recompiled successfully!")

# Dice loss
#%%

# freeze the model
for layer in unet.layers:
    layer.trainable = False

# unfreeze the lasts layers
for i in range(1, UNFROZEN_LAYERS_ENCODER):
    unet.layers[i].trainable = True

for i in range(1, UNFROZEN_LAYERS_DECODER):
    unet.layers[-i].trainable = True

unet.summary()


# dataset_len = len(data)
# 80% training 20% validation. NOTE: We are not using the test set for that
#%%
train_size = int(TRAIN_SIZE * DATASET_SIZE)
val_size = DATASET_SIZE-train_size
#%%

# create the dataset

# I will train the model changing the dataset hopping that this would prevent the model to overfitting thanks to 
# augmentation. Also the memory constraints doesn't allow me to create a big dataset with a lot of augmentation 
# so this is a work around of it.
preproc = Preprocessing()
feature_extractor = FeatureExtractor(images_path="./data/Train")
indexes = np.array(range((DATASET_SIZE)))
np.random.shuffle(indexes)
train_indexes = indexes[:train_size]
val_indexes = indexes[train_size:]
images_val = []
masks_val = []


# Create the val dataset and get the metrics of the val set.
preproc = Preprocessing()
for j in [0,2]:
    for i in val_indexes:
        for k in range(data[i].shape[-1]):
            image = data[i][j,:,:,k]
            image = preproc.preprocess(image)
            image = F.interpolate(image.reshape(1,1,220,220), size=(128,128),
                                mode='bilinear', align_corners=False).reshape(1,128,128)
            mask = F.interpolate((data[i][j+1,:,:,0]==1).to(float).reshape(1,1,220,220), size=(128,128),
                                mode='bilinear', align_corners=False).reshape(1,128,128)
            images_val.append(image)
            masks_val.append(mask)


images_val = torch.cat(images_val, dim=0)
images_val = tf.convert_to_tensor(images_val.numpy())
images_val = tf.cast(images_val, tf.float32)

masks_val = torch.cat(masks_val, dim=0)
masks_val = tf.convert_to_tensor(masks_val.numpy())
masks_val = tf.cast(masks_val, tf.float32)

# Test the model with validation set
val_dataset = tf.data.Dataset.from_tensor_slices((images_val, masks_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

for iteration in range(10):
    images_train = []
    masks_train = []
    print(f"THE ITERATION IS: {iteration}")
    for j in [0,2]:
        for i in train_indexes:
            for k in range(data[i].shape[-1]):
                # image = feature_extractor.preprocessing(i)
                image = data[i][j,:,:,k]
                # image = image[0,0,:,:]
                # image = preproc.preprocess(image)
                image = F.interpolate(image.reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
                mask = F.interpolate((data[i][j+1,:,:,0]==1).to(float).reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
                images_train.append(image)
                masks_train.append(mask)


    print("The length of the dataset is: ", len(images_train))
    print("The length of the mask is: ", len(masks_train))

    # Compile the model
    # Image to tensorflow
    # masks to tensorflow
    images_train = torch.cat(images_train, dim=0)
    images_train = tf.convert_to_tensor(images_train.numpy())
    images_train = tf.cast(images_train, tf.float32)

    masks_train = torch.cat(masks_train, dim=0)
    masks_train = tf.convert_to_tensor(masks_train.numpy())
    masks_train = tf.cast(masks_train, tf.float32)


    train_dataset = tf.data.Dataset.from_tensor_slices((images_train, masks_train))
    train_dataset = train_dataset.batch(BATCH_SIZE)

    unet.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=dice_loss, metrics=["accuracy", dice_coef])
    unet.fit(train_dataset, epochs=EPOCHS)


val_metrics = unet.evaluate(val_dataset)

    #%%

    # Save the model


print(f"Validation loss: {val_metrics[0]}, validation dice score {val_metrics[-1]}, validation accuracy {val_metrics[1]}")
#%%

print("✅ Model trained successfully!")
print("Saving the model...")
unet.save(f"unet_trained_model_2_dice{val_metrics[-1]}.h5")
print("✅ Model saved successfully!")



