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

# Configuration
data = NiftiDataset("./data/Train", augment=True)
UNFRONTZED_LAYERS = 25
DATASET_SIZE = len(data)
TRAIN_SIZE = 0.8
BATCH_SIZE = 1
EPOCHS = 5
LEARNING_RATE = 1e-4
model_path = "unet_trained_model_2_dice0.3379485607147217.h5"


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

# unfreeze the lasts layers
for i in range(1, UNFRONTZED_LAYERS):
    unet.layers[-i].trainable = True



# dataset_len = len(data)
# 80% training 20% validation. NOTE: We are not using the test set for that
train_size = int(TRAIN_SIZE * DATASET_SIZE)
val_size = DATASET_SIZE-train_size

# create the dataset

# I will train the model changing the dataset hopping that this would prevent the model to overfitting thanks to 
# augmentation. Also the memory constraints doesn't allow me to create a big dataset with a lot of augmentation 
# so this is a work around of it.
for iteration in range(8):
    images_train = []
    masks_train = []
    print(f"THE ITERATION IS: {iteration}")
    for j in [0, 2]:
        for i in range(train_size):
            for k in range(data[i].shape[-1]):
                image = F.interpolate(data[i][j,:,:,k].reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
                mask = F.interpolate((data[i][j+1,:,:,0]==1).to(float).reshape(1,1,220,220), size=(128,128),
                                    mode='bilinear', align_corners=False).reshape(1,128,128)
                images_train.append(image)
                masks_train.append(mask)


    print("The length of the dataset is: ", len(images_train))

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

    unet.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=["accuracy", dice_coef])
    #%%
    # Train the model
    unet.fit(train_dataset, epochs=EPOCHS)

#%%
unet.save(f"unet_trained_model_2.h5")
# See model's performance in the validation set
images_val = []
masks_val = []


# Create the val dataset and get the metrics of the val set.
for j in [0, 2]:
    for i in range(train_size, DATASET_SIZE):
        for k in range(data[i].shape[-1]):
            image = F.interpolate(data[i][j,:,:,k].reshape(1,1,220,220), size=(128,128),
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
val_metrics = unet.evaluate(val_dataset)

print(f"Validation loss: {val_metrics[0]}, validation dice score {val_metrics[-1]}, validation accuracy {val_metrics[1]}")

print("✅ Model trained successfully!")
print("Saving the model...")
unet.save(f"unet_trained_model_2_dice{val_metrics[-1]}.h5")
print("✅ Model saved successfully!")



