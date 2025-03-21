import torch
import tensorflow as tf
import os
# import niidataloader
import sys
# sys.path.append('../niidataloader')
from niidataloader import NiftiDataset
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from torch.nn import functional as F

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

# ✅ Download the model from Hugging Face
model_path = hf_hub_download(repo_id="amal90888/unet-segmentation-model", filename="unet_model.keras")

# ✅ Load the model with registered custom objects
unet = load_model(model_path, custom_objects={"dice_coef": dice_coef, "gl_sl": gl_sl}, compile=False)

# ✅ Recompile with fresh optimizer and correct loss function
from tensorflow.keras.optimizers import Adam
unet.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=["accuracy", dice_coef])

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
for i in range(1, 24):
    unet.layers[-i].trainable = True


data = NiftiDataset("./data/Train")

images = [F.interpolate(data[i][0,:,:,0].reshape(1,1,220,220), size=(128,128), 
                        mode='bilinear', align_corners=False).reshape(1,128,128)
           for i in range(32)]
print(images[0].shape)
masks = [F.interpolate(data[i][1,:,:,0].reshape(1,1,220,220), size=(128,128),
                       mode='bilinear', align_corners=False).reshape(1,128,128)
           for i in range(32)]

# Compile the model
# Image to tensorflow
# masks to tensorflow
images = torch.cat(images, dim=0)
masks = torch.cat(masks, dim=0)
images = tf.convert_to_tensor(images.numpy())
masks = tf.convert_to_tensor(masks.numpy())

masks = tf.cast(masks, tf.float32)
images = tf.cast(images, tf.float32)

train_dataset = tf.data.Dataset.from_tensor_slices((images, masks))
train_dataset = train_dataset.batch(32)
unet.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=["accuracy", dice_coef])
#%%
# Train the model
unet.fit(train_dataset, epochs=1000)

