# MODEL HUB: https://huggingface.co/amal90888/unet-segmentation-model
#%%
import torch
import tensorflow as tf
import os
# import niidataloader
import sys
from niidataloader import NiftiDataset
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from torch.nn import functional as F
#%%

# ✅ Set Keras backend (optional)
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

#%%
unet.summary()

#%%
# Dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



#%%
# freeze the model
for layer in unet.layers:
    layer.trainable = False

# unfreeze the lasts layers
for i in range(1, 24):
    unet.layers[-i].trainable = True
unet.layers[-1].trainable = True
unet.layers[-2].trainable = True
unet.layers[-3].trainable = True
unet.layers[-4].trainable = True
unet.layers[-5].trainable = True
unet.layers[-6].trainable = True
unet.layers[-7].trainable = True
unet.layers[-8].trainable = True
unet.layers[-9].trainable = True
unet.layers[-10].trainable = True
unet.layers[-11].trainable = True
unet.layers[-12].trainable = True
unet.layers[-13].trainable = True
unet.layers[-14].trainable = True
unet.layers[-15].trainable = True
unet.layers[-16].trainable = True
unet.layers[-17].trainable = True
unet.layers[-18].trainable = True
unet.layers[-19].trainable = True
unet.layers[-20].trainable = True
unet.layers[-21].trainable = True
unet.layers[-22].trainable = True
unet.layers[-23].trainable = True
# unet.layers[-24].trainable = True
# unet.layers[-25].trainable = True
# unet.layers[-26].trainable = True

unet.summary()

#%%
data = NiftiDataset("./data/Train")

#%%
print(data[0][0,:,:,0].shape)
plt.imshow(data[0][0,:,:,0], cmap="gray")
plt.show()
plt.imshow(data[0][1,:,:,0], cmap="gray")
plt.show()
#%%
# Resize the image
print(data[0][0,:,:,0].shape)
image = data[0][0,:,:,0].reshape(1, 1, 220, 220)
print(image.shape)
resized_image = F.interpolate(image, size=(128, 128), mode='bilinear', align_corners=False).reshape(128,128)
print(resized_image.shape)

#%%
# Pass the image trough the model
predicted_mask = unet.predict(resized_image.reshape(1, 128, 128, 1))
print(predicted_mask.shape)
print("PREDICTED")
plt.imshow(predicted_mask[0,:,:,0], cmap="gray")
plt.show()
print("ORIGINAL")
plt.imshow(data[0][1,:,:,0], cmap="gray")
plt.show()

#%%
print(data[32])
#%%
# Train the model Toy example
# Prepare the data
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

#%%
print(images.shape)
plt.imshow(images[0], cmap="gray")
plt.show()
#%%
# inference
predicted_mask = unet.predict(images)
#%%
print(predicted_mask.shape)
#%%
plt.imshow(predicted_mask[0,:,:,0], cmap="gray")
plt.show()
# plt.imshow(images[0], cmap="gray")
#%%
print(masks.shape)
plt.imshow(masks[0,:,:], cmap="gray")
plt.show()