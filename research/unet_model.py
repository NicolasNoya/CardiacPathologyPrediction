# MODEL HUB: https://huggingface.co/amal90888/unet-segmentation-model
#%%
import tensorflow as tf
import os
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

# freeze the model
for layer in unet.layers:
    layer.trainable = False

# unfreeze the lasts layers
for i in range(1, 24):
    unet.layers[-i].trainable = True



#%%
unet.summary()



