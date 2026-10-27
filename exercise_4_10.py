# Exercise 4.10
# Modify the Python program from Example 4.11 to use VGG19 for classification and
# compare its performance with VGG16.

import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x_raw = image.img_to_array(img)
x_raw = np.expand_dims(x_raw, axis=0)

# --- VGG16 Classification ---
model_vgg16 = VGG16(weights='imagenet')
x_16 = preprocess_vgg16(x_raw.copy())
t0 = time.time()
preds_16 = model_vgg16.predict(x_16)
t_16 = time.time() - t0
results_16 = decode_predictions(preds_16)
print("=== VGG16 Results ===")
print(f"Inference Time: {t_16:.4f}s")
print(results_16)

# --- VGG19 Classification ---
model_vgg19 = VGG19(weights='imagenet')
x_19 = preprocess_vgg19(x_raw.copy())
t0 = time.time()
preds_19 = model_vgg19.predict(x_19)
t_19 = time.time() - t0
results_19 = decode_predictions(preds_19)
print("\n=== VGG19 Results ===")
print(f"Inference Time: {t_19:.4f}s")
print(results_19)
