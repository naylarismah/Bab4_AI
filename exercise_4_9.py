# Exercise 4.9
# Modify the Python program from Example 4.10 so that it loads the VGG19 model,
# instead of the VGG16 model.

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np

model = VGG19(weights='imagenet')
print(model.summary())
