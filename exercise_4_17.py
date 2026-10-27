# Exercise 4.17
# Modify the previous example Python code (Example 4.17 Part 5 & 6), select the second or
# third or fourth convolutional layer for visualization by changing n = 1 to n = 2 and so on,
# and comment on the results.
#
# Comment on results:
# As we move deeper into the network (e.g. n = 2 for block1_conv2, n = 4 for block2_conv1,
# n = 5 for block2_conv2), the feature maps become more abstract and highlight higher-level
# contours and textures rather than raw pixel intensities, and spatial resolution decreases
# after pooling layers.

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from numpy import expand_dims

def plot_feature_maps(feature_maps):
    # plot all feature maps
    col = 8
    row = int(feature_maps.shape[3] / col)
    ix = 1
    plt.figure(figsize=(20, 20))
    for _ in range(row):
        for _ in range(col):
            # specify subplot and turn of axis
            ax = plt.subplot(row, col, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()

# load the model
base_model = VGG19()

# Select the second convolutional layer to visualize (n = 2 -> block1_conv2)
# You can also change n to 4 (block2_conv1) or 5 (block2_conv2)
n = 2

# redefine model to output right after the selected hidden layer
model = Model(inputs=base_model.inputs, outputs=base_model.layers[n].output)
model.summary()

# load the image with the required shape
img = load_img('Elephant.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)

# get feature map for the selected hidden layer
feature_maps = model.predict(img)
print(f"Feature maps for layer {n} ({base_model.layers[n].name}): ", feature_maps.shape)

# plot all the feature maps
plot_feature_maps(feature_maps)
