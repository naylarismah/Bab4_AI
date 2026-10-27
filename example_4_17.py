# Example 4.17 CNN Visualize Filters.ipynb (Part 1 - Part 6)
# Modified from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from matplotlib import pyplot
import matplotlib.pyplot as plt
from numpy import expand_dims

# --- PART 1: Load VGG19 model and display summary ---
model = VGG19()
model.summary()

# --- PART 2: Display all layers and layer numbers ---
n = 0
for layer in model.layers:
    print(n, layer.name)
    n += 1

# --- PART 3: Display filter shape and bias shape for convolutional layers ---
n = 0
for layer in model.layers:
    if 'conv' in layer.name:
        filters, biases = layer.get_weights()
        print(n, layer.name, filters.shape, biases.shape)
    n += 1

# --- PART 4: Visualize filters in the first convolutional layer (n = 1) ---
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
# n_filters, ix = s[3], 1
n_filters, ix = 4, 1

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]

    # plot each channel separately
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[j, :, :], cmap='gray')
        ix += 1
# show the figure
pyplot.show()

# --- PART 5: Function to draw feature maps ---
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

# --- PART 6: Visualize feature maps of the first layer (n = 1) ---
# Select the hidden layer to visualize
n = 1
# redefine model to output right after the hidden layer
sub_model = Model(inputs=model.inputs, outputs=model.layers[n].output)
sub_model.summary()

# load the image with the required shape
img = load_img('Elephant.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = sub_model.predict(img)
print("Feature maps: ", feature_maps.shape)
# plot all the feature maps
plot_feature_maps(feature_maps)
