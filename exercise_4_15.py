# Exercise 4.15
# Modify the Python code from Example 4.17 (Part 4), select the second convolutional
# layer to visualize by changing n = 1 to n = 2, and comment on the results.
#
# Comment on results:
# In Layer 1 (block1_conv1, n = 1), the filter shape is (3, 3, 3, 64), meaning 3 input color channels.
# In Layer 2 (block1_conv2, n = 2), the filter shape is (3, 3, 64, 64), meaning there are 64 input
# channels (from the 64 feature maps of Layer 1) and 64 output filters of size 3x3.

from tensorflow.keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# load the model
model = VGG19()

# retrieve weights from the second convolutional layer (n = 2)
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Layer name:", model.layers[n].name)
print("Input channels: ", s[2])
print("Filter size: ", s[0], s[1])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first 4 filters (plotting first 3 input channels of each filter for display)
n_filters, ix = 4, 1
n_channels_to_plot = 3

pyplot.figure(figsize=(10, 10))
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]

    # plot first 3 channels separately
    for j in range(n_channels_to_plot):
        ax = pyplot.subplot(n_filters, n_channels_to_plot, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1

# show the figure
pyplot.show()
