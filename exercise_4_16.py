# Exercise 4.16
# Modify the Python code from the previous example (Example 4.17 Part 4), display all the
# filters by changing n_filters, ix = 4, 1 to n_filters, ix = s[3], 1, and comment on the results.
#
# Comment on results:
# Setting n_filters = s[3] displays all 64 filters (64 rows x 3 color channels = 192 subplots),
# showing the full variety of 3x3 edge, gradient, and color detectors learned in the first layer.

from tensorflow.keras.applications.vgg19 import VGG19
from matplotlib import pyplot

# load the model
model = VGG19()

# retrieve weights from the first hidden layer
n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape
print("Color channels: ", s[0])
print("Filter size: ", s[1], s[2])
print("Total number of filters : ", s[3])

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot all filters
n_filters, ix = s[3], 1

pyplot.figure(figsize=(10, 40))
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
