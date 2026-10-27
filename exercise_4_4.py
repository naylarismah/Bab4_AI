# Exercise 4.4
# Modify Python LeNet program from Example 4.4 to use a different number of filters
# and see how this affects the total number of parameters.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D

def build_lenet(filters_1, filters_2):
    model = Sequential()
    model.add(Conv2D(filters=filters_1, kernel_size=(3, 3), activation='relu',
                     input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())
    model.add(Conv2D(filters=filters_2, kernel_size=(3, 3), activation='relu'))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model

print("=== Original LeNet-5 (filters: 6 and 16) ===")
model_orig = build_lenet(6, 16)
model_orig.summary()

print("\n=== Modified LeNet-5 (filters: 12 and 32) ===")
model_mod = build_lenet(12, 32)
model_mod.summary()

print(f"\nOriginal Total Parameters: {model_orig.count_params():,}")
print(f"Modified Total Parameters: {model_mod.count_params():,}")
