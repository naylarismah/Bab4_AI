# Exercise 4.5
# Modify the Python AlexNet program from Example 4.5, to use a different number of
# dense layer units and see how this affects the total number of parameters.

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

def build_alexnet(dense_1=2048, dense_2=2048, dense_3=512):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11),
                     activation='relu', strides=(4, 4), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), activation='relu',
                     strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',
                     strides=(1, 1), padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 1st Fully Connected layer
    model.add(Flatten())
    model.add(Dense(dense_1, activation='relu'))
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(dense_2, activation='relu'))
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(dense_3, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam', metrics=["accuracy"])
    return model

print("=== Original AlexNet (Dense units: 4096, 4096, 1000) ===")
model_orig = build_alexnet(4096, 4096, 1000)
model_orig.summary()

print("\n=== Modified AlexNet (Dense units: 2048, 2048, 512) ===")
model_mod = build_alexnet(2048, 2048, 512)
model_mod.summary()

print(f"\nOriginal Total Parameters: {model_orig.count_params():,}")
print(f"Modified Total Parameters: {model_mod.count_params():,}")
