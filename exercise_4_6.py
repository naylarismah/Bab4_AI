# Exercise 4.6
# Modify the Python program from Example 4.7, and add another Conv2D, MaxPooling,
# and Dropout layer before the Flatten layer. Compare the number of parameters.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Original model from Example 4.7
model_orig = Sequential()
model_orig.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model_orig.add(MaxPooling2D())
model_orig.add(Dropout(0.2))
model_orig.add(Flatten())
model_orig.add(Dense(128, activation='relu'))
model_orig.add(Dense(2, activation='softmax'))
model_orig.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("=== Original Model (1 Conv2D block) ===")
model_orig.summary()

# Modified model with an additional Conv2D, MaxPooling2D, and Dropout layer
model_mod = Sequential()
model_mod.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model_mod.add(MaxPooling2D())
model_mod.add(Dropout(0.2))

# Added 2nd Conv2D, MaxPooling2D, and Dropout layer
model_mod.add(Conv2D(64, (3, 3), activation='relu'))
model_mod.add(MaxPooling2D())
model_mod.add(Dropout(0.2))

model_mod.add(Flatten())
model_mod.add(Dense(128, activation='relu'))
model_mod.add(Dense(2, activation='softmax'))
model_mod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n=== Modified Model (2 Conv2D blocks) ===")
model_mod.summary()

print("\n=== Parameter Comparison ===")
print(f"Original Total Parameters: {model_orig.count_params():,}")
print(f"Modified Total Parameters: {model_mod.count_params():,}")
print("Note: Adding another Conv2D + MaxPooling reduces the spatial dimensions before Flatten,")
print("which significantly reduces the number of parameters in the Dense(128) layer!")
