# Exercise 4.8
# Modify the Example 4.9 Python program, add another Conv2D, MaxPooling, and
# Dropout layer before the flatten layer. Compare the performances.

import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32') / 255.0
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255.0

# one hot encode outputs
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create modified model with an extra Conv2D, MaxPooling2D, and Dropout layer
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

# Added second Conv2D, MaxPooling2D, and Dropout block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Fit the model
start_time = time.time()
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=10, batch_size=200)
elapsed = time.time() - start_time

# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(f"Modified Model (2 Conv blocks) -> Accuracy: {scores[1]*100:.2f}%, CNN Error: {100-scores[1]*100:.2f}%, Time: {elapsed:.2f}s")
