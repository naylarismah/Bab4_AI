# Exercise 4.7
# Modify the Python program from Example 4.9, change the kernel size (number of filters)
# of 32 to 64 in the Conv2D layer, and compare their performances.
# Change the filter size (5,5) to (3,3) and compare the performances.

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

def run_experiment(filters, kernel_size, label):
    print(f"\n--- Running: {label} (filters={filters}, kernel_size={kernel_size}) ---")
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    start_time = time.time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=10, batch_size=200, verbose=1)
    elapsed = time.time() - start_time

    scores = model.evaluate(X_test, y_test, verbose=0)
    error = 100 - scores[1] * 100
    print(f"{label} -> Accuracy: {scores[1]*100:.2f}%, CNN Error: {error:.2f}%, Time: {elapsed:.2f}s")
    return scores[1], error, elapsed

# 1. Baseline (Example 4.9): filters=32, kernel_size=(5, 5)
acc_1, err_1, t_1 = run_experiment(32, (5, 5), "Baseline (32 filters, 5x5)")

# 2. Change filters from 32 to 64: filters=64, kernel_size=(5, 5)
acc_2, err_2, t_2 = run_experiment(64, (5, 5), "Modified 1 (64 filters, 5x5)")

# 3. Change filter size (5,5) to (3,3): filters=64, kernel_size=(3, 3)
acc_3, err_3, t_3 = run_experiment(64, (3, 3), "Modified 2 (64 filters, 3x3)")

print("\n=== Summary Comparison ===")
print(f"1. 32 filters, (5,5): Accuracy={acc_1*100:.2f}%, Error={err_1:.2f}%, Time={t_1:.2f}s")
print(f"2. 64 filters, (5,5): Accuracy={acc_2*100:.2f}%, Error={err_2:.2f}%, Time={t_2:.2f}s")
print(f"3. 64 filters, (3,3): Accuracy={acc_3*100:.2f}%, Error={err_3:.2f}%, Time={t_3:.2f}s")
