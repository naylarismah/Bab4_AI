# Exercise 4.3
# Modify the Python program from Example 4.3 so that it uses five neurons for the dense
# layer, instead of 10 neurons. Compare the performances, such as the training time and
# the accuracy.

import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
X_data = whole_data.data
y_data = whole_data.target
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=0.3, random_state=7)

def build_and_run_model(num_neurons):
    print(f"\n--- Running MLP with {num_neurons} neurons in Dense layers ---")
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=[X_train.shape[1]]))
    model.add(Dense(num_neurons, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error',
                  metrics=['mse', 'accuracy'])

    start_time = time.time()
    model.fit(X_train, y_train, batch_size=50, validation_split=0.2,
              epochs=100, verbose=0)
    training_time = time.time() - start_time

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neurons: {num_neurons} | Training Time: {training_time:.4f}s | Loss (MSE): {results[0]:.4f} | Accuracy: {results[2]:.4f}")
    return training_time, results

# Run modified model with 5 neurons
time_5, res_5 = build_and_run_model(5)

# Run original model with 10 neurons for comparison
time_10, res_10 = build_and_run_model(10)

print("\n=== Performance Comparison ===")
print(f"5 Neurons  -> Time: {time_5:.4f}s, Loss: {res_5[0]:.4f}, Accuracy: {res_5[2]:.4f}")
print(f"10 Neurons -> Time: {time_10:.4f}s, Loss: {res_10[0]:.4f}, Accuracy: {res_10[2]:.4f}")
