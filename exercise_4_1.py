# Exercise 4.1
# Modify Python program from Example 4.1 so that it performs a logical OR of the inputs.
# Note: Added a bias input (column of 1s) so that [0, 0] can properly map to 0
# instead of 0.5 when passed through the sigmoid function 1 / (1 + exp(-z)).

from numpy import exp, array, random, dot

# Set up parameters ===============================================
# Third column is bias = 1
inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
outputs = array([[0, 1, 1, 1]]).T
random.seed(1)
weights = 2 * random.random((3, 1)) - 1

# Define a Single Neuron function =================================
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron ================================================
for iteration in range(50000):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    weights += dot(inputs.T, (outputs - output) * output * (1 - output))

# Test the Neuron =================================================
for x_in in [[0, 0], [1, 0], [0, 1], [1, 1]]:
    x = array(x_in + [1])
    pred = neuron(x, weights)
    print(f"Input: {x_in} -> Predicted OR: {pred[0]:.6f} (Rounded: {int(pred[0] > 0.5)})")
