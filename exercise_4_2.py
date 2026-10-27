# Exercise 4.2
# Modify the Python program from Example 4.2 so that it has three inputs, instead of two inputs.

from sklearn.neural_network import MLPClassifier

# 3 inputs truth table (Logical OR / AND example with 3 inputs)
X = [
    [0., 0., 0.],
    [0., 0., 1.],
    [0., 1., 0.],
    [1., 0., 0.],
    [0., 1., 1.],
    [1., 0., 1.],
    [1., 1., 0.],
    [1., 1., 1.]
]
y = [0, 1, 1, 1, 1, 1, 1, 1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

print("Predictions for [[2., 2., 2.], [-1., -2., -1.]]:")
print(clf.predict([[2., 2., 2.], [-1., -2., -1.]]))
print("Coefficient shapes:")
print([coef.shape for coef in clf.coefs_])
