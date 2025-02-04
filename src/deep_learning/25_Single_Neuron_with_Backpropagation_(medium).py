# Single Neuron with Backpropagation (medium)
# https://www.deep-ml.com/problem/Single%20Neuron%20with%20Backpropagation
# Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
'''
Example:
        input: features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2], initial_bias = 0.0, learning_rate = 0.1, epochs = 2
        output: updated_weights = [0.0808, -0.1916], updated_bias = -0.0214, mse_values = [0.2386, 0.2348]
        reasoning: The neuron receives feature vectors and computes predictions using the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights and bias are computed and used to update the model parameters across epochs.
'''

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(features: np.ndarray, weights: np.ndarray, bias: float):
    probabilities = sigmoid(np.dot(features, weights) + bias)
    return probabilities

def mse_loss(probabilities: np.ndarray, labels: np.ndarray):
    mse = np.mean((labels - probabilities) ** 2)
    return mse

def backward(features: np.ndarray, probabilities: np.ndarray, labels: np.ndarray):
    # actually, error = 2 * (probabilities - labels) for mse loss derivative, but use (probabilities - labels) in this problem.
    error = probabilities - labels
    grad_w = np.dot(error * probabilities * (1 - probabilities), features)
    grad_b = sum(error * (probabilities * (1 - probabilities)))
    return grad_w, grad_b

def update_parameter(current_weights: np.ndarray, current_bias: float, learning_rate: float, grad_w: np.ndarray, grad_b: float, batch_size: int):
    weights = current_weights - learning_rate * grad_w / batch_size
    bias = current_bias - learning_rate * grad_b / batch_size
    return weights, bias

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    # Your code here
    updated_weights, updated_bias = initial_weights, initial_bias
    mse_values = []
    batch_size = len(labels)
    for i in range(epochs):
        probabilities = forward(features, updated_weights, updated_bias)
        loss = mse_loss(probabilities, labels)
        grad_w, grad_b = backward(features, probabilities, labels)
        updated_weights, updated_bias = update_parameter(updated_weights, updated_bias, learning_rate, grad_w, grad_b, batch_size)
        mse_values.append(loss)
    updated_weights = [round(weight, 4) for weight in updated_weights]
    updated_bias = round(updated_bias, 4)
    mse_values = [round(value, 4) for value in mse_values]
    return updated_weights, updated_bias, mse_values

