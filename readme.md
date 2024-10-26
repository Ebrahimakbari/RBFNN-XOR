# Radial Basis Function Neural Network (RBFNN) - Python Implementation

This project implements a basic Radial Basis Function Neural Network (RBFNN) in Python using NumPy and Matplotlib. The network is designed to learn and predict outputs for simple datasets, particularly the XOR dataset, by mapping input vectors to outputs using radial basis functions (Gaussian functions).

## About RBFNN

The Radial Basis Function Neural Network (RBFNN) is a type of artificial neural network that uses radial basis functions as activation functions. Each neuron in the hidden layer has a radial basis function centered on a particular point, known as a center, and outputs a value based on the distance between the input and the center. RBFNNs are particularly useful for function approximation tasks and can learn complex mappings from inputs to outputs. Here, the network is trained to approximate the XOR function.

## Code Structure

The implementation consists of the following components:

### 1. **Class `RBFNN`**

   - `__init__(self, sigma)`: Initializes the RBFNN model with a specific `sigma` parameter (controls the spread of the Gaussian function) and predefined centers for the RBFs.
   - `_gaussian(self, x, c)`: Computes the Gaussian activation for a given input `x` and center `c`.
   - `_calculate_activation(self, X)`: Calculates the activations for all inputs in `X` based on the predefined centers.
   - `fit(self, X, y)`: Trains the network by calculating the optimal weights to minimize the error between predicted and target values.
   - `predict(self, X)`: Uses the trained weights to predict the outputs for new inputs in `X`.

### 2. **Example usage (XOR problem)**

   - Defines a simple XOR dataset and initializes the RBFNN model with `sigma = 0.1`.
   - Trains the model using the `fit` function.
   - Uses `predict` to generate predictions for the inputs and calculates the Mean Squared Error (MSE).
   - Visualizes the results using Matplotlib.

## Usage Example

```python
# Define XOR dataset
X = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]])
y = np.array([0, 1, 1, 0])

# Initialize and train RBFNN
rbfnn = RBFNN(sigma=0.1)
rbfnn.fit(X, y)

# Predict
predictions = rbfnn.predict(X)
print("Predictions:", predictions)

# Calculate mean squared error
mse = np.mean((predictions - y) ** 2)
print("Mean Squared Error:", mse)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis')
plt.colorbar(label='Predicted Output')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('RBFNN Predictions for XOR ')
plt.show()
