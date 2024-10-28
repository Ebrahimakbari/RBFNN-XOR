# Explain:

### 1. Importing Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
```

- `numpy`: Handles array operations and data management.
- `matplotlib.pyplot`: Provides data visualization capabilities.
- `KMeans` from `sklearn.cluster`: Used to perform clustering on input data to determine the Gaussian function centers.
- `train_test_split` from `sklearn.model_selection`: Splits the dataset into training and test sets.

### 2. RBFNN Class Definition

This class defines the core structure and methods of the RBFNN model.

#### `__init__(self, sigma, n_centers)`

The constructor initializes the RBFNN with parameters:

- **`sigma`**: Defines the width of the Gaussian function. It controls the spread of each RBF neuron and thus the influence of each neuron over the input space.
- **`n_centers`**: Number of Gaussian centers (i.e., RBF neurons). These centers represent the prototypes for clusters in the input data space.
- **`centers`** and **`weights`**: Initialized as `None`, these attributes will be set during training. `centers` stores the Gaussian centers, while `weights` represents the linear weights applied to the output of the Gaussian neurons.

```python
class RBFNN():
    def __init__(self, sigma, n_centers):
        self.sigma = sigma
        self.n_centers = n_centers
        self.centers = None
        self.weights = None
```

#### `_gaussian(self, x, c)`

This private method calculates the output of a Gaussian function for a given input `x` and center `c`. It uses the Euclidean distance between `x` and `c` to compute the Gaussian response, which is high when `x` is near `c`.

```python
def _gaussian(self, x, c):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * self.sigma ** 2))
```

#### `_calculate_activation(self, X)`

This private method computes the activation matrix for a given dataset `X`. Each row in the matrix corresponds to a data point, and each column represents the activation output of a Gaussian function centered at one of the centers.

```python
def _calculate_activation(self, X):
    activations = np.zeros((X.shape[0], self.centers.shape[0]))
    for i, center in enumerate(self.centers):
        for j, x in enumerate(X):
            activations[j, i] = self._gaussian(x, center)
    return activations
```

#### `fit(self, X, y)`

The fit method trains the RBFNN model by performing the following steps:
1. **KMeans Clustering**: Clusters the input data `X` to determine Gaussian centers.
2. **Activation Calculation**: Computes the activation matrix for `X` based on the selected centers.
3. **Weight Calculation**: Uses the pseudo-inverse of the activation matrix to calculate the weights, minimizing the mean squared error between the model predictions and the target values `y`.

```python
def fit(self, X, y):
    kmeans = KMeans(n_clusters=self.n_centers, random_state=0)
    kmeans.fit(X)
    self.centers = kmeans.cluster_centers_
    activations = self._calculate_activation(X)
    self.weights = np.linalg.pinv(
        activations.T @ activations) @ activations.T @ y
```

#### `predict(self, X)`

Generates predictions for new input data `X` using the trained model. If the model hasn’t been trained, it raises an error. This method also utilizes the activation matrix for `X` and applies the weights computed during training.

```python
def predict(self, X):
    if self.weights is None:
        raise ValueError('''
                Model not trained yet. Call fit method first.
        ''')

    activations = self._calculate_activation(X)
    return activations @ self.weights
```

## Summary

This RBFNN implementation demonstrates how radial basis functions can effectively approximate complex, non-linear patterns. The model identifies clusters in input data and places Gaussian functions at these clusters to produce a flexible approximation of the target function. 

The RBFNN is particularly suited for regression and classification problems, where clustering can help identify local patterns within the data. The combination of clustering and linear regression on the output layer provides the RBFNN model with the power to generalize well with sufficient training data.