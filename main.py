import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

class RBFNN():
    def __init__(self, sigma, n_centers):
        self.sigma = sigma
        self.n_centers = n_centers
        self.centers = None
        self.weights = None

    def _gaussian(self, x, c):
        return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * self.sigma ** 2))

    def _calculate_activation(self, X):
        activations = np.zeros((X.shape[0], self.centers.shape[0]))
        for i, center in enumerate(self.centers):
            for j, x in enumerate(X):
                activations[j, i] = self._gaussian(x, center)
        return activations

    def fit(self, X, y):
        kmeans = KMeans(n_clusters=self.n_centers, random_state=0)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        activations = self._calculate_activation(X)
        self.weights = np.linalg.pinv(activations.T @ activations) @ activations.T @ y

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained yet. Call fit method first.")
        
        activations = self._calculate_activation(X)
        return activations @ self.weights


# Example usage:
if __name__ == "__main__":
    X_original = np.array([
        [0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9],
        [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7],
        [0.7, 0.3], [0.15, 0.85], [0.85, 0.15], [0.25, 0.75],
        [0.75, 0.25], [0.4, 0.6], [0.6, 0.4], [0.1, 0.3],
        [0.3, 0.9], [0.9, 0.3], [0.3, 0.5], [0.7, 0.5],
        [0.6, 0.2], [0.2, 0.6], [0.8, 0.8], [0.25, 0.85],
        [0.85, 0.25]
    ])
    y_original = np.array([0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 
                        0, 1, 1, 0, 1, 0, 1, 0, 1, 1])

    np.random.seed(0) 
    n_repeats = 4
    X_repeated = np.tile(X_original, (n_repeats, 1))
    y = np.tile(y_original, n_repeats)

    # افزودن نویز ۰.۰۲ به داده‌های تکرار شده
    noise = np.random.uniform(-0.02, 0.02, X_repeated.shape)
    X = X_repeated + noise

    # تقسیم داده‌ها به آموزش و تست با نسبت 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rbfnn = RBFNN(sigma=0.3, n_centers=50)
    rbfnn.fit(X_train, y_train)

    # پیش‌بینی بر روی داده‌های آموزشی
    train_predictions = rbfnn.predict(X_train)
    train_mse = np.mean((train_predictions - y_train) ** 2)
    print("Training Predictions:", train_predictions)
    print("Training Mean Squared Error:", train_mse)

    # پیش‌بینی بر روی داده‌های تست
    test_predictions = rbfnn.predict(X_test)
    test_mse = np.mean((test_predictions - y_test) ** 2)
    print("Test Predictions:", test_predictions)
    print("Test Mean Squared Error:", test_mse)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=train_predictions, cmap='viridis', marker='o', label='Train Data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=test_predictions, cmap='viridis', marker='x', label='Test Data')
    plt.colorbar(label='Predicted Output')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('RBFNN Predictions for Train and Test Data')
    plt.legend()
    plt.show()
