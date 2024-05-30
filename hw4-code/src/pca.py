import numpy as np
import matplotlib.pyplot as plt


# def plot_vector(vector, filepath):
#     plt.imshow(vector.reshape(61, 80), cmap='gray')
#     print(filepath)
#     plt.savefig(filepath)
#     plt.clf()

"""
Implementation of Principal Component Analysis.
"""
class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X: np.ndarray) -> None:
        #TODO: 10%
        self.mean = np.mean(X, axis=0)
        # plot_vector(self.mean, "mean_vector")
        X_centered = X - self.mean

        # Covariance matrix.
        covariance_matrix = np.cov(X_centered, rowvar=False)
        
        # calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # select top n_components eigenvectors
        self.components = sorted_eigenvectors[:, 0:self.n_components]

        # for i in range(min(self.n_components, 4)):
        #     plot_vector(self.components[:, i], "eigenvec_#{}".format(i + 1))
        

    def transform(self, X: np.ndarray) -> np.ndarray:
        #TODO: 2%
        # center the data
        X_centered = X - self.mean
        
        # project the data onto the principal components
        return X_centered @ self.components

    def reconstruct(self, X):
        #TODO: 2%
        # transform the data
        X_transformed = self.transform(X)

        # reconstruct the data back to its original space
        return X_transformed @ self.components.T + self.mean

    def get_components(self):
        # Return the principal components (eigenfaces)
        return self.components

    def get_mean_vector(self):
        # Return the mean vector
        return self.mean
        