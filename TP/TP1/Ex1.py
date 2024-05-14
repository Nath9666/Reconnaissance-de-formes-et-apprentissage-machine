import numpy as np
import sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardizes the input data using the StandardScaler from sklearn.preprocessing.

    Parameters:
        data (np.ndarray): The input data to be standardized.

    Returns:
        np.ndarray: The standardized data.

    """
    data = sklearn.preprocessing.StandardScaler().fit_transform(data)
    return data

def cov_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculates the covariance matrix of the input data.

    Parameters:
        data (np.ndarray): The input data.

    Returns:
        np.ndarray: The covariance matrix of the input data.

    """
    return np.cov(data, rowvar=False)

def decompose_cov_matrix(data: np.ndarray) -> tuple:
    """
    Decomposes the covariance matrix of the given data.

    Parameters:
        data (np.ndarray): The input data array.

    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors of the covariance matrix.
    """
    cov_mat = cov_matrix(data)
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    return eigenvalues, eigenvectors

def compute_ACP(data : np.ndarray, n_components=None):
    """
    Computes the Principal Component Analysis (PCA) on the given data.

    Parameters:
    - data: numpy.ndarray
        The input data for PCA.
    - n_components: int or None, optional (default=None)
        The number of components to keep. If None, all components are kept.

    Returns:
    - transformed_data: numpy.ndarray
        The transformed data after applying PCA.
    - explained_variance_ratio: numpy.ndarray
        The ratio of explained variance for each principal component.
    - pca: sklearn.decomposition.PCA
        The PCA object used for the transformation.

    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    explained_variance_ratio = pca.explained_variance_ratio_
    return transformed_data, explained_variance_ratio, pca

if __name__ == "__main__":
    # Générer 500 vecteurs dans l'espace 3D
    data = np.random.randn(500,3)
    #? data = standardize_data(data)

    # Visualiser les points générés
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2])
    plt.show()

    # Appliquer la fonction compute_ACP aux données générées
    transformed_data, explained_variance_ratio, pca = compute_ACP(data)
    print("Transformed Data:\n", transformed_data)
    print("Explained Variance Ratio:\n", explained_variance_ratio)