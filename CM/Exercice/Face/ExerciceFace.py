from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

min_person = 10
n_component = 20

def load_dataset(min_faces_per_person:int):
    """
    Load the LFW (Labeled Faces in the Wild) dataset.

    Parameters:
    - min_faces_per_person: Minimum number of face images per person.

    Returns:
    - data: Array-like, shape (n_samples, n_features)
        The flattened images as a 2D array, where each row represents a face image.
    - target: Array-like, shape (n_samples,)
        The labels corresponding to each face image.
    - target_names: Array-like, shape (n_classes,)
        The names of the people (target labels).
    - images: Array-like, shape (n_samples, h, w)
        The original images as a 3D array, where each element represents a pixel value.

    """
    dataset = fetch_lfw_people(min_faces_per_person=min_faces_per_person)
    return dataset.data, dataset.target, dataset.target_names, dataset.images

def display_images(images, target, target_names):
    """
    Display a grid of images with their corresponding labels.

    Parameters:
    images (list): A list of images to be displayed.
    target (list): A list of target labels for each image.
    target_names (list): A list of names corresponding to the target labels.

    Returns:
    None
    """
    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(images[i], cmap='bone')
        axi.set(xticks=[], yticks=[],
                xlabel=target_names[target[i]])

def plot_histogram(target, target_names):
    """
    Plots a histogram of the target values.

    Parameters:
    - target: numpy array or list of target values
    - target_names: list of target names corresponding to the target values

    Returns:
    None
    """
    plt.figure(figsize=(min_person, 5))
    plt.hist(target, bins=np.arange(target.max()+2)-0.5, rwidth=0.7)
    plt.xticks(np.unique(target), target_names, rotation=90)

def apply_pca(X, n_components):
    """
    Applies Principal Component Analysis (PCA) to the input data.

    Parameters:
    - X: The input data matrix of shape (n_samples, n_features).
    - n_components: The number of components to keep.

    Returns:
    - X_pca: The transformed data matrix after applying PCA, of shape (n_samples, n_components).
    - pca: The fitted PCA object.

    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def apply_pca_whiten(X, n_components):
    """
    Applies Principal Component Analysis (PCA) with whitening to the input data.

    Parameters:
    - X: The input data matrix of shape (n_samples, n_features).
    - n_components: The number of components to keep after dimensionality reduction.

    Returns:
    - X_pca_whiten: The transformed data matrix after applying PCA with whitening.
    - pca: The PCA object fitted on the input data.

    """
    pca = PCA(n_components=n_components, whiten=True)
    X_pca_whiten = pca.fit_transform(X)
    return X_pca_whiten, pca

def rebalance_dataset(X, y, target, min_faces_per_person):
    """
    Rebalances the dataset by selecting a minimum number of faces per person.

    Parameters:
    X (array-like): The input data.
    y (array-like): The target labels.
    target (int): The target label to rebalance.
    min_faces_per_person (int): The minimum number of faces per person to keep.

    Returns:
    X_balanced (array-like): The rebalanced input data.
    y_balanced (array-like): The rebalanced target labels.
    """
    mask = np.zeros(y.shape, dtype=bool)
    for target in np.unique(y):
        mask[np.where(y==target)[0][:min_faces_per_person]] = True

    X_balanced = X[mask]
    y_balanced = y[mask]
    return X_balanced, y_balanced

def process_faces():
    X, y, target_names, images = load_dataset(min_person)
    display_images(images, y, target_names)
    plot_histogram(y, target_names)
    X_pca, pca = apply_pca(X, 200)
    # affiche le cumul de la variance expliquée en fonction du nombre de composantes
    plt.figure(figsize=(min_person, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    X_pca_whiten, pca_whiten = apply_pca_whiten(X, n_component)
    X_balanced, y_balanced = rebalance_dataset(X, y, target_names, min_person)
    plot_histogram(y_balanced, target_names)
    plt.show()

if __name__ == "__main__":
    process_faces()