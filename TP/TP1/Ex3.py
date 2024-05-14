import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import sklearn.decomposition
import sklearn.preprocessing

def compute_ACP(data, n_components = None):
    return sklearn.decomposition.PCA(n_components = n_components).fit_transform(data)

def variance_percentage(data, n_components = None):
    data = sklearn.decomposition.PCA(n_components = n_components).fit(data)
    return data.explained_variance_ratio_


    


## Exercice 2

def acp_with_components(data, n_components):
    pca = sklearn.decomposition.PCA(n_components = n_components)
    pca.fit_transform(data)
    transformed_data = compute_ACP(data, n_components)
    
    print("PCA with ", n_components, " principal components:")
    print(transformed_data)

    print("Principal Components:")
    print(pca.components_)

    return transformed_data



def display_PCA(data, dimensions, color = None): # Display PCA with 2 or 3 principal components
    transformed_data = compute_ACP(data, dimensions)

    if dimensions == 2:
        plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c = color)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
    elif dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c = color)
    plt.title('PCA of Iris dataset')
    plt.show()



def dimensions_needed(data, threshold):
    pca = sklearn.decomposition.PCA(n_components = len(data[0]))
    pca.fit_transform(data)
    variance_cumulative = np.cumsum(pca.explained_variance_ratio_)
    dimensions_needed = np.argmax(variance_cumulative >= threshold) + 1
    print("Number of dimensions needed to preserve 95% of the variance:", dimensions_needed)



def dimensions_needed2(data, threshold):
    n_components = 1
    pca = sklearn.decomposition.PCA(n_components = n_components)
    pca.fit_transform(data)
    while pca.explained_variance_ratio_.sum() < threshold:
        n_components += 1
        pca = sklearn.decomposition.PCA(n_components = n_components)
        pca.fit_transform(data)
    print("Number of dimensions needed to preserve 95% of the variance:", n_components)



def display_explained_variance(data):
    pca = sklearn.decomposition.PCA(n_components = len(data[0]))
    pca.fit_transform(data)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    for i in range(len(cumulative_explained_variance)):
        print("Explained variance of component", i + 1, ":", cumulative_explained_variance[i])



def display_cumulative_explained_variance(data):
    pca = sklearn.decomposition.PCA(n_components = len(data[0]))
    pca.fit_transform(data)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
    plt.xlabel('Number of dimensions')
    plt.ylabel('Cumulative explained variance')
    plt.title('Contribution of variance according to the number of dimensions')
    plt.grid()
    plt.show()


## Exercice 3

def display_eigenvalues_decrease(data):
    pca = sklearn.decomposition.PCA(n_components = len(data[0]))
    pca.fit_transform(data)
    eigenvalues = pca.explained_variance_
    plt.figure(figsize = (8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues)
    plt.xlabel('Number of dimensions')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues according to the number of dimensions')
    plt.grid()
    plt.show()



def mainEx3():
    data = np.genfromtxt('C:\\Users\\Nathan\\Documents\\Projet\\Reconnaissance de formes et apprentissage machine\\TP\\TP1\\leaf\\leaf.csv', delimiter=',')
    display_eigenvalues_decrease(data[:, 1:])
    display_PCA(data[:, 1:], 3, data[:, 0])

if __name__ == '__main__':
    mainEx3()

