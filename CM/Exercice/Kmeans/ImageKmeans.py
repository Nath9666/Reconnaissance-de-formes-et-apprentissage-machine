from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def kmeans_segmentation(path, k_values, color=False):
    # Lire l'image
    if color:
        img = Image.open(path)
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 1)
    else:
        img = Image.open(path).convert('L')
        img_np = np.array(img)
        pixels = img_np.reshape(-1, 1)

    # Afficher l'histogramme
    plt.hist(img_np.ravel(), bins=256, color='black')
    plt.xlabel('Niveau de gris')
    plt.ylabel('Nombre de pixels')
    #plt.show()

    # Appliquer KMeans avec différentes valeurs de k
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)
        
        # Remplacer chaque pixel par le centre de son cluster
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        segmented_img = segmented_img.reshape(img_np.shape)
        
        # Normaliser l'image segmentée pour qu'elle soit dans l'intervalle [0, 1]
        segmented_img = (segmented_img - segmented_img.min()) / (segmented_img.max() - segmented_img.min())
        
        # Afficher l'image segmentée
        plt.figure()
        plt.imshow(segmented_img, cmap='gray' if not color else None)
    plt.show()

# Utilisation de la fonction
kmeans_segmentation('CM\\Exercice\\Kmeans\\arbre.jpg', [2, 3, 4, 5], color=False)
kmeans_segmentation('CM\\Exercice\\Kmeans\\arbre.jpg', [2, 3, 4, 5], color=True)