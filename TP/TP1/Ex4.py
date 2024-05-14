from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Charger les données de chiffres
digits = load_digits()

# Afficher la première image
plt.figure(figsize=(2, 2))
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title('Label: %i' % digits.target[0])
plt.show()

from sklearn.decomposition import PCA

# Obtenir la dimension des données
data = digits.images.reshape((digits.images.shape[0], -1))
print("Dimension des données : ", data.shape)

# Appliquer l'ACP avec 2 composantes principales
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

# Créer un graphique pour les deux composantes principales
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Blues_r', 10))
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.colorbar()
plt.show()

n_components = [1, 2, 3, 10, 20, 30, 50, 64]
images_reconstructed = []

# Appliquer l'ACP pour chaque nombre de composantes principales
for n in n_components:
    pca = PCA(n_components=n)
    X_transformed = pca.fit_transform(data)
    X_reconstructed = pca.inverse_transform(X_transformed)
    images_reconstructed.append(X_reconstructed)

# Visualiser la reconstruction d'un chiffre
plt.figure(figsize=(10, 20))
for i, images in enumerate(images_reconstructed):
    plt.subplot(len(n_components), 1, i + 1)
    plt.imshow(images[0].reshape((8, 8)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i composantes principales' % n_components[i])
plt.show()