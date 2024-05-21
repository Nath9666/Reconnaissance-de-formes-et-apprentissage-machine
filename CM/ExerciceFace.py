from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

min_person = 10
n_component = 20

# 1. Lire le dataset
dataset = fetch_lfw_people(min_faces_per_person=min_person)
X = dataset.data
y = dataset.target
target_names = dataset.target_names

# 2. Afficher quelques images
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(dataset.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=target_names[y[i]])

# 3. Afficher le nombre d'images par personne sous forme d'histogramme
plt.figure(figsize=(min_person, 5))
plt.hist(y, bins=np.arange(y.max()+2)-0.5, rwidth=0.7)
plt.xticks(np.unique(y), target_names, rotation=90)
#plt.show()

# 4. Appliquer PCA pour réduire la dimension des données
pca = PCA(n_components=n_component)
X_pca = pca.fit_transform(X)

## Calcule de la variance expliquée
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.figure()
plt.plot(explained_variance_ratio_cumsum)
plt.xlabel('n_components')
plt.ylabel('Explained variance ratio')
plt.grid()
#plt.show()

# 5. Appliquer PCA avec l'option whiten=True
pca = PCA(n_components=n_component, whiten=True)
X_pca_whiten = pca.fit_transform(X)

# 6. Rééquilibrer le jeu de données en se limitant à 50 images par personne
mask = np.zeros(y.shape, dtype=bool)
for target in np.unique(y):
    mask[np.where(y==target)[0][:min_person]] = True

X_balanced = X[mask]
y_balanced = y[mask]

## afficher l'histogramme des classes
plt.figure(figsize=(min_person, 5))
plt.hist(y_balanced, bins=np.arange(y_balanced.max()+2)-0.5, rwidth=0.7)
plt.xticks(np.unique(y_balanced), target_names, rotation=90)
plt.show()