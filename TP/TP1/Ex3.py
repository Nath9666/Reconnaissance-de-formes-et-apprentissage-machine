import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('C:\\Users\\Nathan\\Documents\\Projet\\Reconnaissance de formes et apprentissage machine\\TP\\TP1\\leaf\\leaf.csv')

# Appliquer l'ACP
pca = PCA()
pca.fit(df)

# Obtenir les valeurs propres
eigenvalues = pca.explained_variance_

# Créer un graphique pour la décroissance des valeurs propres
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues)
plt.xlabel('Nombre de composantes')
plt.ylabel('Valeurs propres')
plt.title('Décroissance des valeurs propres')
plt.grid()
plt.show()

from mpl_toolkits.mplot3d import Axes3D

# Appliquer l'ACP avec 3 composantes principales
pca = PCA(n_components=3)
pca_result = pca.fit_transform(df.iloc[:, :-1])

# Créer un graphique 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Utiliser l'étiquette de classe pour donner une couleur à chaque classe
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=df.iloc[:, -1])

# Ajouter une légende
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# Ajouter des étiquettes d'axe
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()