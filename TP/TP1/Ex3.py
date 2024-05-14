import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('./leaf/leaf.csv')

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