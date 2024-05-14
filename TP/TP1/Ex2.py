from Ex1 import *
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Charger le dataset iris
iris = load_iris()

# Créer un DataFrame à partir des données iris
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

def explore_dataset (df: pd.DataFrame) -> None:
    """
    Explore the given dataset by displaying the first 5 rows, the information and the descriptive statistics of the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be explored.

    Returns:
        None

    """
    print("First 5 rows of the DataFrame:")
    print(df.head())
    print("-----------------------------")

    print("Information about the DataFrame:")
    print(df.info())
    print("-----------------------------")

    print("Descriptive statistics of the DataFrame:")
    print(df.describe())
    print("-----------------------------")

#? Explore the dataset
#explore_dataset(iris_df)

#? Appliquer l'ACP avec 2 composantes principales
transformed_data, explained_variance_ratio = compute_ACP(iris.data, n_components=2)

# Vérifier la dimension des données après application de l'ACP
print("Dimension des donnees apres application de l'ACP :", transformed_data.shape)

#? Affiche l'acp et les composantes principales
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(iris.data)

# Afficher les composantes principales
print("Composantes principales :\n", pca.components_)

# Créer un DataFrame à partir des données transformées
transformed_df = pd.DataFrame(data=transformed_data, columns=['Principal Component 1', 'Principal Component 2'])

# Ajouter les labels des classes au DataFrame
transformed_df['Target'] = iris.target

# Afficher chaque classe avec une couleur différente
colors = ['r', 'g', 'b']
targets = [0, 1, 2]

for target, color in zip(targets, colors):
    indicesToKeep = transformed_df['Target'] == target
    plt.scatter(transformed_df.loc[indicesToKeep, 'Principal Component 1'], transformed_df.loc[indicesToKeep, 'Principal Component 2'], c=color)

# Calculer la proportion de variance expliquée pour chaque composante principale
explained_variance_ratio = pca.explained_variance_ratio_

# Convertir la proportion de variance expliquée en pourcentage
explained_variance_ratio_percentage = explained_variance_ratio * 100

# Afficher la proportion de variance expliquée pour chaque composante principale en pourcentage
print("Explained Variance Ratio (in %): ", explained_variance_ratio_percentage)

plt.xlabel('Principal Component 1 ({}%)'.format(round(explained_variance_ratio_percentage[0], 2)))
plt.ylabel('Principal Component 2 ({}%)'.format(round(explained_variance_ratio_percentage[1], 2)))
plt.title('2D PCA of Iris Dataset')
plt.legend(iris.target_names)
plt.grid()
plt.show()