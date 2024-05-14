from Ex1 import *
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA

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
pca.fit(iris.data)

# Afficher les composantes principales
print("Composantes principales :\n", pca.components_)