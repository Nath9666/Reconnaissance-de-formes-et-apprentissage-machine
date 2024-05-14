import Ex1
from sklearn.datasets import load_iris
import pandas as pd

# Charger le dataset iris
iris = load_iris()

# Créer un DataFrame à partir des données iris
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Afficher les premières lignes du DataFrame
print("First 5 rows of the DataFrame:")
print(iris_df.head())
print("-----------------------------")

# Afficher les informations sur le DataFrame
print("Information about the DataFrame:")
print(iris_df.info())
print("-----------------------------")

# Afficher les statistiques descriptives du DataFrame
print("Descriptive statistics of the DataFrame:")
print(iris_df.describe())
print("-----------------------------")
