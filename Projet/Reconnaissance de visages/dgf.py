#3.3 Travail demandé


"""1. Ecrire un programme Python permettant de reconnaitre un visage en utilisant l’ACP."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random
import cv2


# Charger les données de visage LFW
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)
X = lfw_dataset.data
y = lfw_dataset.target
target_names = lfw_dataset.target_names

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Appliquer l'ACP pour réduire la dimensionnalité des données de visage
n_components = 150  # nombre de composantes principales
pca = PCA(n_components=n_components, whiten=True, random_state=42)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Entraîner un classifieur sur les données réduites
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train_pca, y_train)

# Prédire les étiquettes sur l'ensemble de test
y_pred = clf.predict(X_test_pca)

# Évaluer la performance du modèle
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred))

# Afficher un résultat de reconnaissance de visage
def plot_single_image(image, title, h, w):
    plt.figure(figsize=(4, 6))
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    plt.xticks(())
    plt.yticks(())
    plt.show()

# Préparer le titre pour l'image prédite
def single_title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'pred: %s\ntrue: %s' % (pred_name, true_name)

# Afficher une seule image avec son titre
h, w = lfw_dataset.images.shape[1:3]
i = random.randint(0, len(X_test) - 1)  # Choisir un index aléatoire
title = single_title(y_pred, y_test, target_names, i)
plot_single_image(X_test[i], title, h, w)

cv2.imwrite('lfw_face.jpg', X_test[i].reshape((h, w)))