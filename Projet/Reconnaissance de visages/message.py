import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from skimage.feature import hog
import cv2

# Fonction pour lire les coordonnées à partir d'un fichier texte
def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates = []
        for line in lines:
            parts = line.strip().split()
            x_center, y_center, width, height = map(float, parts[1:])
            coordinates.append((x_center, y_center, width, height))
        return coordinates

# Chemin du dossier contenant les images et les coordonnées
image_folder = '../assets/archive/images/train/'
coordinates_folder = '../assets/archive/labels2/'

# Lister tous les fichiers d'image
image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

# Taille fixe pour les images de visage
fixed_size = (64, 64)

# Initialiser les listes pour les images de visages et les étiquettes
face_images = []
labels = []  # Remplacer par les étiquettes réelles associées à chaque image

# Lire et traiter chaque image et ses coordonnées associées
for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_folder, image_file)
    coord_file_path = os.path.join(coordinates_folder, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))
    
    # Lire l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image au chemin {image_path}")
        continue
    
    # Lire les coordonnées
    coordinates = read_coordinates(coord_file_path)
    
    # Découper les visages en utilisant les coordonnées
    for (x_center, y_center, width, height) in coordinates:
        h, w = image.shape[:2]
        x = int((x_center - width / 2) * w)
        y = int((y_center - height / 2) * h)
        w = int(width * w)
        h = int(height * h)
        face = image[y:y+h, x:x+w]
        resized_face = cv2.resize(face, fixed_size)  # Redimensionner à la taille fixe
        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
        face_images.append(gray_face)
        # Ajoutez l'étiquette correspondante ici, par exemple :
        labels.append(idx % 2)  # Exemple : alterner entre deux classes

# Convertir les listes en tableaux numpy
face_images = np.array(face_images)
labels = np.array(labels)

# Appliquer l'ACP sur les images de visages
pca = PCA(n_components=100)
pca_features = pca.fit_transform([face.flatten() for face in face_images])

# Calculer les descripteurs HOG
hog_features = [hog(face, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False) for face in face_images]

# Diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
pca_train, pca_test, labels_train, labels_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)
hog_train, hog_test, labels_train, labels_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Approches classiques de reconnaissance de visages

# Utiliser l'ACP + Decision Tree
clf_pca = DecisionTreeClassifier()
clf_pca.fit(pca_train, labels_train)
predictions_pca = clf_pca.predict(pca_test)

# Utiliser HOG + Decision Tree
clf_hog_dt = DecisionTreeClassifier()
clf_hog_dt.fit(hog_train, labels_train)
predictions_hog_dt = clf_hog_dt.predict(hog_test)

# Utiliser HOG + Random Forest
clf_hog_rf = RandomForestClassifier()
clf_hog_rf.fit(hog_train, labels_train)
predictions_hog_rf = clf_hog_rf.predict(hog_test)

# Évaluer les performances
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy PCA: ", accuracy_score(labels_test, predictions_pca))
print("Report PCA:\n", classification_report(labels_test, predictions_pca))

print("Accuracy HOG + Decision Tree: ", accuracy_score(labels_test, predictions_hog_dt))
print("Report HOG + Decision Tree:\n", classification_report(labels_test, predictions_hog_dt))

print("Accuracy HOG + Random Forest: ", accuracy_score(labels_test, predictions_hog_rf))
print("Report HOG + Random Forest:\n", classification_report(labels_test, predictions_hog_rf))
