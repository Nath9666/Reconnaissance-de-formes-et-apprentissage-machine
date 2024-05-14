import numpy as np
import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('C:\\Users\\Nathan\\Documents\\Projet\\Reconnaissance de formes et apprentissage machine\\TP\\TP1\\MPEG-7_data_arc_length parametrization.csv')

# Afficher les premières lignes du DataFrame
print(df.head())

original_contour = df.iloc[0].values

def fourier_descriptors(contour):
    # Calculer la transformée de Fourier discrète
    fourier_transform = np.fft.fft(contour)

    # Ignorer le premier descripteur pour l'invariance à la translation
    fourier_transform = fourier_transform[1:]

    # Utiliser le module pour l'invariance à la rotation
    fourier_transform = np.abs(fourier_transform)

    # Normaliser pour l'invariance à l'échelle
    fourier_transform = fourier_transform / fourier_transform[0]

    # Retourner les descripteurs de Fourier
    return fourier_transform

# Calculer les descripteurs de Fourier du contour original
original_descriptors = fourier_descriptors(original_contour)

tolerance = 1e-8  # Définir la tolérance

# Déplacer, faire pivoter et redimensionner le contour
translated_contour = original_contour + 10
rotated_contour = np.roll(original_contour, 1)
rescaled_contour = original_contour * 2

# Calculer les descripteurs de Fourier des contours modifiés
translated_descriptors = fourier_descriptors(translated_contour)
rotated_descriptors = fourier_descriptors(rotated_contour)
rescaled_descriptors = fourier_descriptors(rescaled_contour)

# Vérifier les propriétés d'invariance avec la tolérance
print("Invariance a la translation : ", np.allclose(original_descriptors[1:], translated_descriptors[1:], atol=tolerance))
print("Invariance a la rotation : ", np.allclose(np.abs(original_descriptors[1:]), np.abs(rotated_descriptors[1:]), atol=tolerance))
print("Invariance a l'echelle : ", np.allclose(original_descriptors[1:] / original_descriptors[1], rescaled_descriptors[1:] / rescaled_descriptors[1], atol=tolerance))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Définir les étiquettes de classe réelles
# Ici, nous supposons que les deux premiers contours appartiennent à la même classe (étiquette 1), et les autres à une autre classe (étiquette 0)
y_true = np.array([1, 1] + [0] * (len(df) - 2))

# Calculer les scores de prédiction en utilisant une mesure de distance
# Ici, nous utilisons la distance euclidienne entre les descripteurs de Fourier du contour original et ceux des autres contours
y_scores = np.array([np.linalg.norm(fourier_descriptors(df.iloc[i].values)[1:] - original_descriptors[1:]) for i in range(len(df))])

# Calculer la précision et le rappel pour différents seuils
precision, recall, _ = precision_recall_curve(y_true, y_scores)

# Tracer la courbe Précision-Rappel
plt.plot(recall, precision)
plt.xlabel('Rappel')
plt.ylabel('Précision')
plt.title('Courbe Précision-Rappel')
plt.show()