import cv2
import mediapipe as mp
import pygame  # Importer pygame
from Gestes import dictionnaire_gestes

# Initialiser pygame pour la lecture de son
pygame.init()
pygame.mixer.init()
sound = pygame.mixer.Sound('chemin/vers/votre/fichier/sonore.wav')  # Charger le fichier sonore

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for geste in dictionnaire_gestes:
                if geste['fonction'](hand_landmarks):
                    sound = pygame.mixer.Sound(geste['sound'])  # Charger le fichier sonore
                    cv2.putText(frame, geste['message'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
                    sound.play()

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()