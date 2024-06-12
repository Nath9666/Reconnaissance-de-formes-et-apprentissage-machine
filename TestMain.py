import cv2
import mediapipe as mp

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialiser MediaPipe DrawingUtils
mp_drawing = mp.solutions.drawing_utils

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l'image en RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Faire la détection
    results = hands.process(image)

    # Dessiner les résultats
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Afficher l'image
    cv2.imshow('MediaPipe Hands', frame)

    # Si l'utilisateur appuie sur 'q', quitter la boucle
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()