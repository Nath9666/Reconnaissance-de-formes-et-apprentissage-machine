import mediapipe as mp

mp_hands = mp.solutions.hands

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    if index_finger_tip.y > thumb_tip.y:
        return True
    return False

def is_middle_finger(hand_landmarks):
    # Exemple simplifié pour détecter un majeur levé
    # Vous pouvez ajouter des règles plus complexes basées sur les positions des landmarks
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    # Vérifier si le bout du majeur est au-dessus du bout de l'annulaire (ceci est une simplification)
    if middle_finger_tip.y < ring_finger_tip.y:
        return True
    return False

def is_victory(hand_landmarks):
    # Exemple simplifié pour détecter un signe de victoire
    # Vous pouvez ajouter des règles plus complexes basées sur les positions des landmarks
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    # Vérifier si les doigts sont pliés ou non (ceci est une simplification)
    if index_finger_tip.y < middle_finger_tip.y and middle_finger_tip.y < ring_finger_tip.y and ring_finger_tip.y < pinky_tip.y:
        return True
    return False

dictionnaire_gestes = [
    {
        'fonction' : is_fist,
        'message' : 'Fist',
        'sound' : 'MiniProjet\sound\punch.mp3'
    },
    {
        'fonction' : is_middle_finger,
        'message' : 'Middle finger',
        'sound' : 'MiniProjet\sound\damn-youwav.mp3'
    },
    {
        'fonction' : is_victory,
        'message' : 'Victory',
        'sound' : 'MiniProjet\sound\goodresult.mp3'
    }
]