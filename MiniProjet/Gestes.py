import mediapipe as mp

mp_hands = mp.solutions.hands

def is_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    if index_finger_tip.y > thumb_tip.y:
        return True
    return False

def is_middle_finger(hand_landmarks):
    # Obtenir les landmarks nécessaires
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]  # Articulation intermédiaire du majeur

    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]  # Articulation interphalangienne du pouce

    # Vérifier si le majeur est tendu
    middle_finger_up = middle_finger_tip.y < middle_finger_pip.y

    # Vérifier si les autres doigts sont pliés
    index_finger_down = index_finger_tip.y > index_finger_pip.y
    ring_finger_down = ring_finger_tip.y > ring_finger_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y
    thumb_down = thumb_tip.y > thumb_ip.y

    # Si le majeur est tendu et les autres doigts pliés
    if middle_finger_up and index_finger_down and ring_finger_down and pinky_down and thumb_down:
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