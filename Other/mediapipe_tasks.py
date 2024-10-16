import cv2
import mediapipe as mp

def is_hand_closed(landmarks):
    # Verifica se tutte le dita sono chiuse confrontando le posizioni delle punte con il polso
    # Landmark 0 è il polso, le punte delle dita sono 8 (indice), 12 (medio), 16 (anulare), 20 (mignolo)
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    finger_tips = [
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
        landmarks[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    # Controlla la distanza tra ogni punta delle dita e il polso
    closed = all(finger_tip.y > wrist.y for finger_tip in finger_tips)
    return closed


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Apri la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converti l'immagine da BGR a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i landmark delle mani
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Disegna i landmark sulla mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmark 0 è il polso, 8 è la punta dell'indice
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Aggiungi qui la logica per riconoscere mano aperta o chiusa
            # Esempio: confrontare la distanza tra polso e la punta delle dita
            # (logica da implementare)

    # Mostra l'immagine
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
