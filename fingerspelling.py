import cv2
import mediapipe as mp
import numpy as np
import joblib


# MediaPipe Hands init
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


# Load saved Model
model = joblib.load('asl_fingerspelling_model.pkl')

cap = cv2.VideoCapture(0)
captured = False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_data_points = []
        for hand in results.multi_hand_landmarks:
            hand_data_point = []
            for point in hand.landmark:
                x = point.x
                y = point.y
                z = point.z
                hand_data_point.extend([x, y, z])
            hand_data_points.append(hand_data_point)

        if captured:
            hand_data_points = np.array(hand_data_points)
            hand_data_points = hand_data_points.reshape(hand_data_points.shape[0], -1)

            prediction = model.predict(hand_data_points)

            predicted_letter = prediction[0]
            print("Predicted Class:", predicted_letter)

            captured = False

    cv2.imshow('Hand Tracking', frame)
    tasto = cv2.waitKey(1)
    if tasto & 0xFF == ord('q'):
        break
    elif tasto & 0xFF == ord('s'):
        captured = True

cap.release()
cv2.destroyAllWindows()
