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

# Initialization of the sentence variable
sentence = ""

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
            landmarks = hand.landmark
            for landmark in landmarks:
                min_x = min([landmark.x for landmark in landmarks])
                max_x = max([landmark.x for landmark in landmarks])
                min_y = min([landmark.y for landmark in landmarks])
                max_y = max([landmark.y for landmark in landmarks])

                # Width & Height of the Hand Bounding Box
                bbox_width = max_x - min_x
                bbox_height = max_y - min_y

                # Normalization of x / y landmarks coordinates in the bounding box.
                normalized_x = (landmark.x - min_x) / bbox_width
                normalized_y = (landmark.y - min_y) / bbox_height
                z = landmark.z
                hand_data_point.extend([normalized_x, normalized_y, z])
            hand_data_points.append(hand_data_point)

        if captured:
            hand_data_points = np.array(hand_data_points)
            hand_data_points = hand_data_points.reshape(hand_data_points.shape[0], -1)

            prediction = model.predict(hand_data_points)

            predicted_letter = prediction[0]

            if predicted_letter == 'del':
                sentence = sentence[:-1]
            elif predicted_letter == 'space':
                sentence += ' '
            else:
                # Concatenate the predicted letter to the sentence
                sentence += predicted_letter

            captured = False

    # Display the sentence on the video frame
    font_scale = 1  # Adjust this based on your video size
    thickness = 2  # Adjust this based on your preference
    color = (0, 0, 255)  # Color RED

    # Calculation of text length
    (text_width, text_height), _ = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = (frame.shape[1] - text_width) // 2

    # Position the text at the bottom
    y = int(frame.shape[0] - 50)  # Adjust the Y position as needed

    cv2.putText(frame, sentence, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    cv2.imshow('Hand Tracking', frame)

    keyboard = cv2.waitKey(1)
    if keyboard & 0xFF == ord('q'):
        break
    elif keyboard & 0xFF == ord('s'):
        captured = True
    # Text cleaning when Enter button is pressed
    elif keyboard & 0xFF == 13:
        sentence = ''

cap.release()
cv2.destroyAllWindows()
