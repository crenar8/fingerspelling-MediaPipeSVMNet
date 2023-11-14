import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
from flask_basicauth import BasicAuth

# MediaPipe Hands init
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load saved Model
model = joblib.load('asl_fingerspelling_model.pkl')

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

app.config['BASIC_AUTH_USERNAME'] = 'Uninettuno'
app.config['BASIC_AUTH_PASSWORD'] = 'Dervishi2811'

basic_auth = BasicAuth(app)

api_lock = threading.Lock()


@app.route('/')
@basic_auth.required
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    with api_lock:
        predicted_letter = None
        # Get the image from the request
        image_stream = request.files['image']

        # Use PIL to open image in memory
        image = Image.open(image_stream)

        # Convert to NumPy array
        image_np = np.array(image)

        # Convert to BGR (OpenCV uses BGR by default)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # Detect the hands in the image
        results = hands.process(image_bgr)

        # Extract the landmarks from the hands
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

            # Classify the landmarks with the model
            hand_data_points = np.array(hand_data_points)
            hand_data_points = hand_data_points.reshape(hand_data_points.shape[0], -1)

            prediction = model.predict(hand_data_points)

            predicted_letter = prediction[0]
            if predicted_letter == 'space':
                predicted_letter = ' '

        if predicted_letter is not None:
            return jsonify({"predicted_letter": predicted_letter})
        return '', 204


if __name__ == '__main__':
    app.run(debug=True)
