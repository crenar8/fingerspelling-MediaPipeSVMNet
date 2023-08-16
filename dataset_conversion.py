import os
import cv2
import mediapipe as mp

# Specifying directories
dataset_root = 'image_dataset'
training_dir = os.path.join(dataset_root, 'training')
test_dir = os.path.join(dataset_root, 'test')
new_dataset_root = 'dataset'
new_training_dir = os.path.join(new_dataset_root, 'training')
new_test_dir = os.path.join(new_dataset_root, 'test')

# MediaPipe Hands init
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)


# Hand Data Points retrieving
def detect_hand_data_points(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # HDP detection
    results = mp_hands.process(image_rgb)

    hand_data_points = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Coordinates extractions
            landmarks = hand_landmarks.landmark
            for landmark in landmarks:
                min_x = min([landmark.x for landmark in landmarks])
                max_x = max([landmark.x for landmark in landmarks])
                min_y = min([landmark.y for landmark in landmarks])
                max_y = max([landmark.y for landmark in landmarks])

                # Width & Height of the Hand Bounding Box
                bbox_width = max_x - min_x
                bbox_height = max_y - min_y

                # Normalization of x / y landmarks coordinates in the bounding box.
                normalized_lm_x = (landmark.x - min_x) / bbox_width
                normalized_lm_y = (landmark.y - min_y) / bbox_height
                hand_data_points.append((normalized_lm_x, normalized_lm_y, landmark.z))

    return hand_data_points


# Dataset creation with HDP coordinates
def create_new_dataset():
    # New dataset directories creation
    os.makedirs(new_training_dir, exist_ok=True)
    os.makedirs(new_test_dir, exist_ok=True)

    # Training dataset elaboration starting ...
    for root, dirs, files in os.walk(training_dir):
        for directory in dirs:
            source_dir = os.path.join(training_dir, directory)
            target_dir = os.path.join(new_training_dir, directory)
            os.makedirs(target_dir, exist_ok=True)
            for file in os.listdir(source_dir):
                image_path = os.path.join(source_dir, file)
                hand_data_points = detect_hand_data_points(image_path)
                if hand_data_points:
                    text_file_path = os.path.join(target_dir, os.path.splitext(file)[0] + '.txt')
                    print('Just printed HDPs in ' + text_file_path)
                    with open(text_file_path, 'w') as f:
                        for point in hand_data_points:
                            f.write(f'{point[0]} {point[1]} {point[2]}\n')

    # Test dataset elaboration starting ...
    for root, dirs, files in os.walk(test_dir):
        for directory in dirs:
            source_dir = os.path.join(test_dir, directory)
            target_dir = os.path.join(new_test_dir, directory)
            os.makedirs(target_dir, exist_ok=True)
            for file in os.listdir(source_dir):
                image_path = os.path.join(source_dir, file)
                hand_data_points = detect_hand_data_points(image_path)
                if hand_data_points:
                    text_file_path = os.path.join(target_dir, os.path.splitext(file)[0] + '.txt')
                    with open(text_file_path, 'w') as f:
                        for point in hand_data_points:
                            f.write(f'{point[0]} {point[1]} {point[2]}\n')


# Dataset creation function call
create_new_dataset()
