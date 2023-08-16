import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Data loading from a .txt file
def load_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            coordinates = [float(valore) for valore in line.strip().split()]
            data.append(coordinates)
    return data

# Data loading from directory and subdirectories
def load_data_from_dir(cartella):
    data = []
    for root, dirs, files in os.walk(cartella):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_data = load_data_from_file(file_path)
                data.append(file_data)
    return np.array(data)

# Specifying directories
training_directory = 'dataset/training'
test_directory = 'dataset/test'

# Training data loading
training_data = load_data_from_dir(training_directory)
test_data = load_data_from_dir(test_directory)

# Labels loading
training_labels = []
test_labels = []
for claz in os.listdir(training_directory):
    training_subdirectory = os.path.join(training_directory, claz)
    training_class_data_number = len(os.listdir(training_subdirectory))
    training_labels.extend([claz] * training_class_data_number)

for claz in os.listdir(test_directory):
    test_subdirectory = os.path.join(test_directory, claz)
    test_class_data_number = len(os.listdir(test_subdirectory))
    test_labels.extend([claz] * test_class_data_number)

# SVM classifier creation
model = SVC(kernel='linear')

# SVM Training
model.fit(training_data.reshape(training_data.shape[0], -1), training_labels)

# Model saving
model_name = 'asl_fingerspelling_model.pkl'
joblib.dump(model, model_name)

# Model Evaluation using the Test Set
y_pred = model.predict(test_data.reshape(test_data.shape[0], -1))
report = classification_report(test_labels, y_pred)

# Print the report to a file
with open('model_evaluation_report.txt', 'w') as f:
    print(report, file=f)