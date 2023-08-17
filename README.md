# ASL Fingerspelling Recognition - Google MediaPipe Hands + Support Vector Machine

This project utilizes Python and TensorFlow to create a Support Vector Machine (SVM) model, using the `SVC` class from the `sklearn.svm` module, for recognizing American Sign Language (ASL) alphabet signs. 
The model classifies the coordinates of 21 Hand Data Points detected using Google MediaPipe Hands. The training data was obtained by applying Google MediaPipe Hands to convert each ASL sign image into a text file containing the coordinates of Hand Data Points (this was done in the `dataset_conversion.py` file). The model is then used in the `fingerspelling.py` file to classify webcam captures triggered by pressing the "s" key. For each captured image, Google MediaPipe Hands is applied to detect and classify the coordinates of Hand Data Points using the trained SVM model.

## Requirements

Before running the program, make sure you have the following requirements installed:

- Python 3.x: [https://www.python.org/downloads/](https://www.python.org/downloads/)
- TensorFlow 2.x: [https://www.tensorflow.org/install](https://www.tensorflow.org/install)
- scikit-learn: [https://scikit-learn.org/stable/install.html](https://scikit-learn.org/stable/install.html)
- Google MediaPipe: [https://google.github.io/mediapipe/getting_started/install.html](https://google.github.io/mediapipe/getting_started/install.html)

It is recommended to use a virtual environment to install and manage the project's dependencies.

## Installation

1. Clone the repository or download the source code: 
```
git clone https://github.com/crenar8/fingerspelling-MediaPipeSVMNet.git
cd fingerspelling-MediaPipeSVMNet
```

2. Install the required dependencies using pip:
```
pip install cv2
pip install numpy
pip install keras
pip install collections
pip install joblib
```

## Dataset Conversion

To convert the ASL sign images into text files containing the coordinates of Hand Data Points, follow these steps:

1. Prepare a dataset of ASL sign images in a directory.

2. Open the `dataset_conversion.py` file and modify the file paths and parameters according to your dataset.

3. Run the `dataset_conversion.py` script:
```
python dataset_conversion.py
```

4. The script will apply Google MediaPipe Hands to each image in the dataset, extract the Hand Data Points coordinates, and save them in text files.

## Training the SVM Model

To train the SVM model using the extracted Hand Data Points coordinates, follow these steps:

1. Prepare the text files containing the Hand Data Points coordinates in the ASL sign dataset directory.

2. Open the `fingerspelling_model_creation.py` file and modify the file paths and parameters according to your dataset.

3. Run the `fingerspelling_model_creation.py` script:
```
python fingerspelling_model_creation.py
```

4. The script will load the Hand Data Points coordinates from the text files, train the SVM model, and save it to a file.



## Usage

To run the program and perform ASL alphabet sign recognition using the webcam, follow these steps:

1. Navigate to the project directory:
```
cd fingerspelling-MediaPipeSVMNet
```

2. Run the `fingerspelling.py` script:
```
python fingerspelling.py
```

3. When the webcam feed appears, press the "s" key to capture an image for classification.

4. Google MediaPipe Hands will be applied to detect the Hand Data Points coordinates in the captured image.

5. The SVM model will classify the coordinates of Hand Data Points and display the predicted ASL alphabet sign on the

![Instructions.png](..%2FInstructions.png)

## Customization

If you want to customize or train your own model, you can modify the code in the `fingerspelling_model_creation.py` file. Adjust the model architecture, hyperparameters, and training configurations according to your requirements.

## Contributing

Contributions to this project are welcome. If you find any issues or want to suggest enhancements, please create an issue or submit a pull request.

## License

This project is licensed under GNU License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- The ASL Alphabet dataset used for training the model was obtained from [source](https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out).
- Google MediaPipe Hands is a computer vision framework developed by Google. For more information about Google MediaPipe Hands, please refer to the following publication: [MediaPipe Hands: On-device real-time hand tracking](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## Contact

For any questions or inquiries, please contact [k.dervishi@students.uninettunouniversity.net](mailto:k.dervishi@students.uninettunouniversity.net)









