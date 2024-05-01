#Hand Gesture Recognition System
This repository contains a Flask application designed for real-time hand gesture recognition. The system utilizes TensorFlow, Keras, and MediaPipe to detect hand gestures from live camera feeds and images uploaded through a web interface. The application can classify hand gestures using pretrained models based on MobileNet and VGG16 architectures.


#Features
Real-time hand detection: Live detection of hand gestures from a webcam stream.
Image upload for gesture recognition: Users can upload images to be processed for hand gesture recognition.
Model retraining: Interface for retraining models based on user feedback.
Secure HTTP deployment: Configured to enforce HTTPS for secure communication.


##Prerequisites
Python 3.6+
Flask
TensorFlow 2.15.0
Keras
OpenCV
MediaPipe
Flask-Session
APScheduler
Flask-Talisman for HTTPS support

Note:
I am using TensorFlow 2.15.0 as that is the version my model was trained and saved with in Colab


#Configuration
Before running the application, ensure the model paths and other configurations are set in __init__.py:

UPLOAD_FOLDER: Directory for storing uploaded images.
CAPTURED_FOLDER: Directory for storing images captured from the live feed.
CLASSIFIED_FOLDER: Directory for storing classified images based on model predictions.


#Features and Routes
/: The home page shows basic information and links to other sections.
/upload: Allows users to upload images for gesture recognition.
/capture: Interface for capturing images from a webcam and detecting gestures.
/retrain: Endpoint for retraining the models based on accumulated feedback.


#Retraining Models
The application supports retraining the MobileNet and VGG16 models. To initiate retraining, navigate to the /retrain route and submit the request. Ensure the dataset and incorrect classifications are appropriately set up for effective retraining.

#Security
The application uses Flask-Talisman to enforce HTTPS. Ensure that proper SSL certificates are configured for deployment in a production environment.

