import uuid

import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Blueprint, current_app
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np

import re
from io import BytesIO
from PIL import Image


from keras.applications.mobilenet import decode_predictions
from tensorflow.keras.models import load_model

import base64

import os
import shutil

import tensorflow as tf


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress deprecated warnings

# If you're using TensorFlow 2.x and still seeing deprecation warnings, you might need to adjust the import or logging setup:
import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import mediapipe as mp
from time import time

mobilenet_path = r'C:\Users\anazi\FYP\app\BSL_MobileNet_HD_build_mobilenet_hyper4.h5'
custom_mobilenet = load_model(mobilenet_path)

vggmodel_path = r'C:\Users\anazi\FYP\app\BSL_VGG16_Cus_FT_HD_Best_Model3.h5'
custom_vgg16 = load_model(vggmodel_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Defining a blueprint
views_blueprint = Blueprint('views', __name__, template_folder='templates')

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['CAPTURED_FOLDER'] = 'capture/'
# app.config['CLASSIFIED_FOLDER'] = 'classified'
# CORS(app)


def process_hand_detection(image):
    # Convert the image from BGR (OpenCV format) to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection.
    results = hands.process(image_rgb)  # Set the timestamp explicitly if needed

    if not results.multi_hand_landmarks:
        return None  # No hands detected

    # Assuming only the first hand is of interest.
    for hand_landmarks in results.multi_hand_landmarks:
        # Assuming landmarks are already normalized.
        image_height, image_width, _ = image.shape
        x_coords = [int(landmark.x * image_width) for landmark in hand_landmarks.landmark]
        y_coords = [int(landmark.y * image_height) for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Optionally, expand the bounding box here by a small margin.
        margin = 10  # Adjust as needed.
        x_min = max(0, x_min - margin)
        x_max = min(image_width, x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(image_height, y_max + margin)

        # Crop and return the hand region from the original image.
        hand_img = image[y_min:y_max, x_min:x_max]
        return hand_img

    # If execution reaches this point, no hands were processed (unexpected).
    return None

# @views_blueprint.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/')
@views_blueprint.route('/')
def home():
    return render_template('home.html')


def save_image(data, folder, prefix):
    # uniqe id
    image_id = str(uuid.uuid4())
    filename = f"{prefix}_{image_id}.png"

    #ensure filename ends with .png only once
    if not filename.endswith(".png"):
        filename += ".png"

    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
    return filepath, filename

@views_blueprint.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'GET':
       return render_template('capture.html')
    elif request.method == 'POST':
        if 'image' not in request.json:
            return jsonify({'error': 'No image data provides'}), 400

        image_data = request.json['image']
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image_data = base64.b64decode(image_data)

        captures_files_path = current_app.config['CAPTURED_FOLDER']
        os.makedirs(captures_files_path, exist_ok=True)

        filepath, image_id = save_image(image_data, current_app.config['CAPTURED_FOLDER'], 'cap')

        img = cv2.imread(filepath)
        hand_img = process_hand_detection(img)

        if hand_img is not None:
            cv2.imwrite(filepath, hand_img) #save cropped hand for classification
            img_for_classification = image.load_img(filepath, target_size=(224, 224))
            img_array = np.asarray(img_for_classification)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            cprediction = custom_vgg16.predict(img_array)
            cpred_result = np.argmax(cprediction[0])

            cprediction2 = custom_mobilenet(img_array)
            cpred2_result = np.argmax(cprediction2[0])

            return jsonify({'vgg16_prediction': str(cpred_result), 'mobilenet_prediction': str(cpred2_result), 'image_id': image_id})
        else:
            return jsonify({'error': 'No hand detected'})
    return jsonify({'error': 'File processing error'}), 500

@views_blueprint.route('/capture/<filename>')
def captured_file(filename):
    return send_from_directory(current_app.config['CAPTURED_FOLDER'], filename)

@views_blueprint.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method =='GET':
        return render_template('upload.html')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = secure_filename(file.filename)
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)

        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(filepath)
        filepath, image_id = save_image(file.read(), current_app.config['UPLOAD_FOLDER'], 'upl')

        img = cv2.imread(filepath)
        hand_img = process_hand_detection(img)

        if hand_img is not None:
            cv2.imwrite(filepath, hand_img)
            img_for_classification = image.load_img(filepath, target_size=(224, 224))
            img_array = np.asarray(img_for_classification)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = custom_vgg16.predict(img_array)
            pred_result = np.argmax(prediction[0])

            prediction2 = custom_mobilenet(img_array)
            pred2_result = np.argmax(prediction2[0])

            return jsonify({'vgg16_prediction': str(pred_result), 'mobilenet_prediction': str(pred2_result), 'image_id': image_id})
        else:
            return jsonify({'error': 'No hand detected'})
    return jsonify({'error': 'File processing error'}), 500


@views_blueprint.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)


@views_blueprint.route('/translate-image', methods=['POST'])
def translate_image():
    # Decode the image from base64
    image_data = request.data.decode('utf-8')
    image_data = re.sub('^data:image/.+;base64,', '', image_data)
    img = Image.open(BytesIO(base64.b64decode(image_data)))
    
    # Preprocess the image for MobileNet and VGG16
    img = img.resize((224, 224))
    img_array = np.asarray(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make predictions with MobileNet and VGG16
    mobilenet_predictions = custom_mobilenet.predict(img_array)
    vgg16_predictions = custom_vgg16.predict(img_array)

   # Decode the predictions
    decoded_mobilenet_predictions = decode_predictions(mobilenet_predictions, top=3)[0]
    decoded_vgg16_predictions = decode_predictions(vgg16_predictions, top=3)[0]

# Format the predictions for response
    response_predictions = {
        'mobilenet': [(pred[1], float(pred[2])) for pred in decoded_mobilenet_predictions],
        'vgg16': [(pred[1], float(pred[2])) for pred in decoded_vgg16_predictions]
    }
    response = jsonify(response_predictions)
    return jsonify(response)

@views_blueprint.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    feedback_data = request.get_json()
    image_id = feedback_data['image_id']  # This now includes the prefix and the '.png'
    correct_class = feedback_data['correct_class']
    feedback_correct = feedback_data['feedback_correct']

    # Log feedback for improvement purposes
    print(f"Feedback received: Image ID {image_id}, Correct Class: {correct_class}, Was Correct: {feedback_correct}")

    # Choose the target folder based on the correctness of the prediction
    if feedback_correct:
        target_folder = os.path.join(current_app.config['CLASSIFIED_FOLDER'], 'correct_predictions')
    else:
        target_folder = os.path.join(current_app.config['CLASSIFIED_FOLDER'], 'incorrect_predictions', correct_class)

    os.makedirs(target_folder, exist_ok=True)

    # Use the full image_id (which includes prefix and .png) directly
    source_path = os.path.join(determine_initial_folder(image_id), image_id)

    try:
        shutil.move(source_path, os.path.join(target_folder, image_id))
    except FileNotFoundError as e:
        print(f"Error moving file: {e}")
        return jsonify({'error': 'File not found'}), 404

    return jsonify({'message': 'Feedback processed successfully'})


def move_image_to_class_folder(image_id, class_name, initial_folder):
    target_folder_path = os.path.join(current_app.config['CLASSIFIED_FOLDER'], class_name)
    os.makedirs(target_folder_path, exist_ok=True)
    source_path = os.path.join(initial_folder, f"{image_id}.png")
    shutil.move(source_path, os.path.join(target_folder_path, f"{image_id}.png"))

def determine_initial_folder(image_id):
    try:
        image_id_without_extension = image_id.rsplit('.png', 1)[0]
        prefix = image_id_without_extension.split('_')[0]
    except IndexError:
        current_app.logger.error(f"Invalid image ID format: {image_id}")
        raise ValueError('Invalid image ID format')
    if prefix == 'cap':
        return current_app.config['CAPTURED_FOLDER']
    elif prefix == 'upl':
        return current_app.config['UPLOAD_FOLDER']
    else:
        current_app.logger.error(f"Invalid image ID prefix: {image_id}")
        raise ValueError('Invalid image ID prefix')

