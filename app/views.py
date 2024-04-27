import tempfile

from flask import current_app, Blueprint, render_template, request, send_from_directory, jsonify
import cv2
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import uuid

from werkzeug.utils import secure_filename

import os
import shutil

import tensorflow as tf

from retrain_models import retrain_mobilenet, retrain_vgg16

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress deprecated warnings

import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import mediapipe as mp

import magic #python-magic to check MIME types

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)


views_blueprint = Blueprint('views', __name__, template_folder='templates')

# custom_mobilenet = load_model(current_app.config['mobilenet_path'])
#custom_vgg16 = load_model(current_app.config['vggmodel_path'])

custom_mobilenet = load_model(current_app.config['mobilenet_save_path'])

custom_vgg16 = load_model(current_app.config['vgg_save_path'])

@views_blueprint.route('/')
def home():
    return render_template('home.html')

def process_hand_detection(image):
    # Convert the image from BGR (OpenCV format) to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform hand detection.
    results = hands.process(image_rgb)  # Set the timestamp explicitly if needed

    if not results.multi_hand_landmarks:
        return None  # No hands detected

    # Detect only 1 hand
    for hand_landmarks in results.multi_hand_landmarks:
        # Assuming landmarks are already normalized.
        image_height, image_width, _ = image.shape
        x_coords = [int(landmark.x * image_width) for landmark in hand_landmarks.landmark]
        y_coords = [int(landmark.y * image_height) for landmark in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # expand the bounding box here by a small margin.
        margin = 10  # Adjust as needed.
        x_min = max(0, x_min - margin)
        x_max = min(image_width, x_max + margin)
        y_min = max(0, y_min - margin)
        y_max = min(image_height, y_max + margin)

        # Crop and return the hand region from the original image.
        hand_img = image[y_min:y_max, x_min:x_max]
        return hand_img

    # no hands were found.
    return None

def save_image(data, folder, prefix):
    # uniqe id
    image_id = str(uuid.uuid4())
    filename = f"{prefix}_{image_id}.png"


    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
    return filepath, filename

def save_image_temporarily(data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    temp_path = temp_file.name
    with open(temp_path, 'wb') as file:
        file.write(data)
    return temp_path

@views_blueprint.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'GET':
        return render_template('capture.html')
    if 'image' not in request.files:
        return jsonify({'error': 'No image data provided'}), 400

    image_file = request.files['image']
    consent_given = request.form.get('consent') =='true'

    if image_file:
        image_data = image_file.read()
        if consent_given:
            captures_files_path = current_app.config['CAPTURED_FOLDER']
            os.makedirs(captures_files_path, exist_ok=True)

            filepath, image_id = save_image(image_data, captures_files_path, 'cap')
        else:
            filepath = save_image_temporarily(image_data) #done just to process the image

        img = cv2.imread(filepath)
        hand_img = process_hand_detection(img)

        if hand_img is not None:
            cv2.imwrite(filepath, hand_img)  # save cropped hand for classification
            img_for_classification = image.load_img(filepath, target_size=(224, 224))
            img_array = np.asarray(img_for_classification)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            mobilenet_prediction = custom_mobilenet.predict(img_array)
            vgg_prediction = custom_vgg16.predict(img_array)

            mobilenet_pred_class = np.argmax(mobilenet_prediction[0])
            mobilenet_confidence = np.max(mobilenet_prediction[0])

            vgg_pred_class = np.argmax(vgg_prediction[0])
            vgg_confidence = np.max(vgg_prediction[0])

            if not consent_given:
                os.remove(filepath) #remove temp img

            return jsonify({
                'mobilenet_prediction': str(mobilenet_pred_class),
                'mobilenet_confidence': float(mobilenet_confidence),
                'vgg16_prediction': str(vgg_pred_class),
                'vgg16_confidence': float(vgg_confidence),
                'image_id': image_id if consent_given else "Consent not given"
            })
        else:
            if not consent_given:
                os.remove(filepath)
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
    consent_given = request.form.get('consent') == 'true'

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)

        if consent_given:
            os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath, image_id = save_image(file.read(), current_app.config['UPLOAD_FOLDER'], 'upl')
        else:
            filepath = save_image_temporarily(file.read())

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
            prediction_confidence = np.max(prediction[0])

            prediction2 = custom_mobilenet(img_array)
            pred2_result = np.argmax(prediction2[0])
            prediction2_confidence = np.max(prediction2[0])

            if not consent_given:
                os.remove(filepath)

            return jsonify({
                'vgg16_prediction': str(pred_result),
                 'vgg16_confidence': float(prediction_confidence),
                 'mobilenet_prediction': str(pred2_result),
                 'mobilenet_confidence': float(prediction2_confidence),
                 'image_id': image_id if consent_given else "Consent not given"
            })
        else:
            if not consent_given:
                os.remove(filepath)
            return jsonify({'error': 'No hand detected'})
    return jsonify({'error': 'File processing error'}), 500

@views_blueprint.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)



@views_blueprint.route('/submit_feedback', methods=['POST'])
def handle_feedback():
    feedback_data = request.get_json()
    image_id = feedback_data['image_id']  # This now includes the prefix and the '.png'
    correct_class_vgg = feedback_data['correct_class_vgg']
    correct_class_mobilenet = feedback_data['correct_class_mobilenet']
    feedback_correct_vgg = feedback_data['feedback_correct_vgg']
    feedback_correct_mobilenet = feedback_data['feedback_correct_mobilenet']

    # Log feedback for improvement purposes
    print(
        f"Feedback received: Image ID {image_id}, VGG Correct Class: {correct_class_vgg}, VGG Was Correct: {feedback_correct_vgg}")
    print(
        f"Feedback received: Image ID {image_id}, MobileNet Correct Class: {correct_class_mobilenet}, MobileNet Was Correct: {feedback_correct_mobilenet}")

    # Define target folders for VGG and MobileNet based on feedback correctness
    target_folder_vgg = os.path.join(current_app.config['VGG_CLASSIFIED'],
                                     'correct' if feedback_correct_vgg else 'incorrect', correct_class_vgg)
    target_folder_mobilenet = os.path.join(current_app.config['MOBILENET_CLASSIFIED'],
                                           'correct' if feedback_correct_mobilenet else 'incorrect',
                                           correct_class_mobilenet)

    os.makedirs(target_folder_vgg, exist_ok=True)
    os.makedirs(target_folder_mobilenet, exist_ok=True)

    # Retrieve the initial folder and copy the image to each model's feedback directory
    initial_folder = determine_initial_folder(image_id)
    source_path = os.path.join(initial_folder, image_id)

    try:
        # Copy file to VGG classified folder
        shutil.copy(source_path, os.path.join(target_folder_vgg, image_id))
        # Move file to MobileNet classified folder (to avoid having duplicate files)
        shutil.move(source_path, os.path.join(target_folder_mobilenet, image_id))
    except FileNotFoundError as e:
        print(f"Error moving file: {e}")
        return jsonify({'error': 'File not found'}), 404

    return jsonify({'message': 'Feedback processed successfully for both models'})

def determine_initial_folder(image_id):
    prefix = image_id.split('_')[0]
    if prefix == 'cap':
        return current_app.config['CAPTURED_FOLDER']
    elif prefix == 'upl':
        return current_app.config['UPLOAD_FOLDER']
    else:
        current_app.logger.error(f"Invalid image ID prefix: {image_id}")
        raise ValueError('Invalid image ID prefix')

def initialize_models():
    global custom_mobilenet, custom_vgg16
    custom_mobilenet = load_model(current_app.config['mobilenet_save_path'])
    custom_vgg16 = load_model(current_app.config['vgg_save_path'])
    print("Models initialized successfully.")

def load_models():
    global custom_mobilenet, custom_vgg16
    if not custom_mobilenet:
        custom_mobilenet = load_model(current_app.config['mobilenet_path'])
    if not custom_vgg16:
        custom_vgg16 = load_model(current_app.config['vggmodel_path'])

@views_blueprint.route('/reload_models', methods=['POST'])
def reload_models():
    load_models()
    return jsonify({"message": "Models reloaded successfully!"})


@views_blueprint.route('/retrain', methods=['POST'])
def retrain_models():
    if 'custom_mobilenet' not in globals() or 'custom_vgg16' not in globals():
        return jsonify({"error": "Models are not loaded"}), 500

    mobilenet_save_path = current_app.config['mobilenet_save_path']
    vgg_save_path = current_app.config['vgg_save_path']

    retrain_mobilenet(
        custom_mobilenet,
        train_dir=current_app.config['MOBILENET_INCORRECT'],
        val_dir=current_app.config['VGG_INCORRECT'],
        save_path=mobilenet_save_path
    )
    retrain_vgg16(
        custom_vgg16,
        train_dir=current_app.config['VGG_INCORRECT'],
        val_dir=current_app.config['MOBILENET_INCORRECT'],
        save_path=vgg_save_path
    )

    initialize_models()
    return jsonify({"message": "Models retrained successfully!"}), 200