import os
import uuid

import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, Blueprint
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
from keras.applications import MobileNet

import base64
import re
from io import BytesIO
from PIL import Image

from flask_cors import CORS

from keras.applications.mobilenet import decode_predictions
import tensorflow.keras.models
from tensorflow.keras.models import load_model

import base64
import time

import os
import shutil

import tensorflow as tf
import logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Suppress deprecated warnings

# If you're using TensorFlow 2.x and still seeing deprecation warnings, you might need to adjust the import or logging setup:
import tensorflow.compat.v1 as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

mobilenet = MobileNet(weights='imagenet')
# vgg16 = VGG16(weights='imagenet')

vggmodel_path = r'C:\Users\anazi\FYP\app\BSL_VGG16_Cus_FT_Best_Model2.h5'
custom_vgg16 = load_model(vggmodel_path)

# Defining a blueprint
views_blueprint = Blueprint('views', __name__, template_folder='templates')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['CAPTURED_FOLDER'] = 'capture/'
app.config['CLASSIFIED_FOLDER'] = 'classified'
CORS(app)


# @app.route('/')
@views_blueprint.route('/')
def home():
    return render_template('home.html')

def save_image(data, folder, prefix):
    image_id = str(uuid.uuid4())
    filename = f"{prefix}_{image_id}.png"
    filepath = os.path.join(folder, filename)
    with open(filepath, 'wb') as file:
        file.write(data)
    return filepath, image_id

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

        captures_files_path = app.config['CAPTURED_FOLDER']
        os.makedirs(captures_files_path, exist_ok=True)

        # filename = f"captured_{int(time.time())}.png"
        # image_id = str(uuid.uuid4())
        filepath, image_id = save_image(image_data, app.config['CAPTURED_FOLDER'], 'cap')
        # filepath = os.path.join(captures_files_path, filename)
        with open(filepath, 'wb') as file:
            file.write(image_data)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        cprediction = custom_vgg16.predict(img_array)
        cpred_result = np.argmax(cprediction[0])

        return jsonify({'prediction': str(cpred_result), 'image_id': image_id})

@views_blueprint.route('/capture/<filename>')
def captured_file(filename):
    return send_from_directory(app.config['CAPTURED_FOLDER'], filename)

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
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(filepath)
        filepath, image_id = save_image(file.read(), app.config['CAPTURED_FOLDER'], 'upl')

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = np.asarray(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = custom_vgg16.predict(img_array)
        pred_result = np.argmax(prediction[0])

        return jsonify({'prediction': str(pred_result)})

    return jsonify({'error': 'File processing error'}), 500


@views_blueprint.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


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
    mobilenet_predictions = mobilenet.predict(img_array)
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
    image_id = feedback_data['image_id']
    correct_class = feedback_data['correct_class']
    initial_folder = determine_initial_folder(image_id)
    move_image_to_class_folder(image_id, correct_class, initial_folder)
    return jsonify({'message': 'Feedback processed successfully'})

def move_image_to_class_folder(image_id, class_name, initial_folder):
    target_folder_path = os.path.join(app.config['CLASSIFIED_FOLDER'], class_name)
    os.makedirs(target_folder_path, exist_ok=True)
    source_path = os.path.join(initial_folder, f"{image_id}.png")
    shutil.move(source_path, os.path.join(target_folder_path, f"{image_id}.png"))

def determine_initial_folder(image_id):
    prefix = image_id.split('_')[0]
    if prefix == 'cap':
        return app.config['CAPTURED_FOLDER']
    elif prefix == 'upl':
        return app.config['UPLOAD_FOLDER']
    else:
        raise ValueError('Invalid image ID')

