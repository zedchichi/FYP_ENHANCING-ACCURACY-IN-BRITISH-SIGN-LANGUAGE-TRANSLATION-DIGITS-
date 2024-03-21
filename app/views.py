import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from keras.preprocessing import image
import numpy as np
from keras.applications import MobileNet, VGG16

import base64
import re
from io import BytesIO
from PIL import Image

from flask_cors import CORS

from keras.applications.mobilenet import decode_predictions


mobilenet = MobileNet(weights='imagenet')
vgg16 = VGG16(weights='imagenet')



app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
CORS(app)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture_image():
    if request.method == 'POST':
        # Get access to the user's webcam
        cap = cv2.VideoCapture(0)

        # Set up the capture button
        capture_button = request.form.get('capture-button')
        if capture_button:
            # Capture an image from the webcam
            ret, frame = cap.read()
            if ret:
                # Save the captured image to a file
                filename = 'captured_image.jpg'
                filepath = os.path.join('uploads', filename)
                cv2.imwrite(filepath, frame)

                # Release the webcam and redirect to the upload page
                cap.release()
                return redirect(url_for('uploaded_file', filename=filename))

        # Release the webcam and render the capture page
        cap.release()
    return render_template('camera_capture.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/translate-image', methods=['POST'])
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
    vgg16_predictions = vgg16.predict(img_array)

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

if __name__ == '__main__':
    app.run(debug=True)