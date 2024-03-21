import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)