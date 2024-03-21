from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/capture', methods=['GET', 'POST'])
def capture_image():
    if request.method == 'POST':
        # Handle image capture here
        pass
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

if __name__ == '__main__':
    app.run(debug=True)