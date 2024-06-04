from flask import Flask, render_template, request, redirect, url_for, jsonify  # Import jsonify
from werkzeug.utils import secure_filename  # Import secure_filename
import requests
import os
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Replace with the public URL of your Colab server
COLAB_SERVER_URL = 'https://h2mpbvbj4jm-496ff2e9c6d22116-5000-colab.googleusercontent.com/'

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(image_array):
    image = Image.fromarray(np.uint8(image_array))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify(error='No file part'), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='No selected file'), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        with open(filepath, 'rb') as f:
            try:
                response = requests.post(f"{COLAB_SERVER_URL}/predict", files={'file': f}, verify=False)
                response.raise_for_status()
            except requests.RequestException as e:
                return jsonify(error=f"Request to Colab server failed: {e}"), 400
        
        try:
            result = response.json()
        except ValueError:
            return jsonify(error='Invalid JSON response from Colab server'), 400

        if 'error' in result:
            return jsonify(error=result['error']), 400

        try:
            preprocessed_image = encode_image(np.array(result['preprocessed_image'], dtype=np.uint8))
            augmented_images = [encode_image(np.array(img, dtype=np.uint8)) for img in result['augmented_images']]
            result_image = result['result_image']
            predicted_class = result['predicted_class']
            dr_levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            return render_template('display.html', 
                                   preprocessed_image=preprocessed_image, 
                                   augmented_images=augmented_images, 
                                   result_image=result_image, 
                                   predicted_class=predicted_class,
                                   dr_levels=dr_levels)
        except KeyError as e:
            return jsonify(error=f"Missing key in response JSON: {e}"), 400

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
