
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
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
COLAB_SERVER_URL = 'https://1fbe-34-73-107-180.ngrok-free.app'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']



def decode_image(encoded_str):
    encoded_bytes = base64.b64decode(encoded_str.encode('ascii'))
    image = Image.open(BytesIO(encoded_bytes))
    return np.array(image)


def encode_image(image_array):
    pil_image = Image.fromarray(image_array)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')



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
            preprocessed_image = decode_image(result['preprocessed_image'])
            augmented_images = [decode_image(img) for img in result['augmented_images']]
            result_image_path = result['result_image_path']
            predicted_class = result['prediction']
            dr_levels = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
            

            preprocessed_image_b64 = encode_image(preprocessed_image)
            augmented_images_b64 = [encode_image(img) for img in augmented_images]

            return render_template('display.html', 
                                   preprocessed_image=preprocessed_image_b64, 
                                   augmented_images=augmented_images_b64, 
                                   result_image_path=result_image_path, 
                                   predicted_class=predicted_class,
                                   dr_levels=dr_levels)
        except KeyError as e:
            return jsonify(error=f"Missing key in response JSON: {e}"), 400

    return jsonify(error='File not allowed'), 400
    # return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
