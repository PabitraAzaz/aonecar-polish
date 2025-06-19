import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def remove_dirt(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Unable to read image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # threshold to detect dark spots (dirt)
    _, mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    clean = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(output_path, clean)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', filename=None)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('image')
    if not file:
        return redirect(url_for('index'))
    filename = file.filename
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_' + filename)
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clean_' + filename)
    file.save(input_path)
    try:
        remove_dirt(input_path, output_path)
    except Exception as e:
        return f"Error processing image: {e}", 400
    return render_template('index.html', filename=os.path.basename(output_path))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
