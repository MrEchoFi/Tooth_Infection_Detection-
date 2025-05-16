from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image
import os
import cv2
import numpy as np
import torch
import json 

# --- Configuration ---
app = Flask(__name__)
app.secret_key = 'replace_with_a_random_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Load infection types & recommendations ---
def load_infection_types(file_path='infection_types.txt'):
    infections = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                name, rec = line.strip().split(':', 1)
                infections[name.strip()] = rec.strip()
    return infections

infection_data = load_infection_types()

def load_json_dataset(json_path='dataset'):
   
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print("JSON dataset loaded with", len(data), "records.")
        return data
    except FileNotFoundError:
        print("Dataset file not found. (This is simulated only)")
        return {}

dataset = load_json_dataset()

# --- Helpers ---
def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )

def is_tooth_image(image_path):
    """
    Three‐stage heuristic—accept if at least TWO of:
      1) Grid‐sample bright, low‐saturation pixels → white_ratio
      2) Bright‐pixel bounding‐box area → box_area_ratio
      3) Edge density via Canny → edge_ratio
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        return False
    h, w = img_cv.shape[:2]

    # 1) Grid sampling
    step_x, step_y = max(1, w//20), max(1, h//20)
    white = total = 0
    for x in range(0, w, step_x):
        for y in range(0, h, step_y):
            b, g, r = img_cv[y, x]
            intensity = (int(r) + int(g) + int(b)) / 3
            saturation = max(int(r), int(g), int(b)) - min(int(r), int(g), int(b))
            total += 1
            if intensity > 200 and saturation < 30:
                white += 1
    white_ratio = (white / total) if total else 0

    # 2) Bright-pixel bounding box
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    x0, y0, w0, h0 = cv2.boundingRect(mask)
    box_area_ratio = ((w0 * h0) / (w * h)) if (w * h) else 0

    # 3) Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.count_nonzero(edges) / edges.size

    # Conditions
    white_cond = (0.01 < white_ratio < 0.8)
    box_cond   = (box_area_ratio > 0.015)
    edge_cond  = (edge_ratio < 0.15)

    return sum([white_cond, box_cond, edge_cond]) >= 2

def detect_infection(image_path):
    img = Image.open(image_path).convert('L')
    pixels = img.load()
    w, h = img.size
    total = w * h
    dark = sum(
        1 for x in range(w) for y in range(h)
        if pixels[x, y] < 128
    )
    damage = round((dark / total) * 100, 2)

    if damage < 10:
        inf = 'healthy'
    elif damage < 30:
        inf = 'cavities'
    elif damage < 50:
        inf = 'enamel erosion'
    elif damage < 70:
        inf = 'gum infection'
    elif damage < 90:
        inf = 'abscess'
    else:
        inf = 'periodontal disease'

    rec = infection_data.get(inf, 'No recommendation available.')
    return inf, damage, rec

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'Mr. Dento' and request.form['password'] == '2211':
            session['logged_in'] = True
            return redirect(url_for('upload'))
        return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html', error='No file selected')
        if not allowed_file(file.filename):
            return render_template('upload.html', error='Type not allowed')

        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        if not is_tooth_image(path):
            os.remove(path)
            return render_template(
                'upload.html',
                error="That doesn't look like a tooth. Please upload a clear tooth image."
            )

        inf, dmg, rec = detect_infection(path)
        return render_template(
            'result.html',
            infection=inf,
            damage=dmg,
            recommendation=rec,
            filename=filename
        )

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
