import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# 📁 Load model
model = load_model("pneumonia_model.h5")

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = 150

# 🏠 Home page
@app.route('/')
def index():
    return render_template('index.html')

# 🔍 Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # 🧠 Preprocess image
    img = image.load_img(filepath, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔮 Prediction
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "🦠 Pneumonia Detected"
    else:
        result = "✅ Normal"

    return render_template('index.html', prediction=result, img_path=filepath)

# 🚀 Run app
if __name__ == '__main__':
    app.run(debug=True)