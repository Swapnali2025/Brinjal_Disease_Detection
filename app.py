from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import json

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# LOAD MODEL
model = load_model("model/brinjal_disease_model.h5")

with open("model/class_labels.json", "r") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())

medicine = {
    "healthy": "No medicine required. Plant is healthy 🌱",
    "leaf_spot": "Spray Mancozeb fungicide",
    "mosaic_virus_disease": "Remove infected plants + Spray Neem oil",
    "small_leaf_disease": "Spray Imidacloprid or Neem oil",
    "wilt_disease": "Apply Carbendazim fungicide"
}

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    prediction = ""
    med = ""
    confidence = ""
    severity = ""
    img_path = ""

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        pred_index = np.argmax(result)

        prediction = classes[pred_index]
        confidence = round(result[0][pred_index] * 100, 2)

        if confidence > 90:
            severity = "HIGH"
        elif confidence > 70:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        med = medicine[prediction]
        img_path = filepath

    return render_template("index.html",
        prediction=prediction,
        medicine=med,
        confidence=confidence,
        severity=severity,
        image_path=img_path
    )

@app.route("/webcam_capture")
def webcam_capture():
    return render_template("webcam.html")

@app.route("/capture", methods=["POST"])
def capture():
    data = request.form["image"]
    encoded_data = data.split(",")[1]
    img_bytes = base64.b64decode(encoded_data)

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], "captured.jpg")
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    pred_index = np.argmax(result)
    confidence = round(result[0][pred_index] * 100, 2)

    prediction = classes[pred_index]

    if confidence > 90:
        severity = "HIGH"
    elif confidence > 70:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    med = medicine[prediction]

    return render_template("webcam_result.html",
        prediction=prediction,
        medicine=med,
        confidence=confidence,
        severity=severity,
        image_path=file_path
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
