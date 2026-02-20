from flask import Flask, render_template, request, send_file, redirect
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
from datetime import datetime
import sqlite3
import base64
import json

app = Flask(__name__)

# ================= SQLITE DATABASE =================

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            severity TEXT,
            image_path TEXT,
            prediction_time TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ================= UPLOAD FOLDER =================

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL =================

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

# ================= SAVE DATA =================

def save_to_database(disease, confidence, severity, img_path):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
        (disease, confidence, severity, image_path, prediction_time)
        VALUES (?,?,?,?,?)
    """, (disease, confidence, severity, img_path, str(datetime.now())))
    conn.commit()
    conn.close()

# ================= DASHBOARD =================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ================= UPLOAD =================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    prediction = med = confidence = severity = img_path = ""

    if request.method == "POST":
        file = request.files["file"]
        if file.filename == "":
            return redirect("/")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        result = model.predict(img_array)
        pred_index = np.argmax(result)

        prediction = classes[pred_index]
        confidence = round(result[0][pred_index] * 100, 2)
        severity = "HIGH" if confidence > 90 else "MEDIUM" if confidence > 70 else "LOW"
        med = medicine[prediction]

        save_to_database(prediction, confidence, severity, filepath)
        img_path = filepath

    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    history = cur.fetchall()
    conn.close()

    return render_template("index.html",
                           prediction=prediction,
                           medicine=med,
                           confidence=confidence,
                           severity=severity,
                           image_path=img_path,
                           history=history)

# ================= HISTORY =================

@app.route("/history")
def history():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC")
    history = cur.fetchall()
    conn.close()
    return render_template("history.html", history=history)

# ================= DELETE =================

@app.route("/delete/<int:id>")
def delete(id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT image_path FROM predictions WHERE id=?", (id,))
    row = cur.fetchone()

    if row and os.path.exists(row[0]):
        os.remove(row[0])

    cur.execute("DELETE FROM predictions WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect("/history")

@app.route("/delete_all")
def delete_all():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    cur.execute("SELECT image_path FROM predictions")
    rows = cur.fetchall()

    for r in rows:
        if os.path.exists(r[0]):
            os.remove(r[0])

    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

    return redirect("/history")

# ================= WEBCAM =================

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

    if confidence < 50:
        return render_template("webcam_result.html", error="Please upload plant images only!")

    prediction = classes[pred_index]
    severity = "HIGH" if confidence > 90 else "MEDIUM" if confidence > 70 else "LOW"
    med = medicine[prediction]

    save_to_database(prediction, confidence, severity, file_path)

    return render_template("webcam_result.html",
                           prediction=prediction,
                           medicine=med,
                           confidence=confidence,
                           severity=severity,
                           image_path=file_path)

# ================= REPORTS =================

@app.route("/reports")
def reports():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT id, disease, confidence, prediction_time FROM predictions ORDER BY id DESC")
    reports = cur.fetchall()
    conn.close()
    return render_template("reports.html", reports=reports)

# ================= PDF =================

@app.route("/download/<int:report_id>")
def download(report_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT disease, confidence, prediction_time FROM predictions WHERE id=?", (report_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return "Report not found", 404

    disease, confidence, time = row

    file_path = f"Report_{report_id}.pdf"
    c = canvas.Canvas(file_path)

    c.drawString(120, 780, "Brinjal Disease Detection Report")
    c.drawString(100, 720, f"Disease: {disease}")
    c.drawString(100, 680, f"Confidence: {confidence}%")
    c.drawString(100, 640, f"Time: {time}")
    c.save()

    return send_file(file_path, as_attachment=True)

# ================= RUN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
