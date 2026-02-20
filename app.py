from flask import Flask, render_template, request, send_file
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
from datetime import datetime
from flask_mysqldb import MySQL
import base64
import json
from flask import redirect


app = Flask(__name__)

# ================= MYSQL CONFIG =================

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'   # change if needed
app.config['MYSQL_DB'] = 'Brinjal_ai'

mysql = MySQL(app)

# ================= UPLOAD FOLDER =================

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================= LOAD MODEL =================

model = load_model("../model/brinjal_disease_model.h5")

with open("../model/class_labels.json", "r") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())

medicine = {
    "healthy": "No medicine required. Plant is healthy 🌱",
    "leaf_spot": "Spray Mancozeb fungicide",
    "mosaic_virus_disease": "Remove infected plants + Spray Neem oil",
    "small_leaf_disease": "Spray Imidacloprid or Neem oil",
    "wilt_disease": "Apply Carbendazim fungicide"
}

# ================= GLOBAL PDF DATA =================

last_prediction = ""
last_medicine = ""
last_confidence = ""

# ================= SAVE TO DATABASE =================

def save_to_database(disease, confidence, severity, img_path):
    cur = mysql.connection.cursor()
    query = """
    INSERT INTO predictions 
    (disease, confidence, severity, image_path, prediction_time)
    VALUES (%s,%s,%s,%s,%s)
    """
    cur.execute(query, (
        disease,
        confidence,
        severity,
        img_path,
        datetime.now()
    ))
    mysql.connection.commit()
    cur.close()

# ================= DASHBOARD =================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ================= UPLOAD + HISTORY PAGE =================

@app.route("/upload", methods=["GET", "POST"])
def upload():

    global last_prediction, last_medicine, last_confidence

    prediction = ""
    med = ""
    confidence = ""
    severity = ""
    img_path = ""

    if request.method == "POST":

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # IMAGE PREPROCESSING
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # MODEL PREDICTION
        result = model.predict(img_array)
        pred_index = np.argmax(result)
        prediction = classes[pred_index]
        confidence = round(result[0][pred_index] * 100, 2)

        # SEVERITY
        if confidence > 90:
            severity = "HIGH"
        elif confidence > 70:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        med = medicine[prediction]

        # SAVE TO DB
        save_to_database(prediction, confidence, severity, filepath)

        # SAVE FOR PDF
        last_prediction = prediction
        last_medicine = med
        last_confidence = confidence

        img_path = filepath

    # FETCH HISTORY
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY prediction_time DESC")
    history = cur.fetchall()
    cur.close()

    return render_template(
        "index.html",
        prediction=prediction,
        medicine=med,
        confidence=confidence,
        severity=severity,
        image_path=img_path,
        history=history
    )
    
# ================= HISTORY PAGE =================

@app.route("/history")
def history():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY prediction_time DESC")
    history = cur.fetchall()
    cur.close()

    return render_template("history.html", history=history)
#=============== Delete Single Record ===============
@app.route("/delete/<int:id>")
def delete(id):
    cur = mysql.connection.cursor()

    # get image path first
    cur.execute("SELECT image_path FROM predictions WHERE id=%s", (id,))
    row = cur.fetchone()

    if row:
        img_path = row[0]
        if os.path.exists(img_path):
            os.remove(img_path)

    cur.execute("DELETE FROM predictions WHERE id=%s", (id,))
    mysql.connection.commit()
    cur.close()

    return redirect("/history")
#======Delete All===================
@app.route("/delete_all")
def delete_all():
    cur = mysql.connection.cursor()

    cur.execute("SELECT image_path FROM predictions")
    rows = cur.fetchall()

    for r in rows:
        if os.path.exists(r[0]):
            os.remove(r[0])

    cur.execute("DELETE FROM predictions")
    mysql.connection.commit()
    cur.close()

    return redirect("/history")



# ================= WEBCAM PAGE =================

@app.route("/webcam_capture")
def webcam_capture():
    return render_template("webcam.html")

# ================= CAPTURE IMAGE =================

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

    # ❌ NOT A PLANT IMAGE
    if confidence < 50:
        return render_template(
            "webcam_result.html",
            error="Please upload plant images only!"
        )

    prediction = classes[pred_index]

    if confidence > 90:
        severity = "HIGH"
    elif confidence > 70:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    med = medicine[prediction]

    save_to_database(prediction, confidence, severity, file_path)

    return render_template(
        "webcam_result.html",
        prediction=prediction,
        medicine=med,
        confidence=confidence,
        severity=severity,
        image_path=file_path
    )


# ============= Reports Page ==============
@app.route("/reports")
def reports():
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, disease, confidence, prediction_time 
        FROM predictions 
        ORDER BY prediction_time DESC
    """)
    reports = cur.fetchall()
    cur.close()

    return render_template("reports.html", reports=reports)

# ================= PDF DOWNLOAD =================

@app.route("/download/<int:report_id>")
def download(report_id):

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT disease, confidence, prediction_time 
        FROM predictions WHERE id=%s
    """, (report_id,))
    row = cur.fetchone()
    cur.close()

    if not row:
        return "Report not found", 404

    disease, confidence, time = row

    file_path = f"Brinjal_Report_{report_id}.pdf"
    c = canvas.Canvas(file_path)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, 780, "Brinjal Disease Detection Report")

    c.setFont("Helvetica", 14)
    c.drawString(100, 720, f"Disease: {disease}")
    c.drawString(100, 680, f"Confidence: {confidence}%")
    c.drawString(100, 640, f"Prediction Time: {time}")

    c.drawString(100, 580, "Generated by Brinjal AI System")
    c.save()

    return send_file(file_path, as_attachment=True)

# ================= DB TEST =================

@app.route("/dbtest")
def dbtest():
    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    count = cur.fetchone()[0]
    cur.close()
    return f"MySQL Connected Successfully ✅ Total Records: {count}"

# ================= RUN APP =================

if __name__ == "__main__":
     app.run(host="0.0.0.0", port=10000)