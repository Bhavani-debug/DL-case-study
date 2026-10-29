import os
import traceback

import numpy as np
import requests
import tensorflow as tf

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "disaster_resnet50.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.txt"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ==========================================
# LOAD AI MODEL
# ==========================================

model = None
class_names = []

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("AI model loaded successfully!")
    else:
        print("Model not found:", MODEL_PATH)

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            class_names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        print("Classes:", class_names)
    else:
        print("class_names.txt not found!")

except Exception:
    print("Error loading model:")
    traceback.print_exc()


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image, dtype=np.float32)

    # Assumes training used pixel normalization to 0-1.
    image_array = image_array / 255.0

    return np.expand_dims(image_array, axis=0)


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


# ==========================================
# EXISTING HTML PAGES
# ==========================================

@app.route("/<page>.html")
def pages(page):

    allowed_pages = [
        "index",
        "about",
        "alerts",
        "dashboard",
        "dataset",
        "detection",
        "model",
        "early-warning"
    ]

    if page not in allowed_pages:
        return "Page not found", 404

    return send_from_directory(BASE_DIR, page + ".html")


# ==========================================
# CSS AND JAVASCRIPT
# ==========================================

@app.route("/<path:filename>")
def static_files(filename):

    safe_path = os.path.abspath(
        os.path.join(BASE_DIR, filename)
    )

    if not safe_path.startswith(os.path.abspath(BASE_DIR) + os.sep):
        return "Invalid path", 403

    if not os.path.isfile(safe_path):
        return "File not found", 404

    return send_from_directory(BASE_DIR, filename)


# ==========================================
# CHECK FLASK AND MODEL STATUS
# ==========================================

@app.route("/api/health")
def health():

    return jsonify({
        "flask": "running",
        "model_loaded": model is not None,
        "classes_loaded": bool(class_names),
        "classes": class_names
    })


# ==========================================
# DISASTER IMAGE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success": False,
            "error": "AI model is not loaded. Check the model folder."
        }), 503

    if not class_names:
        return jsonify({
            "success": False,
            "error": "Class names are missing."
        }), 503

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "Upload an image first."
        }), 400

    uploaded_file = request.files["image"]

    if uploaded_file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({
            "success": False,
            "error": "Use JPG, JPEG, PNG or WEBP."
        }), 400

    filename = secure_filename(uploaded_file.filename)

    image_path = os.path.join(UPLOAD_FOLDER, filename)

    try:
        uploaded_file.save(image_path)

        image_array = prepare_image(image_path)

        predictions = model.predict(image_array, verbose=0)

        scores = np.asarray(predictions)

        if scores.ndim != 2 or scores.shape[0] != 1:
            raise ValueError("Unexpected model output shape.")

        scores = scores[0]

        if len(scores) != len(class_names):
            raise ValueError(
                "Model outputs do not match the class names."
            )

        index = int(np.argmax(scores))

        predicted_class = class_names[index]

        confidence = float(scores[index]) * 100

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "class_name": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": "Image analysis failed. Check the terminal."
        }), 500

    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass


# ==========================================
# LOCATION SEARCH
# ==========================================

@app.route("/geocode")
def geocode():

    location = request.args.get("location", "").strip()

    if not location:
        return jsonify({
            "success": False,
            "error": "Enter a location."
        }), 400

    try:
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": location,
                "count": 5,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = []

        for item in data.get("results", []):
            results.append({
                "name": item.get("name"),
                "country": item.get("country"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude")
            })

        return jsonify({
            "success": True,
            "results": results
        })

    except requests.RequestException:
        return jsonify({
            "success": False,
            "error": "Location service unavailable."
        }), 502


# ==========================================
# FLOOD FORECAST
# ==========================================

def get_flood_forecast(latitude, longitude):

    response = requests.get(
        "https://flood-api.open-meteo.com/v1/flood",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "river_discharge",
            "forecast_days": 4,
            "timezone": "GMT"
        },
        timeout=20
    )

    response.raise_for_status()

    daily = response.json().get("daily", {})

    dates = daily.get("time", [])
    discharge = daily.get("river_discharge", [])

    return [
        {
            "date": date,
            "river_discharge": value
        }
        for date, value in zip(dates, discharge)
    ]


@app.route("/early-warning", methods=["GET", "POST"])
def early_warning():

    if request.method == "GET":
        return send_from_directory(BASE_DIR, "early-warning.html")

    data = request.get_json(silent=True) or {}

    try:
        latitude = float(data.get("latitude"))
        longitude = float(data.get("longitude"))

        if not -90 <= latitude <= 90:
            raise ValueError()

        if not -180 <= longitude <= 180:
            raise ValueError()

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "error": "Enter valid latitude and longitude."
        }), 400

    try:
        forecast = get_flood_forecast(latitude, longitude)

        return jsonify({
            "success": True,
            "forecast": forecast,
            "risk": "Insufficient Data",
            "message": (
                "Forecast retrieved. A validated local flood threshold "
                "is needed to determine the actual risk level."
            )
        })

    except requests.RequestException:
        return jsonify({
            "success": False,
            "risk": "Insufficient Data",
            "error": "Flood forecast service unavailable."
        }), 502


# ==========================================
# EARTHQUAKE MONITORING
# ==========================================

@app.route("/api/earthquakes")
def earthquakes():

    try:
        response = requests.get(
            "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        events = []

        for feature in data.get("features", []):

            properties = feature.get("properties", {})

            events.append({
                "place": properties.get("place"),
                "magnitude": properties.get("mag"),
                "time": properties.get("time"),
                "url": properties.get("url")
            })

        return jsonify({
            "success": True,
            "count": len(events),
            "events": events,
            "message": (
                "Recorded earthquake events. This feed does not "
                "predict future earthquakes."
            )
        })

    except requests.RequestException:
        return jsonify({
            "success": False,
            "error": "Earthquake data unavailable."
        }), 502


# ==========================================
# START APPLICATION
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("       AI DISASTERGUARD")
    print("================================")
    print("Model loaded:", model is not None)
    print("Classes:", class_names)
    print("URL: http://127.0.0.1:5000")
    print("================================\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )