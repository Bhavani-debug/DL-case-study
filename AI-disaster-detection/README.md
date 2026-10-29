# AI DisasterGuard

AI DisasterGuard is a Flask-based disaster image classification and risk-awareness project. It uses a saved ResNet50 transfer-learning model to classify an uploaded image into one of five categories. A separate early-warning dashboard presents weather-based screening indicators and recent earthquake events from external data services.

## Problem statement

Disaster-related images and changing environmental conditions can be difficult to review quickly. This project demonstrates how image classification and public forecast/event data can be presented together to support awareness. It is a prototype, not an emergency-response or official warning service.

## Objectives

- Classify an uploaded image into one of the model's five configured classes.
- Present the prediction and confidence in a local web interface.
- Show location-based weather/flood screening information and recent earthquake activity.
- Demonstrate a ResNet50 transfer-learning workflow.

## Technologies

- Python and Flask
- TensorFlow / Keras and a saved `.keras` model
- NumPy, Pillow, and Requests
- HTML, CSS, JavaScript, and Bootstrap 5
- Open-Meteo geocoding, weather, and flood forecast services
- USGS earthquake GeoJSON feed

## Image classification model

The training script loads ResNet50 with ImageNet weights and `include_top=False`, freezes the base layers, and adds global average pooling, dropout, and a five-output softmax layer. The saved model is loaded by the Flask application; the application does not retrain it when serving predictions.

The application resizes an uploaded image to 224 × 224 pixels and returns the highest-scoring class and its score as a confidence percentage. The class list is read from `model/class_names.txt`.

The repository includes the trained model file. The training script uses a local dataset directory and expects `train`, `validation`, and `test` splits, each arranged in class subfolders. The dataset itself is not included, and this project does not document its original source or license. Confirm the dataset's terms and add an appropriate citation before redistributing it.

### Classification categories

- Fire
- Flooded Areas
- Collapsed Building
- Normal
- Traffic Incident

## Features

- Image upload, preview, and prediction through the Flask API.
- Prediction confidence display and browser-local prediction history.
- Home, detection, dashboard, alerts, model, dataset, about, and early-warning pages.
- Location search and flood/weather screening indicators.
- Recent earthquake activity from the USGS past-day feed.
- A worldwide overview for a selected set of monitored locations.

The early-warning page uses forecast and recent-event data for screening and awareness. It does not provide authoritative alerts or predict earthquakes.

## Project structure

```text
.
├── about.html
├── alerts.html
├── app.py
├── dashboard.html
├── dataset.html
├── detection.html
├── early-warning.html
├── index.html
├── model.html
├── script.js
├── style.css
├── train_model.py
├── model/
│   ├── class_names.txt
│   └── disaster_resnet50.keras
└── uploads/                 # Runtime upload directory; not committed
```

## Run locally

Use a Python version supported by the TensorFlow release available for your platform.

1. Open a terminal in the project directory.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   In Command Prompt, activate it with:

   ```bat
   .venv\Scripts\activate.bat
   ```

3. Install the application dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install Flask numpy requests tensorflow Pillow
   ```

4. Start Flask:

   ```powershell
   python app.py
   ```

5. Open [http://127.0.0.1:5000/detection.html](http://127.0.0.1:5000/detection.html) in a browser. The home page is available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

The early-warning page needs an internet connection to retrieve external forecast, geocoding, and earthquake data. The prediction endpoint accepts JPG, JPEG, PNG, or WEBP images up to 10 MB.

## Limitations and responsible use

- Image classification identifies a class in an image; it does **not** predict that a future disaster will occur.
- Predictions depend on the training data and model and may be incorrect or unsuitable for real emergency decisions.
- Forecast indicators, selected monitored locations, and recent earthquake records are not a complete global warning system. Earthquake records are observations, not predictions.
- Follow official local emergency services and instructions; do not rely on this prototype for life-safety decisions.
- The training script applies ResNet50's `preprocess_input`, while the Flask inference code scales pixels to the 0–1 range. This difference should be evaluated before using predictions beyond a demonstration.
- Training data is not included. `train_model.py` refers to a local `AIDER_Split` directory, so its path and dataset access must be arranged locally before training.
- External services require network access and may be unavailable or change their responses.

## Acknowledgements and references

- TensorFlow/Keras ResNet50 application: <https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50>
- Open-Meteo APIs used by the project: <https://open-meteo.com/en/docs>, <https://open-meteo.com/en/docs/geocoding-api>, and <https://open-meteo.com/en/docs/flood-api>
- USGS past-day earthquake feed: <https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson>
- Bootstrap: <https://getbootstrap.com/>

No dataset citation or dataset license is present in the project files.
