"""
app.py
------
OptiCrop: Smart Agricultural Production Optimization Engine
Flask backend that loads the trained Logistic Regression model (model.pkl)
and serves the Home, About, and FindYourCrop pages, handling crop
predictions submitted through the FindYourCrop form.

Run with:
    python3 app.py
Then open the URL printed in the terminal (defaults to http://127.0.0.1:5000/).
"""

import os
import pickle

import numpy as np
from flask import Flask, request, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/findyourcrop")
def findyourcrop():
    return render_template("findyourcrop.html")


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    prediction = model.predict(features)
    output = prediction[0]
    return render_template(
        "findyourcrop.html",
        prediction_text="Best crop for given conditions is {}".format(output),
    )


if __name__ == "__main__":
    app.run(debug=True)
