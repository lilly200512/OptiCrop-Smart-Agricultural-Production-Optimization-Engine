# OptiCrop: Smart Agricultural Production Optimization Engine

A Flask + scikit-learn web app that recommends the best crop to plant given
soil and climate parameters (Nitrogen, Phosphorous, Potassium, temperature,
humidity, pH, rainfall), following the project plan (ER diagram, workflow
epics, EDA, K-Means clustering, Logistic Regression, Flask deployment).

## Project structure

```
OptiCrop/
├── app.py                     # Flask backend (Home / About / FindYourCrop / predict)
├── requirements.txt
├── model/
│   └── model.pkl               # trained Logistic Regression model
├── dataset/
│   └── Crop_recommendation.csv # training data (see note below)
├── notebook/
│   ├── generate_dataset.py     # builds Crop_recommendation.csv
│   └── model_training.py       # full EDA + preprocessing + training pipeline
├── static/
│   ├── css/style.css
│   └── images/                 # EDA plots produced by model_training.py
└── templates/
    ├── home.html
    ├── about.html
    └── findyourcrop.html
```

## Running it

```bash
pip install -r requirements.txt
python3 app.py
```

Then open the URL printed in the terminal (default `http://127.0.0.1:5000/`)
and navigate Home → About → FindYourCrop to submit conditions and get a
crop recommendation.

## Regenerating the model

```bash
cd notebook
python3 generate_dataset.py   # (re)builds dataset/Crop_recommendation.csv
python3 model_training.py     # runs EDA, saves plots, trains & saves model/model.pkl
```

## Note on the dataset

The project plan references the Kaggle dataset *"Smart Agricultural
Production Optimizing Engine"*. This build environment has no internet
access, so `generate_dataset.py` instead generates a synthetic dataset with
the same schema, the same 22 crop labels, and the same size (2200 rows) using
agronomically realistic parameter ranges per crop, so the whole pipeline
(EDA → clustering → classification → Flask deployment) runs end-to-end.

**If you have internet access**, download the real dataset from
https://www.kaggle.com/datasets/chitrakumari25/smart-agricultural-production-optimizing-engine
and save it as `dataset/Crop_recommendation.csv` (with columns
`N,P,K,temperature,humidity,ph,rainfall,label`), then re-run
`model_training.py` to retrain `model.pkl` on the real data before deploying.

## Note on page backgrounds

The screenshots in the project plan use licensed stock photography for the
Home/About/FindYourCrop backgrounds. Those specific images can't be
reproduced here, so `static/css/style.css` recreates the same layout,
color story (green farmland hero, yellow "Find Your Best Crop Here" banner,
farmer-in-a-field About page) using CSS gradients and a small inline SVG
instead. Drop your own royalty-free photos into `static/images/` and point
the `.hero`, `.about-hero`, and `.crop-page` background-image rules at them
for a pixel-exact match to the screenshots.
