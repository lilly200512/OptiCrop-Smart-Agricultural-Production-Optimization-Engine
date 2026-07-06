"""
generate_dataset.py
--------------------
OptiCrop: Smart Agricultural Production Optimization Engine

This script generates the Crop_recommendation.csv dataset used throughout the
project. The public Kaggle dataset referenced in the project plan
("smart-agricultural-production-optimizing-engine") could not be downloaded
in this offline build environment, so a synthetic dataset with the same
schema (N, P, K, temperature, humidity, ph, rainfall, label), the same 22
crop classes, and the same size (2200 rows / 100 rows per crop) is generated
here using agronomically realistic parameter ranges for each crop.

If you have network access, simply download the real dataset from Kaggle
and drop it into dataset/Crop_recommendation.csv, replacing this file's
output -- the rest of the pipeline (model_training.py / app.py) works
unchanged either way.
"""

import numpy as np
import pandas as pd

np.random.seed(0)

# Approximate agronomic parameter ranges (min, max) for each crop.
# N, P, K in kg/ha ; temperature in Celsius ; humidity in % ; ph 0-14 ; rainfall in mm
CROP_RANGES = {
    "rice":        dict(N=(80, 120), P=(40, 60),  K=(40, 60),  temperature=(20, 27), humidity=(80, 95), ph=(5.0, 6.5), rainfall=(180, 300)),
    "maize":       dict(N=(60, 100), P=(35, 60),  K=(15, 25),  temperature=(18, 27), humidity=(55, 75), ph=(5.5, 7.0), rainfall=(60, 110)),
    "chickpea":    dict(N=(20, 45),  P=(60, 80),  K=(75, 85),  temperature=(17, 21), humidity=(14, 20), ph=(6.0, 8.0), rainfall=(65, 100)),
    "kidneybeans": dict(N=(0, 25),   P=(60, 80),  K=(15, 25),  temperature=(15, 25), humidity=(18, 25), ph=(5.5, 6.0), rainfall=(60, 150)),
    "pigeonpeas":  dict(N=(0, 30),   P=(60, 90),  K=(15, 25),  temperature=(18, 37), humidity=(30, 70), ph=(4.5, 7.5), rainfall=(90, 200)),
    "mothbeans":   dict(N=(0, 40),   P=(35, 60),  K=(15, 25),  temperature=(24, 32), humidity=(30, 65), ph=(3.5, 10.0), rainfall=(35, 65)),
    "mungbean":    dict(N=(0, 40),   P=(35, 60),  K=(15, 25),  temperature=(27, 32), humidity=(80, 90), ph=(6.2, 7.2), rainfall=(45, 65)),
    "blackgram":   dict(N=(20, 60),  P=(60, 80),  K=(15, 25),  temperature=(25, 35), humidity=(60, 70), ph=(6.5, 7.5), rainfall=(65, 75)),
    "lentil":      dict(N=(0, 30),   P=(60, 80),  K=(15, 25),  temperature=(18, 30), humidity=(60, 70), ph=(6.0, 7.0), rainfall=(40, 65)),
    "pomegranate": dict(N=(0, 40),   P=(5, 20),   K=(35, 45),  temperature=(18, 25), humidity=(85, 95), ph=(6.0, 7.0), rainfall=(35, 110)),
    "banana":      dict(N=(80, 120), P=(70, 95),  K=(45, 55),  temperature=(25, 30), humidity=(75, 85), ph=(5.5, 6.5), rainfall=(95, 120)),
    "mango":       dict(N=(0, 40),   P=(15, 40),  K=(25, 35),  temperature=(27, 36), humidity=(45, 55), ph=(4.5, 7.0), rainfall=(65, 100)),
    "grapes":      dict(N=(0, 40),   P=(120, 145),K=(190, 205),temperature=(8, 30),  humidity=(80, 85), ph=(5.5, 6.5), rainfall=(65, 75)),
    "watermelon":  dict(N=(80, 120), P=(5, 20),   K=(45, 55),  temperature=(24, 27), humidity=(80, 90), ph=(6.0, 7.0), rainfall=(40, 55)),
    "muskmelon":   dict(N=(80, 120), P=(5, 20),   K=(45, 55),  temperature=(27, 30), humidity=(90, 95), ph=(6.0, 7.0), rainfall=(20, 30)),
    "apple":       dict(N=(0, 40),   P=(120, 145),K=(195, 205),temperature=(21, 24), humidity=(90, 95), ph=(5.5, 6.5), rainfall=(100, 125)),
    "orange":      dict(N=(0, 40),   P=(5, 20),   K=(5, 15),   temperature=(10, 35), humidity=(90, 95), ph=(6.0, 7.5), rainfall=(100, 120)),
    "papaya":      dict(N=(30, 70),  P=(45, 70),  K=(45, 55),  temperature=(23, 44), humidity=(90, 95), ph=(6.5, 7.0), rainfall=(40, 250)),
    "coconut":     dict(N=(0, 40),   P=(5, 30),   K=(25, 35),  temperature=(25, 30), humidity=(90, 100),ph=(5.5, 6.5), rainfall=(150, 230)),
    "cotton":      dict(N=(100, 140),P=(35, 60),  K=(15, 25),  temperature=(22, 26), humidity=(75, 85), ph=(5.8, 8.0), rainfall=(60, 100)),
    "jute":        dict(N=(60, 100), P=(35, 60),  K=(35, 45),  temperature=(23, 27), humidity=(70, 90), ph=(6.0, 7.5), rainfall=(150, 200)),
    "coffee":      dict(N=(80, 120), P=(15, 40),  K=(25, 35),  temperature=(23, 28), humidity=(50, 70), ph=(6.0, 7.5), rainfall=(150, 200)),
}

rows = []
for crop, ranges in CROP_RANGES.items():
    for _ in range(100):
        row = {param: round(np.random.uniform(lo, hi), 6) for param, (lo, hi) in ranges.items()}
        row["N"] = int(round(row["N"]))
        row["P"] = int(round(row["P"]))
        row["K"] = int(round(row["K"]))
        row["label"] = crop
        rows.append(row)

df = pd.DataFrame(rows)
df = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]]

# a handful of Potassium outliers to mirror the "Handling Outliers" story in the project plan
outlier_idx = np.random.choice(df.index, size=8, replace=False)
df.loc[outlier_idx, "K"] = (df.loc[outlier_idx, "K"] * np.random.uniform(3, 5, size=8)).astype(int)
df["K"] = df["K"].astype(int)

df = df.sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle rows

df.to_csv("/home/claude/OptiCrop/dataset/Crop_recommendation.csv", index=False)
print(df.shape)
print(df.head())
