"""
model_training.py
------------------
OptiCrop: Smart Agricultural Production Optimization Engine

This script reproduces, end-to-end, every story in the project plan:
  - Importing the libraries
  - Reading the dataset
  - Univariate / Bivariate / Multivariate analysis (saved as PNGs)
  - Checking for null values
  - Handling outliers (IQR on Potassium)
  - Extracting seasonal crops
  - Splitting data into train/test sets
  - K-Means clustering (elbow method)
  - Logistic Regression model training
  - Evaluating model performance
  - Saving the best model as model.pkl for the Flask app

Run with:  python3 model_training.py
Outputs go to ../static/images (plots) and ../model/model.pkl
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (12, 8)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Read the dataset
# ---------------------------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
df = df.rename(columns={"N": "nitrogen", "P": "phosphorous", "K": "potassium"})
print("Dataset shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------------------
# 2. Checking for null values
# ---------------------------------------------------------------------------
print("\nNull values per column:\n", df.isnull().sum())
print("\ndf.info():")
df.info()

# ---------------------------------------------------------------------------
# 3. Univariate Analysis
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
numeric_cols = ["nitrogen", "phosphorous", "potassium", "temperature", "humidity", "ph", "rainfall"]
for ax, col in zip(axes.flat, numeric_cols):
    sns.histplot(df[col], kde=True, ax=ax)
    ax.set_title(f"Distribution of {col}")
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "univariate_analysis.png"))
plt.close()

# ---------------------------------------------------------------------------
# 4. Bivariate Analysis (humidity vs label)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 10))
sns.scatterplot(x=df["humidity"], y=df["label"])
plt.title("Humidity vs Crop Label")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "bivariate_analysis.png"))
plt.close()

# ---------------------------------------------------------------------------
# 5. Multivariate Analysis
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.countplot(data=df[numeric_cols].melt(), x="variable")
plt.title("Feature counts")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "multivariate_countplot.png"))
plt.close()

# Descriptive analysis
desc = df[numeric_cols].describe()
print("\nDescriptive statistics:\n", desc)
desc.to_csv(os.path.join(IMAGES_DIR, "..", "..", "dataset", "descriptive_stats.csv"))

# ---------------------------------------------------------------------------
# 6. Handling Outliers (IQR method on Potassium)
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 8))
sns.boxplot(y=df["potassium"])
plt.title("Potassium - Boxplot before outlier removal")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "potassium_boxplot.png"))
plt.close()

Q1 = df["potassium"].quantile(0.25)
Q3 = df["potassium"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
before = df.shape[0]
df = df.loc[(df["potassium"] >= lower_bound) & (df["potassium"] <= upper_bound)]
print(f"\nRemoved {before - df.shape[0]} potassium outliers "
      f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")

# ---------------------------------------------------------------------------
# 7. Extracting Seasonal Crops
# ---------------------------------------------------------------------------
print("\nSummer crops:")
print(df[(df["temperature"] > 30) & (df["humidity"] > 50)]["label"].unique())
print("\nWinter crops:")
print(df[(df["temperature"] < 20) & (df["humidity"] > 30)]["label"].unique())
print("\nRainy crops:")
print(df[(df["rainfall"] > 200) & (df["humidity"] > 50)]["label"].unique())

# ---------------------------------------------------------------------------
# 8. Splitting Data into Train and Test Sets
# ---------------------------------------------------------------------------
y = df["label"]
X = df.drop(["label"], axis=1)
print("\nShape of X:", X.shape, "Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train:", y_train.shape, "y_test:", y_test.shape)

# ---------------------------------------------------------------------------
# 9. K-Means Clustering (Elbow Method)
# ---------------------------------------------------------------------------
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=10, random_state=0)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "elbow_graph.png"))
plt.close()

km = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(X)
cluster_df = pd.concat(
    [pd.Series(y_means, index=X.index, name="cluster"), df["label"]], axis=1
)
for c in range(4):
    crops_in_cluster = cluster_df[cluster_df["cluster"] == c]["label"].unique()
    print(f"\nCrops in cluster {c + 1}: {crops_in_cluster}")

# ---------------------------------------------------------------------------
# 10. Logistic Regression
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------------------------------------------------------------------
# 11. Evaluating Model Performance
# ---------------------------------------------------------------------------
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Greens", xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"))
plt.close()

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# ---------------------------------------------------------------------------
# 12. Saving the Best Model
# ---------------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"\nModel saved to {model_path}")

# ---------------------------------------------------------------------------
# 13. Predict the Best Crop Based on Given Parameters (sanity check)
# ---------------------------------------------------------------------------
sample = np.array([[90, 42, 43, 20.87, 82.0, 6.5, 202.9]])
prediction = model.predict(sample)
print(f"\nThe suggested crop for the given climatic condition is: {prediction}")
