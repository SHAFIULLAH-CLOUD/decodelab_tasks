# ============================================================
#  Basic Classification Model — Iris Flower Dataset
#  Decode Labs Internship — Project
#
#  Requirements covered:
#  - Load and understand a dataset
#  - Split data into training and testing sets
#  - Apply a simple classification algorithm
#
#  Skills demonstrated:
#  - Data handling (pandas, sklearn datasets)
#  - Supervised learning basics
#  - Model training and evaluation
#
#  Install first:
#    pip install scikit-learn pandas
#
#  Run:
#    python classification_model.py
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────
import pandas as pd                                  # for data handling
from sklearn.datasets import load_iris                # the dataset itself
from sklearn.model_selection import train_test_split  # to split data
from sklearn.neighbors import KNeighborsClassifier     # our classification algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
#  STEP 1 — LOAD THE DATASET
#
#  The Iris dataset is a classic beginner dataset.
#  It contains measurements of 150 flowers from 3 species:
#  Setosa, Versicolor, and Virginica.
#
#  Each flower has 4 measurements (features):
#  sepal length, sepal width, petal length, petal width
# ============================================================

print("=" * 55)
print("  STEP 1: Loading the Iris dataset")
print("=" * 55)

iris = load_iris()

# Features (X) — the measurements used to make predictions
X = iris.data

# Labels (y) — the correct species for each flower (what we predict)
y = iris.target

# Convert to a pandas DataFrame so we can inspect it like a table
df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in y]

print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Species classes: {list(iris.target_names)}")
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nHow many flowers per species:")
print(df["species"].value_counts())


# ============================================================
#  STEP 2 — UNDERSTAND THE DATASET
#
#  Before building a model, always check basic statistics.
#  This helps you understand the scale and spread of the data.
# ============================================================

print("\n" + "=" * 55)
print("  STEP 2: Understanding the dataset")
print("=" * 55)

print("\nBasic statistics for each feature:")
print(df.describe())

print("\nAny missing values?")
print(df.isnull().sum())


# ============================================================
#  STEP 3 — SPLIT DATA INTO TRAINING AND TESTING SETS
#
#  We never train and test on the same data — that would be
#  like grading a student on questions they already saw the
#  answers to. We hold back 20% of the data purely for testing.
# ============================================================

print("\n" + "=" * 55)
print("  STEP 3: Splitting data into train/test sets")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% of data reserved for testing
    random_state=42      # fixed seed so results are reproducible
)

print(f"\nTraining set size: {len(X_train)} samples (80%)")
print(f"Testing set size:  {len(X_test)} samples (20%)")


# ============================================================
#  STEP 4 — APPLY A CLASSIFICATION ALGORITHM
#
#  We use K-Nearest Neighbors (KNN) — one of the simplest
#  classification algorithms to understand.
#
#  How KNN works:
#  To classify a new flower, look at its K closest neighbours
#  (most similar flowers) in the training data, and predict
#  whichever species is most common among them.
# ============================================================

print("\n" + "=" * 55)
print("  STEP 4: Training the classification model")
print("=" * 55)

# Create the model — K=3 means "look at the 3 closest flowers"
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training data
# This is where the model "learns" the patterns
model.fit(X_train, y_train)

print("\nModel trained successfully using K-Nearest Neighbors (K=3)")


# ============================================================
#  STEP 5 — EVALUATE THE MODEL
#
#  Now we test the model on data it has NEVER seen before
#  (the test set) to see how well it actually learned.
# ============================================================

print("\n" + "=" * 55)
print("  STEP 5: Evaluating the model")
print("=" * 55)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy — what % of predictions were correct
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test data: {accuracy * 100:.2f}%")

print("\nDetailed performance report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, y_pred))


# ============================================================
#  STEP 6 — TEST ON A NEW, UNSEEN SAMPLE
#
#  Let's predict the species of a brand new flower
#  that wasn't in the dataset at all.
# ============================================================

print("\n" + "=" * 55)
print("  STEP 6: Predicting a brand new flower")
print("=" * 55)

# A new flower measurement: [sepal length, sepal width, petal length, petal width]
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)
predicted_species = iris.target_names[prediction[0]]

print(f"\nNew flower measurements: {new_flower[0]}")
print(f"Predicted species: {predicted_species}")

print("\n" + "=" * 55)
print("  Done!")
print("=" * 55)
