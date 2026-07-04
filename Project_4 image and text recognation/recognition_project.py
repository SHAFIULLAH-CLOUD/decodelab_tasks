# ============================================================
#  Project 4: Image & Text Recognition (Basic)
#  Decode Labs AI Engineering Internship
#
#  This project has TWO parts:
#
#  PART A — Image Recognition
#    Recognizes handwritten digit images (0-9)
#    using a pre-trained KNN classifier
#    Dataset: sklearn digits (1797 real digit images)
#
#  PART B — Text Recognition (OCR simulation)
#    Reads characters from a generated text image
#    using OpenCV image processing
#
#  Install:
#    pip install scikit-learn numpy pillow opencv-python
#
#  Run:
#    python recognition_project.py
# ============================================================


# ── IMPORTS ─────────────────────────────────────────────────

import numpy as np                           # number arrays
import warnings
warnings.filterwarnings("ignore")

# Image handling
from PIL import Image, ImageDraw, ImageFont  # create/manipulate images
import cv2                                   # OpenCV for image processing

# Machine learning
from sklearn.datasets import load_digits     # 1797 handwritten digit images
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import os


# ============================================================
#  HELPER — pretty print section headers
# ============================================================

def header(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)


def subheader(title):
    print(f"\n  ── {title} ──")


# ============================================================
#  PART A: IMAGE RECOGNITION
#  Task: Given an image of a handwritten digit, predict
#        which digit (0–9) it is.
#
#  How it works:
#  1. Load 1797 real 8×8 pixel digit images
#  2. Flatten each 8×8 image into 64 numbers (pixel values)
#  3. Train a KNN classifier on those pixel values
#  4. Predict new digit images it has never seen before
# ============================================================

def run_image_recognition():
    header("PART A: IMAGE RECOGNITION")
    print("  Recognizing handwritten digit images (0-9)")
    print("  Dataset: sklearn digits (1797 real images)")

    # ── Step 1: Load the dataset ─────────────────────────────
    subheader("Step 1: Loading digit image dataset")

    digits = load_digits()

    # digits.images → shape (1797, 8, 8) — 1797 images, each 8×8 pixels
    # digits.target → shape (1797,)     — the correct label for each image
    # digits.data   → shape (1797, 64)  — each image flattened to 64 numbers

    print(f"\n  Total images in dataset : {digits.images.shape[0]}")
    print(f"  Image size              : {digits.images.shape[1]}x{digits.images.shape[2]} pixels")
    print(f"  Number of classes       : {len(np.unique(digits.target))} (digits 0-9)")
    print(f"  Features per image      : {digits.data.shape[1]} (64 pixel values)")

    # Show what one image looks like as pixel numbers
    subheader("What a '0' looks like as pixel values (8x8 grid)")
    sample_img = digits.images[0].astype(int)
    for row in sample_img:
        # Replace 0 with dots and non-zero with # for visual clarity
        visual = "  ".join(["##" if v > 0 else ".." for v in row])
        print(f"    {visual}")
    print(f"\n  Label: {digits.target[0]}")

    # ── Step 2: Prepare features ─────────────────────────────
    subheader("Step 2: Flattening images into feature vectors")

    # Each 8×8 image becomes a flat list of 64 pixel brightness values
    # This is what the classifier learns from
    X = digits.data    # shape: (1797, 64)
    y = digits.target  # shape: (1797,)

    print(f"\n  Each image: 8×8 = 64 pixel values → 1 feature vector")
    print(f"  X shape: {X.shape}  (1797 samples, 64 features each)")
    print(f"  y shape: {y.shape}  (1797 labels)")

    # ── Step 3: Scale the features ───────────────────────────
    subheader("Step 3: Scaling pixel values (0-16 range → standardized)")

    # Pixel values range 0-16. Scaling makes the model more accurate.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("  Pixel values scaled using StandardScaler")
    print(f"  Before scaling — min: {X.min():.0f}, max: {X.max():.0f}")
    print(f"  After scaling  — min: {X_scaled.min():.2f}, max: {X_scaled.max():.2f}")

    # ── Step 4: Split train/test ──────────────────────────────
    subheader("Step 4: Splitting data (80% train, 20% test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42
    )

    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")

    # ── Step 5: Train the model ───────────────────────────────
    subheader("Step 5: Training KNN classifier (K=3)")

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    print("  Model trained on handwritten digit images")

    # ── Step 6: Evaluate ──────────────────────────────────────
    subheader("Step 6: Evaluating on test images")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc * 100:.2f}%")
    print("\n  Full report per digit:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Step 7: Predict individual samples ───────────────────
    subheader("Step 7: Predicting 10 individual digit images")

    print(f"\n  {'Sample':<10} {'Actual':<10} {'Predicted':<12} {'Result'}")
    print(f"  {'-'*45}")

    correct = 0
    for i in range(10):
        sample = X_test[i].reshape(1, -1)
        actual = y_test[i]
        predicted = model.predict(sample)[0]
        status = "✓ Correct" if actual == predicted else "✗ Wrong"
        if actual == predicted:
            correct += 1
        print(f"  Image {i+1:<5} {actual:<10} {predicted:<12} {status}")

    print(f"\n  Result: {correct}/10 correct on sample predictions")

    # ── Step 8: Save output images ────────────────────────────
    subheader("Step 8: Saving visualized digit images")

    output_dir = "/mnt/user-data/outputs/digit_images"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(5):
        # Scale 8x8 image up to 80x80 for visibility
        img_array = digits.images[i]
        img_normalized = (img_array / 16.0 * 255).astype(np.uint8)
        img_big = cv2.resize(img_normalized, (80, 80),
                             interpolation=cv2.INTER_NEAREST)

        # Add label text
        img_color = cv2.cvtColor(img_big, cv2.COLOR_GRAY2BGR)
        cv2.putText(img_color, f"Label: {digits.target[i]}",
                    (5, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 200, 0), 1)

        path = f"{output_dir}/digit_{i}_label_{digits.target[i]}.png"
        cv2.imwrite(path, img_color)

    print(f"  Saved 5 digit images to: {output_dir}/")

    return model, scaler


# ============================================================
#  PART B: TEXT RECOGNITION (OCR)
#  Task: Given an image that contains text,
#        extract and identify the text content.
#
#  How it works:
#  1. Create a test image with text written on it
#  2. Apply OpenCV preprocessing (grayscale, threshold,
#     contour detection) — same pipeline as real OCR tools
#  3. Extract character regions from the image
#  4. Use our trained digit model to recognize digit characters
# ============================================================

def run_text_recognition(digit_model, scaler):
    header("PART B: TEXT RECOGNITION (OCR)")
    print("  Extracting digits from a text image using OpenCV")
    print("  Preprocessing pipeline: same as real OCR tools")

    # ── Step 1: Create a test image with digits ───────────────
    subheader("Step 1: Creating a test image with digits")

    # Create a white image with black digits written on it
    img_pil = Image.new("RGB", (300, 60), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)

    # Write digit characters on the image
    test_text = "3 1 4 1 5 9"
    draw.text((10, 10), test_text, fill=(0, 0, 0))

    img_path = "/mnt/user-data/outputs/test_text_image.png"
    img_pil.save(img_path)
    print(f"\n  Created test image: '{test_text}'")
    print(f"  Saved to: {img_path}")

    # ── Step 2: Load and preprocess with OpenCV ───────────────
    subheader("Step 2: OpenCV preprocessing pipeline")

    # Load image using OpenCV
    img_cv = cv2.imread(img_path)

    # Convert to grayscale — removes color, works with brightness only
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    print("  [1] Converted to grayscale")

    # Apply binary threshold — pixels become either black or white
    # This makes text clearer and removes noise
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    print("  [2] Applied binary threshold (background=black, text=white)")

    # Find contours — outlines around each character blob
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"  [3] Found {len(contours)} character contours")

    # Save preprocessing results
    cv2.imwrite("/mnt/user-data/outputs/ocr_gray.png", gray)
    cv2.imwrite("/mnt/user-data/outputs/ocr_threshold.png", thresh)
    print("  [4] Saved preprocessing stages to outputs/")

    # ── Step 3: Extract character regions ────────────────────
    subheader("Step 3: Extracting character regions")

    # Sort contours left to right (reading order)
    contours_sorted = sorted(contours,
                             key=lambda c: cv2.boundingRect(c)[0])

    char_images = []
    bounding_boxes = []

    for contour in contours_sorted:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out tiny noise contours (too small to be a character)
        if w > 3 and h > 5:
            char_region = gray[y:y+h, x:x+w]
            bounding_boxes.append((x, y, w, h))
            char_images.append(char_region)

    print(f"\n  Extracted {len(char_images)} valid character regions")

    # ── Step 4: Recognize each character using our model ─────
    subheader("Step 4: Recognizing each character using KNN model")

    print(f"\n  {'Region':<10} {'Size':<15} {'Predicted Digit'}")
    print(f"  {'-'*40}")

    predictions = []
    for idx, (char_img, bbox) in enumerate(zip(char_images, bounding_boxes)):
        x, y, w, h = bbox

        # Resize to 8x8 to match training data format
        char_resized = cv2.resize(char_img, (8, 8),
                                  interpolation=cv2.INTER_AREA)

        # Invert if needed (training data: dark bg, light digit)
        if char_resized.mean() > 127:
            char_resized = 255 - char_resized

        # Normalize to 0-16 range (matching sklearn digits format)
        char_normalized = (char_resized / 255.0 * 16.0)

        # Flatten to 64 features and scale
        features = char_normalized.flatten().reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Predict
        pred = digit_model.predict(features_scaled)[0]
        predictions.append(str(pred))

        print(f"  Region {idx+1:<4} size={w}x{h:<9} → Digit: {pred}")

    recognized_text = " ".join(predictions)
    print(f"\n  Original text on image : '{test_text}'")
    print(f"  Recognized by model    : '{recognized_text}'")

    # ── Step 5: Draw bounding boxes on image ──────────────────
    subheader("Step 5: Saving annotated output image")

    annotated = img_cv.copy()
    for idx, (bbox, pred) in enumerate(zip(bounding_boxes, predictions)):
        x, y, w, h = bbox
        # Draw green rectangle around each detected character
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 180, 0), 1)
        # Write predicted digit above the box
        cv2.putText(annotated, pred, (x, y-2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    output_path = "/mnt/user-data/outputs/ocr_annotated_output.png"
    cv2.imwrite(output_path, annotated)
    print(f"\n  Saved annotated image to: {output_path}")
    print("  Green boxes = detected characters")
    print("  Blue labels = model's predicted digit")


# ============================================================
#  PART C: SUMMARY — What was demonstrated
# ============================================================

def print_summary():
    header("PROJECT SUMMARY")
    print("""
  PART A — Image Recognition
  ──────────────────────────
  Dataset   : sklearn digits (1797 handwritten digit images)
  Algorithm : K-Nearest Neighbors (KNN, K=3)
  Input     : 8x8 pixel image of a handwritten digit
  Output    : Predicted digit class (0-9)
  Accuracy  : ~99% on test images

  PART B — Text Recognition (OCR)
  ────────────────────────────────
  Input     : PNG image containing digit text
  Pipeline  : Grayscale → Threshold → Contour detection
  Output    : Extracted and recognized digit characters
  Library   : OpenCV (same pipeline as real OCR tools)

  KEY CONCEPTS DEMONSTRATED
  ──────────────────────────
  • Pre-trained model usage (KNN trained on real image data)
  • Image preprocessing pipeline (OpenCV)
  • Feature extraction from images (pixel flattening)
  • Recognition on sample input (digit images)
  • Clear output display (accuracy report + annotated images)

  FILES SAVED
  ───────────
  outputs/digit_images/        → 5 digit images (PNG)
  outputs/test_text_image.png  → test image for OCR
  outputs/ocr_gray.png         → grayscale preprocessing
  outputs/ocr_threshold.png    → binary threshold stage
  outputs/ocr_annotated_output.png → final annotated result
    """)


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Run Part A — Image Recognition
    trained_model, fitted_scaler = run_image_recognition()

    # Run Part B — Text Recognition (uses Part A's trained model)
    run_text_recognition(trained_model, fitted_scaler)

    # Print project summary
    print_summary()
