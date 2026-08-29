import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 64

BATCH_SIZE = 32

TEST_DIR = "dataset/test"

MODEL_PATH = "models/drowsiness_cnn.keras"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained CNN...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(

    TEST_DIR,

    labels="inferred",

    label_mode="binary",

    color_mode="grayscale",

    image_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    shuffle=False
)


class_names = test_dataset.class_names


print(
    "\nClasses:"
)

print(
    class_names
)


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    test_dataset
)


print("\n")
print("=" * 50)
print("TEST RESULTS")
print("=" * 50)

print(
    f"Test Loss     : {loss:.4f}"
)

print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predictions = (
        predictions.flatten() >= 0.5
    ).astype(int)

    y_pred.extend(
        predictions
    )

    y_true.extend(
        labels.numpy().astype(int).flatten()
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n")
print("Confusion Matrix:")

print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=class_names
)


disp.plot()

plt.title(
    "CNN Eye State Confusion Matrix"
)

plt.savefig(
    "confusion_matrix.png"
)

plt.show()