import os
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "dataset/train"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "drowsiness_cnn.keras"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="binary",

    color_mode="grayscale",

    image_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=42,

    validation_split=0.20,

    subset="training"
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="binary",

    color_mode="grayscale",

    image_size=(
        IMG_SIZE,
        IMG_SIZE
    ),

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=42,

    validation_split=0.20,

    subset="validation"
)


# ============================================================
# DISPLAY CLASS NAMES
# ============================================================

print("\nClass names:")

print(
    train_dataset.class_names
)

# Expected:
# ['closed', 'open']


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.05
    ),

    layers.RandomZoom(
        0.10
    )

])


# ============================================================
# CNN MODEL
# ============================================================

model = models.Sequential([

    # Input
    layers.Input(
        shape=(
            IMG_SIZE,
            IMG_SIZE,
            1
        )
    ),

    # Normalize pixels
    layers.Rescaling(
        1.0 / 255
    ),

    # Augmentation
    data_augmentation,

    # --------------------------------------------------------
    # CNN BLOCK 1
    # --------------------------------------------------------

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 2
    # --------------------------------------------------------

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # CNN BLOCK 3
    # --------------------------------------------------------

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    layers.Dropout(
        0.30
    ),

    # --------------------------------------------------------
    # FLATTEN
    # --------------------------------------------------------

    layers.Flatten(),

    # --------------------------------------------------------
    # FULLY CONNECTED LAYER
    # --------------------------------------------------------

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.50
    ),

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    layers.Dense(
        1,
        activation="sigmoid"
    )

])


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(

        MODEL_PATH,

        monitor="val_accuracy",

        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 50)
print("STARTING CNN TRAINING")
print("=" * 50)


history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_PATH
)


print("\nModel saved to:")

print(
    MODEL_PATH
)


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure()

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CNN Training and Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid()

plt.savefig(
    "training_accuracy.png"
)

plt.show()


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure()

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "CNN Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid()

plt.savefig(
    "training_loss.png"
)

plt.show()


print("\nTraining completed successfully.")