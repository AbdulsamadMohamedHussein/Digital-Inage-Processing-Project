# ============================================================
# MobileNetV2_noisy.py
# ============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# PATHS
# ============================================================

dataset_path = r"C:\Users\hp\Desktop\DIP-G7\chest_xray\balanced_dataset"

train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\mobilenet_noisy"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# ============================================================
# NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(image):

    row, col, ch = image.shape

    mean = 0
    sigma = 25

    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss

    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image):

    noisy = np.copy(image)

    prob = 0.02

    # Salt
    num_salt = np.ceil(prob * image.size * 0.5)

    coords = [
        np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape
    ]

    noisy[tuple(coords)] = 255

    # Pepper
    num_pepper = np.ceil(prob * image.size * 0.5)

    coords = [
        np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape
    ]

    noisy[tuple(coords)] = 0

    return noisy

# ============================================================
# CUSTOM PREPROCESSING
# ============================================================

def noisy_preprocessing(image):

    image = image.astype(np.uint8)

    image = add_gaussian_noise(image)

    image = add_salt_pepper_noise(image)

    image = image / 255.0

    return image

# ============================================================
# DATA GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing,

    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing
)

# ============================================================
# LOAD DATA
# ============================================================

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_data = val_test_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

test_data = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ============================================================
# LOAD MOBILENETV2
# ============================================================

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    BatchNormalization(),

    Dropout(0.5),

    Dense(128, activation='relu'),

    BatchNormalization(),

    Dropout(0.4),

    Dense(1, activation='sigmoid')

])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ============================================================
# CALLBACKS
# ============================================================

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=4,
    min_lr=0.000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(
        model_dir,
        "MobileNetV2_noisy_best.h5"
    ),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[
        reduce_lr,
        checkpoint
    ]
)

# ============================================================
# EVALUATE
# ============================================================

test_loss, test_accuracy = model.evaluate(test_data)

print("\n================================")
print("TEST RESULTS")
print("================================")

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(test_data)

predicted_classes = (predictions > 0.5).astype(int)

true_classes = test_data.classes

class_labels = list(test_data.class_indices.keys())

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Reds',
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("MobileNetV2 Noisy Confusion Matrix")

plt.savefig(
    os.path.join(
        plot_dir,
        "confusion_matrix.png"
    )
)

plt.close()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
)

print(report)

with open(
    os.path.join(
        plot_dir,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)

# ============================================================
# ACCURACY PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("MobileNetV2 Noisy Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig(
    os.path.join(
        plot_dir,
        "accuracy.png"
    )
)

plt.close()

# ============================================================
# LOSS PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("MobileNetV2 Noisy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig(
    os.path.join(
        plot_dir,
        "loss.png"
    )
)

plt.close()

# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    os.path.join(
        model_dir,
        "MobileNetV2_noisy_final.h5"
    )
)

print("\n================================")
print("NOISY MODEL TRAINING COMPLETED")
print("================================")