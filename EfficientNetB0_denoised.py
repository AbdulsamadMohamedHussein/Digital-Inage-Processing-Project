# ===========================================================
# EfficientNetB0_Denoised.py
# Chest X-Ray Pneumonia Classification with Denoising
# ===========================================================

import os
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

# ===========================================================
# PATHS
# ===========================================================
dataset_path = r"C:\Users\hp\Desktop\DIP-G7\chest_xray\balanced_dataset"
train_path = os.path.join(dataset_path, "train")
val_path   = os.path.join(dataset_path, "val")
test_path  = os.path.join(dataset_path, "test")

plot_dir   = r"C:\Users\hp\Desktop\DIP-G7\plots\EfficientNetB0_denoised"
model_dir  = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ===========================================================
# SETTINGS
# ===========================================================
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 50

# ===========================================================
# DENOISING FUNCTION
# ===========================================================
def denoise_image(image):
    # Gaussian blur
    image = cv2.GaussianBlur(image, (3,3), 0)
    # Median blur
    image = cv2.medianBlur(image, 3)
    return image

# Custom preprocessing function for ImageDataGenerator
def preprocess_input(img):
    img = img.astype(np.uint8)
    img = denoise_image(img)
    img = img / 255.0
    return img

# ===========================================================
# DATA AUGMENTATION
# ===========================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# ===========================================================
# BUILD MODEL
# ===========================================================
# Base EfficientNetB0
base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze for first stage

# Head: VGG-style with Dropout and BatchNorm
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===========================================================
# CALLBACKS
# ===========================================================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, "EfficientNetB0_denoised_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ===========================================================
# TRAIN MODEL - Stage 1: frozen backbone
# ===========================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[reduce_lr, checkpoint]
)

# ===========================================================
# UNFREEZE TOP LAYERS FOR FINE-TUNING
# ===========================================================
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train remaining epochs
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS-20,
    callbacks=[reduce_lr, checkpoint]
)

# Combine histories
history.history.update(history2.history)

# ===========================================================
# EVALUATION
# ===========================================================
pred = model.predict(test_data)
y_pred = (pred > 0.5).astype(int).reshape(-1)

cm = confusion_matrix(test_data.classes, y_pred)
class_labels = list(test_data.class_indices.keys())

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title("EfficientNetB0 Denoised Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

report = classification_report(test_data.classes, y_pred, target_names=class_labels)
with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write(report)

test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# ===========================================================
# PLOT ACCURACY & LOSS
# ===========================================================
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("EfficientNetB0 Denoised Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("EfficientNetB0 Denoised Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss.png"))
plt.close()

# ===========================================================
# SAVE FINAL MODEL
# ===========================================================
model.save(os.path.join(model_dir, "EfficientNetB0_denoised_final.h5"))

print("\n✅ EfficientNetB0 Denoised Training Completed")
print("Saved files:")
print("1. Confusion Matrix")
print("2. Accuracy Plot")
print("3. Loss Plot")
print("4. Classification Report")
print("5. Best Model")
print("6. Final Model")