# ============================================================
# ViT_base_original.py
# Chest X-Ray Pneumonia Classification
# ORIGINAL DATASET
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import vit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# =========================
# PATHS
# =========================
dataset_path = r"C:\Users\hp\Desktop\DIP-G7\chest_xray\balanced_dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\vit_base_original"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

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

# =========================
# LOAD ViT BASE
# =========================
base_model = vit.ViT(
    include_top=False,
    pretrained=True,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers
base_model.trainable = False

# =========================
# BUILD MODEL
# =========================
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# CALLBACKS
# =========================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.000001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, "ViT_base_original_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# TRAIN MODEL
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[reduce_lr, checkpoint]
)

# =========================
# EVALUATE MODEL
# =========================
test_loss, test_accuracy = model.evaluate(test_data)
print("\n===================================")
print("TEST RESULTS")
print("===================================")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# =========================
# PREDICTIONS
# =========================
predictions = model.predict(test_data)
predicted_classes = (predictions > 0.5).astype(int)
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("ViT Base Original Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# =========================
# CLASSIFICATION REPORT
# =========================
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print("\nClassification Report:\n")
print(report)

with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# =========================
# ACCURACY & LOSS PLOTS
# =========================
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("ViT Base Original Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("ViT Base Original Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss.png"))
plt.close()

# =========================
# SAVE FINAL MODEL
# =========================
model.save(os.path.join(model_dir, "ViT_base_original_final.h5"))

print("\n===================================")
print("ViT BASE ORIGINAL TRAINING COMPLETED")
print("===================================")
print("\nSaved Files:")
print("1. Confusion Matrix")
print("2. Accuracy Plot")
print("3. Loss Plot")
print("4. Classification Report")
print("5. Best Model")
print("6. Final Model")