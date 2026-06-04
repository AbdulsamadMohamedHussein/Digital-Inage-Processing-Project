# ============================================================
# EfficientNetB0_noisy.py
# Chest X-Ray Pneumonia Classification with Noise
# ============================================================

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# PATHS
# =========================
dataset_path = r"C:\Users\hp\Desktop\DIP-G7\chest_xray\balanced_dataset"
train_path = os.path.join(dataset_path, "train")
val_path   = os.path.join(dataset_path, "val")
test_path  = os.path.join(dataset_path, "test")

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\EfficientNetB0_noisy"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =========================
# SETTINGS
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50

# ============================================================
# CUSTOM NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(img):
    img = img.astype(np.float32)
    mean = 0
    sigma = 0.05
    gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img + gauss
    return np.clip(noisy, 0.0, 1.0)

def add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01):
    img = img.astype(np.float32)
    noisy = img.copy()
    total_pixels = img.shape[0] * img.shape[1]

    # Salt
    num_salt = int(salt_prob * total_pixels)
    coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 1.0

    # Pepper
    num_pepper = int(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 0.0

    return noisy

def noisy_preprocessing(img):
    img = add_gaussian_noise(img)
    img = add_salt_pepper_noise(img)
    return img

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=noisy_preprocessing
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
# BUILD MODEL
# =========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze pretrained layers

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
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(model_dir, "EfficientNetB0_noisy_best.h5"),
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
# EVALUATION
# =========================
test_loss, test_acc = model.evaluate(test_data)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
pred = model.predict(test_data)
y_pred = (pred > 0.5).astype(int).reshape(-1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("EfficientNetB0 Noisy Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"))
plt.close()

# Classification report
report = classification_report(y_true, y_pred)
with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Accuracy & Loss Plots
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("EfficientNetB0 Noisy Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(plot_dir, "accuracy.png"))
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("EfficientNetB0 Noisy Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(plot_dir, "loss.png"))
plt.close()

# Save final model
model.save(os.path.join(model_dir, "EfficientNetB0_noisy_final.h5"))

print("✅ EfficientNetB0 Noisy Training Completed")