# ============================================================
# ResNet50_noisy.py - 50 EPOCHS FULL
# Chest X-Ray Pneumonia Classification
# NOISY DATASET - RESNET50 WITH GAUSSIAN + SALT & PEPPER NOISE
# ============================================================

import os
import numpy as np
import random
import tensorflow as tf
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2


# ============================================================
# FIX RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seeds(42)

# ============================================================
# PATHS
# ============================================================
dataset_path = r"C:\Users\hp\Desktop\DIP-G7\chest_xray\balanced_dataset"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\resnet50_noisy"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # FULL 50 EPOCHS - NOT DIVIDED


# ============================================================
# NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(image):
    """
    Add Gaussian noise to image
    Args:
        image: Input image in range [0, 255] (uint8)
    Returns:
        Noisy image with Gaussian noise added
    """
    row, col, ch = image.shape
    mean = 0
    sigma = 25  # Noise intensity (25 is noticeable but not destructive)
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image):
    """
    Add Salt and Pepper noise to image
    Args:
        image: Input image in range [0, 255] (uint8)
    Returns:
        Noisy image with salt & pepper noise
    """
    noisy = np.copy(image)
    prob = 0.03  # 3% of pixels affected (balanced between too little and too much)

    # Salt noise (white pixels - value 255)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords_salt)] = 255

    # Pepper noise (black pixels - value 0)
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0

    return noisy.astype(np.uint8)


def apply_both_noises(image):
    """
    Apply both Gaussian and Salt & Pepper noise to image
    This ensures the image is degraded enough to see accuracy drop
    """
    # First apply Gaussian noise
    image = add_gaussian_noise(image)
    # Then apply Salt & Pepper noise
    image = add_salt_pepper_noise(image)
    return image


def noisy_preprocessing(image):
    """
    Complete preprocessing for noisy images:
    1. Apply both noises
    2. Normalize to [-1, 1] for ResNet50
    """
    # Apply both noises
    image = apply_both_noises(image)

    # Normalize to [-1, 1] as expected by ResNet50
    image = image / 127.5 - 1

    return image


# ============================================================
# DATA GENERATORS WITH NOISE
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing,
    rotation_range=10,  # Reduced augmentation because noise already adds variability
    horizontal_flip=True,
    fill_mode='nearest'
)

# For validation and test, ONLY add noise (no augmentation)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing
)

print("=" * 60)
print("LOADING DATASETS WITH NOISE")
print("=" * 60)

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

print(f"\n✅ Training samples: {train_data.samples}")
print(f"✅ Validation samples: {val_data.samples}")
print(f"✅ Test samples: {test_data.samples}")
print(f"✅ Classes: {train_data.class_indices}")

# ============================================================
# VERIFY NOISE IS BEING APPLIED (Save sample noisy images)
# ============================================================
print("\n" + "=" * 60)
print("VERIFYING NOISE APPLICATION")
print("=" * 60)

# Get one batch of images
sample_images, sample_labels = next(train_data)

# Save sample noisy images for verification
sample_dir = os.path.join(plot_dir, "noise_samples")
os.makedirs(sample_dir, exist_ok=True)

for i in range(min(5, len(sample_images))):
    # Convert back to [0, 255] range for saving
    img_display = ((sample_images[i] + 1) * 127.5).astype(np.uint8)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    plt.title(f"Noisy Image {i + 1} - Label: {'PNEUMONIA' if sample_labels[i] == 1 else 'NORMAL'}")
    plt.axis('off')
    plt.savefig(os.path.join(sample_dir, f"noisy_sample_{i + 1}.png"))
    plt.close()

print(f"✅ Sample noisy images saved to: {sample_dir}")
print("✅ Noise verification: Both Gaussian and Salt & Pepper applied")

# ============================================================
# BUILD RESNET50 MODEL (SAME ARCHITECTURE AS ORIGINAL)
# ============================================================
print("\n" + "=" * 60)
print("BUILDING RESNET50 MODEL FOR NOISY DATA")
print("=" * 60)

# Load base model
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model (no fine-tuning needed for noisy data)
base_model.trainable = False

# Build complete model (SAME architecture as original for fair comparison)
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile with slightly higher learning rate for noisy data
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS (NO EARLY STOPPING - GUARANTEE 50 EPOCHS)
# ============================================================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "ResNet50_noisy_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

csv_logger = CSVLogger(
    os.path.join(plot_dir, "training_log.csv"),
    separator=',',
    append=False
)

# NO EARLY STOPPING - WILL RUN ALL 50 EPOCHS
callbacks = [reduce_lr, checkpoint, csv_logger]

# ============================================================
# TRAIN FOR 50 EPOCHS DIRECTLY (NOT DIVIDED)
# ============================================================
print("\n" + "=" * 60)
print(f"TRAINING RESNET50 ON NOISY DATA FOR {EPOCHS} EPOCHS")
print("⚠️ EXPECTED ACCURACY DROP COMPARED TO ORIGINAL")
print("=" * 60)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,  # FULL 50 EPOCHS - NOT SPLIT
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training completed for all {len(history.history['accuracy'])} epochs")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON NOISY TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\n{'=' * 60}")
print(f"TEST RESULTS - NOISY DATA")
print(f"{'=' * 60}")
print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"{'=' * 60}")

# ============================================================
# PREDICTIONS
# ============================================================
print("\nGenerating predictions...")
predictions = model.predict(test_data, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'size': 14})
plt.title(f"ResNet50 - Noisy Dataset (50 Epochs)\nTest Accuracy: {test_accuracy:.2%}",
          fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
report = classification_report(true_classes, predicted_classes,
                               target_names=class_labels, digits=4)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT - NOISY DATA")
print("=" * 60)
print(report)

with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write("RESNET50 - NOISY DATASET (50 EPOCHS)\n")
    f.write("=" * 60 + "\n")
    f.write(f"NOISE TYPE: Gaussian (σ=25) + Salt & Pepper (3%)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# ============================================================
# PLOT: ACCURACY & LOSS (Full 50 epochs)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
ax1.set_title(f'ResNet50 on Noisy Data - Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
ax2.set_title(f'ResNet50 on Noisy Data - Loss (50 Epochs)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "training_history_50epochs.png"), dpi=300)
plt.close()

# Individual accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
plt.title(f"ResNet50 Noisy Data - Accuracy (50 Epochs)\nFinal Test Accuracy: {test_accuracy:.2%}",
          fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig(os.path.join(plot_dir, "accuracy_50epochs.png"), dpi=300)
plt.close()

# Individual loss plot
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.title("ResNet50 Noisy Data - Loss (50 Epochs)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "loss_50epochs.png"), dpi=300)
plt.close()

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model.save(os.path.join(model_dir, "ResNet50_noisy_50epochs_final.h5"))
print(f"\n✅ Final model saved to: {os.path.join(model_dir, 'ResNet50_noisy_50epochs_final.h5')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY - 50 EPOCHS")
print("=" * 60)
print(f"📊 Total Epochs Completed: {len(history.history['accuracy'])} / 50")
print(f"📈 Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"📉 Final Test Loss: {test_loss:.4f}")
print("\n🔊 NOISE PARAMETERS:")
print(f"   • Gaussian Noise: σ = 25")
print(f"   • Salt & Pepper Noise: {3}% density")
print("\n📁 Saved Files:")
print(f"   • Confusion Matrix: {os.path.join(plot_dir, 'confusion_matrix.png')}")
print(f"   • Accuracy Plot: {os.path.join(plot_dir, 'accuracy_50epochs.png')}")
print(f"   • Loss Plot: {os.path.join(plot_dir, 'loss_50epochs.png')}")
print(f"   • Training History: {os.path.join(plot_dir, 'training_history_50epochs.png')}")
print(f"   • Classification Report: {os.path.join(plot_dir, 'classification_report.txt')}")
print(f"   • Sample Noisy Images: {os.path.join(plot_dir, 'noise_samples')}")
print("=" * 60)