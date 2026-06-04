# ============================================================
# ViT_base_noisy.py - 50 EPOCHS GUARANTEED
# Chest X-Ray Pneumonia Classification
# NOISY DATASET - ViT BASE with GAUSSIAN + SALT & PEPPER NOISE
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
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Input, LayerNormalization, \
    MultiHeadAttention, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.regularizers import l2


# ============================================================
# FIX RANDOM SEEDS
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

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\vit_base_noisy"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create directory for noise samples
noise_samples_dir = os.path.join(plot_dir, "noise_samples")
os.makedirs(noise_samples_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # ALL 50 EPOCHS


# ============================================================
# NOISE FUNCTIONS
# ============================================================

def add_gaussian_noise(image):
    """Add Gaussian noise to image"""
    row, col, ch = image.shape
    mean = 0
    sigma = 25  # Noise intensity
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image):
    """Add Salt and Pepper noise to image"""
    noisy = np.copy(image)
    prob = 0.03  # 3% of pixels affected
    # Salt noise (white pixels)
    num_salt = np.ceil(prob * image.size * 0.5)
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords_salt)] = 255
    # Pepper noise (black pixels)
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0
    return noisy.astype(np.uint8)


def apply_both_noises(image):
    """Apply both Gaussian and Salt & Pepper noise"""
    image = add_gaussian_noise(image)
    image = add_salt_pepper_noise(image)
    return image


def noisy_preprocessing(image):
    """Complete preprocessing for noisy images"""
    # Apply both noises
    image = apply_both_noises(image)
    # Normalize to [-1, 1] for ViT
    image = image / 127.5 - 1
    return image


# ============================================================
# DATA GENERATORS WITH NOISE
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=noisy_preprocessing
)

print("=" * 60)
print("LOADING DATASETS WITH NOISE")
print("=" * 60)
print("Noise Type: Gaussian (σ=25) + Salt & Pepper (3%)")

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
# VERIFY NOISE (Save sample noisy images)
# ============================================================
print("\n" + "=" * 60)
print("SAVING SAMPLE NOISY IMAGES")
print("=" * 60)

sample_images, sample_labels = next(train_data)
for i in range(min(5, len(sample_images))):
    # Convert back to [0, 255] for saving
    img_display = ((sample_images[i] + 1) * 127.5).astype(np.uint8)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_display)
    label = "PNEUMONIA" if sample_labels[i] == 1 else "NORMAL"
    plt.title(f"Noisy Image {i + 1} - {label}", fontsize=10)
    plt.axis('off')
    plt.savefig(os.path.join(noise_samples_dir, f"noisy_sample_{i + 1}.png"), dpi=150, bbox_inches='tight')
    plt.close()

print(f"✅ Sample noisy images saved to: {noise_samples_dir}")

# ============================================================
# CUSTOM ViT IMPLEMENTATION (SAME AS ORIGINAL)
# ============================================================
print("\n" + "=" * 60)
print("BUILDING ViT BASE MODEL FOR NOISY DATA")
print("=" * 60)


class PatchExtractor(tf.keras.layers.Layer):
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


def create_vit_model():
    patch_size = 16
    num_patches = (IMG_SIZE // patch_size) ** 2
    projection_dim = 768
    num_heads = 12
    transformer_layers = 12

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Extract patches
    patches = PatchExtractor(patch_size)(inputs)

    # Linear projection
    x = Dense(projection_dim)(patches)

    # Add positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = tf.keras.layers.Embedding(
        input_dim=num_patches,
        output_dim=projection_dim
    )(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)
    x = x + pos_embedding

    x = Dropout(0.1)(x)

    # Transformer blocks
    for _ in range(transformer_layers):
        # Attention
        norm1 = LayerNormalization(epsilon=1e-6)(x)
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim // num_heads,
            dropout=0.1
        )(norm1, norm1)
        x = Add()([attention, x])

        # MLP
        norm2 = LayerNormalization(epsilon=1e-6)(x)
        mlp = Dense(projection_dim * 4, activation='gelu')(norm2)
        mlp = Dropout(0.1)(mlp)
        mlp = Dense(projection_dim)(mlp)
        mlp = Dropout(0.1)(mlp)
        x = Add()([mlp, x])

    # Classification head
    x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Create and compile model
model = create_vit_model()
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ============================================================
# CALLBACKS - NO EARLY STOPPING
# ============================================================
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "ViT_base_noisy_best.h5"),
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
# TRAIN ALL 50 EPOCHS
# ============================================================
print("\n" + "=" * 60)
print(f"TRAINING ViT BASE ON NOISY DATA FOR {EPOCHS} EPOCHS")
print("ALL 50 EPOCHS WILL RUN COMPLETELY")
print("⚠️ EXPECTED ACCURACY DROP COMPARED TO ORIGINAL")
print("=" * 60)

start_time = datetime.now()
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
end_time = datetime.now()
training_time = end_time - start_time

print(f"\n✅ SUCCESS! All {len(history.history['accuracy'])} epochs completed")
print(f"✅ Training time: {str(training_time).split('.')[0]}")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON NOISY TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\n✅ Test Accuracy (Noisy): {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"✅ Test Loss: {test_loss:.4f}")

# Predictions
predictions = model.predict(test_data, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()
predicted_probabilities = predictions.flatten()
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

# ROC Curve
fpr, tpr, _ = roc_curve(true_classes, predicted_probabilities)
roc_auc = auc(fpr, tpr)

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'size': 16})
plt.title(f"ViT Base - Noisy Dataset\nTest Accuracy: {test_accuracy:.2%}",
          fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - ViT Base Noisy', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "roc_curve.png"), dpi=300)
plt.close()

# Plot 3: Accuracy & Loss (Side by Side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc)
ax1.scatter(best_epoch, best_val_acc, color='red', s=100, label=f'Best: {best_val_acc:.4f}')
ax1.set_title('Accuracy on Noisy Data (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
best_val_loss = min(history.history['val_loss'])
best_loss_epoch = history.history['val_loss'].index(best_val_loss)
ax2.scatter(best_loss_epoch, best_val_loss, color='red', s=100, label=f'Best: {best_val_loss:.4f}')
ax2.set_title('Loss on Noisy Data (50 Epochs)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "training_history.png"), dpi=300)
plt.close()

# Plot 4: Detailed Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
best_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_acc)
plt.scatter(best_epoch, best_acc, color='red', s=100, label=f'Best: {best_acc:.4f}')
plt.title(f'ViT Base Noisy - Accuracy (50 Epochs)\nTest Accuracy: {test_accuracy:.2%}',
          fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig(os.path.join(plot_dir, "accuracy_50epochs.png"), dpi=300)
plt.close()

# Plot 5: Detailed Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
best_loss = min(history.history['val_loss'])
best_loss_epoch = history.history['val_loss'].index(best_loss)
plt.scatter(best_loss_epoch, best_loss, color='red', s=100, label=f'Best: {best_loss:.4f}')
plt.title('ViT Base Noisy - Loss (50 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "loss_50epochs.png"), dpi=300)
plt.close()

# Plot 6: Learning Rate Schedule
import pandas as pd

csv_path = os.path.join(plot_dir, "training_log.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    if 'learning_rate' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['learning_rate'], linewidth=2, color='green')
        plt.title("Learning Rate Schedule - Noisy Data", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "learning_rate_schedule.png"), dpi=300)
        plt.close()

# ============================================================
# SAVE RESULTS
# ============================================================

# Classification Report
report = classification_report(true_classes, predicted_classes,
                               target_names=class_labels, digits=4)
with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write("ViT BASE - NOISY DATASET (50 EPOCHS)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Noise: Gaussian (σ=25) + Salt & Pepper (3%)\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Sensitivity: {sensitivity:.4f}\n")
    f.write(f"Specificity: {specificity:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# Final Model
model.save(os.path.join(model_dir, "ViT_base_noisy_final.h5"))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ 50 EPOCHS COMPLETED SUCCESSFULLY - NOISY DATA")
print("=" * 60)
print(f"📊 Epochs completed: {len(history.history['accuracy'])} / 50")
print(f"🎯 Test Accuracy (Noisy): {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"📈 Best Val Accuracy: {best_val_acc:.4f}")
print(f"📉 Best Val Loss: {best_val_loss:.4f}")
print(f"🎨 AUC-ROC: {roc_auc:.4f}")
print(f"⏱️  Training time: {str(training_time).split('.')[0]}")
print("\n🔊 NOISE PARAMETERS:")
print(f"   • Gaussian Noise: σ = 25")
print(f"   • Salt & Pepper: 3% density")
print("=" * 60)
print(f"📁 All plots saved to: {plot_dir}")
print(f"💾 Model saved to: {model_dir}")
print("=" * 60)