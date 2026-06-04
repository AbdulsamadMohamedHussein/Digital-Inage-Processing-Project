# ============================================================
# ViT_base_denoised.py - 50 EPOCHS FULL
# Chest X-Ray Pneumonia Classification
# DENOISED DATASET - ViT BASE with MEDIAN + GAUSSIAN FILTERS
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

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\vit_base_denoised"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Create subdirectories
denoising_comparison_dir = os.path.join(plot_dir, "denoising_comparison")
os.makedirs(denoising_comparison_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # FULL 50 EPOCHS


# ============================================================
# NOISE FUNCTIONS (SAME AS NOISY SCRIPT)
# ============================================================

def add_gaussian_noise(image):
    """Add Gaussian noise to image"""
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image):
    """Add Salt and Pepper noise to image"""
    noisy = np.copy(image)
    prob = 0.03
    num_salt = np.ceil(prob * image.size * 0.5)
    coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords_salt)] = 255
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords_pepper)] = 0
    return noisy.astype(np.uint8)


# ============================================================
# DENOISING FUNCTIONS
# ============================================================

def denoise_with_median(image, kernel_size=3):
    """Apply Median filter - removes salt & pepper noise"""
    return cv2.medianBlur(image, kernel_size)


def denoise_with_gaussian(image, kernel_size=(3, 3), sigma=1.5):
    """Apply Gaussian filter - smooths Gaussian noise"""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_denoising_pipeline(image):
    """
    Complete denoising process:
    1. Add noise (same as noisy script)
    2. Apply Median filter (removes salt & pepper)
    3. Apply Gaussian filter (smooths Gaussian noise)
    """
    # Add both noises
    image = add_gaussian_noise(image)
    image = add_salt_pepper_noise(image)

    # Apply denoising
    image = denoise_with_median(image, kernel_size=3)
    image = denoise_with_gaussian(image, kernel_size=(3, 3), sigma=1.5)

    return image


def denoised_preprocessing(image):
    """Complete preprocessing for denoised images"""
    # Apply noise + denoising
    image = apply_denoising_pipeline(image)
    # Normalize to [-1, 1] for ViT
    image = image / 127.5 - 1
    return image


# ============================================================
# DATA GENERATORS WITH DENOISING
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=denoised_preprocessing,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=denoised_preprocessing
)

print("=" * 60)
print("LOADING DATASETS WITH DENOISING")
print("=" * 60)
print("Denoising Pipeline: Median Filter (3x3) + Gaussian Filter (3x3, σ=1.5)")

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
# VISUALIZE DENOISING EFFECT (Save comparison images)
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZING DENOISING EFFECT")
print("=" * 60)

# Get sample images
sample_images_paths = []
for category in ["NORMAL", "PNEUMONIA"]:
    category_path = os.path.join(train_path, category)
    if os.path.exists(category_path):
        images = os.listdir(category_path)[:2]
        for img_name in images:
            sample_images_paths.append(os.path.join(category_path, img_name))

for idx, img_path in enumerate(sample_images_paths[:4]):
    # Read original image
    original = cv2.imread(img_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_resized = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

    # Add noise only
    noisy = add_gaussian_noise(original_resized)
    noisy = add_salt_pepper_noise(noisy)

    # Apply denoising
    denoised = denoise_with_median(noisy, kernel_size=3)
    denoised = denoise_with_gaussian(denoised, kernel_size=(3, 3), sigma=1.5)

    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_resized)
    axes[0].set_title('Original (Clean)', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(noisy)
    axes[1].set_title('Noisy (Gaussian + S&P)', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(denoised)
    axes[2].set_title('Denoised (Median + Gaussian)', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'ViT Base - Denoising Comparison {idx + 1}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(denoising_comparison_dir, f"denoising_comparison_{idx + 1}.png"), dpi=300,
                bbox_inches='tight')
    plt.close()

print(f"✅ Denoising comparison images saved to: {denoising_comparison_dir}")

# ============================================================
# CUSTOM ViT IMPLEMENTATION (SAME AS ORIGINAL)
# ============================================================
print("\n" + "=" * 60)
print("BUILDING ViT BASE MODEL FOR DENOISED DATA")
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
    os.path.join(model_dir, "ViT_base_denoised_best.h5"),
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
# TRAIN FOR 50 EPOCHS DIRECTLY
# ============================================================
print("\n" + "=" * 60)
print(f"TRAINING ViT BASE ON DENOISED DATA FOR {EPOCHS} EPOCHS")
print("ALL 50 EPOCHS WILL RUN COMPLETELY")
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

print(f"\n✅ Training completed for all {len(history.history['accuracy'])} epochs")
print(f"✅ Total training time: {str(training_time).split('.')[0]}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON DENOISED TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\n{'=' * 60}")
print(f"TEST RESULTS - ViT BASE DENOISED")
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
predicted_probabilities = predictions.flatten()
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(true_classes, predicted_classes)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision_pneumonia = tp / (tp + fp) if (tp + fp) > 0 else 0
precision_normal = tn / (tn + fn) if (tn + fn) > 0 else 0
f1_pneumonia = 2 * (precision_pneumonia * sensitivity) / (precision_pneumonia + sensitivity) if (
                                                                                                            precision_pneumonia + sensitivity) > 0 else 0
f1_normal = 2 * (precision_normal * specificity) / (precision_normal + specificity) if (
                                                                                                   precision_normal + specificity) > 0 else 0

# ROC Curve
fpr, tpr, _ = roc_curve(true_classes, predicted_probabilities)
roc_auc = auc(fpr, tpr)

# ============================================================
# PLOTS
# ============================================================

# Plot 1: Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'size': 16})
plt.title(f"ViT Base - Denoised Dataset (50 Epochs)\nTest Accuracy: {test_accuracy:.2%}",
          fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - ViT Base Denoised', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "roc_curve.png"), dpi=300)
plt.close()

# Plot 3: Accuracy & Loss (Side by Side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
best_val_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_val_acc)
ax1.scatter(best_epoch, best_val_acc, color='red', s=100, zorder=5, label=f'Best: {best_val_acc:.4f}')
ax1.set_title(f'ViT Base Denoised - Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
best_val_loss = min(history.history['val_loss'])
best_loss_epoch = history.history['val_loss'].index(best_val_loss)
ax2.scatter(best_loss_epoch, best_val_loss, color='red', s=100, zorder=5, label=f'Best: {best_val_loss:.4f}')
ax2.set_title(f'ViT Base Denoised - Loss (50 Epochs)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "training_history.png"), dpi=300)
plt.close()

# Plot 4: Detailed Accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
best_acc = max(history.history['val_accuracy'])
best_epoch = history.history['val_accuracy'].index(best_acc)
plt.scatter(best_epoch, best_acc, color='red', s=100, label=f'Best: {best_acc:.4f}')
plt.title(f'ViT Base Denoised - Accuracy (50 Epochs)\nFinal Test Accuracy: {test_accuracy:.2%}',
          fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])
plt.savefig(os.path.join(plot_dir, "accuracy_50epochs.png"), dpi=300)
plt.close()

# Plot 5: Detailed Loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
best_loss = min(history.history['val_loss'])
best_loss_epoch = history.history['val_loss'].index(best_loss)
plt.scatter(best_loss_epoch, best_loss, color='red', s=100, label=f'Best: {best_loss:.4f}')
plt.title('ViT Base Denoised - Loss (50 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
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
        plt.title("Learning Rate Schedule - ViT Base Denoised", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, "learning_rate_schedule.png"), dpi=300)
        plt.close()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================
report = classification_report(true_classes, predicted_classes,
                               target_names=class_labels, digits=4)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT - ViT BASE DENOISED")
print("=" * 60)
print(report)

with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write("ViT BASE - DENOISED DATASET (50 EPOCHS)\n")
    f.write("=" * 60 + "\n")
    f.write(f"DENOISING METHOD: Median Filter (3x3) + Gaussian Filter (3x3, σ=1.5)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Sensitivity (Recall - Pneumonia): {sensitivity:.4f}\n")
    f.write(f"Specificity (Recall - Normal): {specificity:.4f}\n")
    f.write(f"Precision (Pneumonia): {precision_pneumonia:.4f}\n")
    f.write(f"Precision (Normal): {precision_normal:.4f}\n")
    f.write(f"F1-Score (Pneumonia): {f1_pneumonia:.4f}\n")
    f.write(f"F1-Score (Normal): {f1_normal:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model.save(os.path.join(model_dir, "ViT_base_denoised_50epochs_final.h5"))
print(f"\n✅ Final model saved to: {os.path.join(model_dir, 'ViT_base_denoised_50epochs_final.h5')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ ViT BASE DENOISED - 50 EPOCHS COMPLETED")
print("=" * 60)
print(f"📊 Total Epochs Completed: {len(history.history['accuracy'])} / 50")
print(f"📈 Best Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch + 1})")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"📉 Final Test Loss: {test_loss:.4f}")
print(f"🎨 AUC-ROC Score: {roc_auc:.4f}")
print(f"⏱️  Total Training Time: {str(training_time).split('.')[0]}")
print("\n🔊 DENOISING PROCESS:")
print(f"   • Step 1: Add Gaussian Noise (σ=25) + Salt & Pepper (3%)")
print(f"   • Step 2: Apply Median Filter (3x3)")
print(f"   • Step 3: Apply Gaussian Filter (3x3, σ=1.5)")
print("=" * 60)