# ============================================================
# EfficientNetB0_original.py - 50 EPOCHS GUARANTEED
# Chest X-Ray Pneumonia Classification
# ORIGINAL DATASET - EFFICIENTNETB0
# ============================================================

import os
import numpy as np
import random
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
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

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\efficientnetb0_original"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
TOTAL_EPOCHS = 50  # FIXED: 50 epochs guaranteed


# ============================================================
# PROPER PREPROCESSING FOR EFFICIENTNET
# ============================================================
def preprocess_input(x):
    return tf.keras.applications.efficientnet.preprocess_input(x)


# ============================================================
# DATA GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# ============================================================
# LOAD DATASETS
# ============================================================
print("=" * 60)
print("LOADING DATASETS")
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
# BUILD MODEL
# ============================================================
print("\n" + "=" * 60)
print("BUILDING EFFICIENTNETB0 MODEL")
print("=" * 60)

# Load base model
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model initially
base_model.trainable = False

# Build complete model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================================
# CALLBACKS (NO EARLY STOPPING - GUARANTEE 50 EPOCHS)
# ============================================================
# Reduce LR on plateau (but don't stop training)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Save best model checkpoint
checkpoint = ModelCheckpoint(
    os.path.join(model_dir, "EfficientNetB0_original_best.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1,
    mode='max'
)

# CSV Logger for training history
csv_logger = CSVLogger(
    os.path.join(plot_dir, "training_log.csv"),
    separator=',',
    append=False
)

# NO EARLY STOPPING - we want all 50 epochs
callbacks = [reduce_lr, checkpoint, csv_logger]

# ============================================================
# PHASE 1: TRAIN TOP LAYERS (Epochs 1-25)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: TRAINING TOP LAYERS (Epochs 1-25)")
print("=" * 60)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,  # First 25 epochs with frozen base
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# PHASE 2: FINE-TUNING (Epochs 26-50)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: FINE-TUNING (Epochs 26-50)")
print("=" * 60)

# Unfreeze last 30 layers for fine-tuning
base_model.trainable = True

# Freeze all layers first
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

print(
    f"✅ Trainable layers after unfreezing: {sum(1 for l in base_model.layers if l.trainable)} out of {len(base_model.layers)}")

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.0001),  # 10x lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training for remaining 25 epochs
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,  # Second 25 epochs
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# COMBINE HISTORIES
# ============================================================
history = {
    'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
    'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
    'loss': history1.history['loss'] + history2.history['loss'],
    'val_loss': history1.history['val_loss'] + history2.history['val_loss']
}

print(f"\n✅ Total epochs completed: {len(history['accuracy'])} / {TOTAL_EPOCHS}")

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\n{'=' * 60}")
print(f"TEST RESULTS")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'size': 14})
plt.title(f"EfficientNetB0 - Original Dataset (50 epochs)\nTest Accuracy: {test_accuracy:.2%}",
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
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report)

with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write("EFFICIENTNETB0 - ORIGINAL DATASET (50 EPOCHS)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)

# ============================================================
# PLOT: ACCURACY & LOSS (Full 50 epochs)
# ============================================================
# Combined accuracy/loss plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
ax1.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
ax1.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
ax1.set_title('Model Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss plot
ax2.plot(history['loss'], label='Train Loss', linewidth=2, color='blue')
ax2.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
ax2.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
ax2.set_title('Model Loss (50 Epochs)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "training_history_50epochs.png"), dpi=300)
plt.close()

# Individual accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'], label='Train Accuracy', linewidth=2, color='blue')
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='orange')
plt.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
plt.fill_between(range(len(history['accuracy'])),
                 np.array(history['accuracy']) - np.array(history['val_accuracy']),
                 0, alpha=0.1, color='red')
plt.title(f"EfficientNetB0 Original - Accuracy (50 Epochs)\nFinal Test Accuracy: {test_accuracy:.2%}",
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
plt.plot(history['loss'], label='Train Loss', linewidth=2, color='blue')
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
plt.axvline(x=25, color='red', linestyle='--', alpha=0.7, label='Fine-tuning Start')
plt.title("EfficientNetB0 Original - Loss (50 Epochs)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "loss_50epochs.png"), dpi=300)
plt.close()

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model.save(os.path.join(model_dir, "EfficientNetB0_original_50epochs_final.h5"))
print(f"\n✅ Final model saved to: {os.path.join(model_dir, 'EfficientNetB0_original_50epochs_final.h5')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY - 50 EPOCHS")
print("=" * 60)
print(f"📊 Total Epochs: {len(history['accuracy'])} / 50")
print(f"📈 Best Validation Accuracy: {max(history['val_accuracy']):.4f}")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"📉 Final Test Loss: {test_loss:.4f}")
print(f"💾 Model saved: EfficientNetB0_original_50epochs_final.h5")
print("\n📁 Saved Files:")
print(f"   • Confusion Matrix: {os.path.join(plot_dir, 'confusion_matrix.png')}")
print(f"   • Accuracy Plot: {os.path.join(plot_dir, 'accuracy_50epochs.png')}")
print(f"   • Loss Plot: {os.path.join(plot_dir, 'loss_50epochs.png')}")
print(f"   • Training History: {os.path.join(plot_dir, 'training_history_50epochs.png')}")
print(f"   • Classification Report: {os.path.join(plot_dir, 'classification_report.txt')}")
print(f"   • Training Log: {os.path.join(plot_dir, 'training_log.csv')}")
print("=" * 60)