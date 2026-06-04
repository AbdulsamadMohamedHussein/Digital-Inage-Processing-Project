# ============================================================
# ResNet50_original.py - WORKING VERSION (50 EPOCHS)
# Chest X-Ray Pneumonia Classification
# ORIGINAL DATASET - RESNET50
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
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization
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

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\resnet50_original"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # FULL 50 EPOCHS


# ============================================================
# PROPER PREPROCESSING FOR RESNET50
# ============================================================
def preprocess_input(x):
    # ResNet50 expects inputs in range [-1, 1]
    return tf.keras.applications.resnet50.preprocess_input(x)


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
# BUILD RESNET50 MODEL
# ============================================================
print("\n" + "=" * 60)
print("BUILDING RESNET50 MODEL")
print("=" * 60)

# Load base model
base_model = ResNet50(
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
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Higher LR for ResNet
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
    os.path.join(model_dir, "ResNet50_original_best.h5"),
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
print(f"TRAINING RESNET50 FOR {EPOCHS} EPOCHS")
print("=" * 60)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,  # DIRECT 50 EPOCHS
    callbacks=callbacks,
    verbose=1
)

print(f"\n✅ Training completed for all {len(history.history['accuracy'])} epochs")

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
plt.title(f"ResNet50 - Original Dataset (50 Epochs)\nTest Accuracy: {test_accuracy:.2%}",
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
    f.write("RESNET50 - ORIGINAL DATASET (50 EPOCHS)\n")
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
ax1.set_title(f'ResNet50 Model Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Loss plot
ax2.plot(history.history['loss'], label='Train Loss', linewidth=2, color='blue')
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
ax2.set_title(f'ResNet50 Model Loss (50 Epochs)', fontsize=14, fontweight='bold')
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
plt.title(f"ResNet50 Original - Accuracy (50 Epochs)\nFinal Test Accuracy: {test_accuracy:.2%}",
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
plt.title("ResNet50 Original - Loss (50 Epochs)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "loss_50epochs.png"), dpi=300)
plt.close()

# ============================================================
# SAVE FINAL MODEL
# ============================================================
model.save(os.path.join(model_dir, "ResNet50_original_50epochs_final.h5"))
print(f"\n✅ Final model saved to: {os.path.join(model_dir, 'ResNet50_original_50epochs_final.h5')}")

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
print("=" * 60)