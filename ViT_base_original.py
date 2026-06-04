# ============================================================
# ViT_base_original.py - 50 EPOCHS GUARANTEED
# Chest X-Ray Pneumonia Classification
# ORIGINAL DATASET - ViT BASE (Vision Transformer)
# ============================================================

import os
import numpy as np
import random
import tensorflow as tf
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

plot_dir = r"C:\Users\hp\Desktop\DIP-G7\plots\vit_base_original"
model_dir = r"C:\Users\hp\Desktop\DIP-G7\models"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ============================================================
# HYPERPARAMETERS
# ============================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # ALL 50 EPOCHS - NO DIVIDING


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_input(x):
    return x / 127.5 - 1  # Normalize to [-1, 1]


# ============================================================
# DATA GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

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
# CUSTOM ViT IMPLEMENTATION
# ============================================================
print("\n" + "=" * 60)
print("BUILDING ViT BASE MODEL")
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
    num_patches = (IMG_SIZE // patch_size) ** 2  # 196
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
    os.path.join(model_dir, "ViT_base_original_best.h5"),
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
print(f"TRAINING ViT BASE FOR {EPOCHS} EPOCHS")
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

print(f"\n✅ SUCCESS! All {len(history.history['accuracy'])} epochs completed")
print(f"✅ Training time: {str(training_time).split('.')[0]}")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\n✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
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
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=class_labels, yticklabels=class_labels,
            annot_kws={'size': 16})
plt.title(f"ViT Base - Confusion Matrix\nTest Accuracy: {test_accuracy:.2%}",
          fontsize=14, fontweight='bold')
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "confusion_matrix.png"), dpi=300)
plt.close()

# Plot 2: ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - ViT Base', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "roc_curve.png"), dpi=300)
plt.close()

# Plot 3: Accuracy & Loss (Side by Side)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
ax1.set_title('Accuracy (50 Epochs)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

ax2.plot(history.history['loss'], label='Train', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
ax2.set_title('Loss (50 Epochs)', fontsize=14, fontweight='bold')
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
plt.title(f'ViT Base - Accuracy (50 Epochs)\nTest Accuracy: {test_accuracy:.2%}',
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
plt.title('ViT Base - Loss (50 Epochs)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(plot_dir, "loss_50epochs.png"), dpi=300)
plt.close()

# ============================================================
# SAVE RESULTS
# ============================================================

# Classification Report
report = classification_report(true_classes, predicted_classes,
                               target_names=class_labels, digits=4)
with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
    f.write(report)

# Final Model
model.save(os.path.join(model_dir, "ViT_base_original_final.h5"))

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ 50 EPOCHS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"📊 Epochs completed: {len(history.history['accuracy'])} / 50")
print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"📈 Best Val Accuracy: {best_acc:.4f}")
print(f"📉 Best Val Loss: {best_loss:.4f}")
print(f"🎨 AUC-ROC: {roc_auc:.4f}")
print(f"⏱️  Training time: {str(training_time).split('.')[0]}")
print("=" * 60)
print(f"📁 All plots saved to: {plot_dir}")
print(f"💾 Model saved to: {model_dir}")
print("=" * 60)