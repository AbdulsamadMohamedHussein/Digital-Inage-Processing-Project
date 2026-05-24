# Digital-Image-Processing-Project
# 🫁 Chest X-Ray Pneumonia Detection: Comparative Study of Deep Learning Architectures Under Noisy Conditions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.13+-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Made with](https://img.shields.io/badge/Made%20with-❤️-red.svg)](https://github.com/yourusername)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Experimental Design](#experimental-design)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Results Summary](#results-summary)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🎯 Project Overview

This project presents a **comprehensive comparative analysis** of five state-of-the-art deep learning architectures for pneumonia detection from chest X-ray images. The study evaluates model performance under three critical conditions:

| Condition | Description |
|-----------|-------------|
| 🟢 **Original** | Clean, unaltered chest X-ray images |
| 🟡 **Noisy** | Images corrupted with Gaussian + Salt & Pepper noise |
| 🔵 **Denoised** | Noisy images restored using Median + Gaussian filters |

### Why This Matters

In real-world clinical settings, medical images often suffer from various types of noise due to:
- Equipment limitations
- Patient movement during capture
- Transmission errors
- Storage compression

Understanding how different AI models perform under these conditions is **critical for deploying reliable diagnostic systems** in actual healthcare environments.

---

## 🏆 Key Findings

| Rank | Model | Best Accuracy | Best Condition | Key Strength |
|------|-------|---------------|----------------|---------------|
| 🥇 | **EfficientNetB0** | **96.50%** | Denoised | Best overall performance |
| 🥈 | **MobileNetV2** | **96.00%** | Denoised | Best denoising recovery (+5.67%) |
| 🥉 | **VGG16** | 95.00% | Denoised | Strong improvement (+5.00%) |
| 4 | **ResNet50** | 94.88% | Denoised | Most noise-robust |
| 5 | **ViT Base** | 94.54% | Denoised | Best sensitivity (96.10%) |

### 🎯 Clinical Recommendation
> **EfficientNetB0 with denoising preprocessing** offers the best balance of pneumonia detection (96.41% sensitivity) and false alarm avoidance (97.06% specificity), making it ideal for clinical deployment.

---

## 📊 Dataset

### Source
- **Dataset:** Chest X-Ray Images (Pneumonia) from Kaggle
- **Total Images:** 5,863
- **Classes:** NORMAL (1,583) | PNEUMONIA (4,273)
- **Image Format:** JPEG

### Data Split

### Balanced Test Set
| Class | Training | Validation | Test |
|-------|----------|------------|------|
| NORMAL | 1,267 | 158 | 238 |
| PNEUMONIA | 3,426 | 428 | 641 |

---

## 🧠 Models Implemented

| Model | Architecture Type | Parameters | Strengths |
|-------|------------------|------------|------------|
| **VGG16** | Convolutional Neural Network | 138M | Classic, well-understood |
| **ResNet50** | Residual CNN | 25M | Skip connections, deep architecture |
| **ViT Base** | Vision Transformer | 86M | Attention mechanism, global context |
| **EfficientNetB0** | Efficient CNN | 5.3M | Compound scaling, efficient |
| **MobileNetV2** | Lightweight CNN | 3.5M | Mobile-optimized, fast inference |

---

## 🔬 Experimental Design

### 3 × 5 Factorial Design

### Noise Parameters
```python
GAUSSIAN_NOISE = {
    'mean': 0,
    'sigma': 25  # Moderate noise intensity
}

SALT_PEPPER_NOISE = {
    'probability': 0.03  # 3% of pixels affected
}
MEDIAN_FILTER = {
    'kernel_size': 3  # Removes salt & pepper
}

GAUSSIAN_FILTER = {
    'kernel_size': (3, 3),
    'sigma': 1.5  # Smooths Gaussian noise
}
chest-xray-pneumonia-detection/
│
├── 📂 models/
│   ├── vgg16_original.py
│   ├── vgg16_noisy.py
│   ├── vgg16_denoised.py
│   ├── resnet50_original.py
│   ├── resnet50_noisy.py
│   ├── resnet50_denoised.py
│   ├── vit_base_original.py
│   ├── vit_base_noisy.py
│   ├── vit_base_denoised.py
│   ├── efficientnetb0_original.py
│   ├── efficientnetb0_noisy.py
│   ├── efficientnetb0_denoised.py
│   ├── mobilenetv2_original.py
│   ├── mobilenetv2_noisy.py
│   └── mobilenetv2_denoised.py
│
├── 📂 plots/
│   ├── vgg16_original/
│   ├── vgg16_noisy/
│   ├── vgg16_denoised/
│   └── ... (similar for all models)
│
├── 📂 results/
│   ├── classification_reports/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── training_logs/
│
├── 📂 data/
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
│
├── README.md
├── requirements.txt
└── LICENSE
🛠 Installation & Setup
Prerequisites
Python 3.8+
CUDA-capable GPU (recommended for faster training)

Step 1: Clone the Repository
git clone https://github.com/AbdulsamadMohamedHussein/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
Step 2: Create Virtual Environment
# Windows
python -m venv tf_env
tf_env\Scripts\activate

# Linux/Mac
python -m venv tf_env
source tf_env/bin/activate
Step 3: Install Dependencies
pip install -r requirements.txt
requirements.txt
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.8.0
pillow>=9.5.0
scipy>=1.10.0

