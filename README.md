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
