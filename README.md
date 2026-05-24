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
Step 4: Download Dataset
# Download from Kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Or manually place data in ./data/chest_xray/
Complete Pipeline: Train All Models
# Original Condition
python models/vgg16_original.py
python models/resnet50_original.py
python models/vit_base_original.py
python models/efficientnetb0_original.py
python models/mobilenetv2_original.py

# Noisy Condition
python models/vgg16_noisy.py
python models/resnet50_noisy.py
python models/vit_base_noisy.py
python models/efficientnetb0_noisy.py
python models/mobilenetv2_noisy.py

# Denoised Condition
python models/vgg16_denoised.py
python models/resnet50_denoised.py
python models/vit_base_denoised.py
python models/efficientnetb0_denoised.py
python models/mobilenetv2_denoised.py
Expected Output Per Model
============================================================
LOADING DATASETS
============================================================
✅ Training samples: 4693
✅ Validation samples: 586
✅ Test samples: 584
✅ Classes: {'NORMAL': 0, 'PNEUMONIA': 1}

============================================================
TRAINING MODEL FOR 50 EPOCHS
============================================================
Epoch 1/50
129/129 ━━━━━━━━━━━━━━━━━━━━ 45s 350ms/step - accuracy: 0.6965 - loss: 0.6596 - val_accuracy: 0.7301 - val_loss: 0.5780

... (training progress)

============================================================
TEST RESULTS
============================================================
✅ Test Accuracy: 0.9613 (96.13%)
✅ Test Loss: 0.1621
✅ AUC-ROC: 0.979

📁 Saved Files:
   • confusion_matrix.png
   • roc_curve.png
   • accuracy_50epochs.png
   • loss_50epochs.png
   • classification_report.txt
📊 Results Summary
Complete Performance Table
Model	Original	Noisy	Denoised	Best
EfficientNetB0	96.13%	93.50%	96.50%	Denoised 🏆
MobileNetV2	91.50%	90.33%	96.00%	Denoised
VGG16	92.00%	90.00%	95.00%	Denoised
ResNet50	93.17%	94.08%	94.88%	Noisy
ViT Base	94.08%	90.33%	94.54%	Original
Denoising Recovery Ranking
Rank	Model	Improvement	Recovery Rate
1	MobileNetV2	+5.67%	100%+ 🏆
2	VGG16	+5.00%	100%+
3	ViT Base	+4.21%	95%
4	EfficientNetB0	+3.00%	100%+
5	ResNet50	+0.80%	30%
Key Metrics by Best Model
Metric	Best Model	Value
Highest Accuracy	EfficientNetB0 Denoised	96.50%
Best Sensitivity	ViT Base Original	96.10%
Best Specificity	ResNet50 Original	99.16%
Fewest False Negatives	ResNet50 Denoised	20
Fewest False Positives	ResNet50 Original	2
Best Denoising Recovery	MobileNetV2	+5.67%
📈 Visualizations
Each model run generates the following visualizations:

1. Confusion Matrix
Shows true vs predicted classifications with counts

2. ROC Curve with AUC
Demonstrates model discrimination ability

3. Training History (Accuracy & Loss)
Side-by-side accuracy and loss curves over 50 epochs

4. Denoising Comparison
Visual comparison of original → noisy → denoised images

🔮 Future Work
Test additional noise types (Poisson, Speckle, Motion blur)

Implement advanced denoising (Autoencoders, DnCNN, BM3D)

Explore ensemble methods combining multiple models

Deploy as web application using TensorFlow.js or Flask

Add explainability using Grad-CAM and attention maps

Expand to multi-class classification (bacterial vs viral pneumonia)

Real-time inference pipeline for clinical deployment

🙏 Acknowledgments
Dataset: Chest X-Ray Images (Pneumonia) by Paul Mooney

TensorFlow/Keras for deep learning framework

OpenCV for image processing operations

Kaggle for dataset hosting and community resources

📧 Contact
Author: Abdulsamad Mohamed Hussein

GitHub: github.com/AbdulsamadMohamedHussein/

LinkedIn: Abdulsamad Mohamed

Email: abdulsamadmohamed09@example.com

⭐  Your Support
If you found this project helpful, please consider:

⭐ Starring the repository on GitHub

🔄 Forking to use in your own projects

📢 Sharing with others in the field

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "This project demonstrates the importance of understanding   │
│    how different AI architectures behave under real-world      │
│    conditions like image noise, and how proper preprocessing   │
│    can significantly improve model reliability for clinical    │
│    deployment."                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
text

---

## 📝 Instructions to Upload to GitHub

1. **Create a new repository** on GitHub
2. **Copy the entire markdown code above**
3. **Create a file named `README.md`** in your repository
4. **Paste the code** into the file
5. **Commit and push** to GitHub

## 📁 Additional Files to Create

### Create `requirements.txt`
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
opencv-python>=4.8.0
pillow>=9.5.0
scipy>=1.10.0

text

### Create `LICENSE`
MIT License

Copyright (c) 2026 [Abdulsamad Mohamed Hussein]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




