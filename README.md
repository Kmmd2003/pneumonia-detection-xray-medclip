# Pneumonia Detection from Chest X-ray Images using CLIP

This repository presents an ongoing research project on **pneumonia detection from chest X-ray images** using vision-language models, with a primary focus on **CLIP**.

The goal of this project is to systematically evaluate and improve the performance of vision-language models for medical image understanding, progressing from **zero-shot evaluation** to **fine-tuned medical domain adaptation**, and eventually extending to multimodal models.

---

## 🔬 Project Overview

Chest X-ray imaging is one of the most widely used diagnostic tools for pneumonia detection. Recently, vision-language models such as CLIP have shown promising generalization abilities. However, their effectiveness in medical imaging—especially without domain-specific training—remains an open research question.

This project investigates:
- The limitations of **zero-shot CLIP** on medical X-ray images
- The impact of **fine-tuning CLIP** on pneumonia classification
- Performance improvements across different experimental settings

---

## 📁 Experiments

### Experiment 1: Zero-shot Evaluation
- Model: `openai/clip-vit-base-patch16`
- Training: None
- Dataset: Chest X-ray Pneumonia (Kaggle)
- Purpose: Evaluate out-of-the-box CLIP performance on medical images

**Results:**
- Accuracy: 0.3798  
- Precision: 1.0000  
- Recall: 0.0077  
- F1-score: 0.0153  

This experiment highlights the severe limitations of zero-shot CLIP in identifying pneumonia cases without domain adaptation.

---

### Experiment 2: Fine-tuning CLIP
- Model: `openai/clip-vit-base-patch16`
- Training: ✔ Fine-tuned on Kaggle Chest X-ray training set
- Evaluation: Kaggle test set

**Results:**
- Accuracy: 0.7356  
- Precision: 0.7103  
- Recall: 0.9744  
- F1-score: 0.8216  

Fine-tuning significantly improves recall and overall classification performance.

---

## 📊 Visualizations
The repository includes:
- Confusion matrices
- Metric comparison plots
- Loss curves for training analysis

All figures are generated using `matplotlib`.

---

## 📦 Dataset

The dataset used in this project is publicly available on Kaggle:

🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Due to size and licensing constraints, the dataset is **not included** in this repository.

---

## 💾 Checkpoints

Trained model checkpoints are saved locally and in Google Drive but are **not included in this repository**, following best academic and open-source practices.

---

## 🚧 Project Status

⚠ **This project is currently under active development.**

Planned future work includes:
- Additional fine-tuning strategies
- More robust evaluation metrics (e.g., ROC-AUC)
- Experiments with larger and multimodal models
- Extension to medical vision-language architectures
- Preparation of results for conference submission

---

## 📌 Reproducibility

All experiments are conducted in Google Colab with GPU support.  
Code and results are structured to ensure reproducibility.

---

## 📬 Contact

For questions, suggestions, or collaboration opportunities, feel free to open an issue or contact the author.

---

*This repository is intended for research and educational purposes.*
