# Diabetic Retinopathy Grading via RETFound

## Objective
This project aims to fine-tune **RETFound**, a medical Vision # Diabetic Retinopathy Grading via RETFound

## 🎯 Objective
This project fine-tunes **RETFound**, a medical Vision Transformer (ViT) foundation model, to classify retinal fundus images across a 5-grade severity scale for Diabetic Retinopathy. The core focus is bridging the gap between statistical accuracy and **clinical safety** by mathematically optimizing decision thresholds to minimize critical False Negatives.

## 🧠 Core Methodologies
* **Foundation Models:** Fine-tuning domain-specific Vision Transformers for medical imaging.
* **Computer Vision:** Ben Graham's preprocessing (unsharp masking) for robust illumination normalization.
* **Imbalance Handling:** Dynamic inverse-frequency weighting strategies (`WeightedRandomSampler`).
* **Optimization:** Bounded Nelder-Mead simplex algorithm for derivative-free, asymmetric threshold tuning.
* **Explainable AI (XAI):** Custom Grad-CAM implementation to extract attention maps and perform spatial failure analysis.

## 🛠️ Tech Stack
* **Core:** PyTorch, Python
* **Data & Vision:** OpenCV, NumPy, Pandas
* **Evaluation:** Scikit-Learn (Quadratic Weighted Kappa, Sensitivity/Recall tracking)

## 📊 Datasets
* **Training & Internal Validation:** APTOS, DDR, Messidor-2
* **External Testing (Out-of-Distribution):** IDRiD