# CBAM-based Autoencoder for Anomaly Detection and Localization

This repository implements a lightweight convolutional autoencoder enhanced with
**Convolutional Block Attention Modules (CBAM)** for image anomaly detection and
pixel-level anomaly localization.

The proposed approach follows a **one-class / unsupervised learning** paradigm:
The model is trained exclusively on normal samples and learns to identify
deviations at both the image and pixel levels.

The focus of this project is on **practical anomaly detection**, interpretability,
and efficiency, making it suitable for research prototyping and real-world
applications with limited computational resources.



## Key Features
- One-class training using only normal samples  
- CBAM attention for enhanced spatial and channel-wise feature modeling  
- Image-level anomaly detection evaluated with AUROC  
- Pixel-level anomaly localization evaluated with IoU and Dice  
- Lightweight architecture with CPU-friendly inference  



## Dataset and Models
- CIFAR-10 is used as the benchmark dataset (normal class: airplane).
- The dataset is **not included** in this repository due to size constraints.
- Trained model checkpoints and generated visualizations are also excluded.

The code can be easily adapted to other image anomaly detection benchmarks or
custom datasets.



## Tasks
- Image-level anomaly detection (OOD detection)
- Pixel-level anomaly localization



## Evaluation Metrics
- AUROC (image-level detection)
- IoU and Dice score (localization)



## Requirements
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn



## Usage
Open and run the main notebook:

code/eval_cbam_ae_ood_and_localization.ipynb


The notebook handles:
- Dataset loading
- Model evaluation
- Anomaly score computation
- Localization map generation
- Visualization of results



## Notes
This repository focuses on **evaluation, analysis, and interpretability**.
Training pipelines and additional extensions (e.g., different attention modules,
fusion strategies, or datasets) can be added on top of this framework.
