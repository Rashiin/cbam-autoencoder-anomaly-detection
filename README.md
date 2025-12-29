# CBAM-based Autoencoder for Anomaly Detection and Localization

This project implements a lightweight convolutional autoencoder augmented
with CBAM attention modules for anomaly detection and pixel-level localization.

The model is trained in a one-class setting on CIFAR-10 and evaluated using
both out-of-distribution samples and synthetic anomalies.

## Features
- One-class training using normal samples only

- CBAM attention for enhanced spatial focus
- Image-level anomaly detection (AUROC)
- Pixel-level anomaly localization (IoU, Dice)
- CPU-friendly evaluation pipeline

## Project Structure
- `evaluate_cbam_autoencoder.py`: main evaluation script
- `visualizations/`: trained checkpoint and visual results
- `results/`: quantitative metrics and analysis

