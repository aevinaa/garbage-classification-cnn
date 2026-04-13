# Garbage Density Classification Using Custom CNN

# Overview
This project focuses on classifying garbage density in images into three categories:

* Low
* Medium
* High

The model is built using a custom Convolutional Neural Network (CNN) trained from scratch without using any pretrained architectures.

# Key Features
* Custom CNN architecture (no pretrained models)
* Image classification into 3 classes: Low / Medium / High
* Data preprocessing and augmentation
* Handling class imbalance using weighted loss
* Achieved ~85% validation accuracy

# Model Architecture
The model consists of:

* 3 Convolutional Layers
* ReLU Activation
* Batch Normalization
* Max Pooling
* Fully Connected Layers with Dropout

Flow:
Conv → ReLU → BatchNorm → Pool → FC → Output

# Performance
* Validation Accuracy: ~85%

# Dataset
* Custom dataset created and labeled manually
* Categories:

  * Low (clean areas)
  * Medium (moderate waste)
  * High (heavy garbage/dumps)
* Only a small sample dataset is included in this repository

# Challenges Faced
* Model initially predicted only one class (bias issue)
* Overfitting due to small dataset
* Difficulty distinguishing between medium and high classes

# Solutions
* Applied class weighting in loss function
* Added data augmentation
* Tuned training epochs and model architecture

# Project Structure
```
garbage-ai-model/
│
├── model.py
├── train.py
├── predict.py
├── model.pth
├── dataset/
│   ├── low/
│   ├── medium/
│   └── high/
└── README.md
```

# Future Improvements
* Larger and more diverse dataset
* Better separation between medium and high classes
* Deploy as a real-time application

# Motivation

This project was built to explore real-world image classification using a custom CNN. 
Unlike pretrained approaches, the model was trained from scratch to better understand 
the full machine learning pipeline.

# Author
Developed as part of a learning project in machine learning and computer vision.
