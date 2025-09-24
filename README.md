# Emotion Detector 
## Introduction

In this project, I designed and trained a convolutional neural network (CNN) to classify facial expressions into distinct emotion categories. I implemented the training pipeline in Python using Keras (MobileNet backbone) and deployed the trained model into a real-time C++/OpenCV pipeline for inference. The system performs face detection, preprocessing, and emotion classification on live video streams with low latency.

## Training Pipeline

- **Data Augmentation**: Leveraged `ImageDataGenerator` with zoom, shear, rescaling, and horizontal flips to improve generalization.  
- **Backbone Architecture**: Fine-tuned a pre-trained **MobileNet** model, replacing the classifier head with a custom dense layer stack.  
- **Optimization**: Used Adam optimizer, categorical cross-entropy loss, and early stopping to converge efficiently.  
- **Evaluation**: Validated on a held-out set with accuracy tracking and plotted training/validation curves.  
- **Export**: Converted the trained model to ONNX/TensorFlow format for cross-platform deployment.  

## Deployment Pipeline (C++ / OpenCV)

- **Face Detection**: Utilized Haar cascades / DNN-based detectors to localize faces in each video frame.  
- **Preprocessing**: Resized ROIs to 224×224, normalized pixel intensities, and converted to tensors.  
- **Model Inference**: Integrated the ONNX model with OpenCV’s DNN module for fast forward passes.  
- **Post-processing**: Mapped output logits to human-readable emotion labels and overlaid results in real time.  
