# Level 2 - Intermediate: Deep Learning Fundamentals

Welcome to Level 2! This level covers deep neural networks and convolutional neural networks.

## Overview

This level focuses on:
- Deep neural networks with multiple layers
- Convolutional Neural Networks (CNNs)
- Activation functions and their properties
- Regularization techniques
- Batch normalization
- Image classification

## Prerequisites

- Completion of Level 0 and Level 1
- Understanding of neural network basics
- Familiarity with gradient descent

## Modules

### 01_deep_neural_network.py
Building deeper networks with more layers.

**Topics:**
- Multi-layer architectures
- Deep vs shallow networks
- Vanishing/exploding gradients
- Weight initialization strategies

### 02_activation_functions.py
Understanding different activation functions.

**Topics:**
- ReLU, Leaky ReLU, ELU
- Sigmoid and Tanh
- Softmax for classification
- Comparing activation functions

### 03_cnn_basics.py
Introduction to Convolutional Neural Networks.

**Topics:**
- Convolutional layers
- Pooling layers
- Feature maps
- Receptive fields

### 04_image_classification.py
Building a CNN for CIFAR-10 classification.

**Topics:**
- Loading CIFAR-10 dataset
- CNN architecture design
- Data augmentation
- Training and evaluation

### 05_regularization.py
Preventing overfitting with regularization.

**Topics:**
- L1 and L2 regularization
- Dropout
- Early stopping
- Data augmentation

### 06_batch_normalization.py
Accelerating training with batch normalization.

**Topics:**
- Batch normalization theory
- Implementation in PyTorch
- Comparing with/without BatchNorm
- Layer normalization

## Learning Objectives

By the end of Level 2, you should be able to:
- Build and train deep neural networks
- Understand and use different activation functions
- Implement CNNs for image classification
- Apply regularization techniques
- Use batch normalization effectively
- Achieve >80% accuracy on CIFAR-10

## Key Concepts

### Convolutional Neural Networks
- **Convolution**: Applying filters to extract features
- **Pooling**: Downsampling feature maps
- **Feature Hierarchy**: Low-level to high-level features

### Regularization
- **Dropout**: Randomly dropping neurons during training
- **L2 Regularization**: Weight decay
- **Data Augmentation**: Artificially expanding dataset

### Normalization
- **Batch Normalization**: Normalizing layer inputs
- **Benefits**: Faster training, higher learning rates

## Time Estimate

4-6 weeks, spending 2-3 hours per day

## Next Steps

Proceed to [Level 3 - Advanced](../level-3-advanced/README.md) for advanced architectures like ResNet, RNNs, and LSTMs!

## Additional Resources

- [CS231n - Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
- [Understanding CNNs](https://poloclub.github.io/cnn-explainer/)
