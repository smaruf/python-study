# Level 3 - Advanced: Advanced Architectures

Welcome to Level 3! This level covers advanced deep learning architectures and techniques.

## Overview

This level focuses on:
- Advanced CNN architectures (ResNet, VGG, Inception)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM) networks
- Transfer learning
- Object detection
- Semantic segmentation

## Prerequisites

- Completion of Levels 0-2
- Strong understanding of CNNs
- Experience with PyTorch

## Modules

### 01_resnet_implementation.py
Implementing ResNet with residual connections.

**Topics:**
- Residual blocks
- Skip connections
- Deep networks (50+ layers)
- Identity vs projection shortcuts

### 02_rnn_basics.py
Introduction to Recurrent Neural Networks.

**Topics:**
- Sequence modeling
- Hidden states
- Backpropagation through time
- Text generation basics

### 03_lstm_text_generation.py
Using LSTMs for sequence generation.

**Topics:**
- LSTM architecture
- Forget, input, output gates
- Text generation
- Character-level models

### 04_transfer_learning.py
Leveraging pre-trained models.

**Topics:**
- ImageNet pre-trained models
- Fine-tuning strategies
- Feature extraction
- Domain adaptation

### 05_object_detection.py
Detecting objects in images.

**Topics:**
- Bounding box prediction
- YOLO architecture basics
- Non-maximum suppression
- mAP metric

### 06_semantic_segmentation.py
Pixel-wise image classification.

**Topics:**
- U-Net architecture
- Encoder-decoder networks
- Skip connections
- Dice loss

## Learning Objectives

By the end of Level 3, you should be able to:
- Implement residual networks from scratch
- Build and train RNN/LSTM models
- Use transfer learning effectively
- Understand object detection pipelines
- Perform semantic segmentation
- Work with sequence data

## Key Concepts

### Residual Learning
- **Skip Connections**: Allowing gradients to flow directly
- **Identity Mapping**: Learning residuals instead of direct mapping
- **Very Deep Networks**: Training 100+ layer networks

### Recurrent Networks
- **Temporal Dependencies**: Modeling sequences
- **Hidden State**: Memory of previous inputs
- **Vanishing Gradients**: LSTM solution

### Transfer Learning
- **Pre-training**: Learning on large datasets
- **Fine-tuning**: Adapting to specific tasks
- **Feature Extraction**: Using as feature extractor

## Time Estimate

6-8 weeks, spending 3-4 hours per day

## Next Steps

Proceed to [Level 4 - Expert](../level-4-expert/README.md) for GANs, Transformers, and cutting-edge architectures!

## Additional Resources

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [CS224n - NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Fast.ai - Transfer Learning](https://docs.fast.ai/tutorial.vision.html)
