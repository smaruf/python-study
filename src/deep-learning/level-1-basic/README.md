# Level 1 - Basic: Neural Network Fundamentals

Welcome to Level 1! This level introduces you to the building blocks of neural networks.

## Overview

This level focuses on:
- Perceptron algorithm
- Linear and logistic regression
- Simple feed-forward neural networks
- Loss functions and optimization
- MNIST digit classification

## Prerequisites

- Completion of Level 0
- Understanding of basic calculus (derivatives)
- Familiarity with linear algebra

## Modules

### 01_perceptron.py
The simplest form of a neural network - a single neuron.

**Topics:**
- Perceptron algorithm
- Binary classification
- Weight updates
- Decision boundaries

**Run:**
```bash
python 01_perceptron.py
```

### 02_linear_regression.py
Building linear regression from scratch using gradient descent.

**Topics:**
- Linear regression theory
- Mean squared error (MSE) loss
- Gradient descent optimization
- PyTorch implementation

**Run:**
```bash
python 02_linear_regression.py
```

### 03_logistic_regression.py
Classification with logistic regression.

**Topics:**
- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent for classification
- Decision boundaries

**Run:**
```bash
python 03_logistic_regression.py
```

### 04_simple_neural_network.py
Your first multi-layer neural network.

**Topics:**
- Multi-layer perceptron (MLP)
- Forward propagation
- Backpropagation
- Hidden layers and activation functions

**Run:**
```bash
python 04_simple_neural_network.py
```

### 05_mnist_classifier.py
Classifying handwritten digits with a neural network.

**Topics:**
- Loading MNIST dataset
- Building a digit classifier
- Training loop
- Evaluation and visualization

**Run:**
```bash
python 05_mnist_classifier.py
```

## Learning Objectives

By the end of Level 1, you should be able to:
- Understand how a perceptron works
- Implement linear and logistic regression from scratch
- Build and train simple neural networks
- Understand forward and backward propagation
- Train a model to classify MNIST digits
- Monitor training progress and evaluate models

## Key Concepts

### Loss Functions
- **MSE (Mean Squared Error)**: For regression tasks
- **Cross-Entropy**: For classification tasks

### Optimizers
- **Gradient Descent**: Basic optimization algorithm
- **Stochastic Gradient Descent (SGD)**: Using mini-batches
- **Learning Rate**: Controls step size in optimization

### Activation Functions
- **Sigmoid**: Squashes values to (0, 1)
- **ReLU**: Most common in modern networks
- **Softmax**: For multi-class classification output

## Time Estimate

2-3 weeks, spending 1-2 hours per day

## Next Steps

Once you're comfortable with basic neural networks, proceed to [Level 2 - Intermediate](../level-2-intermediate/README.md) where you'll learn about deep neural networks and CNNs!

## Additional Resources

- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Gradient Descent Video](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
