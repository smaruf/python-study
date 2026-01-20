# Quick Start Guide - Deep Learning in Python

Get started with deep learning in just a few minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- 10GB free disk space

## Installation

### 1. Navigate to the deep learning directory

```bash
cd src/deep-learning
```

### 2. Create a virtual environment (recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
# Install core dependencies
pip install numpy matplotlib seaborn scikit-learn scipy

# Install PyTorch (CPU version)
pip install torch torchvision

# For GPU support (NVIDIA CUDA required):
# Visit: https://pytorch.org/get-started/locally/
# and follow instructions for your system
```

## Your First Deep Learning Program

### Level 0: NumPy Basics

```bash
cd level-0-beginner
python 01_numpy_basics.py
```

This will show you:
- Array creation and manipulation
- Broadcasting
- Mathematical operations
- Linear algebra basics

### Level 0: Tensor Operations

```bash
python 02_tensor_operations.py
```

This introduces PyTorch tensors and automatic differentiation.

### Level 1: Your First Neural Network

```bash
cd ../level-1-basic
python 01_perceptron.py
```

Trains a perceptron on linearly separable data.

### Level 1: MNIST Digit Classification

```bash
# This will download MNIST dataset (~10MB)
python 05_mnist_classifier.py
```

Trains a neural network to recognize handwritten digits!

## Learning Path

### Recommended Order

1. **Week 1-2**: Complete Level 0 (Beginner)
   - Focus on NumPy and PyTorch basics
   - Understand tensors and autograd
   - Learn data preprocessing

2. **Week 3-5**: Complete Level 1 (Basic)
   - Build simple neural networks
   - Understand backpropagation
   - Train MNIST classifier

3. **Week 6-11**: Complete Level 2 (Intermediate)
   - Deep neural networks
   - Convolutional Neural Networks
   - Image classification with CIFAR-10

4. **Week 12-19**: Complete Level 3 (Advanced)
   - ResNet and advanced architectures
   - RNNs and LSTMs
   - Transfer learning

5. **Week 20-31**: Complete Level 4 (Expert)
   - GANs and VAEs
   - Transformers and BERT
   - Attention mechanisms

6. **Week 32-43**: Complete Level 5 (Master)
   - Model optimization
   - Production deployment
   - MLOps practices

## Quick Examples

### Creating a Simple Neural Network

```python
import torch
import torch.nn as nn

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleNet()
print(model)
```

### Training Loop

```python
import torch.optim as optim

# Setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

## Common Issues

### Issue: ModuleNotFoundError

**Solution**: Install the missing package
```bash
pip install <package-name>
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size or use CPU
```python
device = torch.device("cpu")  # Force CPU usage
```

### Issue: Training is slow

**Solutions**:
1. Use GPU if available
2. Reduce model size
3. Use smaller batch size
4. Use fewer epochs

## Getting Help

1. **Documentation**: Read the README files in each level
2. **Code Comments**: All code is well-commented
3. **Online Resources**: See resources section in main README
4. **Community**: Join PyTorch Forums, r/MachineLearning

## Next Steps

After completing the quick start:

1. Read the [main README](./README.md) for detailed information
2. Start with Level 0 and progress sequentially
3. Practice by modifying the examples
4. Build your own projects using the learned concepts

## Tips for Success

1. **Code along**: Don't just read, type the code yourself
2. **Experiment**: Modify parameters and see what happens
3. **Take notes**: Document your learnings
4. **Be patient**: Deep learning takes time to master
5. **Stay curious**: Read papers and explore new ideas

Happy Learning! 🚀
