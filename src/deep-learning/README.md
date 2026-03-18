[← Back to Python Study Repository](../../README.md)

# Deep Learning in Python - Zero to Expert

A comprehensive learning path for mastering deep learning from fundamentals to advanced production-ready systems. This project provides hands-on examples, detailed explanations, and real-world applications using PyTorch, TensorFlow, and other popular frameworks.

## Table of Contents

- [Introduction](#introduction)
- [Learning Path](#learning-path)
- [Technologies Covered](#technologies-covered)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
- [Learning Levels](#learning-levels)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Contributing](#contributing)

## Introduction

This project offers a structured learning path from zero to expert level in deep learning. Whether you're a beginner or looking to advance your skills, this repository provides comprehensive examples covering neural networks, computer vision, natural language processing, and deployment strategies.

## Learning Path

The learning path is divided into 6 progressive levels:

1. **[Level 0 - Beginner](level-0-beginner/README.md)**: Python basics for ML, NumPy, tensors, and basic math
2. **[Level 1 - Basic](level-1-basic/README.md)**: Perceptron, linear/logistic regression, simple neural networks
3. **[Level 2 - Intermediate](level-2-intermediate/README.md)**: Deep neural networks, CNNs, activation functions, backpropagation
4. **[Level 3 - Advanced](level-3-advanced/README.md)**: Advanced CNNs, RNNs, LSTMs, GRUs, transfer learning
5. **[Level 4 - Expert](level-4-expert/README.md)**: GANs, Transformers, attention mechanisms, advanced architectures
6. **[Level 5 - Master](level-5-master/README.md)**: Production deployment, optimization, distributed training, MLOps

## Technologies Covered

### Core Frameworks
- **PyTorch**: Primary framework for most examples
- **TensorFlow/Keras**: Alternative implementations
- **NumPy**: Fundamental numerical computing
- **scikit-learn**: Classical ML and preprocessing

### Specialized Libraries
- **torchvision**: Computer vision datasets and models
- **transformers**: Hugging Face transformers library
- **opencv-python**: Image processing
- **matplotlib/seaborn**: Visualization
- **tensorboard**: Training visualization
- **ONNX**: Model export and interoperability

### Focus Areas
- Neural network fundamentals
- Computer vision (CNNs, object detection, segmentation)
- Natural language processing (RNNs, LSTMs, Transformers)
- Generative models (GANs, VAEs, Diffusion models)
- Reinforcement learning basics
- Model optimization and deployment
- Production best practices

## Directory Structure

```
deep-learning/
├── README.md
├── requirements.txt
├── datasets/                    # Sample datasets and data loaders
├── utils/                       # Shared utilities and helpers
├── level-0-beginner/           # Introduction to ML/DL fundamentals
│   ├── 01_numpy_basics.py
│   ├── 02_tensor_operations.py
│   ├── 03_data_preprocessing.py
│   ├── 04_visualization.py
│   └── README.md
├── level-1-basic/              # Basic neural networks
│   ├── 01_perceptron.py
│   ├── 02_linear_regression.py
│   ├── 03_logistic_regression.py
│   ├── 04_simple_neural_network.py
│   ├── 05_mnist_classifier.py
│   └── README.md
├── level-2-intermediate/       # Deep learning fundamentals
│   ├── 01_deep_neural_network.py
│   ├── 02_activation_functions.py
│   ├── 03_cnn_basics.py
│   ├── 04_image_classification.py
│   ├── 05_regularization.py
│   ├── 06_batch_normalization.py
│   └── README.md
├── level-3-advanced/           # Advanced architectures
│   ├── 01_resnet_implementation.py
│   ├── 02_rnn_basics.py
│   ├── 03_lstm_text_generation.py
│   ├── 04_transfer_learning.py
│   ├── 05_object_detection.py
│   ├── 06_semantic_segmentation.py
│   └── README.md
├── level-4-expert/             # Cutting-edge models
│   ├── 01_gan_implementation.py
│   ├── 02_vae_implementation.py
│   ├── 03_transformer_basics.py
│   ├── 04_bert_fine_tuning.py
│   ├── 05_attention_mechanisms.py
│   ├── 06_diffusion_models.py
│   └── README.md
└── level-5-master/             # Production and deployment
    ├── 01_model_optimization.py
    ├── 02_quantization.py
    ├── 03_onnx_export.py
    ├── 04_distributed_training.py
    ├── 05_model_serving.py
    ├── 06_mlops_pipeline.py
    └── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with linear algebra and calculus (recommended)
- GPU recommended for advanced levels (but not required)

### Installation

1. Navigate to the deep learning directory:
   ```bash
   cd src/deep-learning
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
   ```

## Learning Levels

### [Level 0 - Beginner](level-0-beginner/README.md)
**Focus**: Fundamentals of numerical computing and data handling

Topics covered:
- NumPy arrays and operations
- Tensor manipulation
- Data loading and preprocessing
- Basic visualization with matplotlib
- Mathematical foundations

**Time estimate**: 1-2 weeks

### [Level 1 - Basic](level-1-basic/README.md)
**Focus**: Introduction to neural networks

Topics covered:
- Perceptron algorithm
- Linear regression from scratch
- Logistic regression
- Simple feed-forward neural networks
- MNIST digit classification
- Loss functions and optimizers

**Time estimate**: 2-3 weeks

### [Level 2 - Intermediate](level-2-intermediate/README.md)
**Focus**: Deep learning fundamentals

Topics covered:
- Deep neural networks
- Activation functions (ReLU, Sigmoid, Tanh)
- Convolutional Neural Networks (CNNs)
- Image classification (CIFAR-10)
- Regularization techniques (Dropout, L1/L2)
- Batch normalization
- Learning rate scheduling

**Time estimate**: 4-6 weeks

### [Level 3 - Advanced](level-3-advanced/README.md)
**Focus**: Advanced architectures and techniques

Topics covered:
- ResNet, VGG, Inception architectures
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRUs)
- Transfer learning
- Object detection (YOLO, R-CNN)
- Semantic segmentation
- Attention mechanisms

**Time estimate**: 6-8 weeks

### [Level 4 - Expert](level-4-expert/README.md)
**Focus**: State-of-the-art models and research topics

Topics covered:
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)
- Transformer architecture
- BERT, GPT models
- Fine-tuning pre-trained models
- Advanced attention mechanisms
- Diffusion models
- Neural architecture search

**Time estimate**: 8-12 weeks

### [Level 5 - Master](level-5-master/README.md)
**Focus**: Production deployment and MLOps

Topics covered:
- Model optimization techniques
- Quantization and pruning
- ONNX model export
- Distributed training (DDP, Horovod)
- Model serving (TorchServe, TensorFlow Serving)
- MLOps pipelines
- Monitoring and logging
- A/B testing
- Cloud deployment (AWS, GCP, Azure)

**Time estimate**: 8-12 weeks

## Best Practices

1. **Start from Level 0**: Even if you have programming experience, start from the beginning to build a solid foundation
2. **Practice regularly**: Code along with examples and try variations
3. **Use GPU when possible**: For levels 3-5, GPU acceleration significantly speeds up training
4. **Experiment**: Modify hyperparameters and architectures to understand their impact
5. **Read papers**: Each level includes references to seminal papers
6. **Join communities**: Participate in forums like PyTorch Forums, TensorFlow Hub, and r/MachineLearning

## Hardware Recommendations

### Minimum (Levels 0-2)
- CPU: Any modern processor
- RAM: 8GB
- GPU: Optional

### Recommended (Levels 3-5)
- CPU: Multi-core processor (4+ cores)
- RAM: 16GB+
- GPU: NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- Storage: SSD with 50GB+ free space

### Cloud Alternatives
- Google Colab (Free tier with GPU)
- Kaggle Kernels (Free tier with GPU)
- AWS SageMaker
- Google Cloud AI Platform
- Azure Machine Learning

## Resources

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with Python" by François Chollet
- "Neural Networks and Deep Learning" by Michael Nielsen (free online)

### Online Courses
- Fast.ai Practical Deep Learning for Coders
- Andrew Ng's Deep Learning Specialization (Coursera)
- Stanford CS231n (Computer Vision)
- Stanford CS224n (NLP)

### Papers
- AlexNet: ImageNet Classification with Deep CNNs (2012)
- ResNet: Deep Residual Learning for Image Recognition (2015)
- LSTM: Long Short-term Memory (1997)
- GAN: Generative Adversarial Networks (2014)
- Attention is All You Need (Transformers) (2017)
- BERT: Pre-training of Deep Bidirectional Transformers (2018)

### Websites
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Papers with Code](https://paperswithcode.com/)
- [Distill.pub](https://distill.pub/) - Visual explanations
- [Towards Data Science](https://towardsdatascience.com/)

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Add well-documented code examples
4. Include README updates if adding new topics
5. Test your code
6. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

Special thanks to the open-source community and the creators of PyTorch, TensorFlow, and other amazing libraries that make deep learning accessible to everyone.

---

**Happy Learning! 🚀🧠**

For questions or discussions, please open an issue or reach out to the maintainers.
