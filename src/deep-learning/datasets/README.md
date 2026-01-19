# Deep Learning Datasets

This directory contains information about datasets used in the deep learning learning path.

## Common Datasets

### Level 0-1: Beginner/Basic
- **MNIST**: Handwritten digits (60k train, 10k test)
- **Fashion-MNIST**: Clothing items (60k train, 10k test)
- **Iris**: Classic ML dataset (150 samples, 3 classes)

### Level 2: Intermediate
- **CIFAR-10**: 32x32 color images, 10 classes (50k train, 10k test)
- **CIFAR-100**: 32x32 color images, 100 classes
- **STL-10**: 96x96 color images, 10 classes

### Level 3: Advanced
- **ImageNet**: Large-scale image classification (1.2M images, 1000 classes)
- **COCO**: Object detection and segmentation
- **IMDB**: Movie reviews for sentiment analysis
- **WikiText**: Text corpus for language modeling

### Level 4: Expert
- **CelebA**: Face attributes dataset (200k images)
- **LSUN**: Large-scale scene understanding
- **Common Voice**: Speech recognition dataset
- **SQuAD**: Question answering dataset

## Dataset Download

All datasets are automatically downloaded when running the examples using PyTorch's built-in dataset loaders:

```python
from torchvision import datasets

# MNIST
mnist = datasets.MNIST(root='./data', train=True, download=True)

# CIFAR-10
cifar10 = datasets.CIFAR10(root='./data', train=True, download=True)

# ImageNet (requires manual download)
# Visit: https://image-net.org/download.php
```

## Storage Requirements

Approximate storage needed for datasets:

- Level 0-1: ~100 MB (MNIST, Fashion-MNIST)
- Level 2: ~500 MB (CIFAR-10, CIFAR-100)
- Level 3: ~150 GB (ImageNet, COCO)
- Level 4: ~50 GB (CelebA, various NLP datasets)

## Custom Datasets

For custom datasets, follow PyTorch's Dataset API:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Load your data here
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return (image, label) or (input, target)
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
```

## Data Augmentation

Common augmentation techniques:

### Images
- Random crop
- Random horizontal flip
- Color jitter
- Random rotation
- Normalization

### Text
- Random word replacement
- Back-translation
- Random deletion
- Synonym replacement

### Audio
- Pitch shifting
- Time stretching
- Adding noise
- Volume adjustment

## References

- [PyTorch Datasets](https://pytorch.org/vision/stable/datasets.html)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)
