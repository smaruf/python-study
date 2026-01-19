"""
Level 0 - Beginner: Data Preprocessing

This module covers essential data preprocessing techniques used in deep learning.
Proper data preprocessing is crucial for model performance and training stability.

Topics covered:
- Data normalization and standardization
- Train/validation/test splits
- Handling missing values
- Feature scaling
- Data augmentation basics

Author: Python Study Repository
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris, make_classification


def demonstrate_normalization():
    """Demonstrate data normalization techniques."""
    print("=" * 60)
    print("1. NORMALIZATION AND STANDARDIZATION")
    print("=" * 60)
    
    # Create sample data
    data = np.array([[1, 2000, 0.5],
                     [2, 3000, 0.7],
                     [3, 1500, 0.3],
                     [4, 2500, 0.9],
                     [5, 4000, 0.6]])
    
    print("Original data:")
    print(data)
    print(f"\nFeature ranges:")
    print(f"  Column 0: {data[:, 0].min():.2f} to {data[:, 0].max():.2f}")
    print(f"  Column 1: {data[:, 1].min():.2f} to {data[:, 1].max():.2f}")
    print(f"  Column 2: {data[:, 2].min():.2f} to {data[:, 2].max():.2f}\n")
    
    # Min-Max Normalization (scales to [0, 1])
    scaler_minmax = MinMaxScaler()
    normalized_data = scaler_minmax.fit_transform(data)
    
    print("Min-Max Normalized (0 to 1):")
    print(normalized_data)
    print(f"\nNew ranges:")
    print(f"  Column 0: {normalized_data[:, 0].min():.2f} to {normalized_data[:, 0].max():.2f}")
    print(f"  Column 1: {normalized_data[:, 1].min():.2f} to {normalized_data[:, 1].max():.2f}")
    print(f"  Column 2: {normalized_data[:, 2].min():.2f} to {normalized_data[:, 2].max():.2f}\n")
    
    # Standardization (zero mean, unit variance)
    scaler_standard = StandardScaler()
    standardized_data = scaler_standard.fit_transform(data)
    
    print("Standardized (mean=0, std=1):")
    print(standardized_data)
    print(f"\nMeans: {standardized_data.mean(axis=0)}")
    print(f"Stds: {standardized_data.std(axis=0)}\n")
    
    # Manual normalization
    manual_norm = (data - data.mean(axis=0)) / data.std(axis=0)
    print("Manual standardization (matches sklearn):")
    print(manual_norm)
    print()


def demonstrate_train_test_split():
    """Demonstrate splitting data into train/validation/test sets."""
    print("=" * 60)
    print("2. TRAIN/VALIDATION/TEST SPLITS")
    print("=" * 60)
    
    # Create sample dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2,
                               random_state=42)
    
    print(f"Total samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}\n")
    
    # Split into train and temp (validation + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print("Split sizes:")
    print(f"  Training: {len(X_train)} samples (70%)")
    print(f"  Validation: {len(X_val)} samples (15%)")
    print(f"  Test: {len(X_test)} samples (15%)\n")
    
    print("Class distribution:")
    print(f"  Train: {np.bincount(y_train)}")
    print(f"  Val: {np.bincount(y_val)}")
    print(f"  Test: {np.bincount(y_test)}\n")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    print(f"PyTorch tensor shapes:")
    print(f"  X_train: {X_train_tensor.shape}")
    print(f"  y_train: {y_train_tensor.shape}\n")


def demonstrate_missing_values():
    """Demonstrate handling missing values."""
    print("=" * 60)
    print("3. HANDLING MISSING VALUES")
    print("=" * 60)
    
    # Create data with missing values
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, np.nan, 6.0],
                     [7.0, 8.0, np.nan],
                     [np.nan, 10.0, 11.0],
                     [12.0, 13.0, 14.0]])
    
    print("Data with missing values (NaN):")
    print(data)
    print(f"\nMissing values per column: {np.isnan(data).sum(axis=0)}\n")
    
    # Strategy 1: Remove rows with missing values
    clean_data = data[~np.isnan(data).any(axis=1)]
    print("After removing rows with NaN:")
    print(clean_data)
    print(f"Remaining samples: {len(clean_data)}\n")
    
    # Strategy 2: Fill with mean
    data_mean_filled = data.copy()
    for col in range(data.shape[1]):
        col_mean = np.nanmean(data[:, col])
        data_mean_filled[np.isnan(data_mean_filled[:, col]), col] = col_mean
    
    print("After filling with column mean:")
    print(data_mean_filled)
    print()
    
    # Strategy 3: Fill with median (more robust to outliers)
    data_median_filled = data.copy()
    for col in range(data.shape[1]):
        col_median = np.nanmedian(data[:, col])
        data_median_filled[np.isnan(data_median_filled[:, col]), col] = col_median
    
    print("After filling with column median:")
    print(data_median_filled)
    print()


def demonstrate_feature_scaling():
    """Demonstrate feature scaling importance."""
    print("=" * 60)
    print("4. FEATURE SCALING")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    
    print("Original Iris dataset:")
    print(f"  Shape: {X.shape}")
    print(f"  Feature names: {iris.feature_names}\n")
    
    print("Original feature statistics:")
    for i, name in enumerate(iris.feature_names):
        print(f"  {name}:")
        print(f"    Min: {X[:, i].min():.2f}, Max: {X[:, i].max():.2f}")
        print(f"    Mean: {X[:, i].mean():.2f}, Std: {X[:, i].std():.2f}")
    print()
    
    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("After standard scaling:")
    for i, name in enumerate(iris.feature_names):
        print(f"  {name}:")
        print(f"    Min: {X_scaled[:, i].min():.2f}, Max: {X_scaled[:, i].max():.2f}")
        print(f"    Mean: {X_scaled[:, i].mean():.2f}, Std: {X_scaled[:, i].std():.2f}")
    print()
    
    # Why scaling matters
    print("Why feature scaling is important:")
    print("  1. Features with larger ranges can dominate the learning")
    print("  2. Gradient descent converges faster with scaled features")
    print("  3. Some algorithms (like neural networks) are sensitive to scale")
    print("  4. Helps prevent numerical instability\n")


def demonstrate_batch_creation():
    """Demonstrate creating batches for training."""
    print("=" * 60)
    print("5. BATCH CREATION")
    print("=" * 60)
    
    # Create sample data
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # binary labels
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Using PyTorch DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Create data loader
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Number of batches: {len(dataloader)}\n")
    
    # Iterate through batches
    print("First 3 batches:")
    for i, (batch_X, batch_y) in enumerate(dataloader):
        if i >= 3:
            break
        print(f"  Batch {i+1}: X shape {batch_X.shape}, y shape {batch_y.shape}")
    print()


def demonstrate_data_augmentation():
    """Demonstrate basic data augmentation concepts."""
    print("=" * 60)
    print("6. DATA AUGMENTATION BASICS")
    print("=" * 60)
    
    # Simulate an image (3x3 for simplicity)
    image = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float32)
    
    print("Original 'image':")
    print(image)
    print()
    
    # Horizontal flip
    flipped = np.fliplr(image)
    print("Horizontally flipped:")
    print(flipped)
    print()
    
    # Add noise
    noise = np.random.randn(*image.shape) * 0.5
    noisy = image + noise
    print("With added noise:")
    print(noisy)
    print()
    
    # Normalize
    normalized = (image - image.mean()) / image.std()
    print("Normalized:")
    print(normalized)
    print()
    
    print("Common data augmentation techniques:")
    print("  Images: rotation, flipping, cropping, color jitter, noise")
    print("  Text: synonym replacement, back-translation, random deletion")
    print("  Time series: time warping, magnitude warping, permutation")
    print("  Audio: pitch shifting, time stretching, adding background noise\n")


def demonstrate_one_hot_encoding():
    """Demonstrate one-hot encoding for categorical variables."""
    print("=" * 60)
    print("7. ONE-HOT ENCODING")
    print("=" * 60)
    
    # Categorical labels
    labels = np.array([0, 1, 2, 0, 1, 2, 1])
    print(f"Original labels: {labels}")
    print(f"Number of classes: {len(np.unique(labels))}\n")
    
    # One-hot encoding using NumPy
    num_classes = len(np.unique(labels))
    one_hot = np.eye(num_classes)[labels]
    
    print("One-hot encoded:")
    print(one_hot)
    print()
    
    # Using PyTorch
    labels_tensor = torch.LongTensor(labels)
    one_hot_tensor = torch.nn.functional.one_hot(labels_tensor, num_classes=num_classes)
    
    print("PyTorch one-hot encoded:")
    print(one_hot_tensor)
    print()
    
    print("When to use one-hot encoding:")
    print("  - Multi-class classification output layer")
    print("  - Categorical features without ordinal relationship")
    print("  - Cross-entropy loss requires one-hot or class indices\n")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING FOR DEEP LEARNING")
    print("=" * 60 + "\n")
    
    demonstrate_normalization()
    demonstrate_train_test_split()
    demonstrate_missing_values()
    demonstrate_feature_scaling()
    demonstrate_batch_creation()
    demonstrate_data_augmentation()
    demonstrate_one_hot_encoding()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Proper data preprocessing is essential for successful deep learning.
Key concepts covered:
1. Normalization and standardization (feature scaling)
2. Train/validation/test splits (proper evaluation)
3. Handling missing values (data quality)
4. Importance of feature scaling (training stability)
5. Batch creation (efficient training)
6. Data augmentation (increasing dataset size)
7. One-hot encoding (categorical variables)

Best practices:
- Always normalize/standardize your features
- Use 70-15-15 or 80-10-10 train-val-test splits
- Keep test set completely separate until final evaluation
- Use stratification for imbalanced datasets
- Apply same preprocessing to train/val/test sets
- Fit scalers only on training data

Next steps:
- Practice preprocessing real datasets
- Understand when to use different scaling methods
- Move to visualization in Level 0.4
    """)


if __name__ == "__main__":
    main()
