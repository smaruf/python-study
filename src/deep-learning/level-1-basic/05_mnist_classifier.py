"""
Level 1 - Basic: MNIST Digit Classifier

This module implements a simple neural network to classify handwritten digits
from the MNIST dataset. This is a classic "Hello World" problem in deep learning.

Topics covered:
- Loading and preprocessing MNIST dataset
- Building a multi-layer perceptron (MLP)
- Training loop with batches
- Evaluation and visualization
- Model saving and loading

Author: Python Study Repository
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


class SimpleMNISTNet(nn.Module):
    """
    Simple neural network for MNIST classification.
    
    Architecture:
    - Input: 784 (28x28 flattened image)
    - Hidden layer 1: 128 neurons with ReLU
    - Hidden layer 2: 64 neurons with ReLU
    - Output: 10 neurons (one per digit class)
    """
    
    def __init__(self):
        super(SimpleMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


def load_data(batch_size=64):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        batch_size: Number of samples per batch
        
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize (MNIST mean and std)
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to train on (CPU or GPU)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Neural network model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        Average loss and accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def visualize_predictions(model, test_dataset, device, num_samples=10):
    """
    Visualize model predictions on random test samples.
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, label = test_dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_input)
            _, predicted = torch.max(output, 1)
            predicted = predicted.item()
            
            # Plot image
            img = image.squeeze().cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'True: {label}, Pred: {predicted}',
                            color='green' if label == predicted else 'red')
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/tmp/mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("✓ Saved predictions: /tmp/mnist_predictions.png")
    plt.close()


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """
    Plot training history.
    
    Args:
        train_losses: Training losses per epoch
        train_accs: Training accuracies per epoch
        test_losses: Test losses per epoch
        test_accs: Test accuracies per epoch
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-s', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-s', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/mnist_training_history.png', dpi=150, bbox_inches='tight')
    print("✓ Saved training history: /tmp/mnist_training_history.png")
    plt.close()


def main():
    """Main function to train and evaluate the model."""
    print("\n" + "=" * 60)
    print("MNIST DIGIT CLASSIFICATION")
    print("=" * 60 + "\n")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5
    
    print("Hyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Number of epochs: {num_epochs}\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader, train_dataset, test_dataset = load_data(batch_size)
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = SimpleMNISTNet().to(device)
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...\n")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n")
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {test_accs[-1]:.2f}%\n")
    
    # Visualize predictions
    print("Visualizing predictions...")
    visualize_predictions(model, test_dataset, device)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Save model
    model_path = '/tmp/mnist_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Successfully trained a neural network to classify MNIST digits!

Key achievements:
1. Loaded and preprocessed MNIST dataset
2. Built a 3-layer neural network (784 → 128 → 64 → 10)
3. Trained using Adam optimizer and cross-entropy loss
4. Achieved >95% accuracy on test set
5. Visualized predictions and training history

Network architecture:
- Input layer: 784 neurons (28×28 image flattened)
- Hidden layer 1: 128 neurons with ReLU activation
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons (one per digit class)

Training details:
- Loss function: Cross-Entropy Loss
- Optimizer: Adam (adaptive learning rate)
- Batch size: 64
- Number of epochs: 5

Next steps:
- Try different architectures (more layers, different sizes)
- Experiment with learning rates
- Add dropout for regularization
- Try different optimizers (SGD, RMSprop)
- Move to Level 2 to learn about CNNs for better performance!
    """)


if __name__ == "__main__":
    main()
