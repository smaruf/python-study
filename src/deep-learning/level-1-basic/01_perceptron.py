"""
Level 1 - Basic: Perceptron

This module implements the perceptron algorithm - the simplest form of a neural network.
A perceptron is a single neuron that can learn to classify linearly separable data.

Topics covered:
- Perceptron algorithm
- Binary classification
- Weight updates
- Decision boundaries

Author: Python Study Repository
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification


class Perceptron:
    """
    Simple Perceptron classifier.
    
    A perceptron learns a linear decision boundary to separate two classes.
    It updates weights based on misclassified examples.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the perceptron.
        
        Args:
            learning_rate: Step size for weight updates
            n_iterations: Maximum number of training iterations
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Train the perceptron on the given data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,) - must be 0 or 1
        """
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to -1 and 1 for easier math
        y_ = np.where(y <= 0, -1, 1)
        
        # Training loop
        for iteration in range(self.n_iterations):
            errors = 0
            
            for idx, x_i in enumerate(X):
                # Calculate linear output
                linear_output = np.dot(x_i, self.weights) + self.bias
                
                # Apply activation (step function)
                y_predicted = np.sign(linear_output)
                
                # Update weights if prediction is wrong
                if y_[idx] * y_predicted <= 0:
                    errors += 1
                    update = self.learning_rate * y_[idx]
                    self.weights += update * x_i
                    self.bias += update
            
            # Stop if no errors
            if errors == 0:
                print(f"Converged after {iteration + 1} iterations")
                break
        
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict (n_samples, n_features)
            
        Returns:
            Predicted labels (0 or 1)
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = np.sign(linear_output)
        return np.where(y_predicted <= 0, 0, 1)
    
    def score(self, X, y):
        """
        Calculate accuracy on the given data.
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def demonstrate_perceptron():
    """Demonstrate perceptron on a simple 2D dataset."""
    print("=" * 60)
    print("1. PERCEPTRON ON LINEARLY SEPARABLE DATA")
    print("=" * 60)
    
    # Generate linearly separable data
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, n_features=2, centers=2, 
                      cluster_std=1.5, random_state=42)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}\n")
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
    perceptron.fit(X, y)
    
    # Evaluate
    accuracy = perceptron.score(X, y)
    print(f"Training accuracy: {accuracy:.4f}")
    print(f"Weights: {perceptron.weights}")
    print(f"Bias: {perceptron.bias}\n")
    
    # Visualize
    plt.figure(figsize=(10, 5))
    
    # Plot data points
    plt.subplot(1, 2, 1)
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.7)
    
    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    w = perceptron.weights
    b = perceptron.bias
    
    # Decision boundary: w1*x1 + w2*x2 + b = 0
    # => x2 = -(w1*x1 + b) / w2
    x1_line = np.array([x_min, x_max])
    x2_line = -(w[0] * x1_line + b) / w[1]
    
    plt.plot(x1_line, x2_line, 'r--', linewidth=2, label='Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron: Linearly Separable Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot decision regions
    plt.subplot(1, 2, 2)
    
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on mesh
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contours
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Class 1', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron: Decision Regions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/perceptron_linear.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization: /tmp/perceptron_linear.png")
    plt.close()
    print()


def demonstrate_perceptron_limitations():
    """Demonstrate perceptron limitations on non-linearly separable data."""
    print("=" * 60)
    print("2. PERCEPTRON LIMITATIONS: XOR PROBLEM")
    print("=" * 60)
    
    # XOR problem - not linearly separable
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR output
    
    print("XOR dataset:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")
    print()
    
    # Try to train perceptron
    perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)
    perceptron.fit(X, y)
    
    # Evaluate
    predictions = perceptron.predict(X)
    accuracy = perceptron.score(X, y)
    
    print("Predictions:")
    for i in range(len(X)):
        print(f"  {X[i]} -> Predicted: {predictions[i]}, Actual: {y[i]}")
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n⚠️  A single perceptron cannot solve the XOR problem!")
    print("This requires a multi-layer network (covered in Level 1.4)\n")
    
    # Visualize
    plt.figure(figsize=(8, 6))
    
    colors = ['red' if label == 0 else 'blue' for label in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, (x, label) in enumerate(zip(X, y)):
        plt.annotate(f'({x[0]},{x[1]}) → {label}', 
                    xy=(x[0], x[1]), 
                    xytext=(10, 10), 
                    textcoords='offset points',
                    fontsize=10)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('XOR Problem: Not Linearly Separable')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/perceptron_xor.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization: /tmp/perceptron_xor.png")
    plt.close()
    print()


def demonstrate_perceptron_convergence():
    """Demonstrate perceptron convergence on different datasets."""
    print("=" * 60)
    print("3. PERCEPTRON CONVERGENCE")
    print("=" * 60)
    
    # Create datasets with different separability
    np.random.seed(42)
    
    datasets = [
        ("Easy", make_blobs(n_samples=100, n_features=2, centers=2, 
                           cluster_std=0.5, random_state=42)),
        ("Medium", make_blobs(n_samples=100, n_features=2, centers=2, 
                             cluster_std=1.5, random_state=42)),
        ("Hard", make_blobs(n_samples=100, n_features=2, centers=2, 
                           cluster_std=2.5, random_state=42))
    ]
    
    for name, (X, y) in datasets:
        perceptron = Perceptron(learning_rate=0.01, n_iterations=1000)
        
        print(f"\nDataset: {name}")
        perceptron.fit(X, y)
        accuracy = perceptron.score(X, y)
        print(f"Final accuracy: {accuracy:.4f}")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("PERCEPTRON - THE SIMPLEST NEURAL NETWORK")
    print("=" * 60 + "\n")
    
    demonstrate_perceptron()
    demonstrate_perceptron_limitations()
    demonstrate_perceptron_convergence()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The perceptron is the foundation of neural networks.
Key concepts covered:
1. Binary classification with a single neuron
2. Linear decision boundaries
3. Weight update rule: w = w + η * y * x
4. Perceptron convergence theorem
5. Limitations: cannot solve non-linearly separable problems

Perceptron algorithm:
1. Initialize weights to zero
2. For each training sample:
   - Compute prediction: ŷ = sign(w·x + b)
   - If wrong: update weights: w = w + η·y·x
3. Repeat until convergence or max iterations

Key insights:
- Works perfectly for linearly separable data
- Fails on XOR and other non-linear problems
- Foundation for more complex neural networks
- Historical importance: first learning algorithm for neural nets

Next steps:
- Understand the weight update rule
- Practice on different datasets
- Move to linear regression in Level 1.2
    """)


if __name__ == "__main__":
    main()
