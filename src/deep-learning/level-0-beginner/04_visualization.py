"""
Level 0 - Beginner: Data Visualization

This module covers essential data visualization techniques for understanding datasets
and monitoring model training in deep learning.

Topics covered:
- Basic plotting with matplotlib
- Statistical visualizations
- Visualizing distributions
- Plotting training metrics
- Confusion matrices and classification metrics

Author: Python Study Repository
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def demonstrate_basic_plotting():
    """Demonstrate basic matplotlib plotting."""
    print("=" * 60)
    print("1. BASIC PLOTTING")
    print("=" * 60)
    
    # Line plot
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y1, label='sin(x)', linewidth=2)
    plt.plot(x, y2, label='cos(x)', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trigonometric Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = 2 * x_scatter + np.random.randn(100)
    
    plt.subplot(1, 2, 2)
    plt.scatter(x_scatter, y_scatter, alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot: Y = 2X + noise')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/01_basic_plots.png', dpi=150, bbox_inches='tight')
    print("✓ Created basic plots: /tmp/01_basic_plots.png")
    plt.close()
    print()


def demonstrate_distributions():
    """Demonstrate visualizing data distributions."""
    print("=" * 60)
    print("2. VISUALIZING DISTRIBUTIONS")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    data_normal = np.random.randn(1000)
    data_uniform = np.random.uniform(-3, 3, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram
    axes[0, 0].hist(data_normal, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Histogram: Normal Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    
    # KDE plot
    axes[0, 1].hist(data_normal, bins=30, alpha=0.5, density=True, label='Histogram')
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data_normal)
    x_range = np.linspace(data_normal.min(), data_normal.max(), 100)
    axes[0, 1].plot(x_range, kde(x_range), linewidth=2, label='KDE')
    axes[0, 1].set_title('Histogram with KDE')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    
    # Box plot
    axes[1, 0].boxplot([data_normal, data_uniform], labels=['Normal', 'Uniform'])
    axes[1, 0].set_title('Box Plot Comparison')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Violin plot using seaborn
    import pandas as pd
    df = pd.DataFrame({
        'Normal': data_normal,
        'Uniform': data_uniform
    })
    df_melted = df.melt(var_name='Distribution', value_name='Value')
    sns.violinplot(data=df_melted, x='Distribution', y='Value', ax=axes[1, 1])
    axes[1, 1].set_title('Violin Plot Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/02_distributions.png', dpi=150, bbox_inches='tight')
    print("✓ Created distribution plots: /tmp/02_distributions.png")
    plt.close()
    print()


def demonstrate_feature_visualization():
    """Demonstrate visualizing features in a dataset."""
    print("=" * 60)
    print("3. FEATURE VISUALIZATION")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create a DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = y
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Pairwise scatter plot for first two features
    for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
        mask = df['species'] == i
        axes[0, 0].scatter(df[mask][iris.feature_names[0]], 
                          df[mask][iris.feature_names[1]],
                          label=species, alpha=0.6, s=50)
    axes[0, 0].set_xlabel(iris.feature_names[0])
    axes[0, 0].set_ylabel(iris.feature_names[1])
    axes[0, 0].set_title('Feature Scatter Plot')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature distributions by class
    for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
        mask = df['species'] == i
        axes[0, 1].hist(df[mask][iris.feature_names[0]], 
                       alpha=0.5, label=species, bins=15)
    axes[0, 1].set_xlabel(iris.feature_names[0])
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Feature Distribution by Class')
    axes[0, 1].legend()
    
    # Correlation heatmap
    correlation = df[iris.feature_names].corr()
    im = axes[1, 0].imshow(correlation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 0].set_xticks(range(len(iris.feature_names)))
    axes[1, 0].set_yticks(range(len(iris.feature_names)))
    axes[1, 0].set_xticklabels([name.split()[0] for name in iris.feature_names], rotation=45)
    axes[1, 0].set_yticklabels([name.split()[0] for name in iris.feature_names])
    axes[1, 0].set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Add correlation values
    for i in range(len(iris.feature_names)):
        for j in range(len(iris.feature_names)):
            text = axes[1, 0].text(j, i, f'{correlation.iloc[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
    
    # Box plots for all features
    df_melted = df.melt(id_vars=['species_name'], 
                        value_vars=iris.feature_names,
                        var_name='Feature', 
                        value_name='Value')
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='species_name', ax=axes[1, 1])
    axes[1, 1].set_title('Feature Distributions by Species')
    axes[1, 1].set_xticklabels([name.split()[0] for name in iris.feature_names], rotation=45)
    axes[1, 1].legend(title='Species', loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/tmp/03_feature_visualization.png', dpi=150, bbox_inches='tight')
    print("✓ Created feature visualization: /tmp/03_feature_visualization.png")
    plt.close()
    print()


def demonstrate_training_curves():
    """Demonstrate visualizing training metrics."""
    print("=" * 60)
    print("4. TRAINING CURVES")
    print("=" * 60)
    
    # Simulate training history
    epochs = np.arange(1, 51)
    
    # Realistic training curves
    train_loss = 2.0 * np.exp(-epochs / 10) + 0.1 + np.random.randn(50) * 0.05
    val_loss = 2.0 * np.exp(-epochs / 10) + 0.2 + np.random.randn(50) * 0.08
    
    train_acc = 1 - np.exp(-epochs / 8) * 0.7 - np.random.randn(50) * 0.02
    val_acc = 1 - np.exp(-epochs / 8) * 0.7 - np.random.randn(50) * 0.03
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, train_acc, label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    axes[1].plot(epochs, val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/04_training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Created training curves: /tmp/04_training_curves.png")
    plt.close()
    
    print("\nKey observations from training curves:")
    print("  - Training loss decreases steadily")
    print("  - Validation loss follows training loss (good generalization)")
    print("  - Both accuracies improve over time")
    print("  - No significant overfitting (val_loss not increasing)\n")


def demonstrate_confusion_matrix():
    """Demonstrate confusion matrix visualization."""
    print("=" * 60)
    print("5. CONFUSION MATRIX")
    print("=" * 60)
    
    # Create and train a simple classifier
    X, y = make_classification(n_samples=500, n_features=10, n_informative=8,
                               n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class 0', 'Class 1', 'Class 2'],
                yticklabels=['Class 0', 'Class 1', 'Class 2'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('/tmp/05_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✓ Created confusion matrix: /tmp/05_confusion_matrix.png")
    plt.close()
    
    # Print metrics
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Class 0', 'Class 1', 'Class 2']))


def demonstrate_activation_functions():
    """Visualize common activation functions."""
    print("=" * 60)
    print("6. ACTIVATION FUNCTIONS")
    print("=" * 60)
    
    x = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0, x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid, linewidth=2, color='blue')
    axes[0, 0].set_title('Sigmoid: σ(x) = 1 / (1 + e^(-x))')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('σ(x)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Tanh
    axes[0, 1].plot(x, tanh, linewidth=2, color='green')
    axes[0, 1].set_title('Tanh: tanh(x)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('tanh(x)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # ReLU
    axes[1, 0].plot(x, relu, linewidth=2, color='red')
    axes[1, 0].set_title('ReLU: max(0, x)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('ReLU(x)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu, linewidth=2, color='purple')
    axes[1, 1].set_title('Leaky ReLU: max(0.01x, x)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Leaky ReLU(x)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('/tmp/06_activation_functions.png', dpi=150, bbox_inches='tight')
    print("✓ Created activation functions plot: /tmp/06_activation_functions.png")
    plt.close()
    print()


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("DATA VISUALIZATION FOR DEEP LEARNING")
    print("=" * 60 + "\n")
    
    demonstrate_basic_plotting()
    demonstrate_distributions()
    demonstrate_feature_visualization()
    demonstrate_training_curves()
    demonstrate_confusion_matrix()
    demonstrate_activation_functions()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Visualization is crucial for understanding data and model performance.
Key concepts covered:
1. Basic plotting (line, scatter plots)
2. Distribution visualization (histograms, KDE, box plots)
3. Feature visualization (scatter, correlation heatmaps)
4. Training curves (loss and accuracy over time)
5. Confusion matrices (classification performance)
6. Activation functions (neural network components)

All visualizations saved to /tmp/ directory:
  - 01_basic_plots.png
  - 02_distributions.png
  - 03_feature_visualization.png
  - 04_training_curves.png
  - 05_confusion_matrix.png
  - 06_activation_functions.png

Best practices:
- Always visualize your data before modeling
- Monitor training curves to detect overfitting
- Use confusion matrices to understand misclassifications
- Choose appropriate visualization types for your data
- Make plots clear and informative with labels and legends

Next steps:
- Practice creating visualizations for different datasets
- Learn to interpret training curves
- Move to Level 1 to build your first neural networks!
    """)


if __name__ == "__main__":
    main()
