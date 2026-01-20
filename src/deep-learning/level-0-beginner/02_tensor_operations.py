"""
Level 0 - Beginner: Tensor Operations

This module introduces tensors using PyTorch. Tensors are the fundamental data structures
in deep learning, similar to NumPy arrays but with GPU support and automatic differentiation.

Topics covered:
- Creating tensors
- Tensor shapes and dimensions
- Element-wise operations
- Matrix operations
- Moving tensors to GPU
- Basic autograd (automatic differentiation)

Author: Python Study Repository
"""

import torch
import numpy as np


def demonstrate_tensor_creation():
    """Demonstrate various ways to create PyTorch tensors."""
    print("=" * 60)
    print("1. TENSOR CREATION")
    print("=" * 60)
    
    # Create tensor from list
    tensor1 = torch.tensor([1, 2, 3, 4, 5])
    print(f"Tensor from list: {tensor1}")
    print(f"Shape: {tensor1.shape}, Dtype: {tensor1.dtype}\n")
    
    # Create 2D tensor
    tensor2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"2D tensor:\n{tensor2}")
    print(f"Shape: {tensor2.shape}, Dtype: {tensor2.dtype}\n")
    
    # Create tensors with specific values
    zeros = torch.zeros(3, 4)
    ones = torch.ones(2, 3)
    identity = torch.eye(3)
    
    print(f"Zeros tensor:\n{zeros}\n")
    print(f"Ones tensor:\n{ones}\n")
    print(f"Identity matrix:\n{identity}\n")
    
    # Create tensor with range
    range_tensor = torch.arange(0, 10, 2)
    print(f"Range tensor: {range_tensor}\n")
    
    # Create tensor with evenly spaced values
    linspace_tensor = torch.linspace(0, 1, 5)
    print(f"Linspace tensor: {linspace_tensor}\n")
    
    # Create random tensors
    random_tensor = torch.rand(3, 3)  # uniform [0, 1)
    print(f"Random tensor (uniform):\n{random_tensor}\n")
    
    random_normal = torch.randn(3, 3)  # standard normal
    print(f"Random tensor (normal):\n{random_normal}\n")
    
    # Create tensor with specific dtype
    float_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
    print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}\n")


def demonstrate_numpy_conversion():
    """Demonstrate conversion between NumPy and PyTorch."""
    print("=" * 60)
    print("2. NUMPY ↔ PYTORCH CONVERSION")
    print("=" * 60)
    
    # NumPy to PyTorch
    np_array = np.array([1, 2, 3, 4, 5])
    torch_tensor = torch.from_numpy(np_array)
    
    print(f"NumPy array: {np_array}")
    print(f"PyTorch tensor: {torch_tensor}\n")
    
    # PyTorch to NumPy
    tensor = torch.tensor([6, 7, 8, 9, 10])
    numpy_array = tensor.numpy()
    
    print(f"PyTorch tensor: {tensor}")
    print(f"NumPy array: {numpy_array}\n")
    
    # Note: They share memory!
    print("Memory sharing demonstration:")
    np_arr = np.array([1, 2, 3])
    torch_t = torch.from_numpy(np_arr)
    print(f"Original NumPy: {np_arr}")
    print(f"PyTorch tensor: {torch_t}")
    
    np_arr[0] = 100
    print(f"After modifying NumPy[0] = 100:")
    print(f"NumPy: {np_arr}")
    print(f"PyTorch: {torch_t} (changed too!)\n")


def demonstrate_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("=" * 60)
    print("3. TENSOR OPERATIONS")
    print("=" * 60)
    
    tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    
    # Basic operations
    print(f"Original tensor: {tensor}")
    print(f"Add 10: {tensor + 10}")
    print(f"Multiply by 2: {tensor * 2}")
    print(f"Square: {tensor ** 2}")
    print(f"Square root: {torch.sqrt(tensor)}\n")
    
    # Statistical operations
    print(f"Sum: {tensor.sum()}")
    print(f"Mean: {tensor.mean()}")
    print(f"Std: {tensor.std()}")
    print(f"Min: {tensor.min()}, Max: {tensor.max()}\n")
    
    # Element-wise operations
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    
    print(f"Tensor 1: {t1}")
    print(f"Tensor 2: {t2}")
    print(f"Addition: {t1 + t2}")
    print(f"Multiplication: {t1 * t2}")
    print(f"Division: {t2 / t1}\n")


def demonstrate_matrix_operations():
    """Demonstrate matrix operations."""
    print("=" * 60)
    print("4. MATRIX OPERATIONS")
    print("=" * 60)
    
    # Matrix multiplication
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    B = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A @ B (matrix multiplication):\n{A @ B}\n")
    print(f"A * B (element-wise):\n{A * B}\n")
    
    # Transpose
    print(f"Transpose of A:\n{A.T}\n")
    
    # Batch matrix multiplication
    batch1 = torch.randn(10, 3, 4)  # batch of 10 matrices (3x4)
    batch2 = torch.randn(10, 4, 5)  # batch of 10 matrices (4x5)
    result = torch.bmm(batch1, batch2)  # batch of 10 matrices (3x5)
    
    print(f"Batch matrix multiplication:")
    print(f"  Input 1 shape: {batch1.shape}")
    print(f"  Input 2 shape: {batch2.shape}")
    print(f"  Result shape: {result.shape}\n")


def demonstrate_reshaping():
    """Demonstrate tensor reshaping and manipulation."""
    print("=" * 60)
    print("5. RESHAPING AND MANIPULATION")
    print("=" * 60)
    
    tensor = torch.arange(12)
    print(f"Original tensor: {tensor}")
    print(f"Shape: {tensor.shape}\n")
    
    # Reshape
    reshaped = tensor.reshape(3, 4)
    print(f"Reshaped to (3, 4):\n{reshaped}\n")
    
    # View (similar to reshape but shares memory)
    viewed = tensor.view(2, 6)
    print(f"View as (2, 6):\n{viewed}\n")
    
    # Squeeze and unsqueeze
    t = torch.zeros(1, 3, 1, 4)
    print(f"Original shape: {t.shape}")
    print(f"After squeeze: {t.squeeze().shape}")  # removes all dimensions of size 1
    print(f"After unsqueeze(0): {t.squeeze().unsqueeze(0).shape}\n")
    
    # Concatenate
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])
    
    cat_dim0 = torch.cat([t1, t2], dim=0)
    cat_dim1 = torch.cat([t1, t2], dim=1)
    
    print(f"Tensor 1:\n{t1}")
    print(f"Tensor 2:\n{t2}")
    print(f"Concatenate along dim=0:\n{cat_dim0}")
    print(f"Concatenate along dim=1:\n{cat_dim1}\n")
    
    # Stack
    stacked = torch.stack([t1, t2], dim=0)
    print(f"Stacked tensors shape: {stacked.shape}")
    print(f"Stacked tensors:\n{stacked}\n")


def demonstrate_indexing():
    """Demonstrate tensor indexing and slicing."""
    print("=" * 60)
    print("6. INDEXING AND SLICING")
    print("=" * 60)
    
    tensor = torch.tensor([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12]])
    
    print(f"Original tensor:\n{tensor}\n")
    
    # Basic indexing
    print(f"Element at [0, 0]: {tensor[0, 0]}")
    print(f"Element at [1, 2]: {tensor[1, 2]}\n")
    
    # Slicing
    print(f"First row: {tensor[0, :]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Sub-tensor [0:2, 1:3]:\n{tensor[0:2, 1:3]}\n")
    
    # Boolean indexing
    mask = tensor > 6
    print(f"Mask (> 6):\n{mask}")
    print(f"Values > 6: {tensor[mask]}\n")
    
    # Advanced indexing
    indices = torch.tensor([0, 2])
    print(f"Rows 0 and 2:\n{tensor[indices]}\n")


def demonstrate_gpu_operations():
    """Demonstrate GPU operations if available."""
    print("=" * 60)
    print("7. GPU OPERATIONS")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU is available: {torch.cuda.get_device_name(0)}")
        
        # Create tensor on GPU
        tensor_gpu = torch.randn(3, 3, device=device)
        print(f"Tensor on GPU:\n{tensor_gpu}")
        
        # Move tensor to GPU
        tensor_cpu = torch.randn(3, 3)
        tensor_gpu2 = tensor_cpu.to(device)
        print(f"Moved to GPU:\n{tensor_gpu2}")
        
        # Operations on GPU
        result = tensor_gpu + tensor_gpu2
        print(f"Result on GPU:\n{result}")
        
        # Move back to CPU
        result_cpu = result.to("cpu")
        print(f"Result on CPU:\n{result_cpu}\n")
    else:
        print("GPU is not available. Operations will run on CPU.")
        print("To enable GPU: Install CUDA and PyTorch with CUDA support\n")
    
    # Best practice: using device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tensor = torch.randn(3, 3).to(device)
    print(f"Tensor on {device}:\n{tensor}\n")


def demonstrate_autograd():
    """Demonstrate automatic differentiation."""
    print("=" * 60)
    print("8. AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
    print("=" * 60)
    
    # Create tensor with gradient tracking
    x = torch.tensor([2.0], requires_grad=True)
    print(f"Input x: {x}")
    print(f"Requires grad: {x.requires_grad}\n")
    
    # Define a simple function: y = x^2 + 3x + 1
    y = x**2 + 3*x + 1
    print(f"y = x^2 + 3x + 1 = {y}\n")
    
    # Compute gradients
    y.backward()
    print(f"dy/dx = 2x + 3 = {x.grad}")
    print(f"At x=2: dy/dx = {x.grad.item()}\n")
    
    # More complex example
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    print(f"Input matrix x:\n{x}\n")
    
    # y = sum(x^2)
    y = (x**2).sum()
    print(f"y = sum(x^2) = {y}\n")
    
    y.backward()
    print(f"Gradients (dy/dx = 2x):\n{x.grad}\n")
    
    print("Autograd is essential for training neural networks!")
    print("It automatically computes gradients for backpropagation.\n")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("TENSOR OPERATIONS FOR DEEP LEARNING")
    print("=" * 60 + "\n")
    
    demonstrate_tensor_creation()
    demonstrate_numpy_conversion()
    demonstrate_tensor_operations()
    demonstrate_matrix_operations()
    demonstrate_reshaping()
    demonstrate_indexing()
    demonstrate_gpu_operations()
    demonstrate_autograd()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Tensors are the fundamental data structure in deep learning.
Key concepts covered:
1. Creating and initializing tensors
2. Converting between NumPy and PyTorch
3. Basic tensor operations
4. Matrix operations (crucial for neural networks)
5. Reshaping and manipulation
6. Indexing and slicing
7. GPU acceleration
8. Automatic differentiation (autograd)

Key differences from NumPy:
- GPU support for acceleration
- Automatic differentiation for training
- Optimized for deep learning workflows

Next steps:
- Practice tensor operations
- Understand autograd for gradient computation
- Move to data preprocessing in Level 0.3
    """)


if __name__ == "__main__":
    main()
