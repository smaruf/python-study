"""
Level 0 - Beginner: NumPy Basics

This module introduces NumPy, the fundamental package for numerical computing in Python.
NumPy provides support for large, multi-dimensional arrays and matrices, along with
a collection of mathematical functions to operate on these arrays.

Topics covered:
- Creating arrays
- Array indexing and slicing
- Broadcasting
- Mathematical operations
- Linear algebra basics

Author: Python Study Repository
"""

import numpy as np


def demonstrate_array_creation():
    """Demonstrate various ways to create NumPy arrays."""
    print("=" * 60)
    print("1. ARRAY CREATION")
    print("=" * 60)
    
    # Create array from list
    arr1 = np.array([1, 2, 3, 4, 5])
    print(f"Array from list: {arr1}")
    print(f"Shape: {arr1.shape}, Dtype: {arr1.dtype}\n")
    
    # Create 2D array
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"2D array:\n{arr2}")
    print(f"Shape: {arr2.shape}, Dtype: {arr2.dtype}\n")
    
    # Create arrays with specific values
    zeros = np.zeros((3, 4))
    ones = np.ones((2, 3))
    identity = np.eye(3)
    
    print(f"Zeros array:\n{zeros}\n")
    print(f"Ones array:\n{ones}\n")
    print(f"Identity matrix:\n{identity}\n")
    
    # Create array with range
    range_arr = np.arange(0, 10, 2)  # start, stop, step
    print(f"Range array: {range_arr}\n")
    
    # Create array with evenly spaced values
    linspace_arr = np.linspace(0, 1, 5)  # start, stop, num_points
    print(f"Linspace array: {linspace_arr}\n")
    
    # Create random arrays
    random_arr = np.random.rand(3, 3)  # uniform distribution [0, 1)
    print(f"Random array (uniform):\n{random_arr}\n")
    
    random_normal = np.random.randn(3, 3)  # standard normal distribution
    print(f"Random array (normal):\n{random_normal}\n")


def demonstrate_indexing_slicing():
    """Demonstrate array indexing and slicing."""
    print("=" * 60)
    print("2. INDEXING AND SLICING")
    print("=" * 60)
    
    arr = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]])
    
    print(f"Original array:\n{arr}\n")
    
    # Basic indexing
    print(f"Element at (0, 0): {arr[0, 0]}")
    print(f"Element at (1, 2): {arr[1, 2]}\n")
    
    # Slicing rows
    print(f"First row: {arr[0, :]}")
    print(f"First two rows:\n{arr[:2, :]}\n")
    
    # Slicing columns
    print(f"First column: {arr[:, 0]}")
    print(f"Last two columns:\n{arr[:, -2:]}\n")
    
    # Advanced slicing
    print(f"Sub-array (rows 0-1, cols 1-2):\n{arr[:2, 1:3]}\n")
    
    # Boolean indexing
    mask = arr > 6
    print(f"Boolean mask (values > 6):\n{mask}")
    print(f"Values greater than 6: {arr[mask]}\n")


def demonstrate_broadcasting():
    """Demonstrate NumPy broadcasting."""
    print("=" * 60)
    print("3. BROADCASTING")
    print("=" * 60)
    
    # Scalar and array
    arr = np.array([1, 2, 3, 4])
    print(f"Original array: {arr}")
    print(f"Array + 10: {arr + 10}")
    print(f"Array * 2: {arr * 2}\n")
    
    # 1D and 2D arrays
    arr1 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    arr2 = np.array([10, 20, 30])
    
    print(f"2D array:\n{arr1}")
    print(f"1D array: {arr2}")
    print(f"2D + 1D (broadcasting):\n{arr1 + arr2}\n")
    
    # Column vector broadcasting
    col_vec = np.array([[1], [2]])
    print(f"Column vector:\n{col_vec}")
    print(f"2D array + column vector:\n{arr1 + col_vec}\n")


def demonstrate_math_operations():
    """Demonstrate mathematical operations on arrays."""
    print("=" * 60)
    print("4. MATHEMATICAL OPERATIONS")
    print("=" * 60)
    
    arr = np.array([1, 2, 3, 4, 5])
    
    # Basic operations
    print(f"Original array: {arr}")
    print(f"Sum: {np.sum(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Std: {np.std(arr)}")
    print(f"Min: {np.min(arr)}, Max: {np.max(arr)}\n")
    
    # Element-wise operations
    print(f"Square: {arr ** 2}")
    print(f"Square root: {np.sqrt(arr)}")
    print(f"Exponential: {np.exp(arr)}")
    print(f"Log: {np.log(arr)}\n")
    
    # Trigonometric functions
    angles = np.array([0, np.pi/4, np.pi/2, np.pi])
    print(f"Angles: {angles}")
    print(f"Sin: {np.sin(angles)}")
    print(f"Cos: {np.cos(angles)}\n")
    
    # Statistical operations
    data = np.random.randn(100)
    print(f"Random data statistics:")
    print(f"  Mean: {np.mean(data):.4f}")
    print(f"  Std: {np.std(data):.4f}")
    print(f"  Median: {np.median(data):.4f}")
    print(f"  25th percentile: {np.percentile(data, 25):.4f}")
    print(f"  75th percentile: {np.percentile(data, 75):.4f}\n")


def demonstrate_linear_algebra():
    """Demonstrate linear algebra operations."""
    print("=" * 60)
    print("5. LINEAR ALGEBRA")
    print("=" * 60)
    
    # Matrix multiplication
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"A @ B (matrix multiplication):\n{A @ B}\n")
    print(f"A * B (element-wise):\n{A * B}\n")
    
    # Transpose
    print(f"Transpose of A:\n{A.T}\n")
    
    # Determinant and inverse
    det_A = np.linalg.det(A)
    print(f"Determinant of A: {det_A}")
    
    inv_A = np.linalg.inv(A)
    print(f"Inverse of A:\n{inv_A}")
    print(f"A @ inv(A) (should be identity):\n{A @ inv_A}\n")
    
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}\n")
    
    # Dot product
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    dot_product = np.dot(v1, v2)
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Dot product: {dot_product}\n")
    
    # Norm (magnitude)
    norm_v1 = np.linalg.norm(v1)
    print(f"Norm of v1: {norm_v1}\n")


def demonstrate_reshaping():
    """Demonstrate array reshaping and manipulation."""
    print("=" * 60)
    print("6. RESHAPING AND MANIPULATION")
    print("=" * 60)
    
    arr = np.arange(12)
    print(f"Original array: {arr}\n")
    
    # Reshape
    reshaped = arr.reshape(3, 4)
    print(f"Reshaped to (3, 4):\n{reshaped}\n")
    
    # Flatten
    flattened = reshaped.flatten()
    print(f"Flattened: {flattened}\n")
    
    # Stack arrays
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    
    vstacked = np.vstack([arr1, arr2])
    hstacked = np.hstack([arr1, arr2])
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    print(f"Vertically stacked:\n{vstacked}")
    print(f"Horizontally stacked: {hstacked}\n")
    
    # Concatenate
    concatenated = np.concatenate([arr1, arr2])
    print(f"Concatenated: {concatenated}\n")


def main():
    """Main function to run all demonstrations."""
    print("\n" + "=" * 60)
    print("NUMPY BASICS FOR DEEP LEARNING")
    print("=" * 60 + "\n")
    
    demonstrate_array_creation()
    demonstrate_indexing_slicing()
    demonstrate_broadcasting()
    demonstrate_math_operations()
    demonstrate_linear_algebra()
    demonstrate_reshaping()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
NumPy is the foundation of numerical computing in Python and essential for deep learning.
Key concepts covered:
1. Array creation and initialization
2. Indexing and slicing for data access
3. Broadcasting for efficient computation
4. Mathematical and statistical operations
5. Linear algebra operations (crucial for neural networks)
6. Reshaping and array manipulation

Next steps:
- Practice creating and manipulating arrays
- Understand broadcasting rules
- Get comfortable with linear algebra operations
- Move to tensor operations in Level 0.2
    """)


if __name__ == "__main__":
    main()
