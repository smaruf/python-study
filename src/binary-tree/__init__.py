"""
Binary Tree Module

This module provides comprehensive implementations of binary trees and balanced binary trees.

Available classes:
- TreeNode: Basic tree node for binary trees
- BinaryTree: Binary Search Tree with various traversal methods
- AVLNode: Node for AVL trees
- AVLTree: Self-balancing AVL tree
- RedBlackNode: Node for Red-Black trees
- RedBlackTree: Self-balancing Red-Black tree

For detailed examples and usage, see the README.md or run the example files.
"""

from .binary_tree import TreeNode, BinaryTree
from .balanced_tree import AVLNode, AVLTree, RedBlackNode, RedBlackTree

__all__ = [
    'TreeNode',
    'BinaryTree',
    'AVLNode',
    'AVLTree',
    'RedBlackNode',
    'RedBlackTree',
]

__version__ = '1.0.0'
