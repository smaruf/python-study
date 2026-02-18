"""
Unit Tests for Binary Tree Module

This file contains comprehensive unit tests for the binary tree implementations
to ensure correctness of operations, traversals, and balancing.

Author: Python Study Repository
"""

import unittest
import sys
import os

# Add the binary-tree directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from binary_tree import TreeNode, BinaryTree
from balanced_tree import AVLTree, RedBlackTree


class TestBinaryTree(unittest.TestCase):
    """Test cases for basic Binary Search Tree."""
    
    def setUp(self):
        """Set up a test tree before each test."""
        self.tree = BinaryTree()
        self.values = [10, 5, 15, 3, 7, 12, 18]
        for val in self.values:
            self.tree.insert(val)
    
    def test_insert(self):
        """Test insertion creates correct structure."""
        tree = BinaryTree()
        tree.insert(10)
        self.assertEqual(tree.root.val, 10)
        tree.insert(5)
        self.assertEqual(tree.root.left.val, 5)
        tree.insert(15)
        self.assertEqual(tree.root.right.val, 15)
    
    def test_search_existing(self):
        """Test searching for existing values."""
        for val in self.values:
            self.assertTrue(self.tree.search(val))
    
    def test_search_non_existing(self):
        """Test searching for non-existing values."""
        self.assertFalse(self.tree.search(100))
        self.assertFalse(self.tree.search(-5))
    
    def test_inorder_traversal(self):
        """Test inorder traversal returns sorted values."""
        result = self.tree.inorder()
        expected = sorted(self.values)
        self.assertEqual(result, expected)
    
    def test_preorder_traversal(self):
        """Test preorder traversal."""
        result = self.tree.preorder()
        # Root should be first
        self.assertEqual(result[0], 10)
        # Should contain all values
        self.assertEqual(sorted(result), sorted(self.values))
    
    def test_postorder_traversal(self):
        """Test postorder traversal."""
        result = self.tree.postorder()
        # Root should be last
        self.assertEqual(result[-1], 10)
        # Should contain all values
        self.assertEqual(sorted(result), sorted(self.values))
    
    def test_level_order_traversal(self):
        """Test level-order traversal."""
        result = self.tree.level_order()
        # Root should be first
        self.assertEqual(result[0], 10)
        # Should contain all values
        self.assertEqual(sorted(result), sorted(self.values))
    
    def test_height(self):
        """Test height calculation."""
        # Tree with 7 nodes should have height 3
        self.assertEqual(self.tree.height(), 3)
        
        # Empty tree
        empty_tree = BinaryTree()
        self.assertEqual(empty_tree.height(), 0)
        
        # Single node
        single_tree = BinaryTree()
        single_tree.insert(10)
        self.assertEqual(single_tree.height(), 1)
    
    def test_is_balanced(self):
        """Test balance checking."""
        self.assertTrue(self.tree.is_balanced())
        
        # Create unbalanced tree
        unbalanced = BinaryTree()
        for i in range(1, 6):
            unbalanced.insert(i)
        self.assertFalse(unbalanced.is_balanced())
    
    def test_balance_factor(self):
        """Test balance factor calculation."""
        bf = self.tree.balance_factor(self.tree.root)
        self.assertIn(bf, [-1, 0, 1])  # Balanced tree should have BF in {-1, 0, 1}
    
    def test_size(self):
        """Test size calculation."""
        self.assertEqual(self.tree.size(), len(self.values))
        
        empty_tree = BinaryTree()
        self.assertEqual(empty_tree.size(), 0)
    
    def test_min_max(self):
        """Test min and max value finding."""
        self.assertEqual(self.tree.min_value(), min(self.values))
        self.assertEqual(self.tree.max_value(), max(self.values))


class TestAVLTree(unittest.TestCase):
    """Test cases for AVL Tree (self-balancing)."""
    
    def test_avl_always_balanced(self):
        """Test that AVL tree maintains balance after insertions."""
        avl = AVLTree()
        # Insert sequential values (worst case for regular BST)
        for i in range(1, 16):
            avl.insert(i)
        
        self.assertTrue(avl.is_balanced())
        # Height should be O(log n)
        self.assertLessEqual(avl.height(), 5)  # log2(15) ≈ 4
    
    def test_avl_rotations_left_left(self):
        """Test Left-Left case triggers right rotation."""
        avl = AVLTree()
        avl.insert(30)
        avl.insert(20)
        avl.insert(10)
        
        # After rotation, 20 should be root
        self.assertEqual(avl.root.val, 20)
        self.assertEqual(avl.root.left.val, 10)
        self.assertEqual(avl.root.right.val, 30)
    
    def test_avl_rotations_right_right(self):
        """Test Right-Right case triggers left rotation."""
        avl = AVLTree()
        avl.insert(10)
        avl.insert(20)
        avl.insert(30)
        
        # After rotation, 20 should be root
        self.assertEqual(avl.root.val, 20)
        self.assertEqual(avl.root.left.val, 10)
        self.assertEqual(avl.root.right.val, 30)
    
    def test_avl_rotations_left_right(self):
        """Test Left-Right case triggers left-right rotation."""
        avl = AVLTree()
        avl.insert(30)
        avl.insert(10)
        avl.insert(20)
        
        # After rotation, 20 should be root
        self.assertEqual(avl.root.val, 20)
        self.assertEqual(avl.root.left.val, 10)
        self.assertEqual(avl.root.right.val, 30)
    
    def test_avl_rotations_right_left(self):
        """Test Right-Left case triggers right-left rotation."""
        avl = AVLTree()
        avl.insert(10)
        avl.insert(30)
        avl.insert(20)
        
        # After rotation, 20 should be root
        self.assertEqual(avl.root.val, 20)
        self.assertEqual(avl.root.left.val, 10)
        self.assertEqual(avl.root.right.val, 30)
    
    def test_avl_inorder_sorted(self):
        """Test AVL tree maintains BST property."""
        avl = AVLTree()
        values = [50, 30, 70, 20, 40, 60, 80]
        for val in values:
            avl.insert(val)
        
        result = avl.inorder()
        self.assertEqual(result, sorted(values))
    
    def test_avl_search(self):
        """Test AVL tree search operation."""
        avl = AVLTree()
        values = [10, 20, 30, 40, 50]
        for val in values:
            avl.insert(val)
        
        for val in values:
            self.assertTrue(avl.search(val))
        self.assertFalse(avl.search(100))


class TestRedBlackTree(unittest.TestCase):
    """Test cases for Red-Black Tree."""
    
    def test_rb_tree_insert(self):
        """Test Red-Black tree insertion."""
        rb = RedBlackTree()
        values = [10, 20, 30, 15, 25, 5]
        for val in values:
            rb.insert(val)
        
        # Check inorder traversal is sorted
        result = rb.inorder()
        self.assertEqual(result, sorted(values))
    
    def test_rb_tree_root_is_black(self):
        """Test that root is always black."""
        rb = RedBlackTree()
        rb.insert(10)
        self.assertEqual(rb.root.color, 'BLACK')
        
        rb.insert(20)
        self.assertEqual(rb.root.color, 'BLACK')
    
    def test_rb_tree_inorder(self):
        """Test Red-Black tree maintains BST property."""
        rb = RedBlackTree()
        values = [10, 20, 30, 15, 25, 5, 1, 35, 40]
        for val in values:
            rb.insert(val)
        
        result = rb.inorder()
        self.assertEqual(result, sorted(values))


class TestTreeComparison(unittest.TestCase):
    """Test cases comparing different tree implementations."""
    
    def test_height_comparison(self):
        """Compare heights of balanced vs unbalanced trees."""
        # Unbalanced tree (sequential insertion)
        unbalanced = BinaryTree()
        for i in range(1, 16):
            unbalanced.insert(i)
        
        # AVL tree (same values)
        avl = AVLTree()
        for i in range(1, 16):
            avl.insert(i)
        
        # AVL should have significantly smaller height
        self.assertLess(avl.height(), unbalanced.height())
        
        # AVL height should be O(log n)
        self.assertLessEqual(avl.height(), 5)  # log2(15) ≈ 4
    
    def test_all_trees_maintain_bst_property(self):
        """Test that all tree types maintain BST property."""
        values = [50, 30, 70, 20, 40, 60, 80]
        
        # Binary Tree
        bt = BinaryTree()
        for val in values:
            bt.insert(val)
        self.assertEqual(bt.inorder(), sorted(values))
        
        # AVL Tree
        avl = AVLTree()
        for val in values:
            avl.insert(val)
        self.assertEqual(avl.inorder(), sorted(values))
        
        # Red-Black Tree
        rb = RedBlackTree()
        for val in values:
            rb.insert(val)
        self.assertEqual(rb.inorder(), sorted(values))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
