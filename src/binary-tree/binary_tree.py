"""
binary_tree.py

Implementation of a basic Binary Search Tree (BST) with various tree traversal methods
and utility functions for checking tree properties.

Features:
- Binary Search Tree operations (insert, search, delete)
- Tree traversal methods (inorder, preorder, postorder, level-order)
- Tree property checks (height, balance, etc.)
- Comprehensive examples and demonstrations
"""

from collections import deque


class TreeNode:
    """
    Node for a binary tree.
    
    Attributes:
        val: The value stored in the node
        left: Reference to the left child
        right: Reference to the right child
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
    
    def __repr__(self):
        return f"TreeNode({self.val})"


class BinaryTree:
    """
    Binary Search Tree implementation with comprehensive tree operations.
    """
    
    def __init__(self):
        """Initialize an empty binary tree."""
        self.root = None
    
    def insert(self, val):
        """
        Insert a value into the binary search tree.
        
        Args:
            val: The value to insert
        
        Time Complexity: O(h) where h is the height of the tree
        """
        if self.root is None:
            self.root = TreeNode(val)
        else:
            self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        """Helper method to recursively insert a value."""
        if val < node.val:
            if node.left is None:
                node.left = TreeNode(val)
            else:
                self._insert_recursive(node.left, val)
        else:
            if node.right is None:
                node.right = TreeNode(val)
            else:
                self._insert_recursive(node.right, val)
    
    def search(self, val):
        """
        Search for a value in the tree.
        
        Args:
            val: The value to search for
        
        Returns:
            bool: True if the value exists, False otherwise
        
        Time Complexity: O(h) where h is the height of the tree
        """
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        """Helper method to recursively search for a value."""
        if node is None:
            return False
        if node.val == val:
            return True
        elif val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)
    
    def height(self, node=None):
        """
        Calculate the height of the tree.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            int: The height of the tree (0 for empty tree)
        
        Time Complexity: O(n)
        """
        if node is None:
            node = self.root
        
        return self._height_helper(node)
    
    def _height_helper(self, node):
        """Helper method to calculate height without default root behavior."""
        if node is None:
            return 0
        return max(self._height_helper(node.left), self._height_helper(node.right)) + 1
    
    def is_balanced(self):
        """
        Check if the tree is balanced.
        A tree is balanced if the height difference between left and right
        subtrees of every node is at most 1.
        
        Returns:
            bool: True if the tree is balanced, False otherwise
        
        Time Complexity: O(n)
        Space Complexity: O(h) where h is height
        """
        def check_height(node):
            if node is None:
                return 0
            
            left_height = check_height(node.left)
            if left_height == -1:
                return -1
            
            right_height = check_height(node.right)
            if right_height == -1:
                return -1
            
            if abs(left_height - right_height) > 1:
                return -1
            
            return max(left_height, right_height) + 1
        
        return check_height(self.root) != -1
    
    def balance_factor(self, node):
        """
        Calculate the balance factor of a node.
        Balance Factor = Height(Left) - Height(Right)
        
        Args:
            node: The node to calculate balance factor for
        
        Returns:
            int: The balance factor
        """
        if node is None:
            return 0
        
        left_height = self._height_helper(node.left)
        right_height = self._height_helper(node.right)
        
        return left_height - right_height
    
    # Traversal Methods
    
    def inorder(self, node=None):
        """
        Perform inorder traversal (Left → Root → Right).
        For BST, this gives elements in sorted order.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            list: List of values in inorder
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        start_node = node if node is not None else self.root
        return self._inorder_helper(start_node)
    
    def _inorder_helper(self, node):
        """Helper for inorder traversal."""
        if node is None:
            return []
        result = []
        result.extend(self._inorder_helper(node.left))
        result.append(node.val)
        result.extend(self._inorder_helper(node.right))
        return result
    
    def preorder(self, node=None):
        """
        Perform preorder traversal (Root → Left → Right).
        Useful for creating a copy of the tree.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            list: List of values in preorder
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        start_node = node if node is not None else self.root
        return self._preorder_helper(start_node)
    
    def _preorder_helper(self, node):
        """Helper for preorder traversal."""
        if node is None:
            return []
        result = []
        result.append(node.val)
        result.extend(self._preorder_helper(node.left))
        result.extend(self._preorder_helper(node.right))
        return result
    
    def postorder(self, node=None):
        """
        Perform postorder traversal (Left → Right → Root).
        Useful for deleting the tree.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            list: List of values in postorder
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        start_node = node if node is not None else self.root
        return self._postorder_helper(start_node)
    
    def _postorder_helper(self, node):
        """Helper for postorder traversal."""
        if node is None:
            return []
        result = []
        result.extend(self._postorder_helper(node.left))
        result.extend(self._postorder_helper(node.right))
        result.append(node.val)
        return result
    
    def level_order(self):
        """
        Perform level-order traversal (BFS).
        Visits nodes level by level from top to bottom.
        
        Returns:
            list: List of values in level order
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if self.root is None:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            node = queue.popleft()
            result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        return result
    
    def level_order_by_levels(self):
        """
        Perform level-order traversal with levels separated.
        
        Returns:
            list: List of lists, where each inner list represents a level
        
        Time Complexity: O(n)
        Space Complexity: O(n)
        """
        if self.root is None:
            return []
        
        result = []
        queue = deque([self.root])
        
        while queue:
            level_size = len(queue)
            current_level = []
            
            for _ in range(level_size):
                node = queue.popleft()
                current_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            
            result.append(current_level)
        
        return result
    
    def visualize(self):
        """
        Print a simple visualization of the tree.
        """
        def _visualize_helper(node, prefix="", is_tail=True):
            if node is None:
                return
            
            print(prefix + ("└── " if is_tail else "├── ") + str(node.val))
            
            children = []
            if node.left:
                children.append((node.left, False))
            if node.right:
                children.append((node.right, True))
            
            for i, (child, is_last) in enumerate(children):
                extension = "    " if is_tail else "│   "
                _visualize_helper(child, prefix + extension, is_last)
        
        if self.root:
            print(f"Root: {self.root.val}")
            _visualize_helper(self.root)
        else:
            print("Empty tree")
    
    def size(self):
        """
        Get the total number of nodes in the tree.
        
        Returns:
            int: Number of nodes
        
        Time Complexity: O(n)
        """
        def _count_nodes(node):
            if node is None:
                return 0
            return 1 + _count_nodes(node.left) + _count_nodes(node.right)
        
        return _count_nodes(self.root)
    
    def min_value(self, node=None):
        """
        Find the minimum value in the tree.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            The minimum value, or None if tree is empty
        """
        if node is None:
            node = self.root
        
        if node is None:
            return None
        
        while node.left:
            node = node.left
        
        return node.val
    
    def max_value(self, node=None):
        """
        Find the maximum value in the tree.
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            The maximum value, or None if tree is empty
        """
        if node is None:
            node = self.root
        
        if node is None:
            return None
        
        while node.right:
            node = node.right
        
        return node.val


# Demonstration and Examples
if __name__ == "__main__":
    print("=" * 60)
    print("Binary Search Tree - Demonstrations")
    print("=" * 60)
    
    # Example 1: Balanced Tree
    print("\n1. Creating a Balanced Tree")
    print("-" * 60)
    balanced_tree = BinaryTree()
    values = [10, 5, 15, 3, 7, 12, 18]
    
    for val in values:
        balanced_tree.insert(val)
    
    print(f"Inserted values: {values}")
    print("\nTree Structure:")
    balanced_tree.visualize()
    
    print(f"\nHeight: {balanced_tree.height()}")
    print(f"Size: {balanced_tree.size()}")
    print(f"Is Balanced: {balanced_tree.is_balanced()}")
    print(f"Balance Factor (root): {balanced_tree.balance_factor(balanced_tree.root)}")
    
    # Example 2: Tree Traversals
    print("\n" + "=" * 60)
    print("2. Tree Traversals")
    print("-" * 60)
    print(f"Inorder (sorted):   {balanced_tree.inorder()}")
    print(f"Preorder:           {balanced_tree.preorder()}")
    print(f"Postorder:          {balanced_tree.postorder()}")
    print(f"Level-order (BFS):  {balanced_tree.level_order()}")
    print(f"\nLevel-order by levels:")
    for level, nodes in enumerate(balanced_tree.level_order_by_levels()):
        print(f"  Level {level}: {nodes}")
    
    # Example 3: Search Operations
    print("\n" + "=" * 60)
    print("3. Search Operations")
    print("-" * 60)
    search_values = [7, 12, 20, 3]
    for val in search_values:
        found = balanced_tree.search(val)
        print(f"Search {val}: {'Found' if found else 'Not Found'}")
    
    print(f"\nMinimum value: {balanced_tree.min_value()}")
    print(f"Maximum value: {balanced_tree.max_value()}")
    
    # Example 4: Unbalanced Tree
    print("\n" + "=" * 60)
    print("4. Creating an Unbalanced Tree (Skewed)")
    print("-" * 60)
    unbalanced_tree = BinaryTree()
    skewed_values = [10, 15, 20, 25, 30]
    
    for val in skewed_values:
        unbalanced_tree.insert(val)
    
    print(f"Inserted values: {skewed_values}")
    print("\nTree Structure:")
    unbalanced_tree.visualize()
    
    print(f"\nHeight: {unbalanced_tree.height()}")
    print(f"Size: {unbalanced_tree.size()}")
    print(f"Is Balanced: {unbalanced_tree.is_balanced()}")
    print(f"Balance Factor (root): {unbalanced_tree.balance_factor(unbalanced_tree.root)}")
    
    # Example 5: Performance Comparison
    print("\n" + "=" * 60)
    print("5. Performance Comparison")
    print("-" * 60)
    print("\n┌─────────────────────────┬──────────────────┬────────────────────┐")
    print("│ Operation               │ Balanced Tree    │ Unbalanced Tree    │")
    print("├─────────────────────────┼──────────────────┼────────────────────┤")
    print("│ Search (avg)            │ O(log n)         │ O(n)               │")
    print("│ Insert (avg)            │ O(log n)         │ O(n)               │")
    print("│ Height                  │ O(log n)         │ O(n)               │")
    print("├─────────────────────────┼──────────────────┼────────────────────┤")
    print(f"│ Actual Height (this ex) │ {balanced_tree.height():16} │ {unbalanced_tree.height():18} │")
    print("└─────────────────────────┴──────────────────┴────────────────────┘")
    
    # Example 6: Balance Factor Analysis
    print("\n" + "=" * 60)
    print("6. Balance Factor Analysis")
    print("-" * 60)
    
    def print_balance_factors(tree, name):
        print(f"\n{name}:")
        if tree.root:
            def _print_bf(node, prefix=""):
                if node:
                    bf = tree.balance_factor(node)
                    status = "✓" if abs(bf) <= 1 else "✗"
                    print(f"{prefix}Node {node.val}: BF = {bf:2} {status}")
                    _print_bf(node.left, prefix + "  ")
                    _print_bf(node.right, prefix + "  ")
            _print_bf(tree.root)
    
    print_balance_factors(balanced_tree, "Balanced Tree")
    print_balance_factors(unbalanced_tree, "Unbalanced Tree")
    
    print("\n" + "=" * 60)
    print("End of Demonstrations")
    print("=" * 60)
