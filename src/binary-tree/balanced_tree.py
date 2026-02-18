"""
balanced_tree.py

Implementation of self-balancing binary search trees, including AVL Tree
and conceptual Red-Black Tree.

AVL Tree: A strictly balanced BST where the balance factor of every node is -1, 0, or +1.
It uses rotations to maintain balance after insertions and deletions.

Features:
- AVL Tree with automatic balancing
- Left and Right rotations
- Balance factor calculation
- All standard BST operations with O(log n) complexity
"""

from collections import deque


class AVLNode:
    """
    Node for an AVL Tree.
    
    Attributes:
        val: The value stored in the node
        left: Reference to the left child
        right: Reference to the right child
        height: The height of the node
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.height = 1  # New node has height 1
    
    def __repr__(self):
        return f"AVLNode({self.val}, h={self.height})"


class AVLTree:
    """
    AVL Tree implementation - a self-balancing binary search tree.
    
    The AVL tree maintains the following properties:
    1. It is a valid BST
    2. For every node, the height difference between left and right subtrees is at most 1
    3. After each insertion/deletion, the tree is rebalanced using rotations
    """
    
    def __init__(self):
        """Initialize an empty AVL tree."""
        self.root = None
    
    def _get_height(self, node):
        """
        Get the height of a node.
        
        Args:
            node: The node to get height from
        
        Returns:
            int: Height of the node (0 if None)
        """
        if node is None:
            return 0
        return node.height
    
    def _get_balance_factor(self, node):
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
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _update_height(self, node):
        """
        Update the height of a node based on its children.
        
        Args:
            node: The node to update
        """
        if node is not None:
            node.height = 1 + max(self._get_height(node.left), 
                                   self._get_height(node.right))
    
    def _rotate_right(self, z):
        """
        Perform a right rotation on node z.
        
        Before:          After:
            z              y
           /              / \
          y              x   z
         /
        x
        
        Args:
            z: The node to rotate
        
        Returns:
            AVLNode: The new root after rotation
        """
        y = z.left
        T3 = y.right
        
        # Perform rotation
        y.right = z
        z.left = T3
        
        # Update heights
        self._update_height(z)
        self._update_height(y)
        
        return y
    
    def _rotate_left(self, z):
        """
        Perform a left rotation on node z.
        
        Before:      After:
          z            y
           \          / \
            y        z   x
             \
              x
        
        Args:
            z: The node to rotate
        
        Returns:
            AVLNode: The new root after rotation
        """
        y = z.right
        T2 = y.left
        
        # Perform rotation
        y.left = z
        z.right = T2
        
        # Update heights
        self._update_height(z)
        self._update_height(y)
        
        return y
    
    def insert(self, val):
        """
        Insert a value into the AVL tree and maintain balance.
        
        Args:
            val: The value to insert
        
        Time Complexity: O(log n)
        """
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        """
        Helper method to recursively insert a value and balance the tree.
        
        Args:
            node: Current node
            val: Value to insert
        
        Returns:
            AVLNode: The (possibly new) root of the subtree
        """
        # 1. Perform normal BST insertion
        if node is None:
            return AVLNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        else:
            # Duplicate values not allowed
            return node
        
        # 2. Update height of current node
        self._update_height(node)
        
        # 3. Get the balance factor
        balance = self._get_balance_factor(node)
        
        # 4. If node is unbalanced, there are 4 cases
        
        # Left-Left Case
        if balance > 1 and val < node.left.val:
            return self._rotate_right(node)
        
        # Right-Right Case
        if balance < -1 and val > node.right.val:
            return self._rotate_left(node)
        
        # Left-Right Case
        if balance > 1 and val > node.left.val:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        
        # Right-Left Case
        if balance < -1 and val < node.right.val:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        
        return node
    
    def search(self, val):
        """
        Search for a value in the tree.
        
        Args:
            val: The value to search for
        
        Returns:
            bool: True if the value exists, False otherwise
        
        Time Complexity: O(log n)
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
    
    def inorder(self, node=None):
        """
        Perform inorder traversal (Left → Root → Right).
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            list: List of values in sorted order
        
        Time Complexity: O(n)
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
        
        Args:
            node: The node to start from (default: root)
        
        Returns:
            list: List of values in preorder
        
        Time Complexity: O(n)
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
    
    def level_order(self):
        """
        Perform level-order traversal (BFS).
        
        Returns:
            list: List of values in level order
        
        Time Complexity: O(n)
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
    
    def is_balanced(self):
        """
        Check if the tree is balanced.
        For AVL tree, this should always return True.
        
        Returns:
            bool: True if balanced (always True for AVL)
        
        Time Complexity: O(n)
        """
        def check_balance(node):
            if node is None:
                return True
            
            balance = self._get_balance_factor(node)
            if abs(balance) > 1:
                return False
            
            return check_balance(node.left) and check_balance(node.right)
        
        return check_balance(self.root)
    
    def height(self):
        """
        Get the height of the tree.
        
        Returns:
            int: Height of the tree
        """
        return self._get_height(self.root)
    
    def visualize(self):
        """
        Print a simple visualization of the tree with heights and balance factors.
        """
        def _visualize_helper(node, prefix="", is_tail=True):
            if node is None:
                return
            
            bf = self._get_balance_factor(node)
            print(prefix + ("└── " if is_tail else "├── ") + 
                  f"{node.val} (h={node.height}, bf={bf})")
            
            children = []
            if node.left:
                children.append((node.left, False))
            if node.right:
                children.append((node.right, True))
            
            for i, (child, is_last) in enumerate(children):
                extension = "    " if is_tail else "│   "
                _visualize_helper(child, prefix + extension, is_last)
        
        if self.root:
            print(f"AVL Tree Root: {self.root.val} (h={self.root.height})")
            _visualize_helper(self.root)
        else:
            print("Empty tree")
    
    def size(self):
        """
        Get the total number of nodes in the tree.
        
        Returns:
            int: Number of nodes
        """
        def _count_nodes(node):
            if node is None:
                return 0
            return 1 + _count_nodes(node.left) + _count_nodes(node.right)
        
        return _count_nodes(self.root)


class RedBlackNode:
    """
    Node for a Red-Black Tree.
    
    Attributes:
        val: The value stored in the node
        color: 'RED' or 'BLACK'
        left: Reference to the left child
        right: Reference to the right child
        parent: Reference to the parent node
    """
    def __init__(self, val=0, color='RED', left=None, right=None, parent=None):
        self.val = val
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent
    
    def __repr__(self):
        return f"RBNode({self.val}, {self.color})"


class RedBlackTree:
    """
    Red-Black Tree implementation (conceptual/educational version).
    
    Red-Black Tree Properties:
    1. Every node is either RED or BLACK
    2. The root is always BLACK
    3. All leaves (NIL) are BLACK
    4. If a node is RED, then both its children are BLACK
    5. Every path from a node to its descendant NIL nodes has the same number of BLACK nodes
    
    Note: This is a simplified educational implementation.
    A full production implementation would include deletion and more complex balancing.
    """
    
    def __init__(self):
        """Initialize an empty Red-Black tree."""
        self.NIL = RedBlackNode(val=None, color='BLACK')
        self.root = self.NIL
    
    def insert(self, val):
        """
        Insert a value into the Red-Black tree.
        
        Args:
            val: The value to insert
        
        Time Complexity: O(log n)
        """
        new_node = RedBlackNode(val, color='RED', left=self.NIL, right=self.NIL)
        
        parent = None
        current = self.root
        
        # Find the position to insert
        while current != self.NIL:
            parent = current
            if new_node.val < current.val:
                current = current.left
            else:
                current = current.right
        
        new_node.parent = parent
        
        if parent is None:
            self.root = new_node
        elif new_node.val < parent.val:
            parent.left = new_node
        else:
            parent.right = new_node
        
        # Fix Red-Black Tree properties
        self._fix_insert(new_node)
    
    def _fix_insert(self, node):
        """
        Fix Red-Black Tree properties after insertion.
        
        Args:
            node: The newly inserted node
        """
        while node.parent and node.parent.color == 'RED':
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right
                if uncle.color == 'RED':
                    # Case 1: Uncle is RED
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.right:
                        # Case 2: Node is right child
                        node = node.parent
                        self._rotate_left(node)
                    # Case 3: Node is left child
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._rotate_right(node.parent.parent)
            else:
                uncle = node.parent.parent.left
                if uncle.color == 'RED':
                    node.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._rotate_right(node)
                    node.parent.color = 'BLACK'
                    node.parent.parent.color = 'RED'
                    self._rotate_left(node.parent.parent)
        
        self.root.color = 'BLACK'
    
    def _rotate_left(self, node):
        """Perform left rotation."""
        y = node.right
        node.right = y.left
        if y.left != self.NIL:
            y.left.parent = node
        y.parent = node.parent
        if node.parent is None:
            self.root = y
        elif node == node.parent.left:
            node.parent.left = y
        else:
            node.parent.right = y
        y.left = node
        node.parent = y
    
    def _rotate_right(self, node):
        """Perform right rotation."""
        y = node.left
        node.left = y.right
        if y.right != self.NIL:
            y.right.parent = node
        y.parent = node.parent
        if node.parent is None:
            self.root = y
        elif node == node.parent.right:
            node.parent.right = y
        else:
            node.parent.left = y
        y.right = node
        node.parent = y
    
    def inorder(self):
        """
        Perform inorder traversal.
        
        Returns:
            list: List of values in sorted order
        """
        result = []
        
        def _inorder_helper(node):
            if node != self.NIL:
                _inorder_helper(node.left)
                result.append(node.val)
                _inorder_helper(node.right)
        
        _inorder_helper(self.root)
        return result
    
    def visualize(self):
        """
        Print a simple visualization of the Red-Black tree.
        """
        def _visualize_helper(node, prefix="", is_tail=True):
            if node == self.NIL:
                return
            
            color_symbol = "🔴" if node.color == 'RED' else "⚫"
            print(prefix + ("└── " if is_tail else "├── ") + 
                  f"{color_symbol} {node.val}")
            
            children = []
            if node.left != self.NIL:
                children.append((node.left, False))
            if node.right != self.NIL:
                children.append((node.right, True))
            
            for i, (child, is_last) in enumerate(children):
                extension = "    " if is_tail else "│   "
                _visualize_helper(child, prefix + extension, is_last)
        
        if self.root != self.NIL:
            color_symbol = "🔴" if self.root.color == 'RED' else "⚫"
            print(f"Red-Black Tree Root: {color_symbol} {self.root.val}")
            _visualize_helper(self.root)
        else:
            print("Empty tree")


# Demonstration and Examples
if __name__ == "__main__":
    print("=" * 60)
    print("AVL Tree - Self-Balancing Binary Search Tree")
    print("=" * 60)
    
    # Example 1: Build AVL Tree with Sequential Insertions
    print("\n1. Building AVL Tree with Sequential Insertions")
    print("-" * 60)
    avl = AVLTree()
    
    # Insert values that would create an unbalanced tree without rotations
    values = [10, 20, 30, 40, 50, 25]
    print(f"Inserting values: {values}")
    
    for val in values:
        print(f"\nInserting {val}...")
        avl.insert(val)
        avl.visualize()
        print(f"Height: {avl.height()}, Balanced: {avl.is_balanced()}")
    
    # Example 2: AVL Tree Properties
    print("\n" + "=" * 60)
    print("2. AVL Tree Properties")
    print("-" * 60)
    print(f"Size: {avl.size()} nodes")
    print(f"Height: {avl.height()}")
    print(f"Is Balanced: {avl.is_balanced()} (always True for AVL)")
    print(f"Inorder (sorted): {avl.inorder()}")
    print(f"Preorder: {avl.preorder()}")
    print(f"Level-order: {avl.level_order()}")
    
    # Example 3: Compare with Unbalanced Tree
    print("\n" + "=" * 60)
    print("3. Comparison: AVL vs Unbalanced BST")
    print("-" * 60)
    
    # The same values would create a right-skewed tree without balancing
    print("\nWithout AVL balancing (right-skewed):")
    print("10 → 20 → 30 → 40 → 50 → 25")
    print("Height would be: 6")
    print("\nWith AVL balancing:")
    print(f"Height is: {avl.height()}")
    print(f"Reduction: {6 - avl.height()} levels")
    
    # Example 4: Red-Black Tree
    print("\n" + "=" * 60)
    print("4. Red-Black Tree Demonstration")
    print("-" * 60)
    rb_tree = RedBlackTree()
    
    rb_values = [10, 20, 30, 15, 25, 5]
    print(f"Inserting values: {rb_values}")
    
    for val in rb_values:
        rb_tree.insert(val)
    
    print("\nRed-Black Tree Structure:")
    rb_tree.visualize()
    print(f"\nInorder: {rb_tree.inorder()}")
    
    # Example 5: Performance Summary
    print("\n" + "=" * 60)
    print("5. Performance Summary")
    print("-" * 60)
    print("\n┌──────────────────────┬────────────┬────────────┬────────────┐")
    print("│ Operation            │ AVL Tree   │ RB Tree    │ Unbalanced │")
    print("├──────────────────────┼────────────┼────────────┼────────────┤")
    print("│ Search               │ O(log n)   │ O(log n)   │ O(n)       │")
    print("│ Insert               │ O(log n)   │ O(log n)   │ O(n)       │")
    print("│ Delete               │ O(log n)   │ O(log n)   │ O(n)       │")
    print("├──────────────────────┼────────────┼────────────┼────────────┤")
    print("│ Rotations (insert)   │ ≤ 2        │ ≤ 2        │ 0          │")
    print("│ Rotations (delete)   │ O(log n)   │ ≤ 3        │ 0          │")
    print("├──────────────────────┼────────────┼────────────┼────────────┤")
    print("│ Balancing            │ Strict     │ Relaxed    │ None       │")
    print("│ Best for             │ Lookups    │ Insertions │ -          │")
    print("└──────────────────────┴────────────┴────────────┴────────────┘")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("-" * 60)
    print("• AVL trees are STRICTLY balanced (balance factor: -1, 0, +1)")
    print("• Red-Black trees are LOOSELY balanced (faster insertions)")
    print("• Both guarantee O(log n) operations")
    print("• Rotations automatically maintain balance")
    print("• AVL is better for search-heavy workloads")
    print("• Red-Black is better for insert-heavy workloads")
    print("=" * 60)
