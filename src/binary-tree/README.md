[← Back to Python Study Repository](../../README.md)

# 🌳 Binary Tree & Balanced Binary Tree

This module provides implementations of binary trees, balanced binary trees, and various tree traversal algorithms with comprehensive examples.

## 🌳 Balanced Binary Tree

![Balanced vs Unbalanced Tree](https://cdn.programiz.com/sites/tutorial2program/files/unbalanced-binary-tree.png)

### 📌 Definition

A **Balanced Binary Tree** is a binary tree where the height difference between the left and right subtree of **every node** is at most **1**.

This difference is called the **balance factor**:

```
Balance Factor = Height(Left) - Height(Right)
```

For a balanced tree:

```
Balance Factor ∈ {-1, 0, +1}
```

---

## 🎯 Why Balanced Trees Matter?

If a binary tree becomes **skewed** (like a linked list), operations degrade:

| Operation | Balanced Tree | Unbalanced Tree |
| --------- | ------------- | --------------- |
| Search    | O(log n)      | O(n)            |
| Insert    | O(log n)      | O(n)            |
| Delete    | O(log n)      | O(n)            |

So balanced trees keep operations **efficient**.

---

## 📚 Types of Balanced Binary Trees

### 1️⃣ AVL Tree

AVL tree:
* Strictly balanced
* Performs rotations after insert/delete
* Faster lookups
* Balance factor: {-1, 0, +1}

### 2️⃣ Red-Black Tree

Red-Black tree:
* Slightly relaxed balancing
* Used in many libraries (e.g., Java TreeMap, C++ map)
* Each node is colored red or black
* Root is always black

### 3️⃣ B-Tree (for databases)

B-tree:
* Used in databases and file systems
* Optimized for disk reads
* Multiple keys per node

---

## 🧠 Examples

### Balanced Tree:

```
        10
       /  \
      5    15
     / \     
    3   7
```

**Balance factors:**
- Node 10: |height(left=2) - height(right=1)| = 1 ✓
- Node 5: |height(left=1) - height(right=1)| = 0 ✓
- Node 15: |height(left=0) - height(right=0)| = 0 ✓

### Unbalanced Tree:

```
    10
      \
       15
         \
          20
```

**Balance factors:**
- Node 10: |height(left=0) - height(right=3)| = 3 ✗
- Node 15: |height(left=0) - height(right=2)| = 2 ✗

---

## 🔁 How Balancing Happens (AVL Example)

If imbalance occurs after insertion:

1. **Left-Left (LL)** → Right Rotation
   ```
       z                y
      /                / \
     y        →       x   z
    /
   x
   ```

2. **Right-Right (RR)** → Left Rotation
   ```
   z                    y
    \                  / \
     y        →       z   x
      \
       x
   ```

3. **Left-Right (LR)** → Left Rotation + Right Rotation
   ```
     z              z              x
    /              /              / \
   y      →       x      →       y   z
    \            /
     x          y
   ```

4. **Right-Left (RL)** → Right Rotation + Left Rotation
   ```
   z            z                x
    \            \              / \
     y    →       x      →     z   y
    /              \
   x                y
   ```

---

## 🌲 Tree Traversal Methods

### 1. Inorder Traversal (Left → Root → Right)
- Visits nodes in sorted order for BST
- **Use case:** Get sorted elements

### 2. Preorder Traversal (Root → Left → Right)
- Visits root before children
- **Use case:** Create a copy of tree, prefix expression

### 3. Postorder Traversal (Left → Right → Root)
- Visits root after children
- **Use case:** Delete tree, postfix expression

### 4. Level-order Traversal (BFS)
- Visits nodes level by level
- **Use case:** Find shortest path, serialize tree

---

## 💻 Algorithms & Complexity

### Check if Tree is Balanced

```python
def isBalanced(root):
    def height(node):
        if not node:
            return 0
        left = height(node.left)
        right = height(node.right)
        if left == -1 or right == -1 or abs(left-right) > 1:
            return -1
        return max(left, right) + 1
    return height(root) != -1
```

**Time Complexity:** O(n) - visits each node once  
**Space Complexity:** O(h) - recursion stack, where h is height

### AVL Tree Operations

| Operation | Time Complexity | Space Complexity |
| --------- | --------------- | ---------------- |
| Insert    | O(log n)        | O(log n)         |
| Delete    | O(log n)        | O(log n)         |
| Search    | O(log n)        | O(1)             |
| Traversal | O(n)            | O(n)             |

---

## 🚀 Usage

### Basic Binary Tree

```python
from binary_tree import TreeNode, BinaryTree

# Create a binary tree
tree = BinaryTree()
tree.insert(10)
tree.insert(5)
tree.insert(15)
tree.insert(3)
tree.insert(7)

# Traversals
print("Inorder:", tree.inorder())      # [3, 5, 7, 10, 15]
print("Preorder:", tree.preorder())    # [10, 5, 3, 7, 15]
print("Postorder:", tree.postorder())  # [3, 7, 5, 15, 10]
print("Level-order:", tree.level_order())  # [10, 5, 15, 3, 7]

# Check if balanced
print("Is balanced:", tree.is_balanced())
```

### AVL Tree (Self-Balancing)

```python
from balanced_tree import AVLTree

# Create an AVL tree
avl = AVLTree()
avl.insert(10)
avl.insert(20)
avl.insert(30)  # This triggers rotation
avl.insert(40)
avl.insert(50)
avl.insert(25)

# AVL tree automatically maintains balance
print("Inorder:", avl.inorder())
print("Is balanced:", avl.is_balanced())  # Always True for AVL
```

---

## 📁 Files

- `binary_tree.py` - Basic binary tree implementation with traversals
- `balanced_tree.py` - AVL Tree and Red-Black Tree implementations
- `examples.py` - Demonstration examples
- `test_binary_tree.py` - Comprehensive unit tests
- `README.md` - This file

---

## 🧪 Running Examples

```bash
cd src/binary-tree
python binary_tree.py
python balanced_tree.py
python examples.py
```

---

## 📖 Learning Resources

1. [Balanced Binary Tree - Programiz](https://www.programiz.com/dsa/balanced-binary-tree)
2. [AVL Tree Visualization](https://www.cs.usfca.edu/~galles/visualization/AVLtree.html)
3. [Red-Black Tree - GeeksforGeeks](https://www.geeksforgeeks.org/red-black-tree-set-1-introduction-2/)
4. [Tree Traversals - LeetCode](https://leetcode.com/explore/learn/card/data-structure-tree/)

---

## 🎓 Key Takeaways

1. **Balanced trees** maintain O(log n) operations
2. **AVL trees** are strictly balanced (balance factor: -1, 0, +1)
3. **Rotations** restore balance after insert/delete
4. **Tree traversals** serve different purposes:
   - Inorder → Sorted order
   - Preorder → Copy tree
   - Postorder → Delete tree
   - Level-order → Shortest path

---

## 🤝 Contributing

Feel free to add more tree implementations (Splay Tree, Treap, etc.) or improve existing ones!

---

## 📝 License

This project is part of the python-study repository and follows the same MIT License.
