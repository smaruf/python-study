"""
examples.py

Comprehensive examples and use cases for binary trees and balanced trees.
This file demonstrates practical applications and comparisons.
"""

from binary_tree import BinaryTree, TreeNode
from balanced_tree import AVLTree, RedBlackTree


def example_basic_operations():
    """Demonstrate basic binary tree operations."""
    print("=" * 70)
    print("Example 1: Basic Binary Tree Operations")
    print("=" * 70)
    
    tree = BinaryTree()
    
    # Insert values
    values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 65]
    print(f"\n1. Inserting values: {values}")
    for val in values:
        tree.insert(val)
    
    # Display tree
    print("\nTree Structure:")
    tree.visualize()
    
    # Search operations
    print("\n2. Search Operations:")
    search_vals = [25, 35, 100]
    for val in search_vals:
        result = tree.search(val)
        print(f"   Search for {val}: {'Found ✓' if result else 'Not found ✗'}")
    
    # Tree properties
    print("\n3. Tree Properties:")
    print(f"   Height: {tree.height()}")
    print(f"   Size: {tree.size()} nodes")
    print(f"   Min value: {tree.min_value()}")
    print(f"   Max value: {tree.max_value()}")
    print(f"   Is balanced: {tree.is_balanced()}")
    
    # Traversals
    print("\n4. Tree Traversals:")
    print(f"   Inorder:    {tree.inorder()}")
    print(f"   Preorder:   {tree.preorder()}")
    print(f"   Postorder:  {tree.postorder()}")
    print(f"   Level-order: {tree.level_order()}")
    
    print()


def example_balanced_vs_unbalanced():
    """Compare balanced and unbalanced trees."""
    print("=" * 70)
    print("Example 2: Balanced vs Unbalanced Trees")
    print("=" * 70)
    
    # Create a balanced tree
    balanced = BinaryTree()
    balanced_vals = [50, 25, 75, 12, 37, 62, 87, 6, 18, 31, 43]
    for val in balanced_vals:
        balanced.insert(val)
    
    # Create an unbalanced tree (right-skewed)
    unbalanced = BinaryTree()
    unbalanced_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
    for val in unbalanced_vals:
        unbalanced.insert(val)
    
    print("\n1. Balanced Tree:")
    print(f"   Values: {balanced_vals}")
    balanced.visualize()
    print(f"   Height: {balanced.height()}")
    print(f"   Is balanced: {balanced.is_balanced()}")
    
    print("\n2. Unbalanced Tree (Right-skewed):")
    print(f"   Values: {unbalanced_vals}")
    unbalanced.visualize()
    print(f"   Height: {unbalanced.height()}")
    print(f"   Is balanced: {unbalanced.is_balanced()}")
    
    # Performance implications
    print("\n3. Performance Impact:")
    print("   ┌─────────────────┬──────────┬────────────┐")
    print("   │ Metric          │ Balanced │ Unbalanced │")
    print("   ├─────────────────┼──────────┼────────────┤")
    print(f"   │ Height          │ {balanced.height():8} │ {unbalanced.height():10} │")
    print(f"   │ Size            │ {balanced.size():8} │ {unbalanced.size():10} │")
    print(f"   │ Is Balanced     │ {'Yes':8} │ {'No':10} │")
    print("   └─────────────────┴──────────┴────────────┘")
    
    print("\n   In the worst case (unbalanced tree):")
    print(f"   - Search might need to check all {unbalanced.size()} nodes")
    print(f"   - Time complexity degrades from O(log n) to O(n)")
    print()


def example_avl_tree_rotations():
    """Demonstrate AVL tree rotations."""
    print("=" * 70)
    print("Example 3: AVL Tree Rotations")
    print("=" * 70)
    
    # Left-Left case (Right Rotation)
    print("\n1. Left-Left Case (triggers Right Rotation):")
    print("   Inserting: 30, 20, 10")
    avl1 = AVLTree()
    for val in [30, 20, 10]:
        avl1.insert(val)
    avl1.visualize()
    print(f"   Result: Balanced tree with height {avl1.height()}")
    
    # Right-Right case (Left Rotation)
    print("\n2. Right-Right Case (triggers Left Rotation):")
    print("   Inserting: 10, 20, 30")
    avl2 = AVLTree()
    for val in [10, 20, 30]:
        avl2.insert(val)
    avl2.visualize()
    print(f"   Result: Balanced tree with height {avl2.height()}")
    
    # Left-Right case
    print("\n3. Left-Right Case (triggers Left-Right Rotation):")
    print("   Inserting: 30, 10, 20")
    avl3 = AVLTree()
    for val in [30, 10, 20]:
        avl3.insert(val)
    avl3.visualize()
    print(f"   Result: Balanced tree with height {avl3.height()}")
    
    # Right-Left case
    print("\n4. Right-Left Case (triggers Right-Left Rotation):")
    print("   Inserting: 10, 30, 20")
    avl4 = AVLTree()
    for val in [10, 30, 20]:
        avl4.insert(val)
    avl4.visualize()
    print(f"   Result: Balanced tree with height {avl4.height()}")
    
    print()


def example_avl_maintains_balance():
    """Show how AVL tree maintains balance with many insertions."""
    print("=" * 70)
    print("Example 4: AVL Tree Automatically Maintains Balance")
    print("=" * 70)
    
    print("\nInserting sequential values: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10")
    print("(This would create a completely unbalanced tree without AVL)")
    
    avl = AVLTree()
    for i in range(1, 11):
        avl.insert(i)
    
    print("\nResulting AVL Tree:")
    avl.visualize()
    
    print(f"\nTree Properties:")
    print(f"   Size: {avl.size()} nodes")
    print(f"   Height: {avl.height()}")
    print(f"   Is balanced: {avl.is_balanced()}")
    print(f"   Theoretical minimum height: 3 (for 10 nodes)")
    print(f"   Actual height: {avl.height()}")
    
    # Compare with unbalanced
    regular = BinaryTree()
    for i in range(1, 11):
        regular.insert(i)
    
    print(f"\nComparison:")
    print(f"   Without AVL balancing, height would be: {regular.height()}")
    print(f"   With AVL balancing, height is: {avl.height()}")
    print(f"   Height reduction: {regular.height() - avl.height()} levels")
    
    print()


def example_tree_traversal_use_cases():
    """Demonstrate practical use cases for different traversals."""
    print("=" * 70)
    print("Example 5: Tree Traversal Use Cases")
    print("=" * 70)
    
    tree = BinaryTree()
    values = [50, 30, 70, 20, 40, 60, 80]
    for val in values:
        tree.insert(val)
    
    print("\nTree Structure:")
    tree.visualize()
    
    print("\n1. Inorder Traversal (Left → Root → Right):")
    print(f"   Result: {tree.inorder()}")
    print("   Use case: Get elements in sorted order")
    print("   → Perfect for getting sorted list from BST")
    
    print("\n2. Preorder Traversal (Root → Left → Right):")
    print(f"   Result: {tree.preorder()}")
    print("   Use case: Create a copy of the tree, prefix expressions")
    print("   → Root comes first, useful for serialization")
    
    print("\n3. Postorder Traversal (Left → Right → Root):")
    print(f"   Result: {tree.postorder()}")
    print("   Use case: Delete tree, postfix expressions, calculate sizes")
    print("   → Children processed before parent")
    
    print("\n4. Level-order Traversal (BFS):")
    print(f"   Result: {tree.level_order()}")
    print("   Use case: Find shortest path, level-wise processing")
    levels = tree.level_order_by_levels()
    for i, level in enumerate(levels):
        print(f"   Level {i}: {level}")
    
    print()


def example_balance_factor_analysis():
    """Analyze balance factors in different tree configurations."""
    print("=" * 70)
    print("Example 6: Balance Factor Analysis")
    print("=" * 70)
    
    def print_tree_with_balance_factors(tree, name):
        print(f"\n{name}:")
        tree.visualize()
        
        def analyze_node(node, path="root"):
            if node:
                bf = tree.balance_factor(node)
                status = "✓ OK" if abs(bf) <= 1 else "✗ UNBALANCED"
                h_left = tree.height(node.left) if node.left else 0
                h_right = tree.height(node.right) if node.right else 0
                print(f"   {path:20} → Node {node.val:3}: BF = {bf:2} (L:{h_left}, R:{h_right}) {status}")
                
                if node.left:
                    analyze_node(node.left, f"{path}/left")
                if node.right:
                    analyze_node(node.right, f"{path}/right")
        
        if tree.root:
            analyze_node(tree.root)
    
    # Perfectly balanced
    balanced = BinaryTree()
    for val in [50, 25, 75, 12, 37, 62, 87]:
        balanced.insert(val)
    print_tree_with_balance_factors(balanced, "1. Perfectly Balanced Tree")
    
    # Left-heavy but still balanced
    left_heavy = BinaryTree()
    for val in [50, 25, 75, 12, 37, 6]:
        left_heavy.insert(val)
    print_tree_with_balance_factors(left_heavy, "2. Left-Heavy but Balanced")
    
    # Unbalanced
    unbalanced = BinaryTree()
    for val in [50, 40, 30, 20, 10]:
        unbalanced.insert(val)
    print_tree_with_balance_factors(unbalanced, "3. Unbalanced Tree")
    
    print()


def example_avl_vs_regular_bst():
    """Compare AVL tree with regular BST for worst-case scenario."""
    print("=" * 70)
    print("Example 7: AVL Tree vs Regular BST (Worst Case)")
    print("=" * 70)
    
    # Insert sorted data (worst case for regular BST)
    values = list(range(1, 16))  # 1 to 15
    
    print(f"\nInserting sorted values: {values}")
    print("This is the WORST CASE for regular BST (becomes linked list)")
    
    # Regular BST
    regular = BinaryTree()
    for val in values:
        regular.insert(val)
    
    # AVL Tree
    avl = AVLTree()
    for val in values:
        avl.insert(val)
    
    print("\n1. Regular BST (Unbalanced):")
    print(f"   Height: {regular.height()}")
    print(f"   Is balanced: {regular.is_balanced()}")
    print(f"   Search complexity: O({regular.height()}) ≈ O(n)")
    
    print("\n2. AVL Tree (Self-Balancing):")
    avl.visualize()
    print(f"   Height: {avl.height()}")
    print(f"   Is balanced: {avl.is_balanced()}")
    print(f"   Search complexity: O({avl.height()}) ≈ O(log n)")
    
    print("\n3. Performance Comparison:")
    print("   ┌────────────────────┬─────────────┬───────────┐")
    print("   │ Metric             │ Regular BST │ AVL Tree  │")
    print("   ├────────────────────┼─────────────┼───────────┤")
    print(f"   │ Height             │ {regular.height():11} │ {avl.height():9} │")
    print(f"   │ Balanced           │ {'No':11} │ {'Yes':9} │")
    print(f"   │ Search worst case  │ {'O(n)':11} │ {'O(log n)':9} │")
    print("   └────────────────────┴─────────────┴───────────┘")
    
    print(f"\n   Height reduction: {regular.height() - avl.height()} levels")
    print(f"   This means AVL reduces operations by ~{100 * (1 - avl.height() / regular.height()):.0f}%!")
    
    print()


def example_red_black_tree():
    """Demonstrate Red-Black tree properties."""
    print("=" * 70)
    print("Example 8: Red-Black Tree")
    print("=" * 70)
    
    rb_tree = RedBlackTree()
    values = [10, 20, 30, 15, 25, 5, 1, 35, 40]
    
    print(f"\nInserting values: {values}")
    for val in values:
        rb_tree.insert(val)
    
    print("\nRed-Black Tree Structure:")
    print("🔴 = RED node, ⚫ = BLACK node")
    rb_tree.visualize()
    
    print(f"\nInorder traversal (sorted): {rb_tree.inorder()}")
    
    print("\nRed-Black Tree Properties:")
    print("   1. Every node is either RED or BLACK ✓")
    print("   2. The root is BLACK ✓")
    print("   3. All leaves (NIL) are BLACK ✓")
    print("   4. RED nodes have BLACK children ✓")
    print("   5. All paths have same number of BLACK nodes ✓")
    
    print("\nKey Differences from AVL:")
    print("   • AVL: Strictly balanced (balance factor ≤ 1)")
    print("   • RB:  Loosely balanced (red-black properties)")
    print("   • AVL: Better for search-heavy workloads")
    print("   • RB:  Better for insert-heavy workloads")
    print("   • RB:  Used in: Java TreeMap, C++ std::map, Linux kernel")
    
    print()


def run_all_examples():
    """Run all examples in sequence."""
    examples = [
        example_basic_operations,
        example_balanced_vs_unbalanced,
        example_avl_tree_rotations,
        example_avl_maintains_balance,
        example_tree_traversal_use_cases,
        example_balance_factor_analysis,
        example_avl_vs_regular_bst,
        example_red_black_tree,
    ]
    
    for i, example in enumerate(examples, 1):
        try:
            example()
            if i < len(examples):
                input("Press Enter to continue to next example...")
                print("\n" * 2)
        except KeyboardInterrupt:
            print("\n\nExamples interrupted by user.")
            break
        except Exception as e:
            print(f"\nError in example: {e}")
            continue
    
    print("=" * 70)
    print("All Examples Completed!")
    print("=" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "BINARY TREE COMPREHENSIVE EXAMPLES" + " " * 19 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\nThis script demonstrates various binary tree concepts and operations.")
    print("Press Ctrl+C at any time to exit.\n")
    
    run_all_examples()
