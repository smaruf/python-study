# Implementation of various search algorithms in Python

def binary_search(arr, target):
    """
    Perform binary search on a sorted array to find the target element.

    Args:
        arr (list): A sorted list of elements to search.
        target (any): The element to search for.

    Returns:
        int: The index of the target element if found, otherwise -1.
    """
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


from collections import deque

def bfs(graph, start, target):
    """
    Perform Breadth-First Search (BFS) on a graph to find the target node.

    Args:
        graph (dict): A dictionary representing the adjacency list of the graph.
        start (any): The starting node for BFS.
        target (any): The node to search for.

    Returns:
        bool: True if the target node is found, otherwise False.
    """
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node] - visited)
    return False


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        """
        Node for a binary tree.

        Args:
            val (int): The value of the node.
            left (TreeNode): The left child of the node.
            right (TreeNode): The right child of the node.
        """
        self.val = val
        self.left = left
        self.right = right


def binary_tree_search(root, target):
    """
    Perform search in a binary search tree to find the target value.

    Args:
        root (TreeNode): The root of the binary search tree.
        target (int): The value to search for.

    Returns:
        bool: True if the target value is found, otherwise False.
    """
    if root is None:
        return False
    if root.val == target:
        return True
    elif target < root.val:
        return binary_tree_search(root.left, target)
    else:
        return binary_tree_search(root.right, target)


def dfs(graph, start, target, visited=None):
    """
    Perform Depth-First Search (DFS) on a graph to find the target node.

    Args:
        graph (dict): A dictionary representing the adjacency list of the graph.
        start (any): The starting node for DFS.
        target (any): The node to search for.
        visited (set): A set of visited nodes (used for recursion).

    Returns:
        bool: True if the target node is found, otherwise False.
    """
    if visited is None:
        visited = set()
    if start == target:
        return True
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            if dfs(graph, neighbor, target, visited):
                return True
    return False


# Test data and execution of algorithms
if __name__ == "__main__":
    # Binary Search Test
    sorted_array = [1, 3, 5, 7, 9, 11]
    print("Binary Search Test")
    print(binary_search(sorted_array, 5))  # Output: 2
    print(binary_search(sorted_array, 12))  # Output: -1

    # Breadth-First Search Test
    graph = {
        'A': {'B', 'C'},
        'B': {'A', 'D', 'E'},
        'C': {'A', 'F'},
        'D': {'B'},
        'E': {'B', 'F'},
        'F': {'C', 'E'}
    }
    print("\nBreadth-First Search Test")
    print(bfs(graph, 'A', 'F'))  # Output: True
    print(bfs(graph, 'A', 'Z'))  # Output: False

    # Binary Tree Search Test
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(15)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(7)
    root.right.left = TreeNode(12)
    root.right.right = TreeNode(18)
    print("\nBinary Tree Search Test")
    print(binary_tree_search(root, 7))  # Output: True
    print(binary_tree_search(root, 20))  # Output: False

    # Depth-First Search Test
    print("\nDepth-First Search Test")
    print(dfs(graph, 'A', 'F'))  # Output: True
    print(dfs(graph, 'A', 'Z'))  # Output: False
