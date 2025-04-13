"""
sorting.py

This module provides implementations of various sorting algorithms. 
Each function takes a list of comparable elements and sorts them in ascending order.

Sorting Algorithms:
1. Bubble Sort
2. Insertion Sort
3. Merge Sort
4. Quick Sort
5. Binary Sort
"""

def bubble_sort(data_list):
    """
    Sorts a list using the Bubble Sort algorithm.

    Args:
        data_list (list): The list of elements to be sorted.

    Returns:
        list: The sorted list.
    """
    list_length = len(data_list)
    for i in range(list_length):
        for j in range(0, list_length - i - 1):
            if data_list[j] > data_list[j + 1]:
                data_list[j], data_list[j + 1] = data_list[j + 1], data_list[j]
    return data_list


def insertion_sort(data_list):
    """
    Sorts a list using the Insertion Sort algorithm.

    Args:
        data_list (list): The list of elements to be sorted.

    Returns:
        list: The sorted list.
    """
    for index in range(1, len(data_list)):
        current_element = data_list[index]
        position = index - 1
        while position >= 0 and current_element < data_list[position]:
            data_list[position + 1] = data_list[position]
            position -= 1
        data_list[position + 1] = current_element
    return data_list


def merge_sort(data_list):
    """
    Sorts a list using the Merge Sort algorithm.

    Args:
        data_list (list): The list of elements to be sorted.

    Returns:
        list: The sorted list.
    """
    if len(data_list) > 1:
        middle_index = len(data_list) // 2
        left_half = data_list[:middle_index]
        right_half = data_list[middle_index:]

        # Recursive call on each half
        merge_sort(left_half)
        merge_sort(right_half)

        left_index = right_index = main_index = 0

        # Merge the sorted halves
        while left_index < len(left_half) and right_index < len(right_half):
            if left_half[left_index] < right_half[right_index]:
                data_list[main_index] = left_half[left_index]
                left_index += 1
            else:
                data_list[main_index] = right_half[right_index]
                right_index += 1
            main_index += 1

        # Check for any remaining elements
        while left_index < len(left_half):
            data_list[main_index] = left_half[left_index]
            left_index += 1
            main_index += 1

        while right_index < len(right_half):
            data_list[main_index] = right_half[right_index]
            right_index += 1
            main_index += 1
    return data_list


def quick_sort(data_list):
    """
    Sorts a list using the Quick Sort algorithm.

    Args:
        data_list (list): The list of elements to be sorted.

    Returns:
        list: The sorted list.
    """
    if len(data_list) <= 1:
        return data_list
    else:
        pivot_element = data_list[0]
        smaller_elements = [element for element in data_list[1:] if element <= pivot_element]
        larger_elements = [element for element in data_list[1:] if element > pivot_element]
        return quick_sort(smaller_elements) + [pivot_element] + quick_sort(larger_elements)


def binary_sort(data_list):
    """
    Sorts a list using the Binary Sort algorithm.

    Args:
        data_list (list): The list of elements to be sorted.

    Returns:
        list: The sorted list.

    Note:
        Binary Sort involves inserting each element into a binary search tree (BST) and then performing an in-order traversal of the BST.
    """
    class BinaryTreeNode:
        def __init__(self, value):
            self.value = value
            self.left_child = None
            self.right_child = None

    def insert_into_tree(root, value):
        if root is None:
            return BinaryTreeNode(value)
        if value < root.value:
            root.left_child = insert_into_tree(root.left_child, value)
        else:
            root.right_child = insert_into_tree(root.right_child, value)
        return root

    def in_order_traversal(node, sorted_list):
        if node is not None:
            in_order_traversal(node.left_child, sorted_list)
            sorted_list.append(node.value)
            in_order_traversal(node.right_child, sorted_list)

    # Build the Binary Search Tree (BST)
    root_node = None
    for element in data_list:
        root_node = insert_into_tree(root_node, element)

    # Perform in-order traversal
    sorted_list = []
    in_order_traversal(root_node, sorted_list)
    return sorted_list


# Example usage
if __name__ == "__main__":
    sample_list = [64, 34, 25, 12, 22, 11, 90]
    print("Original list:", sample_list)
    
    print("Bubble Sort:", bubble_sort(sample_list.copy()))
    print("Insertion Sort:", insertion_sort(sample_list.copy()))
    print("Merge Sort:", merge_sort(sample_list.copy()))
    print("Quick Sort:", quick_sort(sample_list.copy()))
    print("Binary Sort:", binary_sort(sample_list.copy()))
