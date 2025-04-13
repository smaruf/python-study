# python interview code problems

def reverse_string(s):
    """
    Reverse a given string.

    Args:
        s (str): The string to reverse.

    Returns:
        str: The reversed string.

    Examples:
        >>> reverse_string("hello")
        'olleh'
    """
    return s[::-1]


def is_palindrome(s):
    """
    Check if a given string is a palindrome.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is a palindrome, False otherwise.

    Examples:
        >>> is_palindrome("radar")
        True
        >>> is_palindrome("hello")
        False
    """
    return s == s[::-1]


def find_duplicates(lst):
    """
    Find duplicate elements in a list.

    Args:
        lst (list): The list to check for duplicates.

    Returns:
        list: A list of duplicate elements.

    Examples:
        >>> find_duplicates([1, 2, 3, 2, 4, 5, 5])
        [2, 5]
    """
    from collections import Counter
    return [item for item, count in Counter(lst).items() if count > 1]


def fibonacci(n):
    """
    Generate the first n Fibonacci numbers.

    Args:
        n (int): The number of Fibonacci numbers to generate.

    Returns:
        list: A list containing the first n Fibonacci numbers.

    Examples:
        >>> fibonacci(5)
        [0, 1, 1, 2, 3]
    """
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[-1] + fib[-2])
    return fib[:n]


def factorial(n):
    """
    Calculate the factorial of a number.

    Args:
        n (int): The number to calculate the factorial of.

    Returns:
        int: The factorial of the number.

    Examples:
        >>> factorial(5)
        120
    """
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)


def is_anagram(s1, s2):
    """
    Check if two strings are anagrams.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        bool: True if the strings are anagrams, False otherwise.

    Examples:
        >>> is_anagram("listen", "silent")
        True
        >>> is_anagram("hello", "world")
        False
    """
    return sorted(s1) == sorted(s2)


def merge_sorted_arrays(arr1, arr2):
    """
    Merge two sorted arrays into one sorted array.

    Args:
        arr1 (list): The first sorted array.
        arr2 (list): The second sorted array.

    Returns:
        list: The merged sorted array.

    Examples:
        >>> merge_sorted_arrays([1, 3, 5], [2, 4, 6])
        [1, 2, 3, 4, 5, 6]
    """
    return sorted(arr1 + arr2)


def find_missing_number(nums):
    """
    Find the missing number in a list of integers from 0 to n.

    Args:
        nums (list): The list of integers.

    Returns:
        int: The missing number.

    Examples:
        >>> find_missing_number([0, 1, 3])
        2
    """
    n = len(nums)
    return n * (n + 1) // 2 - sum(nums)


def max_subarray_sum(nums):
    """
    Find the maximum sum of a contiguous subarray.

    Args:
        nums (list): The list of integers.

    Returns:
        int: The maximum subarray sum.

    Examples:
        >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])
        6
    """
    max_sum = nums[0]
    current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum


def count_vowels(s):
    """
    Count the number of vowels in a string.

    Args:
        s (str): The string to count vowels in.

    Returns:
        int: The number of vowels in the string.

    Examples:
        >>> count_vowels("hello")
        2
    """
    return sum(1 for char in s.lower() if char in "aeiou")


def binary_search(arr, target):
    """
    Perform binary search to find the target in a sorted array.

    Args:
        arr (list): The sorted array.
        target (int): The target value to find.

    Returns:
        int: The index of the target, or -1 if not found.

    Examples:
        >>> binary_search([1, 2, 3, 4, 5], 3)
        2
        >>> binary_search([1, 2, 3, 4, 5], 6)
        -1
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


def is_prime(n):
    """
    Check if a number is prime.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is prime, False otherwise.

    Examples:
        >>> is_prime(7)
        True
        >>> is_prime(10)
        False
    """
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def rotate_matrix(matrix):
    """
    Rotate a matrix 90 degrees clockwise.

    Args:
        matrix (list): A 2D list representing the matrix.

    Returns:
        list: The rotated matrix.

    Examples:
        >>> rotate_matrix([[1, 2], [3, 4]])
        [[3, 1], [4, 2]]
    """
    return [list(row) for row in zip(*matrix[::-1])]


def longest_common_prefix(strs):
    """
    Find the longest common prefix among a list of strings.

    Args:
        strs (list): A list of strings.

    Returns:
        str: The longest common prefix.

    Examples:
        >>> longest_common_prefix(["flower", "flow", "flight"])
        'fl'
        >>> longest_common_prefix(["dog", "car", "race"])
        ''
    """
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def validate_parentheses(s):
    """
    Validate if a string contains valid parentheses.

    Args:
        s (str): The string to validate.

    Returns:
        bool: True if the parentheses are valid, False otherwise.

    Examples:
        >>> validate_parentheses("()")
        True
        >>> validate_parentheses("(]")
        False
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack
