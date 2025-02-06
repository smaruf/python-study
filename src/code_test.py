# Imports
from itertools import permutations

# Sorting Algorithms
class Sorting:
    
    @staticmethod
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            swapped = False
            for j in range(0, n-i-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    swapped = True
            if not swapped:
                break
    
    @staticmethod
    def insertion_sort(arr):
        for i in range(1, len(arr)):
            key = arr[i]
            j = i-1
            while j >=0 and key < arr[j]:
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = key

    @staticmethod
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        else:
            pivot = arr.pop()
            greater = [x for x in arr if x > pivot]
            lesser = [x for x in arr if x <= pivot]
            return Sorting.quick_sort(lesser) + [pivot] + Sorting.quick_sort(greater)

# Matrix Manipulations
class MatrixManipulations:
    
    @staticmethod
    def rotate_matrix(matrix):
        n = len(matrix)
        for layer in range(n // 2):
            first, last = layer, n-layer-1
            for i in range(first, last):
                top = matrix[layer][i]
                matrix[layer][i] = matrix[-i-1][layer]
                matrix[-i-1][layer] = matrix[-layer-1][-i-1]
                matrix[-layer-1][-i-1] = matrix[i][-layer-1]
                matrix[i][-layer-1] = top

    @staticmethod
    def spiral_order(matrix):
        return matrix and [*matrix.pop(0)] + MatrixManipulations.spiral_order([*zip(*matrix)][::-1])

# String Manipulations
class StringManipulations:
    
    @staticmethod
    def reverse_string(s):
        return s[::-1]
    
    @staticmethod
    def is_palindrome(s):
        cleaned = ''.join(filter(str.isalnum, s.lower()))
        return cleaned == cleaned[::-1]
    
    @staticmethod
    def string_compression(s):
        compressed = []
        count = 1
        for i in range(1, len(s)+1):
            if i < len(s) and s[i] == s[i-1]:
                count += 1
            else:
                compressed.append(s[i-1] + (str(count) if count > 1 else ''))
                count = 1
        return ''.join(compressed)
    
    @staticmethod
    def check_anagram(s1, s2):
        return sorted(s1) == sorted(s2)
    
    @staticmethod
    def permutations_string(s):
        return [''.join(p) for p in permutations(s)]

# Searching Algorithms
class Searching:
    
    @staticmethod
    def binary_search(arr, target):
        low, high = 0, len(arr) - 1
        while low <= high:
            mid = (low + high) // 2
            if arr[mid] < target:
                low = mid + 1
            elif arr[mid] > target:
                high = mid - 1
            else:
                return mid
        return -1

# Set Manipulations
class SetManipulations:
    
    @staticmethod
    def union(s1, s2):
        return s1 | s2
    
    @staticmethod
    def intersection(s1, s2):
        return s1 & s2
    
    @staticmethod
    def difference(s1, s2):
        return s1 - s2
    
    @staticmethod
    def symmetric_difference(s1, s2):
        return s1 ^ s2

# Testing
if __name__ == "__main__":
    # Functions can be tested by invoking them here with example input.
    pass
