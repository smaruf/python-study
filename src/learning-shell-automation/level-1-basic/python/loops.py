#!/usr/bin/env python3
"""
Level 1 - Basic: Loops in Python
"""

def main():
    print("=== For Loop Example ===")
    # Simple for loop
    for i in [1, 2, 3, 4, 5]:
        print(f"Number: {i}")
    
    print("\n=== For Loop with Range ===")
    # For loop with range
    for i in range(1, 6):
        print(f"Count: {i}")
    
    print("\n=== While Loop Example ===")
    # While loop
    counter = 1
    while counter <= 5:
        print(f"Counter: {counter}")
        counter += 1
    
    print("\n=== Loop through list ===")
    # Loop through list
    languages = ["Python", "Bash", "PowerShell", "Batch"]
    for lang in languages:
        print(f"Language: {lang}")
    
    print("\n=== Loop with enumerate ===")
    # Loop with index
    for index, lang in enumerate(languages, start=1):
        print(f"{index}. {lang}")
    
    print("\n=== List comprehension ===")
    # List comprehension
    squares = [x**2 for x in range(1, 6)]
    print(f"Squares: {squares}")
    
    print("\n=== Loop through dictionary ===")
    # Loop through dictionary
    tools = {"CI": "Jenkins", "CD": "ArgoCD", "Container": "Docker"}
    for key, value in tools.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
