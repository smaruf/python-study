#!/usr/bin/env python3
"""
Level 0 - Beginner: Basic User Input in Python
"""

def main():
    name = input("What is your name? ")
    print(f"Hello, {name}! Nice to meet you.")
    
    from datetime import datetime
    print(f"Today is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
