#!/usr/bin/env python3
"""
Level 1 - Basic: Functions in Python
"""

import os
import platform

def greet():
    """Simple function"""
    print("Hello from a function!")

def greet_user(name, role):
    """Function with parameters"""
    print(f"Hello, {name}! Welcome to {role} learning.")

def add_numbers(num1, num2):
    """Function with return value"""
    return num1 + num2

def check_directory(path):
    """Function to check if directory exists"""
    if os.path.exists(path):
        print(f"Directory {path} exists")
        return True
    else:
        print(f"Directory {path} does not exist")
        return False

def get_system_info():
    """Function with multiple return values"""
    return {
        'hostname': platform.node(),
        'os': platform.system(),
        'version': platform.version(),
        'python_version': platform.python_version()
    }

def calculate_stats(*numbers):
    """Function with variable arguments"""
    if not numbers:
        return None
    
    return {
        'sum': sum(numbers),
        'average': sum(numbers) / len(numbers),
        'min': min(numbers),
        'max': max(numbers)
    }

def create_user(username, email, **kwargs):
    """Function with keyword arguments"""
    user = {
        'username': username,
        'email': email
    }
    user.update(kwargs)
    return user

def main():
    print("=== Calling Functions ===")
    greet()
    print()
    
    greet_user("DevOps Engineer", "Automation")
    print()
    
    result = add_numbers(10, 20)
    print(f"10 + 20 = {result}")
    print()
    
    print("Checking directories:")
    check_directory("/tmp")
    check_directory("/nonexistent")
    print()
    
    print("System Information:")
    info = get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()
    
    print("Statistics:")
    stats = calculate_stats(10, 20, 30, 40, 50)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    
    print("Create user:")
    user = create_user("johndoe", "john@example.com", role="Admin", active=True)
    print(f"  User: {user}")

if __name__ == "__main__":
    main()
