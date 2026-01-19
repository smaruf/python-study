#!/usr/bin/env python3
"""
Level 1 - Basic: Variables and Conditionals in Python
"""

def main():
    # Variables
    name = "DevOps Engineer"
    age = 25
    is_learning = True
    
    # Display variables
    print(f"Role: {name}")
    print(f"Age: {age}")
    
    # Conditional statements
    if age >= 18:
        print("You are an adult")
    else:
        print("You are a minor")
    
    # Check if learning
    if is_learning:
        print("Keep learning and growing!")
    
    # Multiple conditions
    if age >= 18 and is_learning:
        print("Perfect! Adult learner on the path to DevOps mastery!")
    
    # Elif statement
    environment = "production"
    if environment == "production":
        print("Running in production mode")
    elif environment == "development":
        print("Running in development mode")
    else:
        print("Unknown environment")
    
    # Ternary operator
    status = "Active" if is_learning else "Inactive"
    print(f"Learning Status: {status}")

if __name__ == "__main__":
    main()
