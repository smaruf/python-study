#!/bin/bash
# Level 1 - Basic: Variables and Conditionals in Bash

# Variables
name="DevOps Engineer"
age=25
is_learning=true

# String variable
echo "Role: $name"
echo "Age: $age"

# Conditional statements
if [ $age -ge 18 ]; then
    echo "You are an adult"
else
    echo "You are a minor"
fi

# Check if learning
if [ "$is_learning" = "true" ]; then
    echo "Keep learning and growing!"
fi

# Multiple conditions
if [ $age -ge 18 ] && [ "$is_learning" = "true" ]; then
    echo "Perfect! Adult learner on the path to DevOps mastery!"
fi
