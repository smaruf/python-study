#!/bin/bash
# Level 1 - Basic: Functions in Bash

# Simple function
greet() {
    echo "Hello from a function!"
}

# Function with parameters
greet_user() {
    local name=$1
    local role=$2
    echo "Hello, $name! Welcome to $role learning."
}

# Function with return value
add_numbers() {
    local num1=$1
    local num2=$2
    local sum=$((num1 + num2))
    echo $sum
}

# Function to check if service is running
check_service() {
    local service_name=$1
    if systemctl is-active --quiet $service_name 2>/dev/null; then
        echo "Service $service_name is running"
        return 0
    else
        echo "Service $service_name is not running"
        return 1
    fi
}

# Call functions
echo "=== Calling Functions ==="
greet
echo ""

greet_user "DevOps Engineer" "Automation"
echo ""

result=$(add_numbers 10 20)
echo "10 + 20 = $result"
echo ""

# Note: systemctl check will work on Linux systems with systemd
echo "Checking services (example):"
check_service "ssh" || echo "SSH service check failed or not available"
