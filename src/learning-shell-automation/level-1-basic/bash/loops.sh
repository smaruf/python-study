#!/bin/bash
# Level 1 - Basic: Loops in Bash

echo "=== For Loop Example ==="
# Simple for loop
for i in 1 2 3 4 5; do
    echo "Number: $i"
done

echo ""
echo "=== For Loop with Range ==="
# For loop with range
for i in {1..5}; do
    echo "Count: $i"
done

echo ""
echo "=== While Loop Example ==="
# While loop
counter=1
while [ $counter -le 5 ]; do
    echo "Counter: $counter"
    ((counter++))
done

echo ""
echo "=== Loop through files ==="
# Loop through files in current directory
for file in *.sh; do
    echo "Script found: $file"
done
