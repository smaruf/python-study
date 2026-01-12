#!/bin/bash
# Run all examples (analysis only - no STL generation)

echo "=========================================="
echo "Running Remote Aircraft Examples"
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "1. Weight and Center of Gravity Calculator"
echo "=========================================="
python examples/weight_calc.py
echo ""
echo ""

echo "2. Stress Analysis"
echo "=========================================="
python examples/stress_analysis.py
echo ""
echo ""

echo "3. Motor Mount Generator"
echo "=========================================="
python examples/generate_motor_mounts.py
echo ""
echo ""

echo "=========================================="
echo "All examples completed!"
echo "=========================================="
