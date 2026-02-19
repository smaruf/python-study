#!/bin/bash
echo "=========================================="
echo "Wind Tunnel Simulation Demo"
echo "=========================================="
echo ""
echo "Scenario 1: Small Trainer (Beginner-friendly)"
echo "------------------------------------------"
python aircraft_designer_cli.py -w 1000 -c 200 --weight 1200 --cruise 12 2>&1 | head -35

echo ""
echo ""
echo "Scenario 2: High-Performance Glider"
echo "------------------------------------------"
python aircraft_designer_cli.py -w 1500 -c 150 --weight 900 --cruise 15 2>&1 | head -35

echo ""
echo ""
echo "Scenario 3: Sport Aerobatic Plane"
echo "------------------------------------------"
python aircraft_designer_cli.py -w 900 -c 180 --weight 1500 --cruise 20 2>&1 | head -35
