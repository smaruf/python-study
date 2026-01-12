# Examples

This directory contains practical examples demonstrating the use of the remote-aircraft CAD system.

## Running Examples

All examples can be run without installing CadQuery (though CadQuery is needed to generate actual STL files).

### Weight and CG Calculator

Calculate component weights and center of gravity:

```bash
cd /home/runner/work/python-study/python-study/remote-aircraft
PYTHONPATH=. python examples/weight_calc.py
```

**Output:**
- Frame component weights
- Total weight calculation
- Center of gravity position
- CG analysis and recommendations
- Thrust-to-weight ratio

### Stress Analysis

Analyze arm stress under various loads:

```bash
PYTHONPATH=. python examples/stress_analysis.py
```

**Output:**
- Bending stress calculations
- Material safety factors
- Recommendations by flight style
- Hollow vs solid arm comparison

### Motor Mount Generator

Generate motor mounts for various motor sizes:

```bash
PYTHONPATH=. python examples/generate_motor_mounts.py
```

**Output:**
- Motor mounts for 1507, 2204, 2306, 2806 motors
- STL files (if CadQuery installed)
- Specifications for each mount

**Note:** This example requires CadQuery to generate STL files. Without CadQuery, it will show what would be generated.

## Installation

To run examples that generate STL files, install CadQuery:

**Option 1: Using conda (recommended)**
```bash
conda create -n cadquery
conda activate cadquery
conda install -c conda-forge -c cadquery cadquery
pip install numpy
```

**Option 2: Using CQ-Editor**
Download from: https://github.com/CadQuery/CQ-editor/releases

## Example Output

All generated STL files will be saved to the `output/` directory.
