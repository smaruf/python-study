# Diagram Generation by Python Code

This directory contains examples of creating various types of diagrams using Python code, including architecture diagrams, PlantUML diagrams, and 3D visualizations.

## Getting Started

To get started with creating architecture diagrams, you can refer to the following link:
[Diagrams Documentation - Getting Started Examples](https://diagrams.mingrammer.com/docs/getting-started/examples)

## Example

Here is a basic example of how to create a simple architecture diagram using the `diagrams` library in Python:

1. Import the necessary modules from the `diagrams` library.
2. Define the components and their relationships within a `Diagram` context.

Example Code:
- Import the `Diagram` class and components:
```python
  from diagrams import Diagram
  from diagrams.aws.compute import EC2
  from diagrams.aws.network import ELB
  from diagrams.aws.database import RDS
```
- Create the diagram:
```python
  with Diagram("Simple Architecture Diagram", show=False):
    ELB("load balancer") >> EC2("web server") >> RDS("database")
```

## Installation

To install the `diagrams` library, you can use the following command:
- `pip install diagrams`

## Usage

1. Create your Python script with the architecture diagram code.
2. Run the script to generate the diagram image.

For more detailed examples and usage, please visit the [official documentation](https://diagrams.mingrammer.com/docs/getting-started/examples).

## Additional Resources

For creating diagrams with PlantUML, you can refer to the following link:
[PlantUML Documentation](https://plantuml.com/documentation)

### PlantUML Files in this Folder

- [action_diagram.puml](action_diagram.puml)
- [flowcharts_diagram.puml](flowcharts_diagram.puml)
- [state_diagram.puml](state_diagram.puml)
- [class_diagram.puml](class_diagram.puml)
- [component_diagram.puml](component_diagram.puml)
- [deployment_diagram.puml](deployment_diagram.puml)

---

## 3D Diagram Generation (DA3)

### Overview

**DA3 (Data Analytics 3D)** is a custom Python module for creating and printing 3D diagrams. This addresses the question: "What is DA3?"

There is **no standard Python library named `DA3`** in the ecosystem. This custom implementation provides comprehensive 3D visualization capabilities using matplotlib's `mplot3d` toolkit.

### What DA3 Provides

The DA3 module offers various types of 3D visualizations:

| Plot Type           | Description                                    | Function                  |
| ------------------- | ---------------------------------------------- | ------------------------- |
| Surface Plot        | Mathematical function surfaces                 | `surface_plot()`          |
| Parametric Surface  | Parametric surfaces (e.g., torus)              | `parametric_surface()`    |
| Scatter Plot        | 3D scatter points with color mapping           | `scatter_plot()`          |
| Cluster Scatter     | Multiple data clusters visualization           | `cluster_scatter()`       |
| Wireframe Plot      | Mesh-based 3D visualization                    | `wireframe_plot()`        |
| Sphere Wireframe    | Geometric sphere shape                         | `sphere_wireframe()`      |
| Line Plot           | 3D parametric curves (helix)                   | `line_plot()`             |
| Spiral Plot         | Double spiral visualization                    | `spiral_plot()`           |
| Lissajous Curve     | Complex parametric curves                      | `lissajous_curve()`       |

### Installation

Install the required dependencies:

```bash
pip install -r requirements_3d.txt
```

Or install individually:

```bash
pip install matplotlib numpy
```

### Quick Start

#### Basic Usage

```python
from da3 import DA3

# Create DA3 instance
da3 = DA3(output_dir='./my_plots')

# Create individual plots
da3.surface_plot()
da3.scatter_plot()
da3.wireframe_plot()

# Print summary
da3.print_summary()
```

#### Export to 3D Printer Format (STL)

**NEW**: Convert diagrams to STL format for 3D printing!

```python
from da3 import DA3

da3 = DA3(output_dir='./my_models')

# Export individual shapes
da3.export_surface_stl('surface.stl')
da3.export_torus_stl('torus.stl')
da3.export_sphere_stl('sphere.stl')
da3.export_helix_stl('helix.stl')

# Or export all at once
da3.export_all_stl()
```

The STL files can be imported into 3D printing software like:
- **Cura** (free, beginner-friendly)
- **PrusaSlicer** (free, advanced features)
- **Simplify3D** (paid, professional)
- **MeshLab** (free, for viewing/editing)

#### Create All Plots at Once

```python
from da3 import DA3

da3 = DA3(output_dir='./output')
da3.create_all_plots()
```

#### Run Demo

```bash
python da3.py
```

This will generate all available 3D plot types and save them to the `da3_output` directory.

### Individual Plot Modules

Each plot type is also available as a standalone module:

- **[3d_surface_plot.py](3d_surface_plot.py)** - Surface and parametric surface plots
- **[3d_scatter_plot.py](3d_scatter_plot.py)** - Scatter and cluster scatter plots  
- **[3d_wireframe_plot.py](3d_wireframe_plot.py)** - Wireframe and sphere plots
- **[3d_line_plot.py](3d_line_plot.py)** - Line, spiral, and Lissajous curves
- **[3d_printer_export.py](3d_printer_export.py)** - STL export for 3D printing

#### Example: Using Individual Modules

```python
from 3d_surface_plot import create_3d_surface_plot

# Create and save a surface plot
create_3d_surface_plot(save_to_file=True, filename='my_surface.png')
```

#### Example: 3D Printer Export Module

```python
from 3d_printer_export import create_surface_mesh_stl, create_torus_stl

# Export meshes with custom parameters
create_surface_mesh_stl(filename='custom_surface.stl')
create_torus_stl(filename='custom_torus.stl', R=4, r=1.5)
```

For complete examples, see **[3d_printer_examples.py](3d_printer_examples.py)**

### Features

✅ **Multiple Plot Types** - 9 different 3D visualization types  
✅ **Save to File** - Export plots as PNG images (300 DPI)  
✅ **3D Printer Export** - Convert to STL format for 3D printing  
✅ **Customizable Output** - Specify custom output directory and filenames  
✅ **Print Summary** - Track all created plots with timestamps  
✅ **Batch Generation** - Create all plots with a single command  
✅ **Well-Documented** - Comprehensive docstrings and examples

### Example Output

When you run `da3.create_all_plots()`, it generates:

```
Creating all 3D plots...
--------------------------------------------------
Plot saved to ./da3_output/01_surface.png
Plot saved to ./da3_output/02_parametric_surface.png
Plot saved to ./da3_output/03_scatter.png
Plot saved to ./da3_output/04_cluster_scatter.png
Plot saved to ./da3_output/05_wireframe.png
Plot saved to ./da3_output/06_sphere.png
Plot saved to ./da3_output/07_line.png
Plot saved to ./da3_output/08_spiral.png
Plot saved to ./da3_output/09_lissajous.png
--------------------------------------------------
All plots created successfully!

==================================================
3D DIAGRAM GENERATION SUMMARY
==================================================
Total plots created: 9
Output directory: ./da3_output
...
==================================================
```

### Common Use Cases

#### Data Visualization
- Visualizing mathematical functions in 3D
- Plotting scientific data with multiple dimensions
- Creating educational materials for mathematics/physics

#### Data Analysis
- Clustering analysis with 3D scatter plots
- Surface fitting and interpolation
- Trajectory and path visualization

#### Presentation Graphics
- High-quality 3D plots for reports and presentations
- Custom parametric curves and surfaces
- Professional wireframe visualizations

### Technical Details

- **Library**: matplotlib with mpl_toolkits.mplot3d
- **Backend**: Agg (for file output) or interactive (for display)
- **Image Format**: PNG with 300 DPI resolution
- **Color Maps**: viridis, plasma, rainbow, coolwarm
- **Customization**: All plots support matplotlib customization options

### Troubleshooting

If you encounter issues with the module imports, run the plots directly:

```bash
# Run individual plot scripts
python 3d_surface_plot.py
python 3d_scatter_plot.py
python 3d_wireframe_plot.py
python 3d_line_plot.py

# Test 3D printer export
python 3d_printer_export.py

# Run comprehensive examples
python 3d_printer_examples.py
```

---

## 3D Printer Export

### Overview

The DA3 library now includes **STL export functionality** to convert 3D diagrams into formats compatible with 3D printers. This feature allows you to physically print mathematical surfaces, geometric shapes, and parametric curves.

### Installation for 3D Printing

```bash
pip install numpy-stl
```

Or install all dependencies at once:

```bash
pip install -r requirements_3d.txt
```

### Supported Shapes for 3D Printing

| Shape               | Description                           | Method                  |
| ------------------- | ------------------------------------- | ----------------------- |
| Surface Mesh        | Mathematical function surfaces        | `export_surface_stl()`  |
| Torus               | Donut shape (parametric surface)      | `export_torus_stl()`    |
| Sphere              | Geometric sphere                      | `export_sphere_stl()`   |
| Helix Tube          | Spring/coil shape                     | `export_helix_stl()`    |

### Quick Start - 3D Printing

```python
from da3 import DA3

da3 = DA3(output_dir='./3d_prints')

# Export individual shapes
da3.export_surface_stl('wave_surface.stl')
da3.export_torus_stl('donut.stl')
da3.export_sphere_stl('ball.stl')
da3.export_helix_stl('spring.stl')

# Or export all shapes at once
da3.export_all_stl()
```

### 3D Printing Workflow

1. **Generate STL Files**
   ```python
   from da3 import DA3
   da3 = DA3(output_dir='./models')
   da3.export_all_stl()
   ```

2. **Open in Slicing Software**
   - Cura (free, recommended for beginners)
   - PrusaSlicer (free, advanced features)
   - Simplify3D (paid, professional)

3. **Configure Print Settings**
   - Layer height: 0.1-0.2mm
   - Infill: 10-20%
   - Support: Auto-generate if needed
   - Print speed: 50-60 mm/s

4. **Slice and Print**
   - Generate G-code
   - Transfer to printer (SD card or USB)
   - Start printing!

### File Format Details

- **Format**: STL (Stereolithography)
- **Units**: Centimeters (cm)
- **Mesh Type**: Manifold (watertight, printable)
- **Resolution**: Configurable triangle count

### Customizing Print Parameters

```python
from 3d_printer_export import create_torus_stl, create_sphere_stl

# Custom torus (R=major radius, r=minor radius in cm)
create_torus_stl(filename='large_torus.stl', R=5, r=2)

# Custom sphere (radius in cm)
create_sphere_stl(filename='small_sphere.stl', radius=1.5)

# Custom helix tube
from 3d_printer_export import create_helix_tube_stl
create_helix_tube_stl(
    filename='custom_spring.stl',
    tube_radius=0.3,    # Tube thickness (cm)
    helix_radius=2.5,   # Spring diameter (cm)
    height=10           # Spring height (cm)
)
```

### Example Files

See **[3d_printer_examples.py](3d_printer_examples.py)** for comprehensive examples including:
- Basic STL export
- Batch export
- Combined visualization + STL workflow
- Custom parameter usage
- 3D printing instructions

### Printing Tips

✓ **Scale models** in your slicer if they're too large/small  
✓ **Add supports** for overhangs greater than 45 degrees  
✓ **Use PLA filament** for beginners (easy to print)  
✓ **Test small prints** first before large models  
✓ **Check mesh integrity** with repair tools in slicer  

### Viewing STL Files

Before printing, you can view/edit STL files in:
- **MeshLab** (free, cross-platform)
- **Blender** (free, powerful)
- **Windows 3D Viewer** (built into Windows 10/11)
- **Online**: viewstl.com, 3dviewer.net

### Related Libraries

For more advanced 3D visualization, consider:

- **Plotly** - Interactive 3D plots in web browsers
- **Mayavi** - Advanced 3D scientific visualization
- **PyVista** - 3D plotting and mesh analysis
- **VTK** - Visualization toolkit for scientific computing

