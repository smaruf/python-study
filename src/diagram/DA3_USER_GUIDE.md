# DA3 Complete User Guide

## Three Ways to Use DA3

DA3 provides three interfaces for creating visualizations and physical outputs:

1. **Script-Based** - Direct Python API usage
2. **CLI-Based** - Command-line interface
3. **GUI-Based** - Graphical user interface

---

## 1. Script-Based Usage

### Basic Python API

```python
from da3 import DA3

# Initialize
da3 = DA3(output_dir='./my_outputs')

# Create visualizations (PNG)
da3.surface_plot()
da3.scatter_plot()
da3.create_all_plots()  # Create all at once

# Export to 2D plotter (SVG)
da3.export_contour_svg()
da3.export_parametric_curve_svg(curve_type='spiral')
da3.export_all_svg()

# Export to 3D printer (STL)
da3.export_surface_stl()
da3.export_torus_stl()
da3.export_all_stl()

# View summary
da3.print_summary()
```

### Complete Workflow Example

```python
from da3 import DA3

# Create DA3 instance
da3 = DA3(output_dir='./complete_project')

# Step 1: Visual data (PNG images)
print("Step 1: Creating visualizations...")
da3.surface_plot('visual_surface.png')
da3.scatter_plot('visual_scatter.png')

# Step 2: 2D plotter export (SVG for cutting/plotting)
print("Step 2: Exporting to 2D plotter format...")
da3.export_contour_svg('plotter_contour.svg')
da3.export_parametric_curve_svg('plotter_spiral.svg', curve_type='spiral')

# Step 3: 3D printer export (STL for printing)
print("Step 3: Exporting to 3D printer format...")
da3.export_surface_stl('printer_surface.stl')
da3.export_torus_stl('printer_torus.stl')

# Summary of all created files
da3.print_summary()
```

### Advanced: Custom Parameters

```python
# Direct module access for custom parameters
from 3d_printer_export import create_torus_stl, create_sphere_stl

# Custom torus (major radius=5cm, minor radius=2cm)
create_torus_stl(filename='large_torus.stl', R=5, r=2)

# Custom sphere (radius=3cm)
create_sphere_stl(filename='large_sphere.stl', radius=3)
```

---

## 2. CLI-Based Usage

### Installation

```bash
# Make CLI executable
chmod +x da3_cli.py
```

### Basic Commands

```bash
# View all options
python da3_cli.py --help

# Create a single surface plot
python da3_cli.py --surface --output plots/

# Create all visualizations
python da3_cli.py --all --output plots/

# Export to 2D plotter (SVG)
python da3_cli.py --export-svg --contour --output plotter/

# Export to 3D printer (STL)
python da3_cli.py --export-stl --torus --output printer/

# Complete workflow
python da3_cli.py --complete-workflow --output complete/
```

### Detailed Examples

#### Creating Individual Plots

```bash
# Surface plot
python da3_cli.py --surface -o plots/ -v

# Multiple plots at once
python da3_cli.py --surface --scatter --wireframe -o plots/ -v

# All visualizations
python da3_cli.py --all -o plots/ --summary
```

#### 2D Plotter Export (SVG)

```bash
# Single contour
python da3_cli.py --export-svg --contour -o plotter/ -v

# Multiple patterns
python da3_cli.py --export-svg --svg-spiral --pattern-hexagon -o plotter/

# Custom text
python da3_cli.py --export-svg --text "Hello World" --text-size 96 -o plotter/

# All SVG patterns
python da3_cli.py --all-svg -o plotter/ --summary
```

#### 3D Printer Export (STL)

```bash
# Single model
python da3_cli.py --export-stl --torus -o printer/ -v

# Multiple models
python da3_cli.py --export-stl --stl-surface --helix -o printer/

# All STL models
python da3_cli.py --all-stl -o printer/ --summary

# Custom parameters
python da3_cli.py --export-stl --torus --torus-R 4 --torus-r 1.5 -o custom/
```

#### Complete Workflow

```bash
# One command to create everything
python da3_cli.py --complete-workflow -o complete_project/ -v --summary
```

### CLI Options Reference

#### Output Options
- `-o, --output` - Output directory (default: ./output)
- `-v, --verbose` - Enable verbose output
- `--summary` - Print summary of created files

#### Visualization Options (PNG)
- `--surface` - Surface plot
- `--parametric-surface` - Parametric surface (torus)
- `--scatter` - Scatter plot
- `--cluster-scatter` - Cluster scatter
- `--wireframe` - Wireframe plot
- `--sphere` - Sphere wireframe
- `--line` - Line plot (helix)
- `--spiral` - Spiral plot
- `--lissajous` - Lissajous curve
- `--all` - Create all visualizations

#### 2D Plotter Export (SVG)
- `--export-svg` - Enable SVG export
- `--contour` - Contour plot
- `--svg-spiral` - Spiral curve
- `--svg-lissajous` - Lissajous curve
- `--pattern-grid` - Grid pattern
- `--pattern-hexagon` - Hexagon pattern
- `--text TEXT` - Export text
- `--all-svg` - Export all SVG patterns

#### 3D Printer Export (STL)
- `--export-stl` - Enable STL export
- `--stl-surface` - Surface mesh
- `--torus` - Torus
- `--stl-sphere` - Sphere
- `--helix` - Helix tube
- `--all-stl` - Export all STL models

#### Custom Parameters
- `--torus-R` - Torus major radius (default: 3)
- `--torus-r` - Torus minor radius (default: 1)
- `--sphere-radius` - Sphere radius (default: 2)
- `--text-size` - Text font size (default: 72)

---

## 3. GUI-Based Usage

### Starting the GUI

```bash
# Launch GUI application
python da3_gui.py
```

### GUI Features

#### Tab 1: 3D Visualizations
- **Plot Type Buttons**: Click to create individual plot types
- **Create All Plots**: Generate all 9 visualization types at once
- **Output Log**: View real-time progress and status
- **Clear Output**: Clear the log window

#### Tab 2: 2D Plotter (SVG)
- **Export Buttons**: One click to export each pattern type
- **Text Export**: Enter custom text and export as SVG
- **Export All SVG**: Generate all 2D plotter patterns
- **Information Panel**: Shows compatible hardware and use cases

#### Tab 3: 3D Printer (STL)
- **Model Buttons**: Export individual 3D printable models
- **Parameter Controls**: Customize model dimensions
- **Export All STL**: Generate all 3D printer models
- **Information Panel**: Printing tips and workflow guidance

#### Tab 4: Settings
- **Output Directory**: Choose where files are saved
- **About DA3**: Version and project information
- **View Summary**: See all created files with timestamps

### GUI Workflow

1. **Set Output Directory** (Settings tab)
   - Click "Browse..." to choose a folder
   - Click "Apply" to save the setting

2. **Create Visualizations** (3D Visualizations tab)
   - Click individual plot buttons OR
   - Click "Create All Plots" for batch generation
   - Watch progress in the output log

3. **Export for 2D Plotter** (2D Plotter tab)
   - Select pattern types to export
   - Customize text if needed
   - Click export buttons
   - Files saved as SVG format

4. **Export for 3D Printer** (3D Printer tab)
   - Select models to export
   - Adjust parameters if needed
   - Click export buttons
   - Files saved as STL format

5. **View Summary** (Settings tab)
   - Click "View Summary of Created Files"
   - See complete list with timestamps

### GUI Screenshots

The GUI provides an intuitive interface with:
- **Organized tabs** for different export types
- **One-click operations** for all common tasks
- **Real-time feedback** in the output log
- **Information panels** with usage tips
- **Batch operations** for efficiency

---

## Output Formats

### PNG - Visual Data
- **Resolution**: 300 DPI
- **Use**: Reports, presentations, documentation
- **Software**: Any image viewer

### SVG - 2D Plotter/Cutter
- **Format**: Scalable vector graphics
- **Use**: Vinyl cutting, laser engraving, pen plotting
- **Software**: Inkscape, Adobe Illustrator, Cricut, Silhouette
- **Features**: Infinite scalability, no quality loss

### STL - 3D Printer
- **Format**: Stereolithography (triangular mesh)
- **Use**: 3D printing, CNC machining
- **Software**: Cura, PrusaSlicer, Simplify3D, MeshLab
- **Features**: Manifold meshes, watertight, ready to print

---

## Complete Workflow Examples

### Example 1: Quick Visualization

**Script:**
```python
from da3 import DA3
da3 = DA3(output_dir='./quick')
da3.surface_plot()
```

**CLI:**
```bash
python da3_cli.py --surface -o quick/
```

**GUI:**
1. Open GUI
2. Click "Surface Plot" button
3. Done!

### Example 2: 2D Plotter Project

**Script:**
```python
from da3 import DA3
da3 = DA3(output_dir='./plotter_project')
da3.export_all_svg()
```

**CLI:**
```bash
python da3_cli.py --all-svg -o plotter_project/
```

**GUI:**
1. Open GUI
2. Go to "2D Plotter (SVG)" tab
3. Click "Export All SVG"

### Example 3: 3D Printing Project

**Script:**
```python
from da3 import DA3
da3 = DA3(output_dir='./printer_project')
da3.export_all_stl()
```

**CLI:**
```bash
python da3_cli.py --all-stl -o printer_project/
```

**GUI:**
1. Open GUI
2. Go to "3D Printer (STL)" tab
3. Click "Export All STL"

### Example 4: Complete Project (All Formats)

**Script:**
```python
from da3 import DA3
da3 = DA3(output_dir='./complete')
da3.create_all_plots()
da3.export_all_svg()
da3.export_all_stl()
da3.print_summary()
```

**CLI:**
```bash
python da3_cli.py --complete-workflow -o complete/ --summary
```

**GUI:**
1. Open GUI
2. Create all plots (Tab 1)
3. Export all SVG (Tab 2)
4. Export all STL (Tab 3)
5. View summary (Tab 4)

---

## Troubleshooting

### Import Errors

```bash
# Install dependencies
pip install matplotlib numpy
pip install numpy-stl  # For 3D printing
```

### CLI Not Found

```bash
# Make executable
chmod +x da3_cli.py

# Or run with python
python da3_cli.py --help
```

### GUI Won't Start

```bash
# Check Tkinter installation (usually comes with Python)
python -m tkinter

# If not available, install:
# Ubuntu/Debian: sudo apt-get install python3-tk
# macOS: Comes with Python
# Windows: Comes with Python
```

### 3D Printer Export Fails

```bash
# Install numpy-stl
pip install numpy-stl

# Verify installation
python -c "import stl; print('STL support available')"
```

---

## Tips and Best Practices

### For Script Users
- Use batch operations (`create_all_plots()`, `export_all_svg()`) for efficiency
- Set output directory at initialization
- Check `da3.created_plots` to see what's been created
- Use custom modules for advanced parameter control

### For CLI Users
- Use `--verbose` to see detailed progress
- Combine multiple operations in one command
- Use `--summary` to verify all files created
- Redirect output to file: `python da3_cli.py --all > log.txt`

### For GUI Users
- Set output directory before creating files
- Use "Create All" buttons for batch operations
- Check output log for real-time status
- View summary periodically to track progress
- Keep GUI open while generating (runs in background)

---

## Support and Documentation

- **Main README**: [README.md](README.md)
- **Examples**: 
  - `physical_output_examples.py` - Complete workflow
  - `3d_printer_examples.py` - 3D printing specifics
- **Module Documentation**: See individual `.py` files
- **Project Summary**: [DA3_PROJECT_SUMMARY.md](DA3_PROJECT_SUMMARY.md)

---

## Quick Reference

| Task | Script | CLI | GUI |
|------|--------|-----|-----|
| Single plot | `da3.surface_plot()` | `--surface` | Click button |
| All plots | `da3.create_all_plots()` | `--all` | "Create All Plots" |
| 2D export | `da3.export_contour_svg()` | `--export-svg --contour` | Click SVG button |
| 3D export | `da3.export_surface_stl()` | `--export-stl --stl-surface` | Click STL button |
| Workflow | Call all methods | `--complete-workflow` | Use all tabs |
| Summary | `da3.print_summary()` | `--summary` | "View Summary" |

---

**DA3 - Data Analytics 3D**  
*Complete visualization and physical output solution*
