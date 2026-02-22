# DA3 GUI Interface Guide

## GUI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  DA3 - Data Analytics 3D Designer                               │
├────────────┬────────────┬────────────┬────────────┬─────────────┤
│ 3D Viz     │ 2D Plotter │ 3D Printer │  Settings  │             │
│ (Active)   │   (SVG)    │   (STL)    │            │             │
├────────────┴────────────┴────────────┴────────────┴─────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌────────────────────────────────────┐  │
│  │  Plot Types      │  │  Output Log                        │  │
│  │                  │  │                                    │  │
│  │ [Surface Plot   ]│  │  [Creating surface plot]          │  │
│  │ [Parametric...]  │  │  Plot saved to ./output/...       │  │
│  │ [Scatter Plot  ] │  │  ✓ Surface plot created           │  │
│  │ [Cluster...]     │  │                                    │  │
│  │ [Wireframe]      │  │  [Creating scatter plot]          │  │
│  │ [Sphere]         │  │  Plot saved to ./output/...       │  │
│  │ [Line Plot]      │  │  ✓ Scatter plot created           │  │
│  │ [Spiral]         │  │                                    │  │
│  │ [Lissajous]      │  │                                    │  │
│  │                  │  │                                    │  │
│  │ ─────────────────│  │                                    │  │
│  │ [Create All]    ]│  │                                    │  │
│  │ [Clear Output]   │  │                                    │  │
│  └──────────────────┘  └────────────────────────────────────┘  │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  Status: Ready                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Tab 1: 3D Visualizations

**Purpose**: Create PNG visualization files

**Controls**:
- 9 plot type buttons (Surface, Parametric, Scatter, etc.)
- Create All Plots button (batch operation)
- Clear Output button

**Output**: High-resolution PNG images (300 DPI)

**Workflow**:
1. Click plot type button
2. Watch progress in output log
3. Files saved to configured directory

---

## Tab 2: 2D Plotter (SVG)

```
┌──────────────────┐  ┌────────────────────────────────────┐
│  2D Exports      │  │  Information                       │
│                  │  │                                    │
│ [Contour Plot   ]│  │  SVG Export for 2D Plotters       │
│ [Spiral Curve]   │  │                                    │
│ [Lissajous...]   │  │  Compatible with:                 │
│ [Grid Pattern]   │  │  • Inkscape                       │
│ [Hexagon...]     │  │  • Vinyl cutters                  │
│                  │  │  • Laser cutters                  │
│  Text Export:    │  │  • Pen plotters                   │
│  [DA3        ]   │  │                                    │
│  [Export Text]   │  │  Vector graphics - scalable       │
│                  │  │  without quality loss.            │
│ ─────────────────│  │                                    │
│ [Export All SVG] │  │  Use cases:                       │
│                  │  │  - Stickers, decals               │
└──────────────────┘  │  - Technical drawings             │
                      │  - Custom signage                 │
                      └────────────────────────────────────┘
```

**Purpose**: Export vector graphics for 2D plotters/cutters

**Controls**:
- 5 pattern export buttons
- Text entry field for custom text
- Export Text button
- Export All SVG button (batch)

**Output**: Scalable Vector Graphics (SVG) files

---

## Tab 3: 3D Printer (STL)

```
┌──────────────────┐  ┌────────────────────────────────────┐
│  3D Models       │  │  Information                       │
│                  │  │                                    │
│ [Surface Mesh]   │  │  STL Export for 3D Printing       │
│ [Torus (Donut)]  │  │                                    │
│ [Sphere]         │  │  Compatible with:                 │
│ [Helix (Spring)] │  │  • Cura (free slicer)            │
│                  │  │  • PrusaSlicer                    │
│  Parameters:     │  │  • Simplify3D                     │
│                  │  │  • MeshLab                        │
│ ─────────────────│  │                                    │
│ [Export All STL] │  │  Manifold meshes ready for        │
│                  │  │  3D printing.                     │
└──────────────────┘  │                                    │
                      │  Workflow:                         │
                      │  1. Export STL here               │
                      │  2. Open in slicer                │
                      │  3. Configure settings            │
                      │  4. Generate G-code               │
                      │  5. Print!                        │
                      └────────────────────────────────────┘
```

**Purpose**: Export 3D models for printing

**Controls**:
- 4 model export buttons
- Parameter controls (future enhancement)
- Export All STL button (batch)

**Output**: STL (Stereolithography) files

---

## Tab 4: Settings

```
┌──────────────────────────────────────────────────────────────┐
│  Output Directory:                                           │
│  [./da3_output                    ] [Browse...] [Apply]      │
│                                                              │
│  ──────────────────────────────────────────────────────────  │
│                                                              │
│  About DA3                                                   │
│  DA3 - Data Analytics 3D Designer                           │
│  Version: 1.0.0                                             │
│                                                              │
│  A comprehensive tool for creating:                         │
│  • 3D visualizations (PNG)                                  │
│  • 2D plotter outputs (SVG)                                 │
│  • 3D printer models (STL)                                  │
│                                                              │
│  [View Summary of Created Files]                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Purpose**: Configuration and information

**Controls**:
- Output directory selector
- Browse button (opens file dialog)
- Apply button (saves setting)
- View Summary button

**Features**:
- Change output location
- View project information
- See summary of all created files

---

## Common Operations

### Create a Single Visualization
1. Go to "3D Visualizations" tab
2. Click desired plot button
3. Wait for "✓ Plot created" message
4. File saved automatically

### Export for 2D Plotter
1. Go to "2D Plotter (SVG)" tab
2. Click export button for desired pattern
3. File created as SVG in output directory
4. Open in Inkscape or plotter software

### Export for 3D Printer
1. Go to "3D Printer (STL)" tab
2. Click export button for desired model
3. File created as STL in output directory
4. Open in Cura or other slicer

### Batch Operations
1. Navigate to appropriate tab
2. Click "Create All", "Export All SVG", or "Export All STL"
3. Watch progress in output log
4. All files created automatically

### Change Output Directory
1. Go to "Settings" tab
2. Click "Browse..." button
3. Select folder
4. Click "Apply"
5. New files will save to new location

### View Summary
1. Go to "Settings" tab
2. Click "View Summary of Created Files"
3. New window shows all created files with:
   - File type
   - Full path
   - Creation timestamp

---

## Tips

### Performance
- Creating all plots takes 10-30 seconds depending on system
- GUI remains responsive during operations
- Status bar shows current operation

### File Management
- Set output directory before creating files
- Use descriptive directory names
- Check output log for file locations

### Troubleshooting
- If button doesn't respond, check output log for errors
- For 3D printer export, install: `pip install numpy-stl`
- For best results, close other heavy applications

---

## Keyboard Shortcuts

Currently, the GUI uses mouse/click interactions. Keyboard shortcuts may be added in future versions.

---

## Example Workflow

**Complete Project in 60 Seconds**:

1. Launch GUI: `python da3_gui.py`
2. Tab 1: Click "Create All Plots" (10-30 sec)
3. Tab 2: Click "Export All SVG" (5-10 sec)
4. Tab 3: Click "Export All STL" (10-20 sec) *
5. Tab 4: Click "View Summary" to see results

\* Requires numpy-stl library

**Result**: Complete set of visualizations, 2D plotter files, and 3D printer models in under a minute!

---

## Getting Help

- **User Guide**: See DA3_USER_GUIDE.md
- **README**: See README.md
- **Examples**: Run example scripts in the directory

---

**DA3 GUI v1.0.0** - Making 3D visualization and physical output simple and accessible.
