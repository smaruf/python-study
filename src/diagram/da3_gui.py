#!/usr/bin/env python3
"""
DA3 Graphical User Interface
A user-friendly GUI for generating 3D visualizations, 2D plotter outputs, and 3D printer files.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import sys
import os
import threading
import matplotlib
matplotlib.use('TkAgg')  # Tk-compatible backend
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from da3 import DA3


class DA3GUI:
    """DA3 Graphical User Interface Application."""
    
    def __init__(self, root):
        """Initialize the GUI application."""
        self.root = root
        self.root.title("DA3 - Data Analytics 3D Designer")
        self.root.geometry("1200x800")
        
        # Initialize DA3 instance
        self.output_dir = "./da3_output"
        self.da3 = DA3(output_dir=self.output_dir)
        
        # Create the UI
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 1: Visualizations
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text='3D Visualizations')
        self.create_visualization_tab()
        
        # Tab 2: 2D Plotter Export
        self.plotter_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plotter_frame, text='2D Plotter (SVG)')
        self.create_plotter_tab()
        
        # Tab 3: 3D Printer Export
        self.printer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.printer_frame, text='3D Printer (STL)')
        self.create_printer_tab()
        
        # Tab 4: Settings
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text='Settings')
        self.create_settings_tab()
        
        # Status bar at bottom
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                    relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_visualization_tab(self):
        """Create the 3D visualization tab."""
        # Left panel: controls
        control_frame = ttk.LabelFrame(self.viz_frame, text="Plot Types", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # Plot type buttons
        self.viz_buttons = []
        plots = [
            ("Surface Plot", self.create_surface),
            ("Parametric Surface", self.create_parametric),
            ("Scatter Plot", self.create_scatter),
            ("Cluster Scatter", self.create_cluster),
            ("Wireframe", self.create_wireframe),
            ("Sphere", self.create_sphere),
            ("Line Plot (Helix)", self.create_line),
            ("Spiral", self.create_spiral),
            ("Lissajous Curve", self.create_lissajous),
        ]
        
        for label, command in plots:
            btn = ttk.Button(control_frame, text=label, command=command, width=20)
            btn.pack(pady=3)
            self.viz_buttons.append(btn)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Batch operations
        ttk.Button(control_frame, text="Create All Plots", 
                  command=self.create_all, width=20).pack(pady=3)
        ttk.Button(control_frame, text="Clear Output", 
                  command=self.clear_output, width=20).pack(pady=3)
        
        # Right panel: log output
        log_frame = ttk.LabelFrame(self.viz_frame, text="Output Log", padding=10)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                  height=30, width=60)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def create_plotter_tab(self):
        """Create the 2D plotter export tab."""
        # Controls
        control_frame = ttk.LabelFrame(self.plotter_frame, text="2D Plotter Exports (SVG)", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        plots = [
            ("Contour Plot", lambda: self.export_svg('contour')),
            ("Spiral Curve", lambda: self.export_svg('spiral')),
            ("Lissajous Curve", lambda: self.export_svg('lissajous')),
            ("Grid Pattern", lambda: self.export_svg('grid')),
            ("Hexagon Pattern", lambda: self.export_svg('hexagon')),
        ]
        
        for label, command in plots:
            ttk.Button(control_frame, text=label, command=command, width=20).pack(pady=3)
        
        # Text export section
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Text Export:").pack()
        self.text_entry = ttk.Entry(control_frame, width=20)
        self.text_entry.insert(0, "DA3")
        self.text_entry.pack(pady=3)
        ttk.Button(control_frame, text="Export Text", 
                  command=self.export_text, width=20).pack(pady=3)
        
        # Batch
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Export All SVG", 
                  command=self.export_all_svg, width=20).pack(pady=3)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.plotter_frame, text="Information", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_text = """SVG Export for 2D Plotters
        
Compatible with:
• Inkscape (vector editor)
• Vinyl cutters (Cricut, Silhouette)
• Laser cutters/engravers
• Pen plotters (AxiDraw, etc.)
• CNC routers

SVG files are vector graphics that can
be scaled to any size without quality loss.

Use cases:
- Stickers and decals
- Wall graphics  
- Technical drawings
- Art prints
- Custom signage
"""
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()
        
    def create_printer_tab(self):
        """Create the 3D printer export tab."""
        # Controls
        control_frame = ttk.LabelFrame(self.printer_frame, text="3D Printer Exports (STL)", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        models = [
            ("Surface Mesh", lambda: self.export_stl('surface')),
            ("Torus (Donut)", lambda: self.export_stl('torus')),
            ("Sphere", lambda: self.export_stl('sphere')),
            ("Helix (Spring)", lambda: self.export_stl('helix')),
        ]
        
        for label, command in models:
            ttk.Button(control_frame, text=label, command=command, width=20).pack(pady=3)
        
        # Parameters
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(control_frame, text="Parameters:").pack()
        
        # Batch
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="Export All STL", 
                  command=self.export_all_stl, width=20).pack(pady=3)
        
        # Info panel
        info_frame = ttk.LabelFrame(self.printer_frame, text="Information", padding=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_text = """STL Export for 3D Printing
        
Compatible with:
• Cura (free slicer)
• PrusaSlicer (free)
• Simplify3D (paid)
• MeshLab (viewing/editing)

All meshes are manifold (watertight)
and ready for 3D printing.

Typical workflow:
1. Export STL file here
2. Open in slicing software
3. Configure print settings
4. Generate G-code
5. Print!

Recommended settings:
- Layer height: 0.1-0.2mm
- Infill: 10-20%
- Supports: Auto-generate
- Filament: PLA for beginners
"""
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack()
        
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_container = ttk.Frame(self.settings_frame, padding=20)
        settings_container.pack(fill=tk.BOTH, expand=True)
        
        # Output directory
        ttk.Label(settings_container, text="Output Directory:", 
                 font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        dir_frame = ttk.Frame(settings_container)
        dir_frame.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.dir_entry = ttk.Entry(dir_frame, width=50)
        self.dir_entry.insert(0, self.output_dir)
        self.dir_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(dir_frame, text="Browse...", 
                  command=self.browse_directory).pack(side=tk.LEFT)
        ttk.Button(dir_frame, text="Apply", 
                  command=self.apply_directory).pack(side=tk.LEFT, padx=5)
        
        # About section
        ttk.Separator(settings_container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=20)
        
        ttk.Label(settings_container, text="About DA3", 
                 font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        
        about_text = """DA3 - Data Analytics 3D Designer
Version: 1.0.0

A comprehensive tool for creating:
• 3D visualizations (PNG)
• 2D plotter outputs (SVG)  
• 3D printer models (STL)

Complete workflow: Visual data → Physical output

© 2026 DA3 Project
"""
        ttk.Label(settings_container, text=about_text, justify=tk.LEFT).pack(anchor=tk.W, pady=10)
        
        # View summary button
        ttk.Button(settings_container, text="View Summary of Created Files",
                  command=self.show_summary, width=30).pack(pady=10)
        
    def log(self, message):
        """Add message to log."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def set_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update()
        
    def run_in_thread(self, func, *args):
        """Run function in background thread."""
        def wrapper():
            try:
                func(*args)
                self.set_status("Ready")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.set_status("Error occurred")
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        
    # Visualization methods
    def create_surface(self):
        self.set_status("Creating surface plot...")
        self.log("[Creating surface plot]")
        self.da3.surface_plot()
        self.log("✓ Surface plot created\n")
        
    def create_parametric(self):
        self.set_status("Creating parametric surface...")
        self.log("[Creating parametric surface]")
        self.da3.parametric_surface()
        self.log("✓ Parametric surface created\n")
        
    def create_scatter(self):
        self.set_status("Creating scatter plot...")
        self.log("[Creating scatter plot]")
        self.da3.scatter_plot()
        self.log("✓ Scatter plot created\n")
        
    def create_cluster(self):
        self.set_status("Creating cluster scatter...")
        self.log("[Creating cluster scatter]")
        self.da3.cluster_scatter()
        self.log("✓ Cluster scatter created\n")
        
    def create_wireframe(self):
        self.set_status("Creating wireframe...")
        self.log("[Creating wireframe]")
        self.da3.wireframe_plot()
        self.log("✓ Wireframe created\n")
        
    def create_sphere(self):
        self.set_status("Creating sphere...")
        self.log("[Creating sphere wireframe]")
        self.da3.sphere_wireframe()
        self.log("✓ Sphere created\n")
        
    def create_line(self):
        self.set_status("Creating line plot...")
        self.log("[Creating line plot]")
        self.da3.line_plot()
        self.log("✓ Line plot created\n")
        
    def create_spiral(self):
        self.set_status("Creating spiral...")
        self.log("[Creating spiral plot]")
        self.da3.spiral_plot()
        self.log("✓ Spiral created\n")
        
    def create_lissajous(self):
        self.set_status("Creating Lissajous curve...")
        self.log("[Creating Lissajous curve]")
        self.da3.lissajous_curve()
        self.log("✓ Lissajous curve created\n")
        
    def create_all(self):
        self.set_status("Creating all plots...")
        self.log("[Creating ALL plots]")
        self.da3.create_all_plots()
        self.log("✓ All plots created\n")
        
    # 2D Plotter methods
    def export_svg(self, svg_type):
        try:
            self.set_status(f"Exporting {svg_type} to SVG...")
            self.log(f"[Exporting {svg_type} to SVG]")
            
            if svg_type == 'contour':
                self.da3.export_contour_svg()
            elif svg_type == 'spiral':
                self.da3.export_parametric_curve_svg(curve_type='spiral')
            elif svg_type == 'lissajous':
                self.da3.export_parametric_curve_svg(curve_type='lissajous')
            elif svg_type == 'grid':
                self.da3.export_pattern_svg(pattern_type='grid')
            elif svg_type == 'hexagon':
                self.da3.export_pattern_svg(pattern_type='hexagon')
            
            self.log(f"✓ {svg_type} exported to SVG\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export SVG: {e}")
            
    def export_text(self):
        text = self.text_entry.get()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to export")
            return
        
        try:
            self.set_status("Exporting text to SVG...")
            self.log(f"[Exporting text '{text}' to SVG]")
            self.da3.export_text_svg(text=text)
            self.log(f"✓ Text exported to SVG\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export text: {e}")
            
    def export_all_svg(self):
        try:
            self.set_status("Exporting all SVG patterns...")
            self.log("[Exporting ALL SVG patterns]")
            self.da3.export_all_svg()
            self.log("✓ All SVG patterns exported\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export SVG: {e}")
            
    # 3D Printer methods
    def export_stl(self, model_type):
        try:
            self.set_status(f"Exporting {model_type} to STL...")
            self.log(f"[Exporting {model_type} to STL]")
            
            if model_type == 'surface':
                self.da3.export_surface_stl()
            elif model_type == 'torus':
                self.da3.export_torus_stl()
            elif model_type == 'sphere':
                self.da3.export_sphere_stl()
            elif model_type == 'helix':
                self.da3.export_helix_stl()
            
            self.log(f"✓ {model_type} exported to STL\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export STL: {e}\nNote: Install numpy-stl")
            
    def export_all_stl(self):
        try:
            self.set_status("Exporting all STL models...")
            self.log("[Exporting ALL STL models]")
            self.da3.export_all_stl()
            self.log("✓ All STL models exported\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export STL: {e}\nNote: Install numpy-stl")
            
    # Settings methods
    def browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir)
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)
            
    def apply_directory(self):
        new_dir = self.dir_entry.get()
        if new_dir:
            self.output_dir = new_dir
            self.da3 = DA3(output_dir=self.output_dir)
            self.log(f"[Output directory changed to: {self.output_dir}]\n")
            messagebox.showinfo("Success", f"Output directory set to:\n{self.output_dir}")
            
    def clear_output(self):
        self.log_text.delete(1.0, tk.END)
        self.log("[Output cleared]\n")
        
    def show_summary(self):
        """Show summary in a new window."""
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Summary of Created Files")
        summary_window.geometry("600x400")
        
        text = scrolledtext.ScrolledText(summary_window, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Redirect da3.print_summary to text widget
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            self.da3.print_summary()
        
        text.insert(tk.END, f.getvalue())
        text.config(state=tk.DISABLED)


def main():
    """Main entry point for GUI."""
    root = tk.Tk()
    app = DA3GUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
