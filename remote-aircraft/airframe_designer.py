#!/usr/bin/env python3
"""
Airframe Designer GUI
A parametric design tool for Fixed Wing Aircraft and Gliders
Supports both foamboard and 3D printing outputs
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os

# Add remote-aircraft to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from materials import PLA, PETG, NYLON, CF_NYLON
from wind_tunnel import run_comprehensive_analysis
from wind_tunnel_window import WindTunnelWindow

# Design constants for calculations
TYPICAL_FIXED_WING_WEIGHT_G = 200  # Typical weight for small fixed wing aircraft in grams
TYPICAL_GLIDER_WEIGHT_G = 150      # Typical weight for small glider in grams
GLIDE_RATIO_EFFICIENCY = 0.8       # Aerodynamic efficiency factor for glide ratio calculation


class AirframeDesignerApp:
    """Main application for airframe design"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Airframe Designer - FPV & Glider CAD System")
        self.root.geometry("800x600")
        
        # Create main container
        self.create_welcome_screen()
    
    def create_welcome_screen(self):
        """Create the welcome/selection screen"""
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()
        
        # Title frame
        title_frame = tk.Frame(self.root, bg="#2c3e50", pady=20)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text="Airframe Designer",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Parametric CAD for Remote Aircraft",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        subtitle_label.pack()
        
        # Main content frame
        content_frame = tk.Frame(self.root, padx=50, pady=30)
        content_frame.pack(expand=True, fill=tk.BOTH)
        
        # Instructions
        instructions = tk.Label(
            content_frame,
            text="Select the type of aircraft you want to design:",
            font=("Arial", 14),
            pady=20
        )
        instructions.pack()
        
        # Button frame
        button_frame = tk.Frame(content_frame)
        button_frame.pack(expand=True)
        
        # Fixed Wing button
        fixed_wing_btn = tk.Button(
            button_frame,
            text="✈️ Fixed Wing Aircraft",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            width=25,
            height=3,
            command=self.open_fixed_wing_designer,
            cursor="hand2"
        )
        fixed_wing_btn.pack(pady=10)
        
        # Glider button
        glider_btn = tk.Button(
            button_frame,
            text="🪂 Glider",
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            width=25,
            height=3,
            command=self.open_glider_designer,
            cursor="hand2"
        )
        glider_btn.pack(pady=10)
        
        # Footer
        footer_frame = tk.Frame(self.root, bg="#ecf0f1", pady=10)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        footer_label = tk.Label(
            footer_frame,
            text="Generate foamboard templates and 3D printable parts",
            font=("Arial", 10),
            bg="#ecf0f1",
            fg="#7f8c8d"
        )
        footer_label.pack()
    
    def open_fixed_wing_designer(self):
        """Open the fixed wing aircraft designer"""
        FixedWingDesigner(self.root, self)
    
    def open_glider_designer(self):
        """Open the glider designer"""
        GliderDesigner(self.root, self)


class FixedWingDesigner:
    """Designer dialog for Fixed Wing Aircraft"""
    
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("Fixed Wing Aircraft Designer")
        self.window.geometry("700x800")
        
        self.create_ui()
    
    def create_ui(self):
        """Create the fixed wing designer UI"""
        # Header
        header = tk.Frame(self.window, bg="#3498db", pady=15)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header,
            text="✈️ Fixed Wing Aircraft Parameters",
            font=("Arial", 16, "bold"),
            bg="#3498db",
            fg="white"
        )
        title.pack()
        
        # Main scrollable frame
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parameters
        self.params = {}
        
        # Wing Section
        self.create_section(scrollable_frame, "Wing Dimensions", 0)
        self.add_param(scrollable_frame, "Wing Span (mm)", "wing_span", "1000", 1)
        self.add_param(scrollable_frame, "Wing Chord (mm)", "wing_chord", "200", 2)
        self.add_param(scrollable_frame, "Wing Thickness (%)", "wing_thickness", "12", 3)
        self.add_param(scrollable_frame, "Dihedral Angle (degrees)", "dihedral", "3", 4)
        
        # Fuselage Section
        self.create_section(scrollable_frame, "Fuselage", 5)
        self.add_param(scrollable_frame, "Length (mm)", "fuse_length", "800", 6)
        self.add_param(scrollable_frame, "Width (mm)", "fuse_width", "60", 7)
        self.add_param(scrollable_frame, "Height (mm)", "fuse_height", "80", 8)
        
        # Tail Section
        self.create_section(scrollable_frame, "Tail Surfaces", 9)
        self.add_param(scrollable_frame, "Horizontal Stabilizer Span (mm)", "h_stab_span", "400", 10)
        self.add_param(scrollable_frame, "Horizontal Stabilizer Chord (mm)", "h_stab_chord", "100", 11)
        self.add_param(scrollable_frame, "Vertical Stabilizer Height (mm)", "v_stab_height", "150", 12)
        self.add_param(scrollable_frame, "Vertical Stabilizer Chord (mm)", "v_stab_chord", "120", 13)
        
        # Motor Section
        self.create_section(scrollable_frame, "Propulsion", 14)
        self.add_param(scrollable_frame, "Motor Diameter (mm)", "motor_diameter", "28", 15)
        self.add_param(scrollable_frame, "Motor Length (mm)", "motor_length", "30", 16)
        self.add_param(scrollable_frame, "Propeller Diameter (inches)", "prop_diameter", "9", 17)
        
        # Build Options Section
        self.create_section(scrollable_frame, "Build Options", 18)
        
        # Material selection for 3D printing
        material_frame = ttk.Frame(scrollable_frame)
        material_frame.grid(row=19, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        
        ttk.Label(material_frame, text="3D Print Material:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.material_var = tk.StringVar(value="PETG")
        materials = ["PLA", "PETG", "NYLON", "CF_NYLON"]
        material_combo = ttk.Combobox(material_frame, textvariable=self.material_var, values=materials, state="readonly", width=15)
        material_combo.pack(side=tk.LEFT, padx=5)
        
        # Output options
        output_frame = ttk.Frame(scrollable_frame)
        output_frame.grid(row=20, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        
        ttk.Label(output_frame, text="Generate:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.generate_3d = tk.BooleanVar(value=True)
        self.generate_foam = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(output_frame, text="3D Print Files (.stl)", variable=self.generate_3d).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(output_frame, text="Foamboard Templates", variable=self.generate_foam).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = tk.Frame(self.window, pady=15)
        button_frame.pack(fill=tk.X)
        
        generate_btn = tk.Button(
            button_frame,
            text="Generate Design",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            width=20,
            height=2,
            command=self.generate_design
        )
        generate_btn.pack(side=tk.LEFT, padx=10)
        
        back_btn = tk.Button(
            button_frame,
            text="Back",
            font=("Arial", 12),
            bg="#95a5a6",
            fg="white",
            width=15,
            height=2,
            command=self.go_back
        )
        back_btn.pack(side=tk.RIGHT, padx=10)
    
    def create_section(self, parent, title, row):
        """Create a section header"""
        section_frame = tk.Frame(parent, bg="#34495e", pady=8)
        section_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        
        label = tk.Label(
            section_frame,
            text=title,
            font=("Arial", 11, "bold"),
            bg="#34495e",
            fg="white"
        )
        label.pack()
    
    def add_param(self, parent, label_text, param_name, default_value, row):
        """Add a parameter input field"""
        label = ttk.Label(parent, text=label_text + ":", font=("Arial", 10))
        label.grid(row=row, column=0, sticky="w", padx=20, pady=5)
        
        entry = ttk.Entry(parent, width=20, font=("Arial", 10))
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, sticky="w", padx=20, pady=5)
        
        self.params[param_name] = entry
    
    def generate_design(self):
        """Generate the fixed wing aircraft design"""
        try:
            # Collect parameters
            params = {}
            for name, entry in self.params.items():
                try:
                    params[name] = float(entry.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {name}")
                    return
            
            material = self.material_var.get()
            
            # Generate output directory
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if not output_dir:
                return
            
            # Create design summary
            summary = self.create_design_summary(params, material)
            
            # Save summary
            summary_file = os.path.join(output_dir, "fixed_wing_design_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            # Generate foamboard templates if requested
            if self.generate_foam.get():
                self.generate_foamboard_templates(params, output_dir)
            
            # Generate 3D print files if requested
            if self.generate_3d.get():
                self.generate_3d_parts(params, material, output_dir)
            
            messagebox.showinfo(
                "Success",
                f"Design generated successfully!\n\nFiles saved to:\n{output_dir}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_design_summary(self, params, material):
        """Create a design summary document"""
        summary = "=" * 60 + "\n"
        summary += "FIXED WING AIRCRAFT DESIGN SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        summary += "--- Wing Specifications ---\n"
        summary += f"Wingspan: {params['wing_span']:.1f} mm\n"
        summary += f"Wing Chord: {params['wing_chord']:.1f} mm\n"
        summary += f"Wing Area: {params['wing_span'] * params['wing_chord'] / 1000:.1f} cm²\n"
        summary += f"Wing Thickness: {params['wing_thickness']:.1f}%\n"
        summary += f"Dihedral: {params['dihedral']:.1f}°\n"
        summary += f"Aspect Ratio: {params['wing_span'] / params['wing_chord']:.2f}\n\n"
        
        summary += "--- Fuselage Specifications ---\n"
        summary += f"Length: {params['fuse_length']:.1f} mm\n"
        summary += f"Width: {params['fuse_width']:.1f} mm\n"
        summary += f"Height: {params['fuse_height']:.1f} mm\n\n"
        
        summary += "--- Tail Specifications ---\n"
        summary += f"H-Stab Span: {params['h_stab_span']:.1f} mm\n"
        summary += f"H-Stab Chord: {params['h_stab_chord']:.1f} mm\n"
        summary += f"V-Stab Height: {params['v_stab_height']:.1f} mm\n"
        summary += f"V-Stab Chord: {params['v_stab_chord']:.1f} mm\n\n"
        
        summary += "--- Propulsion ---\n"
        summary += f"Motor Diameter: {params['motor_diameter']:.1f} mm\n"
        summary += f"Motor Length: {params['motor_length']:.1f} mm\n"
        summary += f"Propeller: {params['prop_diameter']:.1f} inches\n\n"
        
        summary += "--- Build Options ---\n"
        summary += f"3D Print Material: {material}\n\n"
        
        summary += "--- Performance Estimates ---\n"
        wing_loading = TYPICAL_FIXED_WING_WEIGHT_G / (params['wing_span'] * params['wing_chord'] / 10000)
        summary += f"Estimated Wing Loading: {wing_loading:.2f} g/dm² (assuming {TYPICAL_FIXED_WING_WEIGHT_G}g weight)\n"
        
        summary += "\n" + "=" * 60 + "\n"
        
        return summary
    
    def generate_foamboard_templates(self, params, output_dir):
        """Generate foamboard cutting templates"""
        template_file = os.path.join(output_dir, "foamboard_templates.txt")
        
        with open(template_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FOAMBOARD CUTTING TEMPLATES - FIXED WING\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MATERIALS NEEDED:\n")
            f.write("- Foamboard sheets (5mm thickness recommended)\n")
            f.write("- Hot glue gun\n")
            f.write("- Hobby knife\n")
            f.write("- Ruler and cutting mat\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("WING TEMPLATES\n")
            f.write("=" * 60 + "\n\n")
            
            # Main wing
            f.write(f"Main Wing (Cut 2 - Left and Right):\n")
            f.write(f"  Length: {params['wing_span']/2:.1f} mm\n")
            f.write(f"  Chord: {params['wing_chord']:.1f} mm\n")
            f.write(f"  Shape: Rectangular with rounded leading edge\n")
            f.write(f"  Airfoil: Flat bottom (5mm foam)\n")
            f.write(f"  Dihedral: {params['dihedral']:.1f}° upward angle when joining at center\n\n")
            
            # Fuselage
            f.write("=" * 60 + "\n")
            f.write("FUSELAGE TEMPLATE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Fuselage Sides (Cut 2):\n")
            f.write(f"  Length: {params['fuse_length']:.1f} mm\n")
            f.write(f"  Height: {params['fuse_height']:.1f} mm\n\n")
            
            f.write(f"Fuselage Top/Bottom (Cut 2):\n")
            f.write(f"  Length: {params['fuse_length']:.1f} mm\n")
            f.write(f"  Width: {params['fuse_width']:.1f} mm\n\n")
            
            # Tail
            f.write("=" * 60 + "\n")
            f.write("TAIL TEMPLATES\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Horizontal Stabilizer:\n")
            f.write(f"  Span: {params['h_stab_span']:.1f} mm\n")
            f.write(f"  Chord: {params['h_stab_chord']:.1f} mm\n\n")
            
            f.write(f"Vertical Stabilizer:\n")
            f.write(f"  Height: {params['v_stab_height']:.1f} mm\n")
            f.write(f"  Chord: {params['v_stab_chord']:.1f} mm\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("ASSEMBLY NOTES\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. Cut all pieces using a sharp hobby knife\n")
            f.write("2. Sand edges smooth\n")
            f.write("3. Glue fuselage box together first\n")
            f.write("4. Attach wings with proper dihedral angle\n")
            f.write("5. Add tail surfaces\n")
            f.write("6. Install motor mount at nose\n")
            f.write("7. Balance CG at 25-30% of wing chord from leading edge\n\n")
    
    def generate_3d_parts(self, params, material, output_dir):
        """Generate 3D printable parts specifications"""
        parts_file = os.path.join(output_dir, "3d_print_parts.txt")
        
        with open(parts_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("3D PRINTABLE PARTS - FIXED WING\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Material: {material}\n\n")
            
            f.write("PRINT SETTINGS:\n")
            if material == "PLA":
                f.write("  Nozzle Temperature: 200-210°C\n")
                f.write("  Bed Temperature: 60°C\n")
            elif material == "PETG":
                f.write("  Nozzle Temperature: 230-240°C\n")
                f.write("  Bed Temperature: 80°C\n")
            elif material == "NYLON":
                f.write("  Nozzle Temperature: 250-260°C\n")
                f.write("  Bed Temperature: 85°C\n")
            else:  # CF_NYLON
                f.write("  Nozzle Temperature: 255-265°C\n")
                f.write("  Bed Temperature: 85°C\n")
            
            f.write("  Layer Height: 0.2mm\n")
            f.write("  Infill: 30% Gyroid\n")
            f.write("  Wall Thickness: 3 perimeters\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("PARTS LIST\n")
            f.write("=" * 60 + "\n\n")
            
            # Motor mount
            f.write(f"1. Motor Mount:\n")
            f.write(f"   - Diameter: {params['motor_diameter'] + 2:.1f} mm\n")
            f.write(f"   - Length: {params['motor_length'] + 5:.1f} mm\n")
            f.write(f"   - Mounting holes: 4x M3\n")
            f.write(f"   - Quantity: 1\n\n")
            
            # Wing joiner
            f.write(f"2. Wing Center Joiner:\n")
            f.write(f"   - Length: {params['wing_chord'] * 0.8:.1f} mm\n")
            f.write(f"   - Width: 20 mm\n")
            f.write(f"   - Thickness: 3 mm\n")
            f.write(f"   - Quantity: 1\n\n")
            
            # Servo mounts
            f.write(f"3. Servo Mounts (9g micro servo):\n")
            f.write(f"   - Standard 9g servo mount\n")
            f.write(f"   - Quantity: 3 (2x wing, 1x tail)\n\n")
            
            # Landing gear
            f.write(f"4. Landing Gear Mounts:\n")
            f.write(f"   - Width: {params['fuse_width']:.1f} mm\n")
            f.write(f"   - Quantity: 2\n\n")
            
            f.write("\nNOTE: STL files would be generated here if CadQuery is installed.\n")
            f.write("To generate actual STL files, ensure CadQuery is properly installed.\n\n")
    
    def open_wind_tunnel(self):
        """Open wind tunnel simulation window"""
        try:
            # Collect parameters
            params = {}
            for name, entry in self.params.items():
                try:
                    params[name] = float(entry.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {name}")
                    return
            
            # Prepare design parameters for wind tunnel
            design_params = {
                'wingspan': params['wing_span'],
                'chord': params['wing_chord'],
                'wing_area': params['wing_span'] * params['wing_chord'],
                'weight': params.get('wing_span', 1000) * 1.2,  # Estimate weight based on size
                'airfoil_type': 'clark_y',
                'fuselage_length': params['fuse_length'],
                'fuselage_diameter': (params['fuse_width'] + params['fuse_height']) / 2
            }
            
            # Open wind tunnel window
            WindTunnelWindow(self.window, design_params)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open wind tunnel: {str(e)}")
    
    def go_back(self):
        """Return to main menu"""
        self.window.destroy()


class GliderDesigner:
    """Designer dialog for Gliders"""
    
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.window = tk.Toplevel(parent)
        self.window.title("Glider Designer")
        self.window.geometry("700x700")
        
        self.create_ui()
    
    def create_ui(self):
        """Create the glider designer UI"""
        # Header
        header = tk.Frame(self.window, bg="#27ae60", pady=15)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header,
            text="🪂 Glider Parameters",
            font=("Arial", 16, "bold"),
            bg="#27ae60",
            fg="white"
        )
        title.pack()
        
        # Main scrollable frame
        canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parameters
        self.params = {}
        
        # Wing Section
        self.create_section(scrollable_frame, "Wing Dimensions", 0)
        self.add_param(scrollable_frame, "Wing Span (mm)", "wing_span", "1200", 1)
        self.add_param(scrollable_frame, "Wing Chord at Root (mm)", "root_chord", "220", 2)
        self.add_param(scrollable_frame, "Wing Chord at Tip (mm)", "tip_chord", "150", 3)
        self.add_param(scrollable_frame, "Wing Thickness (%)", "wing_thickness", "14", 4)
        self.add_param(scrollable_frame, "Dihedral Angle (degrees)", "dihedral", "5", 5)
        
        # Fuselage Section
        self.create_section(scrollable_frame, "Fuselage", 6)
        self.add_param(scrollable_frame, "Length (mm)", "fuse_length", "700", 7)
        self.add_param(scrollable_frame, "Width (mm)", "fuse_width", "50", 8)
        self.add_param(scrollable_frame, "Height (mm)", "fuse_height", "60", 9)
        
        # Tail Section
        self.create_section(scrollable_frame, "Tail Surfaces", 10)
        self.add_param(scrollable_frame, "Horizontal Stabilizer Span (mm)", "h_stab_span", "350", 11)
        self.add_param(scrollable_frame, "Horizontal Stabilizer Chord (mm)", "h_stab_chord", "90", 12)
        self.add_param(scrollable_frame, "Vertical Stabilizer Height (mm)", "v_stab_height", "130", 13)
        self.add_param(scrollable_frame, "Vertical Stabilizer Chord (mm)", "v_stab_chord", "100", 14)
        
        # Build Options Section
        self.create_section(scrollable_frame, "Build Options", 15)
        
        # Material selection for 3D printing
        material_frame = ttk.Frame(scrollable_frame)
        material_frame.grid(row=16, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        
        ttk.Label(material_frame, text="3D Print Material:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        self.material_var = tk.StringVar(value="PLA")
        materials = ["PLA", "PETG", "NYLON", "CF_NYLON"]
        material_combo = ttk.Combobox(material_frame, textvariable=self.material_var, values=materials, state="readonly", width=15)
        material_combo.pack(side=tk.LEFT, padx=5)
        
        # Output options
        output_frame = ttk.Frame(scrollable_frame)
        output_frame.grid(row=17, column=0, columnspan=2, sticky="ew", padx=20, pady=5)
        
        ttk.Label(output_frame, text="Generate:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.generate_3d = tk.BooleanVar(value=True)
        self.generate_foam = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(output_frame, text="3D Print Files (.stl)", variable=self.generate_3d).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(output_frame, text="Foamboard Templates", variable=self.generate_foam).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = tk.Frame(self.window, pady=15)
        button_frame.pack(fill=tk.X)
        
        generate_btn = tk.Button(
            button_frame,
            text="Generate Design",
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            width=20,
            height=2,
            command=self.generate_design
        )
        generate_btn.pack(side=tk.LEFT, padx=10)
        
        wind_tunnel_btn = tk.Button(
            button_frame,
            text="🌪️ Wind Tunnel",
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            width=18,
            height=2,
            command=self.open_wind_tunnel
        )
        wind_tunnel_btn.pack(side=tk.LEFT, padx=10)
        
        back_btn = tk.Button(
            button_frame,
            text="Back",
            font=("Arial", 12),
            bg="#95a5a6",
            fg="white",
            width=15,
            height=2,
            command=self.go_back
        )
        back_btn.pack(side=tk.RIGHT, padx=10)
    
    def create_section(self, parent, title, row):
        """Create a section header"""
        section_frame = tk.Frame(parent, bg="#16a085", pady=8)
        section_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 5))
        
        label = tk.Label(
            section_frame,
            text=title,
            font=("Arial", 11, "bold"),
            bg="#16a085",
            fg="white"
        )
        label.pack()
    
    def add_param(self, parent, label_text, param_name, default_value, row):
        """Add a parameter input field"""
        label = ttk.Label(parent, text=label_text + ":", font=("Arial", 10))
        label.grid(row=row, column=0, sticky="w", padx=20, pady=5)
        
        entry = ttk.Entry(parent, width=20, font=("Arial", 10))
        entry.insert(0, default_value)
        entry.grid(row=row, column=1, sticky="w", padx=20, pady=5)
        
        self.params[param_name] = entry
    
    def generate_design(self):
        """Generate the glider design"""
        try:
            # Collect parameters
            params = {}
            for name, entry in self.params.items():
                try:
                    params[name] = float(entry.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {name}")
                    return
            
            material = self.material_var.get()
            
            # Generate output directory
            output_dir = filedialog.askdirectory(title="Select Output Directory")
            if not output_dir:
                return
            
            # Create design summary
            summary = self.create_design_summary(params, material)
            
            # Save summary
            summary_file = os.path.join(output_dir, "glider_design_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(summary)
            
            # Generate foamboard templates if requested
            if self.generate_foam.get():
                self.generate_foamboard_templates(params, output_dir)
            
            # Generate 3D print files if requested
            if self.generate_3d.get():
                self.generate_3d_parts(params, material, output_dir)
            
            messagebox.showinfo(
                "Success",
                f"Design generated successfully!\n\nFiles saved to:\n{output_dir}"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def create_design_summary(self, params, material):
        """Create a design summary document"""
        summary = "=" * 60 + "\n"
        summary += "GLIDER DESIGN SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        
        # Calculate wing area (trapezoidal wing)
        wing_area = (params['root_chord'] + params['tip_chord']) / 2 * params['wing_span'] / 1000
        mean_chord = (params['root_chord'] + params['tip_chord']) / 2
        aspect_ratio = params['wing_span'] / mean_chord
        
        summary += "--- Wing Specifications ---\n"
        summary += f"Wingspan: {params['wing_span']:.1f} mm\n"
        summary += f"Root Chord: {params['root_chord']:.1f} mm\n"
        summary += f"Tip Chord: {params['tip_chord']:.1f} mm\n"
        summary += f"Mean Chord: {mean_chord:.1f} mm\n"
        summary += f"Wing Area: {wing_area:.1f} cm²\n"
        summary += f"Wing Thickness: {params['wing_thickness']:.1f}%\n"
        summary += f"Dihedral: {params['dihedral']:.1f}°\n"
        summary += f"Aspect Ratio: {aspect_ratio:.2f}\n"
        summary += f"Taper Ratio: {params['tip_chord']/params['root_chord']:.2f}\n\n"
        
        summary += "--- Fuselage Specifications ---\n"
        summary += f"Length: {params['fuse_length']:.1f} mm\n"
        summary += f"Width: {params['fuse_width']:.1f} mm\n"
        summary += f"Height: {params['fuse_height']:.1f} mm\n\n"
        
        summary += "--- Tail Specifications ---\n"
        summary += f"H-Stab Span: {params['h_stab_span']:.1f} mm\n"
        summary += f"H-Stab Chord: {params['h_stab_chord']:.1f} mm\n"
        summary += f"V-Stab Height: {params['v_stab_height']:.1f} mm\n"
        summary += f"V-Stab Chord: {params['v_stab_chord']:.1f} mm\n\n"
        
        summary += "--- Build Options ---\n"
        summary += f"3D Print Material: {material}\n\n"
        
        summary += "--- Performance Estimates ---\n"
        wing_loading = TYPICAL_GLIDER_WEIGHT_G / wing_area
        summary += f"Estimated Wing Loading: {wing_loading:.2f} g/dm² (assuming {TYPICAL_GLIDER_WEIGHT_G}g weight)\n"
        summary += f"Estimated Glide Ratio: {aspect_ratio * GLIDE_RATIO_EFFICIENCY:.1f}:1 (typical)\n"
        summary += f"Stall Speed (estimated): {(wing_loading * 2)**0.5:.1f} m/s\n"
        
        summary += "\n" + "=" * 60 + "\n"
        
        return summary
    
    def generate_foamboard_templates(self, params, output_dir):
        """Generate foamboard cutting templates"""
        template_file = os.path.join(output_dir, "foamboard_templates.txt")
        
        with open(template_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("FOAMBOARD CUTTING TEMPLATES - GLIDER\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("MATERIALS NEEDED:\n")
            f.write("- Foamboard sheets (5mm thickness recommended)\n")
            f.write("- Hot glue gun\n")
            f.write("- Hobby knife\n")
            f.write("- Ruler and cutting mat\n")
            f.write("- Sandpaper (for shaping)\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("WING TEMPLATES (Tapered Design)\n")
            f.write("=" * 60 + "\n\n")
            
            # Main wing - tapered
            f.write(f"Main Wing (Cut 2 - Left and Right):\n")
            f.write(f"  Half-span: {params['wing_span']/2:.1f} mm\n")
            f.write(f"  Root Chord: {params['root_chord']:.1f} mm\n")
            f.write(f"  Tip Chord: {params['tip_chord']:.1f} mm\n")
            f.write(f"  Shape: Tapered planform with rounded leading edge\n")
            f.write(f"  Airfoil: Flat bottom or undercambered (5mm foam)\n")
            f.write(f"  Dihedral: {params['dihedral']:.1f}° upward angle when joining at center\n\n")
            
            f.write(f"Wing Construction Notes:\n")
            f.write(f"  - Cut wing shape from template\n")
            f.write(f"  - Bevel leading edge to create rounded profile\n")
            f.write(f"  - Sand trailing edge to thin taper\n")
            f.write(f"  - Join left/right wings at center with dihedral\n\n")
            
            # Fuselage
            f.write("=" * 60 + "\n")
            f.write("FUSELAGE TEMPLATE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Fuselage Sides (Cut 2):\n")
            f.write(f"  Length: {params['fuse_length']:.1f} mm\n")
            f.write(f"  Height at front: {params['fuse_height']:.1f} mm\n")
            f.write(f"  Height at rear: {params['fuse_height']*0.7:.1f} mm (tapered)\n\n")
            
            f.write(f"Fuselage Top/Bottom (Cut 2):\n")
            f.write(f"  Length: {params['fuse_length']:.1f} mm\n")
            f.write(f"  Width: {params['fuse_width']:.1f} mm\n\n")
            
            # Tail
            f.write("=" * 60 + "\n")
            f.write("TAIL TEMPLATES\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Horizontal Stabilizer:\n")
            f.write(f"  Span: {params['h_stab_span']:.1f} mm\n")
            f.write(f"  Chord: {params['h_stab_chord']:.1f} mm\n")
            f.write(f"  Shape: Rectangular or slightly tapered\n\n")
            
            f.write(f"Vertical Stabilizer:\n")
            f.write(f"  Height: {params['v_stab_height']:.1f} mm\n")
            f.write(f"  Chord: {params['v_stab_chord']:.1f} mm\n")
            f.write(f"  Shape: Swept back recommended\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("ASSEMBLY NOTES\n")
            f.write("=" * 60 + "\n\n")
            f.write("1. Cut all pieces carefully following templates\n")
            f.write("2. Sand and shape wing airfoil profile\n")
            f.write("3. Glue fuselage box together\n")
            f.write("4. Attach wings with proper dihedral (use jig for accuracy)\n")
            f.write("5. Add tail surfaces (ensure perpendicular alignment)\n")
            f.write("6. Balance CG at 25-33% of mean chord from leading edge\n")
            f.write("7. Test glide with gentle hand launch\n")
            f.write("8. Adjust CG position for optimal glide performance\n\n")
    
    def generate_3d_parts(self, params, material, output_dir):
        """Generate 3D printable parts specifications"""
        parts_file = os.path.join(output_dir, "3d_print_parts.txt")
        
        with open(parts_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("3D PRINTABLE PARTS - GLIDER\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Material: {material}\n\n")
            
            f.write("PRINT SETTINGS:\n")
            if material == "PLA":
                f.write("  Nozzle Temperature: 200-210°C\n")
                f.write("  Bed Temperature: 60°C\n")
            elif material == "PETG":
                f.write("  Nozzle Temperature: 230-240°C\n")
                f.write("  Bed Temperature: 80°C\n")
            elif material == "NYLON":
                f.write("  Nozzle Temperature: 250-260°C\n")
                f.write("  Bed Temperature: 85°C\n")
            else:  # CF_NYLON
                f.write("  Nozzle Temperature: 255-265°C\n")
                f.write("  Bed Temperature: 85°C\n")
            
            f.write("  Layer Height: 0.2mm\n")
            f.write("  Infill: 20% Gyroid (lightweight for gliders)\n")
            f.write("  Wall Thickness: 2-3 perimeters\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("PARTS LIST\n")
            f.write("=" * 60 + "\n\n")
            
            # Wing joiner
            f.write(f"1. Wing Center Joiner:\n")
            f.write(f"   - Length: {params['root_chord'] * 0.7:.1f} mm\n")
            f.write(f"   - Width: 15 mm\n")
            f.write(f"   - Thickness: 3 mm\n")
            f.write(f"   - Dihedral angle: {params['dihedral']:.1f}°\n")
            f.write(f"   - Quantity: 1\n\n")
            
            # Nose weight holder
            f.write(f"2. Nose Weight Holder:\n")
            f.write(f"   - Fits in fuselage nose\n")
            f.write(f"   - Diameter: {params['fuse_width'] - 5:.1f} mm\n")
            f.write(f"   - Depth: 40 mm\n")
            f.write(f"   - For ballast adjustment\n")
            f.write(f"   - Quantity: 1\n\n")
            
            # Servo mounts
            f.write(f"3. Servo Mounts (9g micro servo):\n")
            f.write(f"   - Standard 9g servo mount\n")
            f.write(f"   - Quantity: 2 (1x elevator, 1x rudder)\n\n")
            
            # Tail boom mount
            f.write(f"4. Tail Boom Mount:\n")
            f.write(f"   - Attaches fuselage to tail\n")
            f.write(f"   - Width: {params['fuse_width']:.1f} mm\n")
            f.write(f"   - Quantity: 1\n\n")
            
            # Wing reinforcement ribs
            f.write(f"5. Wing Reinforcement Ribs (optional):\n")
            f.write(f"   - Root rib chord: {params['root_chord']:.1f} mm\n")
            f.write(f"   - Tip rib chord: {params['tip_chord']:.1f} mm\n")
            f.write(f"   - Thickness: 2mm\n")
            f.write(f"   - Quantity: 4-6 per wing half\n\n")
            
            f.write("\nNOTE: STL files would be generated here if CadQuery is installed.\n")
            f.write("To generate actual STL files, ensure CadQuery is properly installed.\n\n")
    
    def go_back(self):
    
    def open_wind_tunnel(self):
        """Open wind tunnel simulation window"""
        try:
            # Collect parameters
            params = {}
            for name, entry in self.params.items():
                try:
                    params[name] = float(entry.get())
                except ValueError:
                    messagebox.showerror("Invalid Input", f"Please enter a valid number for {name}")
                    return
            
            # Prepare design parameters for wind tunnel
            design_params = {
                "'wingspan"': params["'wing_span"'],
                "'chord"': (params["'root_chord"'] + params["'tip_chord"']) / 2,
                "'wing_area"': params["'wing_span"'] * (params["'root_chord"'] + params["'tip_chord"']) / 2,
                "'weight"': params.get("'wing_span"', 1000) * 0.8,
                "'airfoil_type"': "'clark_y"'
            }
            
            # Open wind tunnel window
            WindTunnelWindow(self.window, design_params)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open wind tunnel: {str(e)}")

        """Return to main menu"""
        self.window.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AirframeDesignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
