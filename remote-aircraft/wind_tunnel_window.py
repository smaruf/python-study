"""
Wind Tunnel Window GUI Module
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from wind_tunnel import run_comprehensive_analysis


class WindTunnelWindow:
    """Wind tunnel simulation window"""
    
    def __init__(self, parent, design_params):
        self.parent = parent
        self.design_params = design_params
        self.window = tk.Toplevel(parent)
        self.window.title("Wind Tunnel Simulation")
        self.window.geometry("900x700")
        
        self.create_ui()
        self.run_simulation()
    
    def create_ui(self):
        """Create the wind tunnel UI"""
        # Header
        header = tk.Frame(self.window, bg="#e74c3c", pady=15)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header,
            text="🌪️ Wind Tunnel Simulation",
            font=("Arial", 18, "bold"),
            bg="#e74c3c",
            fg="white"
        )
        title.pack()
        
        subtitle = tk.Label(
            header,
            text="Aerodynamic Analysis & Performance Prediction",
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white"
        )
        subtitle.pack()
        
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
        
        # Content will be added dynamically
        self.content_frame = scrollable_frame
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Button frame
        button_frame = tk.Frame(self.window, pady=10)
        button_frame.pack(fill=tk.X)
        
        save_btn = tk.Button(
            button_frame,
            text="Save Results",
            font=("Arial", 11),
            bg="#27ae60",
            fg="white",
            width=15,
            command=self.save_results
        )
        save_btn.pack(side=tk.LEFT, padx=10)
        
        close_btn = tk.Button(
            button_frame,
            text="Close",
            font=("Arial", 11),
            bg="#95a5a6",
            fg="white",
            width=15,
            command=self.window.destroy
        )
        close_btn.pack(side=tk.RIGHT, padx=10)
    
    def run_simulation(self):
        """Run the wind tunnel simulation"""
        try:
            # Run comprehensive analysis
            self.results = run_comprehensive_analysis(self.design_params, cruise_speed=15.0)
            self.display_results()
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Could not run simulation: {str(e)}")
    
    def display_results(self):
        """Display simulation results"""
        # Clear existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        row = 0
        
        # Design parameters
        self.add_section("Design Parameters", row)
        row += 1
        
        params_text = f"Wingspan: {self.design_params['wingspan']:.0f} mm\n"
        params_text += f"Chord: {self.design_params['chord']:.0f} mm\n"
        params_text += f"Wing Area: {self.design_params['wing_area']/100:.1f} cm²\n"
        params_text += f"Weight: {self.design_params['weight']:.0f} g\n"
        params_text += f"Aspect Ratio: {self.results['aspect_ratio']:.2f}\n"
        params_text += f"Airfoil: {self.design_params['airfoil_type']}"
        
        self.add_text_block(params_text, row)
        row += 1
        
        # Stall characteristics
        self.add_section("Stall Characteristics", row)
        row += 1
        
        stall = self.results['stall_characteristics']
        stall_text = f"Stall Speed: {stall['stall_speed_ms']:.1f} m/s ({stall['stall_speed_ms']*3.6:.1f} km/h)\n"
        stall_text += f"Approach Speed: {stall['approach_speed_ms']:.1f} m/s ({stall['approach_speed_ms']*3.6:.1f} km/h)\n"
        stall_text += f"Maximum CL: {stall['cl_max']:.3f}"
        
        self.add_text_block(stall_text, row, bg="#ecf0f1")
        row += 1
        
        # Trim condition
        self.add_section("Level Flight Performance", row)
        row += 1
        
        trim = self.results['trim_condition']
        if trim.get('converged'):
            trim_text = f"✓ Trim Achieved at {self.results['cruise_speed_ms']:.1f} m/s\n"
            trim_text += f"Trim Angle of Attack: {trim['trim_aoa']:.1f}°\n"
            trim_text += f"Trim CL: {trim['trim_cl']:.3f}\n"
            trim_text += f"Trim CD: {trim['trim_cd']:.4f}\n"
            trim_text += f"L/D Ratio: {trim['trim_ld']:.1f}\n"
            trim_text += f"Drag Force: {trim['drag_g']:.1f} g"
        else:
            trim_text = f"⚠ Could not achieve trim\n{trim.get('message', '')}"
        
        self.add_text_block(trim_text, row, bg="#d5f4e6")
        row += 1
        
        # Best L/D
        self.add_section("Optimal Glide Performance", row)
        row += 1
        
        best = self.results['best_ld_condition']
        best_text = f"Best L/D Ratio: {best['ld_ratio']:.1f}\n"
        best_text += f"At Angle of Attack: {best['angle_of_attack']:.1f}°\n"
        best_text += f"CL at best L/D: {best['cl']:.3f}\n"
        best_text += f"CD at best L/D: {best['cd']:.4f}\n"
        best_text += f"Glide Ratio: 1:{best['ld_ratio']:.1f} (travels {best['ld_ratio']:.1f}m forward per 1m descent)"
        
        self.add_text_block(best_text, row, bg="#fff3cd")
        row += 1
        
        # Stability
        self.add_section("Stability Analysis", row)
        row += 1
        
        stability = self.results['stability_analysis']
        if stability.get('stable') is not None:
            status_symbol = "✓" if stability['stable'] else "✗"
            status_color = "#d5f4e6" if stability['stable'] else "#f8d7da"
            
            stab_text = f"{status_symbol} Status: {stability['assessment']}\n"
            stab_text += f"Static Margin: {stability['static_margin']*100:.1f}%\n"
            stab_text += f"CL_alpha: {stability['cl_alpha']:.2f} per radian\n"
            
            if stability['stable']:
                stab_text += "\n✓ This design is longitudinally stable"
            else:
                stab_text += "\n⚠ Design may require tail adjustment for stability"
        else:
            status_color = "#f8d7da"
            stab_text = "⚠ Could not analyze stability"
        
        self.add_text_block(stab_text, row, bg=status_color)
        row += 1
        
        # Angle of attack sweep table
        self.add_section("Angle of Attack Sweep", row)
        row += 1
        
        # Create table
        table_frame = tk.Frame(self.content_frame, bg="white", relief=tk.SOLID, borderwidth=1)
        table_frame.grid(row=row, column=0, sticky="ew", padx=20, pady=5)
        
        # Header row
        headers = ["AoA (°)", "CL", "CD", "L/D", "Lift (g)", "Drag (g)", "Status"]
        for col, header in enumerate(headers):
            label = tk.Label(table_frame, text=header, font=("Arial", 9, "bold"), 
                           bg="#34495e", fg="white", padx=8, pady=4)
            label.grid(row=0, column=col, sticky="ew")
        
        # Data rows (show every 2 degrees for readability)
        data_row = 1
        for i, data in enumerate(self.results['aoa_sweep_data']):
            if i % 2 == 0:  # Show every other row
                bg_color = "#ecf0f1" if data_row % 2 == 0 else "white"
                status = "⚠ STALL" if data['stalled'] else "OK"
                status_color = "#e74c3c" if data['stalled'] else "#27ae60"
                
                values = [
                    f"{data['angle_of_attack']:.0f}",
                    f"{data['cl']:.3f}",
                    f"{data['cd']:.4f}",
                    f"{data['ld_ratio']:.1f}",
                    f"{data['lift_g']:.0f}",
                    f"{data['drag_g']:.1f}",
                    status
                ]
                
                for col, value in enumerate(values):
                    fg_color = status_color if col == 6 else "black"
                    label = tk.Label(table_frame, text=value, font=("Arial", 9),
                                   bg=bg_color, fg=fg_color, padx=8, pady=3)
                    label.grid(row=data_row, column=col, sticky="ew")
                
                data_row += 1
        
        row += 1
        
        # Recommendations
        self.add_section("Recommendations", row)
        row += 1
        
        recommendations = self.generate_recommendations()
        self.add_text_block(recommendations, row, bg="#d5f4e6")
    
    def add_section(self, title, row):
        """Add a section header"""
        section_frame = tk.Frame(self.content_frame, bg="#34495e", pady=8)
        section_frame.grid(row=row, column=0, sticky="ew", padx=20, pady=(10, 5))
        
        label = tk.Label(
            section_frame,
            text=title,
            font=("Arial", 12, "bold"),
            bg="#34495e",
            fg="white"
        )
        label.pack()
    
    def add_text_block(self, text, row, bg="white"):
        """Add a text block"""
        frame = tk.Frame(self.content_frame, bg=bg, relief=tk.SOLID, borderwidth=1)
        frame.grid(row=row, column=0, sticky="ew", padx=20, pady=5)
        
        label = tk.Label(
            frame,
            text=text,
            font=("Arial", 10),
            bg=bg,
            justify=tk.LEFT,
            padx=15,
            pady=10
        )
        label.pack(anchor="w")
    
    def generate_recommendations(self):
        """Generate design recommendations"""
        recommendations = []
        
        # Check stall speed
        stall_speed = self.results['stall_characteristics']['stall_speed_ms']
        if stall_speed < 8:
            recommendations.append("✓ Excellent low-speed performance - suitable for small flying fields")
        elif stall_speed > 15:
            recommendations.append("⚠ High stall speed - requires larger flying area and experienced pilot")
        
        # Check L/D
        best_ld = self.results['best_ld_condition']['ld_ratio']
        if best_ld > 12:
            recommendations.append("✓ Excellent glide performance - great for thermal soaring")
        elif best_ld < 8:
            recommendations.append("⚠ Limited glide performance - consider increasing aspect ratio")
        
        # Check stability
        if self.results['stability_analysis'].get('stable'):
            recommendations.append("✓ Stable design - good for beginners")
        else:
            recommendations.append("⚠ Requires tail adjustment or CG repositioning for stability")
        
        # Check aspect ratio
        ar = self.results['aspect_ratio']
        if ar < 5:
            recommendations.append("• Low aspect ratio - good for aerobatics, less efficient cruise")
        elif ar > 8:
            recommendations.append("• High aspect ratio - efficient cruise, but may be fragile")
        
        return "\n".join(recommendations) if recommendations else "Design parameters are within normal ranges."
    
    def save_results(self):
        """Save simulation results to file"""
        try:
            import json
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                output = {
                    'design_params': self.design_params,
                    'cruise_speed_ms': self.results['cruise_speed_ms'],
                    'aspect_ratio': self.results['aspect_ratio'],
                    'stall_characteristics': self.results['stall_characteristics'],
                    'trim_condition': self.results['trim_condition'],
                    'stability_analysis': self.results['stability_analysis'],
                    'best_ld_condition': self.results['best_ld_condition'],
                    'aoa_sweep_data': self.results['aoa_sweep_data']
                }
                
                with open(filename, 'w') as f:
                    json.dump(output, f, indent=2)
                
                messagebox.showinfo("Success", f"Results saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save results: {str(e)}")

