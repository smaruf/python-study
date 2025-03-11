import os
import tkinter as tk
from tkinter import filedialog, messagebox
from plantuml import PlantUML
from PIL import Image, ImageTk

class PumlConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PUML to Diagram Converter")
        
        self.label = tk.Label(root, text="Select a PUML file to convert")
        self.label.pack(pady=10)
        
        self.select_button = tk.Button(root, text="Browse", command=self.browse_file)
        self.select_button.pack(pady=5)
        
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)
        
        self.save_button = tk.Button(root, text="Save Diagram", command=self.save_diagram, state=tk.DISABLED)
        self.save_button.pack(pady=5)
        
        self.plantuml = PlantUML(url='http://www.plantuml.com/plantuml/img/')
        self.current_diagram = None

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PUML files", "*.puml")])
        if file_path:
            self.convert_puml_to_diagram(file_path)

    def convert_puml_to_diagram(self, puml_file):
        try:
            output_path = puml_file.replace('.puml', '.png')
            self.plantuml.processes_file(puml_file, output_path)
            self.display_diagram(output_path)
            self.current_diagram = output_path
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert {puml_file}: {e}")

    def display_diagram(self, image_path):
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def save_diagram(self):
        if self.current_diagram:
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                os.rename(self.current_diagram, save_path)
                messagebox.showinfo("Success", f"Diagram saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PumlConverterApp(root)
    root.mainloop()
