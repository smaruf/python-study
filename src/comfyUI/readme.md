# Working with ComfyUI

## Introduction

ComfyUI is a node-based user interface (UI) primarily designed for building, visualizing, and customizing stable diffusion AI workflows. It allows users to create complex pipelines for AI-based image generation intuitively and modularly. ComfyUI simplifies the process of managing AI models, connecting nodes, and experimenting with different configurations to produce high-quality results.

---

## Prerequisites

### System Requirements:
- **GPU Support**:
  - NVIDIA GPUs (compatible with **CUDA**).
  - AMD GPUs (compatible with **ROCm**).
- **Python 3.10 or higher**.
- Operating Systems: Windows, macOS, Linux (Linux recommended for maximum performance).

---

## Installation

### Step 1: Clone the Repository

Download the ComfyUI project using Git:
```sh
git clone https://github.com/comfyanonymous/ComfyUI.git
```
Navigate to the project directory:
```sh
cd ComfyUI
```
### Step 2: Set up a Python Virtual Environment

Create and activate a virtual environment to isolate dependencies:
```sh
python -m venv venv  
source venv/bin/activate     (Linux/MacOS)  
venv\Scripts\activate        (Windows)
```

### Step 3: Install Dependencies

Install required Python dependencies:
```sh
pip install -r requirements.txt
```
### Step 4: Launch ComfyUI

Run the server:
```sh
python main.py
```

### Step 5: Open the UI

Open your browser and go to:
http://127.0.0.1:8188

---

## Features of ComfyUI

### Node-Based Workflow:
- Design workflows visually by dragging and dropping **nodes** (components of the Stable Diffusion pipeline).
- Connect nodes to define the relationships between tasks.

### Key Capabilities:
1. **Model Management**:
   - Load and switch between **Stable Diffusion models** (checkpoints, `.ckpt`, or `.safetensors` files).
   - Manage **VAE (Variational Autoencoder)** for enhancing outputs.
2. **Image Generation**:
   - Create images using **Text-to-Image (T2I)** or **Image-to-Image (I2I)** workflows.
   - Configure parameters such as resolution, sampling method, CFG scale, and steps.
3. **Customizable Pipelines**:
   - Add nodes for specialized tasks (e.g., prompt preprocessing, model loading, sampler, output).
4. **Batch Processing**:
   - Process multiple prompts, seeds, or images simultaneously.
5. **Advanced Imaging**:
   - Use additional nodes like **ControlNet** for advanced image editing capabilities (inpainting, sketch-to-image, etc.).

---

## Using ComfyUI: Example Workflow

### Basic Text-to-Image Workflow:

1. Drag **Text Prompt Node** to the canvas.
   - Input your desired prompt, such as "a futuristic cityscape".
2. Add **Model Loader Node** and select your preferred AI model (e.g., Stable Diffusion checkpoint).
3. Attach **Sampler Node** to specify the sampling method (e.g., DDIM, Euler).
4. Connect an **Image Output Node** at the end to display/save the generated image.
5. Run the pipeline after connecting the nodes!

---

## Adding Custom Models and VAEs

### Loading Models:
Place your `.ckpt` or `.safetensors` files under the appropriate `models` directory.

### Adding VAEs:
Place your VAE files into the `vae` directory.

### Adding ControlNet Models:
- Download pre-trained **ControlNet checkpoint files** and add them to the models directory.
- Configure ControlNet nodes in the ComfyUI interface to enable advanced workflows.

---

## Configuration File

Customize the `config.json` file for better performance:
```json
{
    "device": "cuda",
    "always_use_CPU": false,
    "workflow_directory": "./workflows",
    "model_directory": "./models"
}
```
---

## Troubleshooting

1. **Python Version Requirements**:
   Ensure you're using **Python 3.10 or higher**.

2. **Missing Dependencies**:
   Run `pip install -r requirements.txt` again to install dependencies properly.

3. **GPU Compatibility Issues**:
   Install the correct **CUDA Toolkit** for NVIDIA GPUs, or use **ROCm** for AMD GPUs.

4. **Performance Tuning**:
   Enable optimizations like **XFormers** to improve memory efficiency (if supported).

---

## Batch Processing Example

To process multiple image prompts:

1. Add a **Batch Prompt Node**.
2. Specify multiple text prompts (e.g., "a sunset over the ocean" and "a forest in winter").
3. Configure the **Sampler Node** and enable batch settings as needed.
4. Generate all images concurrently.

---

## Deployment and Launch Guide

### Launch ComfyUI:

Run the main server script:
```sh
python main.py
```
### Open Web Interface:

Access http://127.0.0.1:8188 in your browser.

---

## Summary Workflow

1. Install dependencies.
2. Open and configure ComfyUI.
3. Build workflows visually by connecting nodes.
4. Experiment with parameters and generate your desired output.
5. Save results and refine workflows iteratively.

---

This Markdown file outlines the features, installation instructions, workflow usage, and troubleshooting tips for getting started with **ComfyUI**. Save the content into a file named `ComfyUI.md` for documentation purposes.
