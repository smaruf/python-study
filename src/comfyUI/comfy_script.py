"""
ComfyUI Automation Script

This script builds and submits workflows to the ComfyUI server to automate Stable Diffusion pipelines using a text-to-image workflow. ComfyUI allows users to design AI workflows visually or programmatically by using node-based systems. This script demonstrates how to create and send workflows programmatically to the ComfyUI API.

================================================================
Features:
1. Build a Text-to-Image workflow programmatically and submit it via the ComfyUI API.
2. Utilize Stable Diffusion models, samplers, and configurable pipeline parameters.
3. Automate workflows using Python and JSON payloads.
4. Process and retrieve results efficiently.

================================================================
Prerequisites:
1. Install Python 3.10 or higher.
2. Clone and set up ComfyUI:
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI

3. Install dependencies:
   pip install -r requirements.txt

4. Start the ComfyUI server:
   Run `python main.py` in the ComfyUI directory.

5. Ensure a Stable Diffusion model file (e.g., `.safetensors` or `.ckpt`) is placed in the correct directory (typically `/models`).

================================================================
Endpoints Used:
- ComfyUI API URL: http://127.0.0.1:8188/api/workflow

================================================================
Usage:
1. Save this script as `comfy_script.py`.
2. Run the script: `python comfy_script.py`.
3. Provide a text prompt (e.g., "a futuristic cityscape") and model name to create and execute a workflow.

================================================================
Example Workflow:
Text Prompt → Model Loader → Sampler → Output Node
"""

import requests
import json

# Define the ComfyUI server URL
COMFYUI_URL = "http://127.0.0.1:8188"

def create_text_to_image_workflow(prompt, model_name="model.safetensors"):
    """
    Creates a Text-to-Image workflow JSON payload for ComfyUI.

    Parameters:
        prompt (str): The text description for the image (e.g., "a futuristic cityscape").
        model_name (str): The name of the Stable Diffusion model file to load.

    Returns:
        dict: Workflow JSON payload for ComfyUI API submission.
    
    Example:
        workflow = create_text_to_image_workflow("a sunset over the ocean", "model.safetensors")
    """
    workflow = {
        "workflow_name": "Text-to-Image Example",
        "nodes": [
            {
                "name": "LoadModel",
                "type": "model_loader",
                "params": {
                    "ckpt_file": model_name,  # Path to the model checkpoint file
                },
                "id": "load_model",  # Unique node ID
            },
            {
                "name": "TextPrompt",
                "type": "text_prompt",
                "params": {
                    "prompt": prompt,
                },
                "id": "text_prompt",
            },
            {
                "name": "Sampler",
                "type": "sampler",
                "params": {
                    "steps": 50,  # Number of sampling steps
                    "cfg_scale": 7.5,  # Classifier-free guidance scale
                    "seed": 42,  # Random seed for deterministic output
                },
                "id": "sampler",
            },
            {
                "name": "ImageOutput",
                "type": "output",
                "id": "output",
            },
        ],
        "connections": [
            ["load_model", "text_prompt"],
            ["text_prompt", "sampler"],
            ["sampler", "output"],
        ],
    }
    return workflow

def submit_workflow(workflow):
    """
    Submits the workflow JSON payload to the ComfyUI API for processing.

    Parameters:
        workflow (dict): Workflow JSON payload for submission.

    Returns:
        dict: Response from ComfyUI server containing success or error details.

    Raises:
        requests.HTTPError: If workflow submission fails.

    Example:
        response = submit_workflow(workflow)
    """
    response = requests.post(f"{COMFYUI_URL}/api/workflow", json=workflow)
    if response.status_code == 200:
        print("Workflow submitted successfully!")
        return response.json()  # Return the server's response
    else:
        print("Failed to submit workflow:", response.text)
        response.raise_for_status()

def main():
    """
    Main function to create a Text-to-Image workflow and submit it to ComfyUI.

    Workflow Steps:
        1. Load Stable Diffusion model using the Model Loader node.
        2. Provide user-defined text prompt via the Text Prompt node.
        3. Configure sampling parameters via the Sampler node.
        4. Output the generated image.

    Example Execution:
        Run `python comfy_script.py` to submit the workflow.

    Returns:
        None
    """
    # Define input parameters
    prompt = "a beautiful futuristic cityscape at sunset"
    model_name = "model.safetensors"  # Ensure the model is placed in the correct directory

    # Create the workflow JSON payload
    workflow = create_text_to_image_workflow(prompt, model_name)

    # Submit the workflow to ComfyUI
    print("Submitting workflow to ComfyUI...")
    result = submit_workflow(workflow)
    print("Server Response:", json.dumps(result, indent=2))

if __name__ == "__main__":
    """
    Entry point for the script. Build and submit a Text-to-Image workflow to ComfyUI.
    
    To execute:
        - Ensure ComfyUI server is running (`python main.py` in ComfyUI directory).
        - Run this script: `python comfy_script.py`.

    Expected Output:
        Success message and JSON response with workflow details.
    """
    main()
