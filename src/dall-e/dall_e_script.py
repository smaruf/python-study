"""
DALL·E Automation Script

This Python script demonstrates how to interact with OpenAI's **DALL·E API** to generate images based on textual descriptions (Text-to-Image), edit existing images, or create multiple variations.

DALL·E is a deep learning model developed by OpenAI to generate digital art and realistic visuals using natural language prompts. Users can work programmatically with the API to automate workflows.

================================================================
Features:
1. Generate images using text prompts (Text-to-Image).
2. Edit images by applying transformations or new visual elements (Image-to-Image editing).
3. Generate multiple variations of images (Batch Generation).
================================================================

# Prerequisites:
1. **Python Version**:
   - Install Python 3.10+.

2. **Install Required Libraries**:
   - Install OpenAI Python Library:
     pip install openai
   - Install Requests Library:
     pip install requests

3. **OpenAI API Key**:
   - Sign up at https://platform.openai.com and generate an API key.
   - Copy the API key into the script.

4. **Ensure Internet Connectivity**:
   - The script interacts with OpenAI's API, requiring active internet.

================================================================
Usage Instructions:
1. Save this script as `dall_e_script.py`.
2. Add your OpenAI API Key to the `openai.api_key` line.
3. Run the script using prompts to generate images or edit existing images.

Command:
    python dall_e_script.py

================================================================
Endpoints:
- **Image Generation** (`Text-to-Image`):
    Generates an image based on a description (prompt).

- **Image Editing** (`Image-to-Image`):
    Modifies an uploaded image using additional textual instructions.

- **Batch Generation**:
    Produces multiple images in a single request.

================================================================
"""
import openai
import requests

# Set up the OpenAI API key (Replace with your actual API key)
openai.api_key = "your-api-key"

def generate_image(prompt, output_file="generated_image.png"):
    """
    Generate an image using DALL·E based on a text prompt.
    
    Parameters:
        prompt (str): A textual description of the image to generate 
                      (e.g., "a futuristic cityscape at sunset").
        output_file (str): The filename to save the generated image.
    
    Returns:
        None: Saves the generated image to the specified file locally.

    Example Usage:
        generate_image(prompt="A serene landscape of mountains under the night sky", 
                       output_file="serene_landscape.png")
    """
    try:
        # Request image generation with DALL·E API
        response = openai.Image.create(
            prompt=prompt,
            n=1,  # Number of images to generate
            size="1024x1024"  # Resolution: "256x256", "512x512", or "1024x1024"
        )
        
        # Get the image URL from the API response
        image_url = response['data'][0]['url']
        
        # Download and save the image locally
        img_data = requests.get(image_url).content
        with open(output_file, "wb") as f:
            f.write(img_data)

        print(f"Image generated and saved successfully: {output_file}")
        
    except Exception as e:
        print(f"Error generating image: {e}")

def edit_image(input_file, edit_description, output_file="edited_image.png"):
    """
    Edit an existing image using DALL·E Image Editing APIs to transform or modify the image.

    Parameters:
        input_file (str): Path to the image file (e.g., "original_image.jpg") to edit.
        edit_description (str): A textual instruction describing how to edit the image 
                                (e.g., "Add a futuristic city to the background").
        output_file (str): The filename to save the edited image.

    Returns:
        None: Saves the edited image to the specified file locally.

    Example Usage:
        edit_image(input_file="original.jpg", 
                   edit_description="Add a rainbow to the sky", 
                   output_file="edited_image.png")
    """
    try:
        # Request image editing via OpenAI API
        response = openai.Image.create_edit(
            image=open(input_file, "rb"),
            prompt=edit_description,
            n=1,  # Number of variations to generate
            size="1024x1024"  # Resolution of generated image
        )
        
        # Get the edited image URL
        edited_image_url = response['data'][0]['url']
        
        # Download and save the edited image locally
        img_data = requests.get(edited_image_url).content
        with open(output_file, "wb") as f:
            f.write(img_data)

        print(f"Edited image saved successfully: {output_file}")
        
    except Exception as e:
        print(f"Error editing image: {e}")

def generate_batch(prompt, num_images=3):
    """
    Generate multiple images from a single prompt via batch generation.

    Parameters:
        prompt (str): A textual description of the image (e.g., "A sunset over the ocean").
        num_images (int): The number of images to generate (default: 3).
    
    Returns:
        None: Saves all the generated images locally.

    Example Usage:
        generate_batch(prompt="A forest in autumn", num_images=5)
    """
    try:
        # Request batch image generation
        response = openai.Image.create(
            prompt=prompt,
            n=num_images,  # Number of images to generate
            size="1024x1024"  # Resolution of each image
        )
        
        # Iterate over the generated image URLs
        for i, data in enumerate(response['data']):
            image_url = data['url']
            img_data = requests.get(image_url).content
            with open(f"batch_image_{i + 1}.png", "wb") as f:
                f.write(img_data)
            print(f"Image {i + 1} saved successfully.")
        
    except Exception as e:
        print(f"Error generating batch images: {e}")

if __name__ == "__main__":
    """
    Main function to demonstrate DALL·E features (generation, editing, and batch processing).
    
    Instructions:
    - Replace the API key and input files with actual values.
    - Select the desired feature by uncommenting one of the following:

        generate_image(prompt="A futuristic cityscape at sunset", 
                       output_file="futuristic_city.png")

        edit_image(input_file="original_image.jpg", 
                   edit_description="Add a hat to the dog", 
                   output_file="edited_dog.png")

        generate_batch(prompt="A beautiful forest in sunlight", num_images=3)
    
    Example:
        Run the script as:
        python dall_e_script.py
    """
    # Uncomment one of the following examples based on your intended operation:

    # Generate an image from a prompt
    generate_image(prompt="A futuristic cityscape at sunset", output_file="futuristic_city.png")

    # Edit an existing image
    # edit_image(input_file="original_image.jpg", edit_description="Add a rainbow in the background", output_file="edited_image.png")

    # Generate a batch of images
    # generate_batch(prompt="A beautiful forest in sunlight", num_images=3)
