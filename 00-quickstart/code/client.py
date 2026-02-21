"""
vLLM-Omni Online Serving Client Example

This demonstrates how to interact with the vLLM-Omni server
using the OpenAI-compatible API.
"""

import base64
import requests
from PIL import Image
from io import BytesIO


def generate_image(
    prompt: str,
    base_url: str = "http://localhost:8091",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42
) -> Image.Image:
    """
    Generate an image using the vLLM-Omni server.

    Args:
        prompt: Text description of the image
        base_url: Server URL
        height: Image height
        width: Image width
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        seed: Random seed

    Returns:
        PIL Image object
    """
    url = f"{base_url}/v1/chat/completions"

    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()

    # Extract the base64 image data
    image_url = result["choices"][0]["message"]["content"][0]["image_url"]["url"]
    image_data = image_url.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    return Image.open(BytesIO(image_bytes))


def main():
    """Generate example images."""

    # Generate an image
    image = generate_image("a cup of coffee on the table")
    image.save("coffee.png")
    print("Image saved to coffee.png")

    # Generate with custom parameters
    image = generate_image(
        prompt="a sunset over mountains",
        height=768,
        width=1024,
        num_inference_steps=30,
        guidance_scale=5.0
    )
    image.save("sunset.png")
    print("Image saved to sunset.png")


if __name__ == "__main__":
    main()
