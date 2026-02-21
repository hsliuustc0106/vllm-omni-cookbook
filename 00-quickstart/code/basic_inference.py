"""
vLLM-Omni Basic Inference Example

This demonstrates vLLM-Omni's text-to-image generation capabilities.
Run this script to see vLLM-Omni in action.
"""

from vllm_omni.entrypoints.omni import Omni


def main():
    """Generate an image from a text prompt."""

    # Initialize the Omni API with a text-to-image model
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")

    # Define your prompt
    prompt = "a cup of coffee on the table"

    # Generate the image
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images

    # Save the result
    images[0].save("coffee.png")
    print(f"Image saved to coffee.png")


def batch_example():
    """Generate multiple images from a list of prompts."""

    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompts = [
        "a cup of coffee on a table",
        "a toy dinosaur on a sandy beach",
        "a fox waking up in bed and yawning",
    ]
    omni_outputs = omni.generate(prompts)

    for i_prompt, prompt_output in enumerate(omni_outputs):
        this_request_output = prompt_output.request_output[0]
        this_images = this_request_output.images
        for i_image, image in enumerate(this_images):
            path = f"p{i_prompt}-img{i_image}.jpg"
            image.save(path)
            print(f"Saved: {path}")


if __name__ == "__main__":
    main()
    # Uncomment to try batch example
    # batch_example()
