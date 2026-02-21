# Basic Omni-Modality Inference

Your first vLLM-Omni inference example - generate images from text prompts.

## Description

This recipe demonstrates the simplest way to use vLLM-Omni for text-to-image generation. You'll learn how to load a model and generate images from text prompts using the `Omni` API.

## Prerequisites

- [ ] Linux OS
- [ ] Python 3.12
- [ ] GPU with CUDA or ROCm support
- [ ] vLLM 0.15.0 installed
- [ ] vLLM-Omni installed from source

### Installation

```bash
# Create virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate

# On CUDA
uv pip install vllm==0.15.0 --torch-backend=auto

# On ROCm
uv pip install vllm==0.15.0 --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700

# Install vLLM-Omni from source
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
```

## Difficulty

Beginner

## Estimated Time

5 minutes

## Steps

### Step 1: Create a basic text-to-image script

Create a new file `basic_inference.py` with the following code:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images
    images[0].save("coffee.png")
```

### Step 2: Run the inference

```bash
python basic_inference.py
```

This will generate an image file `coffee.png` based on your prompt.

### Step 3: Batch processing

You can pass a list of prompts to process multiple images:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
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
            image.save(f"p{i_prompt}-img{i_image}.jpg")
            print("saved to", f"p{i_prompt}-img{i_image}.jpg")
```

> **Note:** Batch inference is not currently recommended for all models as it may not provide significant performance improvements. This feature is primarily for interface compatibility.

## Expected Output

After running the script, you should see:

```text
saved to p0-img0.jpg
saved to p1-img0.jpg
saved to p2-img0.jpg
```

And corresponding image files will be created in your directory.

## Complete Example

```python
"""
vLLM-Omni Basic Inference Example
Text-to-image generation using the Omni API
"""

from vllm_omni.entrypoints.omni import Omni


def generate_image(model: str, prompt: str, output_path: str):
    """
    Generate an image from a text prompt.

    Args:
        model: Model name (e.g., "Tongyi-MAI/Z-Image-Turbo")
        prompt: Text description of the image to generate
        output_path: Where to save the generated image
    """
    omni = Omni(model=model)
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images
    images[0].save(output_path)
    print(f"Image saved to {output_path}")


def generate_batch(model: str, prompts: list, output_dir: str = "."):
    """
    Generate multiple images from a list of prompts.

    Args:
        model: Model name
        prompts: List of text prompts
        output_dir: Directory to save images
    """
    omni = Omni(model=model)
    omni_outputs = omni.generate(prompts)

    for i_prompt, prompt_output in enumerate(omni_outputs):
        this_request_output = prompt_output.request_output[0]
        this_images = this_request_output.images
        for i_image, image in enumerate(this_images):
            path = f"{output_dir}/p{i_prompt}-img{i_image}.jpg"
            image.save(path)
            print(f"Saved: {path}")


if __name__ == "__main__":
    # Single image generation
    generate_image(
        model="Tongyi-MAI/Z-Image-Turbo",
        prompt="a cup of coffee on the table",
        output_path="coffee.png"
    )

    # Batch generation
    # generate_batch(
    #     model="Tongyi-MAI/Z-Image-Turbo",
    #     prompts=[
    #         "a cup of coffee on a table",
    #         "a toy dinosaur on a sandy beach",
    #         "a fox waking up in bed and yawning",
    #     ]
    # )
```

## vLLM-Omni Key Features

| Feature | Description |
|---------|-------------|
| **Omni-modality** | Process text, image, video, and audio |
| **Non-AR Support** | Diffusion Transformers (DiT) and parallel models |
| **Heterogeneous Outputs** | Generate text, images, or multimodal content |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Module not found | Ensure vLLM-Omni is installed: `uv pip install -e .` from source |
| CUDA out of memory | Use a smaller model or enable quantization |
| Slow generation | Check GPU utilization and consider batch size settings |
| Import errors | Verify vLLM 0.15.0 is installed before vLLM-Omni |

## Related Recipes

- [Online Serving](./online-serving.md) - OpenAI-compatible API server
- [Text-to-Video](../01-inference/text-to-video.md) - Video generation
- [Image-to-Image](../03-multimodal/image-to-image.md) - Image transformation

## Further Reading

- [vLLM-Omni Documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [Offline Inference Guide](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/)
- [Supported Models](https://docs.vllm.ai/projects/vllm-omni/en/latest/models/supported_models/)
