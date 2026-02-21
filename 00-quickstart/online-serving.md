# Online Serving with OpenAI-Compatible API

Serve vLLM-Omni models with an OpenAI-compatible HTTP API.

## Description

This recipe shows how to start a vLLM-Omni server and make requests using the OpenAI-compatible API. The server supports text-to-image generation and other omni-modality tasks.

## Prerequisites

- [ ] Linux OS
- [ ] Python 3.12
- [ ] GPU with CUDA or ROCm support
- [ ] vLLM 0.15.0 installed
- [ ] vLLM-Omni installed from source

## Difficulty

Beginner

## Estimated Time

3 minutes

## Steps

### Step 1: Start the vLLM-Omni Server

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091
```

The `--omni` flag enables vLLM-Omni mode for omni-modality support.

### Step 2: Make a text-to-image request

Use curl to generate an image:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "a cup of coffee on the table"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > coffee.png
```

### Step 3: Try different parameters

Experiment with generation parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `height` | Output image height | Model default |
| `width` | Output image width | Model default |
| `num_inference_steps` | Number of denoising steps | Model default |
| `guidance_scale` | Classifier-free guidance scale | Model default |
| `seed` | Random seed for reproducibility | Random |

## Expected Output

A `coffee.png` file will be created with the generated image.

## Complete Example: Python Client

```python
"""
vLLM-Omni Online Serving Client Example
"""

import base64
import json
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


if __name__ == "__main__":
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
```

## Server Configuration Options

```bash
vllm serve <model> --omni [options]

Common options:
  --port PORT              Server port (default: 8091)
  --host HOST              Server host (default: 0.0.0.0)
  --tensor-parallel-size N  Number of GPUs for tensor parallelism
  --max-model-len LEN      Maximum model length
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Ensure the server is running and the port is correct |
| CUDA out of memory | Use tensor parallelism or a smaller model |
| Slow generation | Reduce `num_inference_steps` or image dimensions |
| Port already in use | Change the port with `--port` |

## Related Recipes

- [Offline Inference](./basic-inference.md) - Direct Python API usage
- [Deployment](../02-deployment/) - Production deployment guides

## Further Reading

- [OpenAI-Compatible API Docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/openai_compatible_api/)
- [Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/openai_compatible_api/image_generation/)
