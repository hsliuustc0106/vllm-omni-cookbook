# vLLM-Omni Cookbook

A practical guide to vLLM-Omni with recipes, examples, and best practices for omni-modality inference and serving.

## Overview

vLLM-Omni extends vLLM to support **omni-modality** model inference and serving. While vLLM was designed for text-based autoregressive generation, vLLM-Omni provides:

- **Omni-modality Support**: Text, image, video, and audio data processing
- **Non-Autoregressive Architectures**: Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous Outputs**: From traditional text generation to multimodal outputs

This cookbook provides hands-on recipes to help you leverage these extended capabilities.

## Key Differences from vLLM

| Feature | vLLM | vLLM-Omni |
|---------|------|-----------|
| Modalities | Text | Text, Image, Video, Audio |
| Architectures | Autoregressive | AR + DiT + Parallel |
| Outputs | Text | Multimodal outputs |
| Use Cases | LLM serving | Omni-modality AI |

## Quick Start

### Prerequisites

- OS: Linux
- Python: 3.12

### Installation

```bash
# Create virtual environment with uv
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

### Offline Inference Example

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
prompt = "a cup of coffee on the table"
outputs = omni.generate(prompt)
images = outputs[0].request_output[0].images
images[0].save("coffee.png")
```

### Online Serving

```bash
# Start the server
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091
```

```bash
# Make a request
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
      "guidance_scale": 4.0
    }
  }'
```

## Table of Contents

| Category | Description | Status |
|----------|-------------|--------|
| [00 - Quickstart](00-quickstart/) | Get started with omni-modality inference | ✅ Available |
| [01 - Inference](01-inference/) | Text, vision, audio generation & streaming | 🚧 Planned |
| [02 - Deployment](02-deployment/) | Production serving for omni-modality models | 🚧 Planned |
| [03 - Multimodal](03-multimodal/) | Cross-modal applications and workflows | 🚧 Planned |
| [04 - DiT Models](04-hardware/) | Diffusion Transformers and parallel generation | 🚧 Planned |
| [05 - Best Practices](05-best-practices/) | Security, monitoring, error handling | 🚧 Planned |
| [06 - Performance](06-performance/) | Benchmarking and optimization strategies | 🚧 Planned |
| [07 - Troubleshooting](07-troubleshooting/) | Common issues and solutions | 🚧 Planned |

See [topics/index.md](topics/index.md) for a detailed table of contents.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to add new recipes, report issues, or improve existing content.

## Resources

- [vLLM-Omni Documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/)
- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)
- [Community Discord](https://discord.gg/vllm)
