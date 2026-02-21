# Contributing to vLLM-Omni Cookbook

Thank you for your interest in contributing! This document provides guidelines for contributing to the vLLM-Omni Cookbook.

## About vLLM-Omni

vLLM-Omni extends vLLM to support:
- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive architectures**: Diffusion Transformers (DiT) and parallel generation models
- **Heterogeneous outputs**: Text, images, and multimodal generation

## Getting Started

1. Fork the repository
2. Create a branch for your contribution (`git checkout -b feature/my-recipe`)
3. Make your changes following the guidelines below
4. Submit a pull request

## Adding New Recipes

### Use the Template

All recipes should follow the [recipe template](templates/recipe-template.md). This ensures consistency across the cookbook.

### Recipe Structure

```
<category-folder>/
├── my-recipe.md           # The main recipe documentation
└── code/
    └── my_example.py      # Associated code examples
```

### Guidelines

- **Focus on vLLM-Omni capabilities**: Highlight omni-modality, DiT, or heterogeneous outputs
- **Beginner-friendly**: Write for users who may be new to vLLM-Omni
- **Concise examples**: Code should be minimal and focused on the concept
- **Test your code**: Verify all examples run before submitting
- **Document dependencies**: List required packages and versions
- **Include troubleshooting**: Anticipate common issues

## Categories

Place your recipe in the appropriate category:

- `00-quickstart/`: Getting started with omni-modality
- `01-inference/`: Text, vision, audio generation & streaming
- `02-deployment/`: Production deployment
- `03-multimodal/`: Cross-modal applications
- `04-hardware/`: DiT models and parallel generation (formerly "Hardware")
- `05-best-practices/`: Security, monitoring, patterns
- `06-performance/`: Benchmarking and profiling
- `07-troubleshooting/`: Common issues and solutions

## Pull Request Process

1. Update the relevant category README.md to include your recipe
2. Update [topics/index.md](topics/index.md) with your new recipe
3. Ensure your recipe follows the template
4. Test all code examples
5. Submit PR using the provided template

## Code Style

- Python code should follow PEP 8
- Use `vllm_omni` imports for vLLM-Omni features (e.g., `from vllm_omni.entrypoints.omni import Omni`)
- Use type hints where appropriate
- Keep examples under 100 lines when possible
- Add docstrings for non-trivial functions

## Example: vLLM vs vLLM-Omni

**vLLM (text-only, autoregressive):**
```python
from vllm import LLM, SamplingParams
llm = LLM(model="gpt2")
outputs = llm.generate(["Hello world"], SamplingParams())
```

**vLLM-Omni (omni-modality, text-to-image):**
```python
from vllm_omni.entrypoints.omni import Omni
omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
outputs = omni.generate("a beautiful landscape")
images = outputs[0].request_output[0].images
images[0].save("output.png")
```

**vLLM-Omni (online serving):**
```bash
# Server
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091

# Client (OpenAI-compatible)
curl http://localhost:8091/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "prompt"}]}'
```

## Questions?

Open an issue for discussion before making significant changes.
