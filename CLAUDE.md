# vLLM-Omni Cookbook - Guidance for Claude Code

This file provides context for Claude Code instances working on this repository.

## What is vLLM-Omni?

vLLM-Omni extends vLLM with support for:
- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive architectures**: Diffusion Transformers (DiT) and parallel generation models
- **Heterogeneous outputs**: From text to multimodal outputs

**Key Difference from vLLM**: vLLM focuses on text-only autoregressive models. vLLM-Omni extends this to support multiple modalities and non-AR architectures.

## Repository Structure

```
vllm-omni-cookbook/
├── README.md              # Project overview and quick links
├── CONTRIBUTING.md        # Contribution guidelines
├── CLAUDE.md              # This file
├── .claude/skills/        # Custom Claude Code skills
├── templates/             # Reusable templates
├── topics/                # Master table of contents
├── 00-quickstart/         # Getting started with omni-modality
├── 01-inference/          # Text, vision, audio inference
├── 02-deployment/         # Production deployment
├── 03-multimodal/         # Cross-modal applications
├── 04-hardware/           # DiT models & parallel generation
├── 05-best-practices/     # Security, monitoring
├── 06-performance/        # Benchmarking, profiling
└── 07-troubleshooting/    # Common issues & solutions
```

## Key Patterns

### Adding New Recipes

1. Use `templates/recipe-template.md` as the starting point
2. Place in the appropriate category folder (00-07)
3. Add code examples in the `code/` subdirectory
4. Update `topics/index.md` to include the new recipe
5. Update the category README.md

### Recipe Format

Each recipe should include:
- Clear title and description
- Prerequisites and difficulty level
- Step-by-step instructions with code
- Expected output examples
- Troubleshooting section
- vLLM-Omni specific features highlighted

### Code Style

- Keep examples minimal and focused
- Follow PEP 8 for Python code
- Use `vllm.omni` imports for omni-modality features
- Include type hints for clarity
- Test all examples before committing

## Common Tasks

### Adding a new omni-modality recipe:
1. Determine which modality (text, vision, audio, video)
2. Choose the appropriate category
3. Read the recipe template
4. Read similar existing recipes for context
5. Create the recipe markdown file
6. Create associated code files
7. Update index and README files

### When asked to improve documentation:
1. Read the existing content
2. Identify gaps or unclear sections
3. Edit in place rather than rewriting entirely
4. Preserve the voice and structure

## vLLM-Omni vs vLLM Quick Reference

| Feature | vLLM | vLLM-Omni |
|---------|------|-----------|
| Import | `from vllm import LLM` | `from vllm_omni.entrypoints.omni import Omni` |
| Offline API | `LLM(model).generate()` | `Omni(model).generate()` |
| Online CLI | `vllm serve <model>` | `vllm serve <model> --omni` |
| Modalities | Text | Text, Image, Video, Audio |
| Models | AR only | AR + DiT + Parallel |
| Output | Text | Heterogeneous (images, audio, etc.) |

## Resources

- vLLM-Omni Documentation: https://docs.vllm.ai/projects/vllm-omni/en/latest/
- GitHub: https://github.com/vllm-project/vllm-omni
