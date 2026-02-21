# add-model

Add a new model example to the vLLM-Omni cookbook.

## Trigger Conditions

Use this skill when the user asks to:
- Add a model example
- Document a new model
- Add model-specific configuration

## Implementation

1. Search for existing documentation of the model
2. Determine the model type (text, vision, audio, video, DiT)
3. Choose the appropriate category
4. Create a new recipe following `templates/recipe-template.md`
5. Include vLLM-Omni specific configuration
6. Document modality requirements and compatibility
7. Update `topics/index.md` and category README.md

## vLLM-Omni Model Types

- **Text Models**: Autoregressive language models
- **Vision Models (VLM)**: Image + text processing
- **Audio Models**: Audio understanding/generation
- **DiT Models**: Diffusion Transformers for parallel generation
- **Multimodal Models**: Cross-modal architectures

## Model Information to Include

- Model name and source
- Modality type (text, vision, audio, video)
- Architecture type (AR or DiT)
- Model size and requirements
- Memory requirements
- vLLM-Omni specific configuration
- Example usage with correct input_type
