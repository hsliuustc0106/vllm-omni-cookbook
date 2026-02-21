# run-dev

Run the vLLM-Omni development server for testing recipes.

## Trigger Conditions

Use this skill when the user asks to:
- Start the development server
- Test a recipe locally
- Run vLLM-Omni locally

## Implementation

1. Check if vLLM-Omni is installed
2. Determine which recipe/model to run
3. Start the vLLM-Omni server with appropriate configuration
4. Provide instructions for testing the endpoint

## Common Commands

```bash
# Start omni-modality server
vllm serve omni --model <model_name>

# Run with specific input type
vllm serve omni --model <vlm_model> --input-type image

# Run with custom port
vllm serve omni --model <model_name> --port 8080

# Run inference programmatically
python -m vllm.omni run --model <model> --input-type <text|image|audio|video>
```

## vLLM-Omni vs vLLM Commands

| vLLM | vLLM-Omni |
|------|-----------|
| `vllm serve <model>` | `vllm serve omni --model <model>` |
| `from vllm import LLM` | `from vllm.omni import OmniProcessor` |
