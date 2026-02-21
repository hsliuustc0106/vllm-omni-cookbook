# new-recipe

Create a new recipe in the vLLM-Omni cookbook.

## Trigger Conditions

Use this skill when the user asks to:
- Add a new recipe
- Create documentation for a feature
- Write an example or guide

## Implementation

1. Read `templates/recipe-template.md`
2. Determine the appropriate category (00-07)
3. Create the recipe markdown file in the category folder
4. Create associated code files in the category's `code/` subdirectory
5. Use vLLM-Omni API (`from vllm.omni import OmniProcessor`)
6. Update `topics/index.md` with the new recipe
7. Update the category README.md

## vLLM-Omni Recipe Checklist

- [ ] Title and description
- [ ] Prerequisites section
- [ ] Difficulty level and estimated time
- [ ] Step-by-step instructions
- [ ] Code examples using `vllm.omni` API
- [ ] Modality specified (text, vision, audio, video)
- [ ] Expected output section
- [ ] Troubleshooting section
- [ ] vLLM-Omni features highlighted
