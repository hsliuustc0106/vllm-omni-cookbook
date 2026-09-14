# GLM-Image v0.20 benchmark artifacts

These files reproduce the v0.20 run used by the GLM-Image cookbook table.

- `test_glm_image_vllm_omni_focused_inline.json`: focused 6-case matrix. v0.20 uses external `--deploy-config` YAML paths because `deploy-config-inline` is not supported by the v0.20 serve CLI.
- `glm_image_1024_1472x1088_steps50_t2i_i2i.json`: shared workload, 1024x1024 first, then 1472x1088, each with t2i and i2i at 50 steps.

Deploy YAML is not shipped here. See the **v0.20.0** section in [`../index.md`](../index.md) for an inline example and bring-your-own YAML steps.
