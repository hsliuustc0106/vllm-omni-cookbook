# GLM-Image v0.22 benchmark artifacts

These files reproduce the v0.22 run used by the GLM-Image cookbook table.

- `test_glm_image_vllm_omni_focused_inline.json`: focused 6-case matrix. v0.22 can use `deploy-config-inline` directly in the perf JSON.
- `glm_image_1024_1472x1088_steps50_t2i_i2i.json`: shared workload, 1024x1024 first, then 1472x1088, each with t2i and i2i at 50 steps.
- `deploy/*.yaml`: equivalent deploy YAMLs for manual `vllm-omni serve --deploy-config ...` examples, especially for multi-stage AR/DiT parallel configs.
