# DiT & Parallel Generation Recipes

Non-autoregressive architectures including Diffusion Transformers (DiT) and parallel generation models with vLLM-Omni.

## Topics

This section will cover:

- Diffusion Transformer (DiT) model inference
- Parallel generation strategies
- Single GPU optimization for DiT
- Multi-GPU with tensor parallelism
- Pipeline parallelism for large models
- CPU fallback and quantization
- Memory management techniques

## vLLM-Omni Extended Architectures

Unlike traditional vLLM which focuses on autoregressive models, vLLM-Omni extends support to:

- **Diffusion Transformers (DiT)**: Parallel denoising for image/video generation
- **Non-AR Models**: Models that don't rely on sequential token generation
- **Hybrid Architectures**: Combining AR and non-AR components

## Status

🚧 Planned - Recipes coming soon

## Related

- [Performance](../06-performance/) - Benchmarking and profiling
- [Deployment](../02-deployment/) - Production hardware setup
