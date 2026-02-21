# vLLM-Omni Cookbook - Table of Contents

This is the master table of contents for all recipes in the vLLM-Omni cookbook.

## About vLLM-Omni

vLLM-Omni extends vLLM with support for:
- **Omni-modality**: Text, image, video, and audio processing
- **Non-autoregressive architectures**: Diffusion Transformers (DiT) and parallel generation
- **Heterogeneous outputs**: Text, images, and multimodal generation

## Status Legend

- ✅ Available - Recipe is complete and ready to use
- 🚧 Planned - Recipe is planned but not yet written
- 🔄 In Progress - Recipe is currently being developed

---

## 00 - Quickstart

Getting started with vLLM-Omni's omni-modality capabilities.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| [Basic Offline Inference](../00-quickstart/basic-inference.md) | Text-to-image generation with Python API | ✅ Available | Beginner |
| [Online Serving](../00-quickstart/online-serving.md) | OpenAI-compatible API server | ✅ Available | Beginner |

---

## 01 - Inference

Text, vision, audio inference with vLLM-Omni's multi-modal support.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Text generation with sampling strategies
- Vision Language Model (VLM) inference
- Audio processing and generation
- Streaming inference for real-time responses
- Batch processing for efficiency

---

## 02 - Deployment

Production deployment strategies for vLLM-Omni servers.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Docker containerization
- Kubernetes deployment
- Multi-modal model serving
- Load balancing strategies

---

## 03 - Multimodal

Advanced cross-modal applications and workflows.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Cross-modal understanding
- Image-to-text, text-to-image workflows
- Video understanding and generation
- Multi-turn multimodal conversations

---

## 04 - DiT & Parallel Generation

Non-autoregressive architectures including Diffusion Transformers.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Diffusion Transformer (DiT) inference
- Parallel generation strategies
- Multi-GPU for DiT models
- Memory optimization for diffusion

---

## 05 - Best Practices

Security patterns, monitoring, and production recommendations.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Multi-modal input validation
- Rate limiting for omni-modality endpoints
- Monitoring multimodal pipelines
- Cost management strategies

---

## 06 - Performance

Benchmarking, profiling, and optimization for vLLM-Omni.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Multimodal throughput benchmarking
- DiT model optimization
- Memory profiling for vision models
- GPU utilization analysis

---

## 07 - Troubleshooting

Common issues, error messages, and solutions for vLLM-Omni.

| Recipe | Description | Status | Difficulty |
|--------|-------------|--------|------------|
| *Coming soon* | - | 🚧 Planned | - |

**Planned Topics:**
- Installation issues
- CUDA and GPU errors for multimodal models
- Model loading failures
- Vision/audio processing errors

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on adding new recipes.

## Key Differences from vLLM

| Feature | vLLM | vLLM-Omni |
|---------|------|-----------|
| Modalities | Text | Text, Image, Video, Audio |
| Architectures | Autoregressive | AR + DiT + Parallel |
| Outputs | Text | Multimodal outputs |
