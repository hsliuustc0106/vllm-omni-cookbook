# HunyuanImage-3.0-Instruct

**Category**: Diffusion (text-to-image, text-image-to-image)  
**Model ID**: `tencent/HunyuanImage-3.0-Instruct`  
**Recipe**: [HunyuanImage-3.0-Instruct.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md)

---

## Key Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| **TTFT** | Time to First Token (AR阶段首token) | ↓ lower is better |
| **TPOT** | Time Per Output Token (AR阶段token生成速度) | ↓ lower is better |
| **E2E Latency (Single)** | End-to-end latency for single request (AR + DIT全流程) | ↓ lower is better |
| **Throughput (Single)** | Single-concurrency throughput (requests/s) | ↑ higher is better |
| **Max Throughput** | Maximum throughput under high load (requests/s) | ↑ higher is better |
| **GPU Memory** | Peak GPU memory usage | ↓ lower is better |

---

## Performance tracks

| Track | Hardware | Source | Purpose |
|-------|----------|--------|---------|
| **Perf (Local L20X)** | L20X | 4 configs: `test_hunyuan_image_{tp4,tp2_sp2,tp2_cfgp2,3_it2i}.json` | Daily performance regression testing |
| **Accuracy (Local L20X)** | L20X | `test_hunyuan_image3.py` | Daily accuracy validation |
| **Accuracy (Nightly CI H100)** | H100 | `test_hunyuan_image3_pixel_accuracy.py` | Full-model pixel-level accuracy |

---

## Standardized Perf Tests (Local Test)

vLLM-Omni runs the following performance test configurations daily:

### Test Configurations

| Config Name | Parallel Strategy | Test File | Pipeline |
|-------------|------------------|-----------|----------|
| **TP4** | Tensor Parallel (TP=4) | `test_hunyuan_image_tp4.json` | DIT-only |
| **TP2+SP2** | TP=2 + Sequence Parallel (SP=2) | `test_hunyuan_image_tp2_sp2.json` | DIT-only |
| **TP2+CFGP2** | TP=2 + CFG Parallel (CFGP=2) | `test_hunyuan_image_tp2_cfgp2.json` | DIT-only |
| **IT2I** | TP=2 (both AR and DIT) | `test_hunyuan_image3_it2i.json` | AR+DIT (Image-to-Image) |

### Reproduce Commands

```bash
# Run any config by specifying the corresponding JSON file
pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_hunyuan_image_{tp4,tp2_sp2,tp2_cfgp2,3_it2i}.json
```

### Baseline Metrics (v0.22.0)

Measured on **L20X** hardware. Performance tests are divided into two categories:

#### Type 1: Single-Concurrency Performance
Evaluates end-to-end latency and throughput for single requests.

| Config | E2E Latency (s) | Throughput (req/s) | GPU Memory (GB) |
|--------|-----------------|--------------------|-----------------| 
| TP4 | 4.67 | 0.21 | 55.58 |
| TP2+SP2 | 4.98 | 0.20 | 94.69 |
| TP2+CFGP2 | 4.72 | 0.21 | 98.30 |
| IT2I | 8.94 | 0.11 | - |

#### Type 2: Maximum Throughput Performance
Evaluates maximum throughput capacity under high load. Only IT2I (AR+DIT pipeline) is tested for this scenario.

| Config | Max Throughput (req/s) | Avg Latency (s) | P95 Latency (s) | P99 Latency (s) |
|--------|------------------------|-----------------|-----------------|-----------------|
| IT2I | 0.21 | 70.84 | 78.18 | 95.08 |

*Measured with max concurrency=16 on L20X*

---

## Retro Comparison

Since HunyuanImage-3.0 is first supported in v0.22.0, there is no cross-version comparison data yet. Delta comparisons will be added when v0.24.0 is released.

**v0.22.0 Baseline:** See [Baseline Metrics](#baseline-metrics-v0220) above for initial support performance data.

---

## Key optimizations

HunyuanImage-3.0's AR+DIT hybrid architecture supports multiple optimization features. The following ablation studies measure the impact of each feature independently:

### AR stage

| Feature | Metric | Without | With | Delta |
|---------|--------|---------|------|-------|
| **Graph Mode** | TTFT (ms) | 1302.54 | 1293.27 | −0.7% |
| | TPOT (ms) | 62.07 | 8.92 | −85.6% |

### DIT stage

| Feature | Metric | Without | With | Delta |
|---------|--------|---------|------|-------|
| **VAE Parallel** | E2E Latency (s) | 4.67 | 4.54 | −2.8% |
| **Piecewise Flash Attention** | E2E Latency (s) | 4.83 | 4.67 | −3.3% |

### AR+DIT joint

| Feature | Metric | Without | With | Delta |
|---------|--------|---------|------|-------|
| **KV Cache Reuse** | E2E Latency (s) | 10.27 | 8.94 | −13.0% |

### Multi-replica scaling

| Config | Total Throughput (req/s) | Per-GPU Throughput (req/s) | Scaling Efficiency |
|--------|-------------------------|----------------------------|-------------------|
| **1:1** (baseline) | 0.21 | 0.105 | - |
| **1:2** | 0.39 | 0.13 | +23.8% |

*Note: 1:1 means 1 AR replica paired with 1 DIT replica (2 GPUs total); 1:2 means 1 AR replica paired with 2 DIT replicas (3 GPUs total). Measured under high load (max concurrency=16). Scaling efficiency = (1:2 total throughput) / (1:1 total throughput) / (GPU ratio).*

---

## Architecture: AR + DIT Hybrid

HunyuanImage-3.0 adopts a **two-stage generation architecture**:

### 1. AR (Autoregressive) Stage
- **Role**: Autoregressively generates initial latent code based on text prompt
- **Key Metrics**: TTFT (time to first token), TPOT (token generation speed)
- **Optimization Points**: Graph mode compilation, KV cache management

### 2. DIT (Diffusion Transformer) Stage
- **Role**: Performs diffusion refinement on AR-generated latent code to produce high-quality images
- **Key Metrics**: Denoising latency, VAE encoding/decoding latency
- **Optimization Points**: VAE parallelism, multi-replica parallelism

### 3. Collaboration
- AR stage output (latent code) serves as conditioning input for DIT stage
- KV cache reuse allows DIT stage to reuse attention cache from AR stage, reducing redundant computation

### 4. Comparison with Pure DIT Models
- **wan2.2** (pure DIT): Single-stage diffusion, no AR component
- **qwen-image** (pure DIT): Single-stage diffusion, no AR component
- **HunyuanImage-3.0** (AR+DIT): Two-stage generation, AR provides stronger semantic understanding, DIT handles visual quality

---

## Release Index

| Version | Date | Highlight |
|---------|------|-----------|
| v0.22.0 | 2026/6/9 | Initial HunyuanImage-3.0 support with AR+DIT architecture |

---

## Serve Commands

```bash
# Serve command for T2I and IT2I tasks
vllm serve tencent/HunyuanImage-3.0-Instruct \
  --omni \
  --port 8091 \
  --deploy-config ./deploy.yaml \
  --init-timeout 600
```

Refer to `deploy.yaml` for specific configurations (TP4, TP2+SP2, TP2+CFGP2 for DIT-only T2I, or IT2I with AR+DIT). Example deploy configs can be found in the vllm-omni repository.

---

## Related Tests

### Accuracy / Correctness
- Daily L20X: `tests/e2e/accuracy/test_hunyuan_image3.py`
- Daily H100 (full model): `tests/e2e/accuracy/test_hunyuan_image3_pixel_accuracy.py -m full_model --run-level full_model`

### Performance
- CI configs: `tests/dfx/perf/tests/test_hunyuan_image_*.json`
- Runner: `tests/dfx/perf/scripts/run_diffusion_benchmark.py`
