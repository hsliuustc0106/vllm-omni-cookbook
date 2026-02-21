# profile

Profile and benchmark vLLM-Omni performance.

## Trigger Conditions

Use this skill when the user asks to:
- Profile performance
- Benchmark a model
- Measure throughput or latency
- Analyze bottlenecks

## Implementation

1. Determine what to profile (latency, throughput, memory)
2. Identify model type (AR, DiT, multimodal)
3. Set up profiling tools
4. Run benchmarks with appropriate parameters
5. Analyze and present results

## Profiling Commands

```bash
# Basic omni-modality profiling
vllm profile omni --model <model_name> --input-type <type>

# Throughput benchmark
vllm benchmark omni --model <model_name> --throughput

# Latency measurement
vllm benchmark omni --model <model_name> --latency

# DiT model profiling
vllm profile omni --model <dit_model> --arch diffusion
```

## Metrics to Capture

- Tokens/frames per second
- Time to first token/frame (TTFT)
- Total request latency
- Memory usage (varies by modality)
- GPU utilization
- Modality-specific metrics (image processing time, etc.)

## vLLM-Omni Considerations

- **Vision models**: Higher memory usage, different bottlenecks
- **DiT models**: Parallel generation affects throughput measurements
- **Audio/Video**: I/O overhead can dominate metrics
