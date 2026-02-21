# deploy

Deploy vLLM-Omni to production environments.

## Trigger Conditions

Use this skill when the user asks to:
- Deploy to production
- Set up Docker containers
- Configure Kubernetes
- Set up load balancing

## Implementation

1. Determine deployment target (Docker, Kubernetes, cloud)
2. Identify model type and resource requirements
3. Create or update deployment configuration
4. Configure environment variables
5. Set up health checks and monitoring
6. Document deployment steps

## Deployment Options

### Docker
```bash
# vLLM-Omni Docker deployment
docker run -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  omni --model <model_name> --input-type <type>
```

### Kubernetes
- Use Deployment with appropriate resource limits (higher for VLMs)
- Configure Service for load balancing
- Set up HorizontalPodAutoscaler based on modality workload

### Cloud Platforms
- Model-specific instance selection (GPU requirements vary)
- Multi-modal endpoints configuration
- Storage setup for image/audio/video uploads

## vLLM-Omni Deployment Considerations

- **Resource Requirements**: VLMs and DiT models need more resources
- **I/O Pipeline**: Handle image/audio/video uploads efficiently
- **API Flexibility**: Support multiple input types in single endpoint
- **Monitoring**: Track modality-specific metrics

### Monitoring
- Configure Prometheus metrics
- Set up health check endpoints
- Configure logging for multimodal pipelines
- Alert on modality-specific failures
