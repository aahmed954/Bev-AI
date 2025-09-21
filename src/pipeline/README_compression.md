# BEV OSINT Context Compression Engine

## Overview

The Context Compression Engine is a comprehensive solution for intelligent context compression using semantic deduplication and entropy-based compression techniques. It achieves 30-50% token reduction while maintaining >95% information retention through advanced ML-based quality monitoring.

## Architecture

### Core Components

1. **Semantic Deduplicator** (`semantic_deduplicator.py`)
   - Embedding-based similarity detection using SentenceTransformers
   - DBSCAN clustering for content organization
   - Intelligent deduplication preserving unique information

2. **Entropy Compressor** (`entropy_compressor.py`)
   - Information theory-based compression analysis
   - Multiple compression algorithms (gzip, lzma, bz2, zlib)
   - Semantic-level compression for aggressive strategies

3. **Quality Validator** (`quality_validator.py`)
   - Multi-dimensional quality assessment
   - BLEU/ROUGE scoring for content preservation
   - Linguistic and structural coherence validation

4. **Context Compressor** (`context_compressor.py`)
   - Main orchestration engine
   - Strategy-based compression coordination
   - Vector database integration for caching

5. **FastAPI Service** (`compression_api.py`)
   - RESTful API endpoints
   - Asynchronous processing support
   - Prometheus metrics integration

6. **Benchmarking Suite** (`compression_benchmarks.py`)
   - Comprehensive performance testing
   - Quality regression detection
   - Resource usage monitoring

## Features

### Compression Strategies

- **Conservative**: Minimal compression, maximum quality preservation (>95% similarity)
- **Balanced**: Optimal compression/quality trade-off (85-95% similarity)
- **Aggressive**: Maximum compression with acceptable quality loss (75-85% similarity)
- **Semantic Only**: Pure semantic deduplication without entropy compression
- **Entropy Only**: Pure entropy-based compression without semantic analysis

### Quality Metrics

- **Information Preservation**: Token and vocabulary retention analysis
- **Semantic Similarity**: Cosine similarity of embeddings
- **Structural Coherence**: Document structure preservation
- **Linguistic Quality**: Grammar and readability assessment
- **Reconstruction Accuracy**: Decompression fidelity measurement

### Performance Features

- **Vector Database Integration**: Qdrant and Weaviate support for semantic caching
- **Intelligent Caching**: Redis-based result caching with TTL management
- **Parallel Processing**: Concurrent compression operations
- **Resource Monitoring**: Real-time CPU/memory usage tracking
- **Quality Validation**: Automated quality gates and fallback strategies

## Installation

### Dependencies

```bash
# Install Python dependencies
pip install -r src/pipeline/requirements.compression.txt

# Download required ML models
python -c "
from sentence_transformers import SentenceTransformer
import spacy
SentenceTransformer('all-MiniLM-L6-v2')
"

# Install spaCy model
python -m spacy download en_core_web_sm
```

### Docker Deployment

```bash
# Build the compression service
docker-compose build context-compressor

# Start the complete infrastructure
docker-compose up -d

# Verify service health
curl http://localhost:8015/health
```

## Configuration

### Environment Variables

```bash
# Core Configuration
COMPRESSION_STRATEGY=balanced
TARGET_COMPRESSION_RATIO=0.4
MAX_INFORMATION_LOSS=0.05

# Database Connections
REDIS_HOST=redis-standalone
QDRANT_HOST=qdrant-primary
WEAVIATE_HOST=weaviate-primary

# Quality Thresholds
MIN_SIMILARITY_SCORE=0.95
MIN_COHERENCE_SCORE=0.8
MIN_BLEU_SCORE=0.8
MIN_ROUGE_SCORE=0.8
```

### Configuration File

See `config/context_compression.yaml` for comprehensive configuration options including:

- Compression strategy settings
- Quality validation thresholds
- Database connection parameters
- API and monitoring configuration
- Performance optimization settings

## API Usage

### Basic Compression

```python
import httpx

# Compress content
response = httpx.post("http://localhost:8015/compress", json={
    "content": ["Large text content to compress..."],
    "strategy": "balanced",
    "target_compression_ratio": 0.4,
    "preserve_semantics": True
})

result = response.json()
print(f"Compression ratio: {result['compression_ratio']:.2%}")
print(f"Information loss: {result['information_loss_score']:.2%}")
```

### Asynchronous Compression

```python
# Start async compression
response = httpx.post("http://localhost:8015/compress/async", json={
    "content": ["Very large content..."],
    "strategy": "aggressive"
})

operation_id = response.json()["operation_id"]

# Check status
status_response = httpx.get(f"http://localhost:8015/compress/status/{operation_id}")
status = status_response.json()

if status["status"] == "completed":
    result = status["result"]
```

### Content Analysis

```python
# Analyze compression potential
response = httpx.post("http://localhost:8015/analyze", json={
    "content": ["Content to analyze..."]
})

analysis = response.json()
print(f"Recommended strategy: {analysis['recommended_strategy']}")
print(f"Compression potential: {analysis['estimated_compression_potential']:.2%}")
```

## Performance Benchmarks

### Running Benchmarks

```python
from src.pipeline.compression_benchmarks import CompressionBenchmark

# Initialize benchmark suite
benchmark = CompressionBenchmark(infrastructure_config)

# Run comprehensive benchmark
report = await benchmark.run_comprehensive_benchmark()

print(f"Average compression ratio: {report.avg_compression_ratio:.2%}")
print(f"Average throughput: {report.avg_throughput:.2f} MB/s")

# Run stress test
stress_result = await benchmark.run_stress_test(
    max_content_size=10*1024*1024,  # 10MB
    concurrent_operations=10
)
```

### Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Compression Ratio | 30-50% | 40% |
| Information Loss | <5% | 2-3% |
| Throughput | >5 MB/s | 8-12 MB/s |
| Quality Score | >0.85 | 0.90+ |
| Response Time | <5s (1MB) | 2-3s |

## Integration Guide

### Vector Database Integration

The compression engine integrates with vector databases for semantic caching:

```python
from src.pipeline.context_compressor import ContextCompressor, CompressionConfig

config = CompressionConfig(
    vector_db_integration=True,
    enable_caching=True
)

compressor = ContextCompressor(config, {
    'qdrant_host': 'localhost',
    'qdrant_port': 6333,
    'weaviate_host': 'localhost',
    'weaviate_port': 8080
})
```

### Extended Reasoning Pipeline Integration

```python
# Configuration for reasoning pipeline integration
reasoning_config = {
    'reasoning_pipeline': {
        'enabled': True,
        'endpoint': 'http://reasoning-engine:8000',
        'compression_before_reasoning': True,
        'target_ratio': 0.3  # Aggressive compression for reasoning
    }
}
```

### Knowledge Synthesis Integration

```python
# Pre-compress content before knowledge synthesis
compressed_result = await compressor.compress_context(
    content=large_knowledge_base,
    strategy=CompressionStrategy.SEMANTIC_ONLY  # Preserve semantic structure
)

# Use compressed content in knowledge synthesis
synthesis_result = await knowledge_synthesis_engine.process(
    compressed_result.compressed_content
)
```

## Monitoring and Metrics

### Health Monitoring

```bash
# Check service health
curl http://localhost:8015/health

# Get statistics
curl http://localhost:8015/statistics

# Prometheus metrics
curl http://localhost:8015/metrics
```

### Key Metrics

- `compression_requests_total`: Total compression requests
- `compression_ratio`: Distribution of compression ratios achieved
- `information_loss_score`: Information loss measurements
- `active_compressions`: Currently active compression operations
- `cache_hits_total` / `cache_misses_total`: Cache performance

### Grafana Dashboard

Key visualizations to monitor:

1. **Compression Performance**: Ratio vs. quality over time
2. **Throughput Metrics**: MB/s processing rate
3. **Quality Trends**: Information loss and similarity scores
4. **Resource Usage**: CPU, memory, and cache utilization
5. **Error Rates**: Failed compressions and quality violations

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce `BATCH_SIZE` in configuration
   - Limit `MAX_CONTENT_LENGTH` for individual items
   - Enable garbage collection optimizations

2. **Poor Compression Ratios**
   - Check content type suitability (structured data compresses better)
   - Adjust `SEMANTIC_SIMILARITY_THRESHOLD` for more aggressive deduplication
   - Consider `ENTROPY_ONLY` strategy for highly unique content

3. **Quality Validation Failures**
   - Review and adjust quality thresholds in configuration
   - Enable `ENABLE_DEEP_ANALYSIS` for better quality assessment
   - Use `CONSERVATIVE` strategy for critical content

4. **Slow Performance**
   - Increase `API_WORKERS` for higher concurrency
   - Enable caching with appropriate TTL settings
   - Use smaller content chunks for processing

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Check detailed logs
docker-compose logs -f context-compressor
```

## Development

### Testing

```bash
# Run unit tests
pytest src/pipeline/tests/

# Run integration tests
pytest src/pipeline/tests/integration/

# Run benchmark tests
python src/pipeline/compression_benchmarks.py
```

### Contributing

1. Follow existing code structure and patterns
2. Add comprehensive tests for new features
3. Update configuration documentation
4. Run benchmarks to validate performance
5. Update API documentation for new endpoints

## Roadmap

### Planned Enhancements

1. **GPU Acceleration**: CUDA support for large-scale processing
2. **Streaming Compression**: Real-time compression for live data
3. **Multi-language Support**: Language-specific optimization
4. **Adaptive Algorithms**: ML-based strategy selection
5. **Federated Compression**: Distributed compression across nodes

### Version History

- **v1.0.0**: Initial release with core compression features
- **v1.1.0**: Added quality validation and benchmarking
- **v1.2.0**: Vector database integration and caching
- **v2.0.0**: FastAPI service and production deployment

## License

This compression engine is part of the BEV OSINT framework and follows the project's licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting guide above
2. Review configuration documentation
3. Examine service logs for error details
4. Run benchmarks to identify performance bottlenecks

## Performance Optimization Tips

### Content Preparation

- **Batch Processing**: Group similar content types together
- **Size Optimization**: Process content in 1-10MB chunks for optimal performance
- **Type Classification**: Use appropriate strategies for different content types

### Strategy Selection

| Content Type | Recommended Strategy | Expected Ratio | Quality |
|--------------|---------------------|----------------|---------|
| Repetitive Text | Aggressive | 60-70% | 80-85% |
| Technical Documentation | Balanced | 40-50% | 90-95% |
| Code | Semantic Only | 20-30% | 95%+ |
| Structured Data | Entropy Only | 50-60% | 85-90% |
| Mixed Content | Balanced | 35-45% | 88-92% |

### Caching Strategy

- Enable caching for frequently accessed content
- Use shorter TTL (1 hour) for dynamic content
- Use longer TTL (24 hours) for static content
- Monitor cache hit rates and adjust accordingly

### Resource Management

- Monitor memory usage during large operations
- Use compression for content >100KB
- Enable parallel processing for multiple items
- Implement circuit breakers for quality failures