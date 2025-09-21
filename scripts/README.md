# BEV OSINT Framework - Predictive Cache Performance Scripts

This directory contains performance testing and optimization scripts for the predictive cache system.

## Scripts Overview

### 1. `benchmark_predictive_cache.py`
Comprehensive benchmarking suite for validating predictive cache performance against targets.

**Features:**
- Multi-phase testing (warmup, mixed workload, stress, ML accuracy)
- Realistic OSINT data simulation
- Performance target validation
- Detailed metrics collection and reporting

**Usage:**
```bash
# Run full benchmark suite
python scripts/benchmark_predictive_cache.py

# Custom configuration
python scripts/benchmark_predictive_cache.py \
  --cache-url http://localhost:8044 \
  --duration 600 \
  --clients 100 \
  --rate 1500
```

**Performance Targets:**
- Cache Hit Rate: >80%
- Response Time: <10ms average
- ML Prediction Accuracy: >80%
- Memory Efficiency: >90%

### 2. `optimize_cache_performance.py`
Automated performance optimization with ML-guided improvements.

**Features:**
- Real-time performance analysis
- Automated optimization application
- Continuous optimization mode
- Impact measurement and reporting

**Usage:**
```bash
# Single optimization cycle
python scripts/optimize_cache_performance.py --mode single

# Continuous optimization
python scripts/optimize_cache_performance.py --mode continuous --interval 15

# Custom endpoints
python scripts/optimize_cache_performance.py \
  --cache-url http://cache-service:8044 \
  --prometheus-url http://prometheus:9090
```

**Optimization Areas:**
- ML model retraining
- Tier allocation optimization
- Cache warming strategy tuning
- Eviction policy optimization
- Memory rebalancing

### 3. `cache_performance_monitor.py`
Real-time performance monitoring with alerting and trend analysis.

**Features:**
- Continuous performance monitoring
- Threshold-based alerting
- Trend analysis and forecasting
- Health status reporting
- Integration with Prometheus metrics

**Usage:**
```bash
# Continuous monitoring
python scripts/cache_performance_monitor.py --interval 30

# Single status report
python scripts/cache_performance_monitor.py --report

# With logging
python scripts/cache_performance_monitor.py \
  --log-file /var/log/cache_performance.log \
  --interval 15
```

**Alert Thresholds:**
- Critical: Hit rate <60%, Response time >50ms, Error rate >5%
- Warning: Hit rate <75%, Response time >25ms, Error rate >2%

### 4. `cache_load_test.py`
Comprehensive load testing suite for performance validation under various conditions.

**Features:**
- Multiple test scenarios (baseline, normal, stress, spike)
- Cache warming efficiency testing
- Realistic load patterns
- Detailed performance analysis

**Usage:**
```bash
# Full load test suite
python scripts/cache_load_test.py --test all

# Specific test scenarios
python scripts/cache_load_test.py --test stress --users 500 --ops 2000
python scripts/cache_load_test.py --test spike --duration 180
python scripts/cache_load_test.py --test warming

# Custom load parameters
python scripts/cache_load_test.py \
  --users 200 \
  --ops 1500 \
  --duration 600
```

**Test Scenarios:**
- **Baseline**: 10 users, 100 ops/sec - Basic functionality validation
- **Normal Load**: 100 users, 1000 ops/sec - Production simulation
- **Stress Test**: 500 users, 2000 ops/sec - Maximum capacity testing
- **Spike Test**: Load spikes with recovery - Elasticity validation
- **Cache Warming**: Efficiency of prefetching strategies

## Performance Targets and Validation

### Target Performance Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| Cache Hit Rate | >80% | <60% |
| Response Time (Avg) | <10ms | >50ms |
| Response Time (P95) | <25ms | >100ms |
| ML Prediction Accuracy | >80% | <60% |
| Memory Efficiency | >90% | <70% |
| Error Rate | <1% | >5% |
| Throughput (Hot Tier) | >100K ops/sec | <50K ops/sec |

### Test Environment Requirements

**System Requirements:**
- Python 3.8+
- Required packages: `aiohttp`, `numpy`, `scikit-learn`
- Cache service running on specified URL
- Prometheus metrics endpoint (optional)

**Network Configuration:**
- Cache service accessible at `http://localhost:8044` (default)
- Prometheus at `http://localhost:9090` (if monitoring enabled)
- Sufficient network bandwidth for load testing

**Resource Requirements:**
- RAM: 2GB minimum for testing scripts
- CPU: 4 cores recommended for concurrent testing
- Network: 1Gbps for high-load testing

## Integration with CI/CD

### Automated Testing Pipeline

```yaml
# Example GitHub Actions workflow
performance_tests:
  runs-on: ubuntu-latest
  steps:
    - name: Start Cache Service
      run: docker-compose up -d predictive-cache

    - name: Wait for Service
      run: sleep 30

    - name: Run Benchmark
      run: python scripts/benchmark_predictive_cache.py

    - name: Run Load Tests
      run: python scripts/cache_load_test.py --test all

    - name: Performance Validation
      run: |
        if [ $? -eq 0 ]; then
          echo "Performance tests passed"
        else
          echo "Performance tests failed"
          exit 1
        fi
```

### Performance Regression Detection

```bash
# Compare against baseline performance
python scripts/benchmark_predictive_cache.py > current_results.txt
python scripts/compare_performance.py baseline_results.txt current_results.txt
```

## Monitoring and Alerting Setup

### Prometheus Integration

The monitoring script integrates with Prometheus to collect additional metrics:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'predictive-cache'
    static_configs:
      - targets: ['localhost:9044']
    scrape_interval: 15s
```

### Alert Manager Rules

```yaml
# cache_alerts.yml
groups:
  - name: cache_performance
    rules:
      - alert: CacheHitRateLow
        expr: bev_cache_hit_rate < 0.75
        for: 5m
        annotations:
          summary: "Cache hit rate below threshold"

      - alert: ResponseTimeHigh
        expr: bev_cache_response_time_seconds > 0.025
        for: 2m
        annotations:
          summary: "Cache response time above threshold"
```

## Troubleshooting

### Common Issues

**Connection Errors:**
```bash
# Verify cache service is running
curl http://localhost:8044/health

# Check Docker container status
docker ps | grep predictive-cache
```

**Performance Issues:**
```bash
# Check resource usage
docker stats bev_predictive_cache

# Review cache metrics
python scripts/cache_performance_monitor.py --report
```

**Load Test Failures:**
```bash
# Reduce load for debugging
python scripts/cache_load_test.py --users 10 --ops 100 --duration 60

# Check service logs
docker logs bev_predictive_cache
```

### Performance Tuning

**For Low Hit Rates:**
1. Run ML model retraining: `curl -X POST http://localhost:8044/admin/retrain`
2. Optimize cache warming: `python scripts/optimize_cache_performance.py --mode single`
3. Review tier allocation in configuration

**For High Response Times:**
1. Check memory utilization: Monitor tier memory usage
2. Optimize eviction policy: Switch to ML-adaptive policy
3. Scale resources: Increase tier sizes or add more memory

**For ML Accuracy Issues:**
1. Collect more training data: Ensure sufficient sample size
2. Update feature engineering: Review feature relevance
3. Retrain models more frequently: Reduce retraining interval

## Development and Customization

### Adding New Test Scenarios

```python
# Example: Custom test scenario
async def _run_custom_test(self) -> LoadTestMetrics:
    """Custom test scenario implementation"""
    custom_config = LoadTestConfig(
        test_duration_seconds=120,
        concurrent_users=75,
        operations_per_second=750,
        read_write_ratio=0.9  # 90% reads
    )
    return await self._run_single_test(custom_config, "Custom")
```

### Custom Metrics Collection

```python
# Example: Additional metric collection
async def collect_custom_metrics(self) -> Dict[str, float]:
    """Collect custom performance metrics"""
    # Implementation for domain-specific metrics
    return {
        "custom_metric_1": value1,
        "custom_metric_2": value2
    }
```

### Performance Script Configuration

All scripts support environment variable configuration:

```bash
export CACHE_SERVICE_URL="http://cache-cluster:8044"
export PROMETHEUS_URL="http://monitoring:9090"
export BENCHMARK_DURATION=600
export LOAD_TEST_USERS=200
```

## Best Practices

### Performance Testing
1. **Baseline First**: Always establish baseline performance before optimization
2. **Gradual Load**: Increase load gradually to identify breaking points
3. **Realistic Data**: Use representative data patterns for accurate results
4. **Multiple Runs**: Average results across multiple test runs
5. **Environment Consistency**: Test in consistent, isolated environments

### Optimization
1. **Measure Before Optimizing**: Collect comprehensive metrics before changes
2. **One Change at a Time**: Apply optimizations incrementally
3. **Validate Impact**: Measure improvement after each optimization
4. **Monitor Continuously**: Use continuous monitoring to detect regressions
5. **Document Changes**: Record optimization history and impact

### Monitoring
1. **Real-time Alerts**: Set up immediate alerts for critical thresholds
2. **Trend Analysis**: Monitor long-term performance trends
3. **Regular Reviews**: Schedule periodic performance reviews
4. **Capacity Planning**: Use metrics for future capacity planning
5. **Incident Response**: Maintain runbooks for performance incidents