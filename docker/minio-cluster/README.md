# MinIO Cluster for ORACLE1

A production-ready, 3-node MinIO cluster providing S3-compatible distributed storage with erasure coding, monitoring, and high availability.

## Architecture

- **3 MinIO Nodes**: Distributed erasure-coded storage
- **Load Balancer**: Nginx-based load balancing and health checks
- **Monitoring**: Prometheus metrics collection and Grafana dashboards
- **Management**: MinIO Client (mc) for administration

## Features

- ✅ **Erasure Coding**: Data protection with 1 parity drive
- ✅ **High Availability**: Automatic failover and recovery
- ✅ **Load Balancing**: Request distribution across nodes
- ✅ **ARM Compatible**: Multi-architecture support
- ✅ **Monitoring**: Comprehensive metrics and dashboards
- ✅ **S3 API**: Full AWS S3 compatibility
- ✅ **Bucket Lifecycle**: Automated data management
- ✅ **Versioning**: Object version control

## Quick Start

### 1. Start the Cluster

```bash
cd docker/minio-cluster
docker-compose up -d
```

### 2. Verify Cluster Health

```bash
# Check cluster status
docker-compose ps

# View logs
docker-compose logs minio1 minio2 minio3

# Test load balancer
curl http://localhost:9000/minio/health/live
```

### 3. Access Interfaces

- **MinIO Console**: http://localhost:9011 (Node 1), 9012 (Node 2), 9013 (Node 3)
- **API Endpoint**: http://localhost:9000 (Load Balanced)
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin123)

## Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
MINIO_ROOT_USER=admin
MINIO_ROOT_PASSWORD=admin123456
MINIO_DISTRIBUTED_MODE_ENABLED=yes
MINIO_ERASURE_SET_DRIVE_COUNT=3
```

### Default Buckets

The cluster automatically creates these buckets:

- `oracle1-documents`: Public bucket for document storage
- `oracle1-models`: Private bucket for ML models
- `oracle1-cache`: Private bucket for cache data
- `oracle1-backups`: Private bucket for backups
- `oracle1-logs`: Private bucket for log storage

### Bucket Policies

- **Documents**: Public read access
- **Models**: Private access only
- **Backups**: Private with versioning enabled
- **Cache**: TTL-based lifecycle policies

## Monitoring

### Prometheus Metrics

The cluster exposes metrics at:
- Cluster metrics: `/minio/v2/metrics/cluster`
- Node metrics: `/minio/v2/metrics/node`
- Bucket metrics: `/minio/v2/metrics/bucket`

### Key Metrics

- **Storage Usage**: Total and per-bucket usage
- **Request Rate**: API requests per second
- **Error Rate**: Failed requests and timeouts
- **Node Health**: Individual node status
- **Network I/O**: Bandwidth utilization

### Grafana Dashboards

Pre-configured dashboards for:
- Cluster overview and health
- Storage capacity and usage
- Request performance and latency
- Error tracking and alerts

## Usage Examples

### Python SDK

```python
from minio import Minio

# Connect to cluster
client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="admin123456",
    secure=False
)

# Upload file
client.fput_object("oracle1-documents", "test.pdf", "/path/to/file.pdf")

# Download file
client.fget_object("oracle1-documents", "test.pdf", "/path/to/download.pdf")
```

### MinIO Client (mc)

```bash
# Configure alias
mc alias set oracle1 http://localhost:9000 admin admin123456

# List buckets
mc ls oracle1

# Upload file
mc cp document.pdf oracle1/oracle1-documents/

# Download file
mc cp oracle1/oracle1-documents/document.pdf ./

# Mirror directory
mc mirror ./local-dir oracle1/oracle1-documents/
```

### Curl API

```bash
# List buckets
curl -X GET http://localhost:9000/

# Upload object
curl -X PUT "http://localhost:9000/oracle1-documents/test.txt" \
     -H "Authorization: AWS4-HMAC-SHA256..." \
     -d "Hello, MinIO!"
```

## Production Deployment

### Security Hardening

1. **Enable TLS**:
   ```bash
   MINIO_API_SECURE=true
   MINIO_TLS_ENABLED=true
   ```

2. **Strong Credentials**:
   ```bash
   MINIO_ROOT_USER=your-secure-username
   MINIO_ROOT_PASSWORD=your-very-strong-password
   ```

3. **Network Security**:
   - Use private networks
   - Configure firewall rules
   - Enable VPN access

### Resource Sizing

**Minimum Production Requirements**:
- CPU: 2 cores per node (6 total)
- Memory: 4GB per node (12GB total)
- Storage: 100GB per node (300GB total)
- Network: 1Gbps between nodes

**Recommended Production**:
- CPU: 4 cores per node (12 total)
- Memory: 8GB per node (24GB total)
- Storage: 1TB per node (3TB total)
- Network: 10Gbps between nodes

### Backup Strategy

1. **Cross-Region Replication**:
   ```bash
   mc replicate add oracle1/oracle1-backups \
     --remote-bucket backup-site/oracle1-backups
   ```

2. **Lifecycle Policies**:
   ```bash
   mc ilm add oracle1/oracle1-documents \
     --expiry-days 365 \
     --transition-days 30 \
     --storage-class GLACIER
   ```

## Troubleshooting

### Common Issues

1. **Split-brain Prevention**:
   - Ensure stable network connectivity
   - Use odd number of nodes (3, 5, 7)
   - Monitor cluster quorum

2. **Performance Optimization**:
   - Tune `MINIO_API_REQUESTS_MAX`
   - Enable caching with `MINIO_CACHE_DRIVES`
   - Use SSD storage for better IOPS

3. **Capacity Planning**:
   - Monitor erasure coding overhead
   - Plan for 33% overhead with EC:1
   - Set up alerting for 80% capacity

### Health Checks

```bash
# Check cluster health
curl http://localhost:9000/minio/health/cluster

# Check individual nodes
curl http://localhost:9001/minio/health/live  # Node 1
curl http://localhost:9002/minio/health/live  # Node 2
curl http://localhost:9003/minio/health/live  # Node 3

# Check load balancer
curl http://localhost:9000/health
```

### Log Analysis

```bash
# View cluster logs
docker-compose logs -f minio1 minio2 minio3

# Check specific issues
docker-compose logs minio1 | grep ERROR
docker-compose logs minio-lb | grep "upstream"
```

## Integration with ORACLE1

The MinIO cluster integrates with ORACLE1 components:

- **OCR Service**: Stores processed documents
- **Document Analyzer**: Caches analysis results
- **Knowledge Worker**: Stores knowledge graphs
- **Genetic Worker**: Stores optimization models
- **Edge Worker**: Distributed model storage

### Service Integration

```python
# OCR Service integration
import boto3

s3_client = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='admin',
    aws_secret_access_key='admin123456'
)

# Store OCR result
s3_client.put_object(
    Bucket='oracle1-documents',
    Key='ocr-results/document-123.json',
    Body=json.dumps(ocr_result)
)
```

## Maintenance

### Cluster Upgrades

1. **Rolling Upgrade**:
   ```bash
   # Upgrade one node at a time
   docker-compose stop minio1
   docker-compose pull minio1
   docker-compose up -d minio1

   # Repeat for minio2 and minio3
   ```

2. **Health Verification**:
   ```bash
   # After each node upgrade
   curl http://localhost:9000/minio/health/cluster
   ```

### Data Integrity

```bash
# Run data integrity check
mc admin heal oracle1 --recursive

# Verify checksums
mc admin info oracle1
```

## Support

- **Documentation**: https://docs.min.io/
- **Community**: https://slack.min.io/
- **GitHub**: https://github.com/minio/minio