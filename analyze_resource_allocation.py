#!/usr/bin/env python3
"""
BEV Platform Resource Allocation Analyzer
Analyzes docker-compose files to calculate total resource allocation per node
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json

def parse_memory_string(mem_str: str) -> float:
    """Convert memory string (e.g., '4G', '512M', '256MB') to GB"""
    if not mem_str:
        return 0.0

    mem_str = str(mem_str).upper().strip()

    # Handle different formats
    if 'GB' in mem_str or 'G' in mem_str:
        return float(re.findall(r'[\d.]+', mem_str)[0])
    elif 'MB' in mem_str or 'M' in mem_str:
        return float(re.findall(r'[\d.]+', mem_str)[0]) / 1024
    elif 'KB' in mem_str or 'K' in mem_str:
        return float(re.findall(r'[\d.]+', mem_str)[0]) / (1024 * 1024)
    else:
        # Assume GB if no unit
        try:
            return float(mem_str)
        except:
            return 0.0

def parse_cpu_string(cpu_str: str) -> float:
    """Convert CPU string to cores"""
    if not cpu_str:
        return 0.0

    cpu_str = str(cpu_str).strip().strip("'\"")
    try:
        return float(cpu_str)
    except:
        return 0.0

def analyze_compose_file(filepath: Path) -> Dict:
    """Analyze a docker-compose file for resource allocations"""

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse YAML
    try:
        compose_data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        print(f"Error parsing {filepath}: {e}")
        return {}

    services = {}

    if 'services' not in compose_data:
        return services

    for service_name, service_config in compose_data['services'].items():
        if not isinstance(service_config, dict):
            continue

        service_info = {
            'container_name': service_config.get('container_name', service_name),
            'image': service_config.get('image', 'custom'),
            'memory': 0.0,
            'cpu': 0.0,
            'gpu': False
        }

        # Check for deploy limits (Docker Swarm/Compose v3)
        if 'deploy' in service_config and 'resources' in service_config['deploy']:
            limits = service_config['deploy']['resources'].get('limits', {})
            reservations = service_config['deploy']['resources'].get('reservations', {})

            # Use limits first, fall back to reservations
            memory = limits.get('memory') or reservations.get('memory')
            if memory:
                service_info['memory'] = parse_memory_string(memory)

            cpus = limits.get('cpus') or reservations.get('cpus')
            if cpus:
                service_info['cpu'] = parse_cpu_string(cpus)

        # Check for mem_limit (Docker Compose v2)
        if 'mem_limit' in service_config:
            service_info['memory'] = parse_memory_string(service_config['mem_limit'])

        # Check for cpus
        if 'cpus' in service_config:
            service_info['cpu'] = parse_cpu_string(service_config['cpus'])

        # Check for GPU usage
        if 'runtime' in service_config and 'nvidia' in service_config['runtime']:
            service_info['gpu'] = True

        if 'environment' in service_config:
            env = service_config['environment']
            if isinstance(env, dict):
                if 'NVIDIA_VISIBLE_DEVICES' in env or 'CUDA_VISIBLE_DEVICES' in env:
                    service_info['gpu'] = True
            elif isinstance(env, list):
                for var in env:
                    if 'NVIDIA_VISIBLE_DEVICES' in str(var) or 'CUDA_VISIBLE_DEVICES' in str(var):
                        service_info['gpu'] = True

        # Estimate memory if not specified based on service type
        if service_info['memory'] == 0.0:
            service_info['memory'] = estimate_memory(service_info['image'], service_name)

        # Estimate CPU if not specified
        if service_info['cpu'] == 0.0:
            service_info['cpu'] = estimate_cpu(service_info['image'], service_name)

        services[service_name] = service_info

    return services

def estimate_memory(image: str, service_name: str) -> float:
    """Estimate memory usage based on service type"""
    image_lower = image.lower()
    name_lower = service_name.lower()

    # Database services
    if 'postgres' in image_lower or 'postgres' in name_lower:
        return 4.0  # PostgreSQL typically needs 4GB minimum
    elif 'neo4j' in image_lower or 'neo4j' in name_lower:
        return 4.0  # Neo4j needs significant memory
    elif 'elasticsearch' in image_lower or 'elastic' in name_lower:
        return 8.0  # Elasticsearch is memory intensive
    elif 'redis' in image_lower or 'redis' in name_lower:
        return 2.0  # Redis default
    elif 'influxdb' in image_lower or 'influx' in name_lower:
        return 2.0  # InfluxDB default

    # Message queues
    elif 'kafka' in image_lower or 'kafka' in name_lower:
        return 4.0  # Kafka needs good memory
    elif 'rabbitmq' in image_lower or 'rabbit' in name_lower:
        return 2.0  # RabbitMQ default
    elif 'zookeeper' in image_lower:
        return 1.0  # Zookeeper is lighter

    # Monitoring
    elif 'prometheus' in image_lower or 'prometheus' in name_lower:
        return 2.0
    elif 'grafana' in image_lower or 'grafana' in name_lower:
        return 1.0
    elif 'node-exporter' in image_lower:
        return 0.256

    # ML/AI services
    elif 'autonomous' in name_lower or 'ml' in name_lower or 'ai' in name_lower:
        return 6.0  # AI services need more memory
    elif 'analyzer' in name_lower or 'processor' in name_lower:
        return 4.0
    elif 'worker' in name_lower:
        return 3.0

    # Web services
    elif 'nginx' in image_lower:
        return 0.5
    elif 'frontend' in name_lower or 'web' in name_lower:
        return 1.0

    # Default
    else:
        return 2.0  # Conservative default

def estimate_cpu(image: str, service_name: str) -> float:
    """Estimate CPU usage based on service type"""
    image_lower = image.lower()
    name_lower = service_name.lower()

    # High CPU services
    if 'autonomous' in name_lower or 'ml' in name_lower or 'ai' in name_lower:
        return 2.0
    elif 'analyzer' in name_lower or 'processor' in name_lower:
        return 1.5
    elif 'kafka' in image_lower:
        return 1.0
    elif 'elasticsearch' in image_lower:
        return 2.0
    elif 'postgres' in image_lower or 'neo4j' in image_lower:
        return 1.0
    elif 'worker' in name_lower:
        return 1.0
    else:
        return 0.5  # Conservative default

def main():
    """Main analysis function"""

    # Define node configurations
    nodes = {
        'THANOS': {
            'file': 'docker-compose-thanos-unified.yml',
            'total_ram': 64,  # GB
            'total_cpu': 16,   # cores
            'gpu_ram': 10,     # GB VRAM (RTX 3080)
            'architecture': 'x86_64'
        },
        'ORACLE1': {
            'file': 'docker-compose-oracle1-unified.yml',
            'total_ram': 24,   # GB
            'total_cpu': 4,    # cores
            'gpu_ram': 0,      # No GPU
            'architecture': 'aarch64'
        },
        'STARLORD': {
            'file': None,  # Development node, no production services
            'total_ram': 64,   # GB (assumed)
            'total_cpu': 24,   # cores (assumed)
            'gpu_ram': 24,     # GB VRAM (RTX 4090)
            'architecture': 'x86_64'
        }
    }

    print("=" * 80)
    print("BEV PLATFORM RESOURCE ALLOCATION ANALYSIS")
    print("=" * 80)
    print()

    total_services = 0
    total_memory_allocated = 0
    total_cpu_allocated = 0

    for node_name, node_config in nodes.items():
        if node_config['file'] is None:
            print(f"\n{'='*60}")
            print(f"NODE: {node_name} (Development/Avatar Node)")
            print(f"{'='*60}")
            print(f"Architecture: {node_config['architecture']}")
            print(f"Total RAM: {node_config['total_ram']} GB")
            print(f"Total CPU: {node_config['total_cpu']} cores")
            print(f"GPU VRAM: {node_config['gpu_ram']} GB")
            print(f"\nNo production services deployed on this node")
            print(f"Reserved for: AI Companion, Avatar System, Development")
            continue

        filepath = Path(node_config['file'])
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            continue

        services = analyze_compose_file(filepath)

        # Calculate totals
        total_mem = sum(s['memory'] for s in services.values())
        total_cpu = sum(s['cpu'] for s in services.values())
        gpu_services = [s for s in services.values() if s['gpu']]

        print(f"\n{'='*60}")
        print(f"NODE: {node_name}")
        print(f"{'='*60}")
        print(f"Architecture: {node_config['architecture']}")
        print(f"Total RAM Available: {node_config['total_ram']} GB")
        print(f"Total CPU Available: {node_config['total_cpu']} cores")
        print(f"GPU VRAM Available: {node_config['gpu_ram']} GB")
        print(f"\nServices Deployed: {len(services)}")
        print(f"Total Memory Allocated: {total_mem:.2f} GB")
        print(f"Total CPU Allocated: {total_cpu:.2f} cores")
        print(f"GPU-Enabled Services: {len(gpu_services)}")

        # Check for over-allocation
        mem_utilization = (total_mem / node_config['total_ram']) * 100
        cpu_utilization = (total_cpu / node_config['total_cpu']) * 100

        print(f"\n📊 Resource Utilization:")
        print(f"  Memory: {mem_utilization:.1f}% ({total_mem:.2f}/{node_config['total_ram']} GB)")
        print(f"  CPU: {cpu_utilization:.1f}% ({total_cpu:.2f}/{node_config['total_cpu']} cores)")

        if mem_utilization > 100:
            print(f"  ⚠️ MEMORY OVER-ALLOCATED by {total_mem - node_config['total_ram']:.2f} GB!")
        elif mem_utilization > 85:
            print(f"  ⚠️ Memory utilization high (>85%)")
        else:
            print(f"  ✅ Memory allocation within limits")

        if cpu_utilization > 100:
            print(f"  ⚠️ CPU OVER-ALLOCATED by {total_cpu - node_config['total_cpu']:.2f} cores!")
        elif cpu_utilization > 85:
            print(f"  ⚠️ CPU utilization high (>85%)")
        else:
            print(f"  ✅ CPU allocation within limits")

        # List top memory consumers
        print(f"\n📈 Top Memory Consumers:")
        sorted_services = sorted(services.items(), key=lambda x: x[1]['memory'], reverse=True)
        for i, (name, info) in enumerate(sorted_services[:10], 1):
            print(f"  {i}. {info['container_name']}: {info['memory']:.2f} GB")

        # List GPU services
        if gpu_services:
            print(f"\n🎮 GPU-Enabled Services:")
            for service in gpu_services:
                print(f"  - {service['container_name']}")

        # Track totals
        total_services += len(services)
        total_memory_allocated += total_mem
        total_cpu_allocated += total_cpu

    # Overall summary
    print(f"\n{'='*80}")
    print(f"PLATFORM-WIDE SUMMARY")
    print(f"{'='*80}")
    print(f"Total Services: {total_services}")
    print(f"Total Memory Allocated: {total_memory_allocated:.2f} GB")
    print(f"Total CPU Allocated: {total_cpu_allocated:.2f} cores")

    # Recommendations
    print(f"\n{'='*80}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*80}")

    recommendations = []

    # Check THANOS
    thanos_services = analyze_compose_file(Path('docker-compose-thanos-unified.yml'))
    thanos_mem = sum(s['memory'] for s in thanos_services.values())
    if thanos_mem > 64:
        recommendations.append(f"1. THANOS is over-allocated by {thanos_mem - 64:.2f} GB. Consider:")
        recommendations.append(f"   - Moving non-GPU services to ORACLE1")
        recommendations.append(f"   - Reducing memory limits for over-provisioned services")
        recommendations.append(f"   - Using memory swap or compression")

    # Check ORACLE1
    oracle_services = analyze_compose_file(Path('docker-compose-oracle1-unified.yml'))
    oracle_mem = sum(s['memory'] for s in oracle_services.values())
    if oracle_mem > 24:
        recommendations.append(f"2. ORACLE1 is over-allocated by {oracle_mem - 24:.2f} GB. Consider:")
        recommendations.append(f"   - Moving heavy services to THANOS")
        recommendations.append(f"   - Using ARM-optimized images")
        recommendations.append(f"   - Reducing service replicas")

    # Check for misplaced services
    gpu_on_oracle = [s for s in oracle_services.values() if s['gpu']]
    if gpu_on_oracle:
        recommendations.append(f"3. Found GPU services on ORACLE1 (no GPU available):")
        for service in gpu_on_oracle:
            recommendations.append(f"   - Move {service['container_name']} to THANOS")

    # Service distribution recommendations
    heavy_services_on_oracle = [s for s in oracle_services.items()
                                if s[1]['memory'] > 3.0]
    if heavy_services_on_oracle:
        recommendations.append(f"4. Heavy services on resource-limited ORACLE1:")
        for name, service in heavy_services_on_oracle:
            recommendations.append(f"   - {service['container_name']}: {service['memory']:.1f} GB → consider moving to THANOS")

    # STARLORD recommendations
    recommendations.append(f"5. STARLORD (RTX 4090) optimization:")
    recommendations.append(f"   - Reserve for AI Companion and Avatar system")
    recommendations.append(f"   - Implement auto-start/stop for resource efficiency")
    recommendations.append(f"   - Use for development and staging only")

    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()