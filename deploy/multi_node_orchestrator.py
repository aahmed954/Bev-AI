#!/usr/bin/env python3
"""
Multi-Node Deployment Orchestrator
Manages deployment across THANOS, Oracle1, and STARLORD
"""

import asyncio
import paramiko
import docker
import yaml
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NodeConfig:
    """Node configuration"""
    def __init__(self, name: str, host: str, user: str, role: str, 
                 architecture: str, services: List[str]):
        self.name = name
        self.host = host
        self.user = user
        self.role = role
        self.architecture = architecture
        self.services = services

class MultiNodeOrchestrator:
    """Orchestrate deployment across multiple nodes"""
    
    def __init__(self):
        self.nodes = {
            'thanos': NodeConfig(
                name='THANOS',
                host='100.122.12.54',
                user='bev',
                role='production',
                architecture='x86_64',
                services=['core', 'database', 'message_queue', 'monitoring']
            ),
            'oracle1': NodeConfig(
                name='Oracle1-Cloud',
                host='100.96.197.84',
                user='oracle',
                role='osint',
                architecture='arm64',
                services=['research', 'crawler', 'osint_tools']
            ),
            'starlord': NodeConfig(
                name='STARLORD',
                host='100.72.73.3',
                user='starlord',
                role='development',
                architecture='x86_64',
                services=['development', 'testing']
            )
        }
        
        self.service_mapping = {
            'core': ['swarm_master', 'agent_coordinator', 'memory_manager'],
            'database': ['postgres', 'redis', 'neo4j', 'qdrant'],
            'message_queue': ['rabbitmq', 'kafka', 'schema_registry'],
            'monitoring': ['prometheus', 'grafana', 'alertmanager'],
            'research': ['research_coordinator', 'alternative_market'],
            'crawler': ['spider', 'tor_proxy', 'selenium_grid'],
            'osint_tools': ['intelowl', 'elasticsearch', 'kibana'],
            'development': ['jupyter', 'vscode_server'],
            'testing': ['pytest', 'locust', 'selenium']
        }
    
    async def deploy_all_nodes(self):
        """Deploy to all nodes in parallel"""
        tasks = []
        
        for node_name, node_config in self.nodes.items():
            if node_config.role != 'development':  # Skip dev for production deploy
                tasks.append(self.deploy_node(node_config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        for node_name, result in zip(self.nodes.keys(), results):
            if isinstance(result, Exception):
                logger.error(f"Failed to deploy {node_name}: {result}")
            else:
                logger.info(f"Successfully deployed {node_name}")
        
        return results
    
    async def deploy_node(self, node: NodeConfig):
        """Deploy to specific node"""
        logger.info(f"Deploying to {node.name} ({node.host})")
        
        # Connect via SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(node.host, username=node.user)
            
            # Stop existing services
            await self.stop_services(ssh, node)
            
            # Deploy new services
            await self.deploy_services(ssh, node)
            
            # Start services
            await self.start_services(ssh, node)
            
            # Verify deployment
            await self.verify_deployment(ssh, node)
            
            return {'node': node.name, 'status': 'success'}
            
        except Exception as e:
            logger.error(f"Deployment failed for {node.name}: {e}")
            raise
        
        finally:
            ssh.close()
    
    async def stop_services(self, ssh, node: NodeConfig):
        """Stop existing services on node"""
        for service_group in node.services:
            services = self.service_mapping[service_group]
            
            for service in services:
                stdin, stdout, stderr = ssh.exec_command(
                    f"docker stop bev_{service} 2>/dev/null || true"
                )
                stdout.read()
    
    async def deploy_services(self, ssh, node: NodeConfig):
        """Deploy services to node"""
        # Create deployment directory
        stdin, stdout, stderr = ssh.exec_command(
            f"mkdir -p /opt/bev && cd /opt/bev"
        )
        
        # Copy docker-compose file
        compose_file = self.generate_compose_for_node(node)
        
        with ssh.open_sftp() as sftp:
            with sftp.open(f'/opt/bev/docker-compose.yml', 'w') as f:
                f.write(yaml.dump(compose_file))
    
    async def start_services(self, ssh, node: NodeConfig):
        """Start services on node"""
        stdin, stdout, stderr = ssh.exec_command(
            "cd /opt/bev && docker-compose up -d"
        )
        
        output = stdout.read().decode()
        logger.info(f"Services started on {node.name}: {output}")
    
    async def verify_deployment(self, ssh, node: NodeConfig):
        """Verify services are running"""
        stdin, stdout, stderr = ssh.exec_command(
            "docker ps --format 'table {{.Names}}\t{{.Status}}'"
        )
        
        output = stdout.read().decode()
        
        # Check all expected services are running
        for service_group in node.services:
            services = self.service_mapping[service_group]
            
            for service in services:
                if f"bev_{service}" not in output:
                    raise Exception(f"Service {service} not running on {node.name}")
    
    def generate_compose_for_node(self, node: NodeConfig) -> Dict:
        """Generate docker-compose configuration for specific node"""
        
        compose = {
            'version': '3.8',
            'services': {},
            'volumes': {},
            'networks': {
                'bev_network': {
                    'driver': 'bridge'
                }
            }
        }
        
        # Add services based on node role
        for service_group in node.services:
            services = self.service_mapping[service_group]
            
            for service in services:
                compose['services'][service] = self.get_service_config(
                    service, 
                    node.architecture
                )
        
        return compose
    
    def get_service_config(self, service: str, architecture: str) -> Dict:
        """Get service configuration for architecture"""
        
        # Service configurations
        configs = {
            'swarm_master': {
                'image': f'bev/swarm-master:{architecture}',
                'container_name': 'bev_swarm_master',
                'environment': [
                    'COORDINATION_MODE=hybrid',
                    'REDIS_URL=redis://redis:6379',
                    'POSTGRES_URL=postgresql://swarm_admin:swarm_password@postgres:5432/ai_swarm'
                ],
                'volumes': [
                    './src/agents:/app/agents',
                    'swarm_data:/data'
                ],
                'ports': ['8000:8000'],
                'networks': ['bev_network'],
                'restart': 'unless-stopped'
            },
            'postgres': {
                'image': 'ankane/pgvector:latest' if architecture == 'x86_64' else 'ankane/pgvector:latest-arm64',
                'container_name': 'bev_postgres',
                'environment': [
                    'POSTGRES_DB=ai_swarm',
                    'POSTGRES_USER=swarm_admin',
                    'POSTGRES_PASSWORD=swarm_password'
                ],
                'volumes': [
                    'postgres_data:/var/lib/postgresql/data',
                    './docker/databases/init-scripts/postgres:/docker-entrypoint-initdb.d'
                ],
                'ports': ['5432:5432'],
                'networks': ['bev_network']
            },
            'redis': {
                'image': 'redis:7-alpine',
                'container_name': 'bev_redis',
                'command': 'redis-server --appendonly yes --maxmemory 4gb --maxmemory-policy allkeys-lru',
                'volumes': ['redis_data:/data'],
                'ports': ['6379:6379'],
                'networks': ['bev_network']
            },
            'rabbitmq': {
                'image': 'rabbitmq:3.12-management-alpine',
                'container_name': 'bev_rabbitmq',
                'environment': [
                    'RABBITMQ_DEFAULT_USER=admin',
                    'RABBITMQ_DEFAULT_PASS=BevSwarm2024!'
                ],
                'volumes': ['rabbitmq_data:/var/lib/rabbitmq'],
                'ports': ['5672:5672', '15672:15672'],
                'networks': ['bev_network']
            },
            'kafka': {
                'image': 'confluentinc/cp-kafka:7.5.0',
                'container_name': 'bev_kafka',
                'environment': [
                    'KAFKA_BROKER_ID=1',
                    'KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181',
                    'KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092',
                    'KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1'
                ],
                'volumes': ['kafka_data:/var/lib/kafka/data'],
                'ports': ['9092:9092'],
                'networks': ['bev_network']
            },
            'research_coordinator': {
                'image': f'bev/research-coordinator:{architecture}',
                'container_name': 'bev_research',
                'environment': [
                    'AGENT_ID=research_1',
                    'TOR_PROXY=socks5://tor-proxy:9050'
                ],
                'volumes': ['research_data:/data'],
                'networks': ['bev_network']
            },
            'spider': {
                'image': f'bev/spider:{architecture}',
                'container_name': 'bev_spider',
                'environment': [
                    'CRAWL_DEPTH=3',
                    'USER_AGENT=BevBot/1.0'
                ],
                'volumes': ['crawl_data:/data/crawls'],
                'networks': ['bev_network']
            }
        }
        
        return configs.get(service, {})

# Health check and monitoring
class NodeHealthMonitor:
    """Monitor health of all nodes"""
    
    def __init__(self, orchestrator: MultiNodeOrchestrator):
        self.orchestrator = orchestrator
        self.health_status = {}
    
    async def monitor_all_nodes(self):
        """Continuous health monitoring"""
        while True:
            for node_name, node_config in self.orchestrator.nodes.items():
                try:
                    health = await self.check_node_health(node_config)
                    self.health_status[node_name] = health
                    
                    if not health['healthy']:
                        logger.warning(f"Node {node_name} unhealthy: {health}")
                        await self.attempt_recovery(node_config)
                        
                except Exception as e:
                    logger.error(f"Health check failed for {node_name}: {e}")
                    self.health_status[node_name] = {'healthy': False, 'error': str(e)}
            
            await asyncio.sleep(60)  # Check every minute
    
    async def check_node_health(self, node: NodeConfig) -> Dict:
        """Check health of specific node"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(node.host, username=node.user, timeout=10)
            
            # Check Docker services
            stdin, stdout, stderr = ssh.exec_command(
                "docker ps --format '{{.Names}},{{.Status}}' | grep '^bev_'"
            )
            
            services = stdout.read().decode().strip().split('\n')
            unhealthy = []
            
            for service_line in services:
                if service_line:
                    name, status = service_line.split(',', 1)
                    if 'unhealthy' in status or 'Exited' in status:
                        unhealthy.append(name)
            
            # Check system resources
            stdin, stdout, stderr = ssh.exec_command(
                "free -m | grep '^Mem:' | awk '{print $3/$2 * 100.0}'"
            )
            memory_usage = float(stdout.read().decode().strip())
            
            stdin, stdout, stderr = ssh.exec_command(
                "df / | tail -1 | awk '{print $5}' | sed 's/%//'"
            )
            disk_usage = float(stdout.read().decode().strip())
            
            return {
                'healthy': len(unhealthy) == 0 and memory_usage < 90 and disk_usage < 90,
                'unhealthy_services': unhealthy,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage
            }
            
        finally:
            ssh.close()
    
    async def attempt_recovery(self, node: NodeConfig):
        """Attempt to recover unhealthy node"""
        logger.info(f"Attempting recovery for {node.name}")
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            ssh.connect(node.host, username=node.user)
            
            # Restart unhealthy containers
            health = self.health_status.get(node.name, {})
            
            for service in health.get('unhealthy_services', []):
                logger.info(f"Restarting {service} on {node.name}")
                stdin, stdout, stderr = ssh.exec_command(
                    f"docker restart {service}"
                )
                stdout.read()
            
            # Clean up if disk usage high
            if health.get('disk_usage', 0) > 90:
                logger.info(f"Cleaning up disk on {node.name}")
                stdin, stdout, stderr = ssh.exec_command(
                    "docker system prune -af --volumes"
                )
                stdout.read()
            
        finally:
            ssh.close()

async def main():
    """Main deployment function"""
    orchestrator = MultiNodeOrchestrator()
    monitor = NodeHealthMonitor(orchestrator)
    
    # Deploy to all nodes
    logger.info("Starting multi-node deployment...")
    await orchestrator.deploy_all_nodes()
    
    # Start health monitoring
    logger.info("Starting health monitoring...")
    monitor_task = asyncio.create_task(monitor.monitor_all_nodes())
    
    # Keep running
    await monitor_task

if __name__ == "__main__":
    asyncio.run(main())
