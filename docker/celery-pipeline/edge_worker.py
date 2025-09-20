#!/usr/bin/env python3
"""
Edge Computing Worker for ORACLE1
Handles distributed processing and edge node coordination
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from celery import Task
from celery_app import app
from pydantic import BaseModel
import redis
import httpx
import docker
import kubernetes
from kubernetes import client, config

# Configure structured logging
logger = structlog.get_logger("edge_worker")

class EdgeTask(BaseModel):
    """Edge computing task model"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 5
    timeout: int = 300
    retry_count: int = 3
    target_nodes: Optional[List[str]] = None
    requirements: Dict[str, Any] = {}

class EdgeNode(BaseModel):
    """Edge node model"""
    node_id: str
    node_type: str
    status: str
    capabilities: List[str]
    resources: Dict[str, Any]
    location: Optional[str] = None
    last_heartbeat: datetime

class EdgeComputingManager:
    """Edge computing coordination manager"""

    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=1)
        self.docker_client = None
        self.k8s_client = None
        self.active_nodes = {}
        self.setup_clients()

    def setup_clients(self):
        """Setup Docker and Kubernetes clients"""
        try:
            # Docker client for local containerization
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized")
        except Exception as e:
            logger.warning("Docker client initialization failed", error=str(e))

        try:
            # Kubernetes client for cluster management
            config.load_incluster_config()  # For in-cluster usage
            self.k8s_client = client.AppsV1Api()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning("Kubernetes client initialization failed", error=str(e))
            try:
                # Try local kubeconfig
                config.load_kube_config()
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized from local config")
            except Exception as e2:
                logger.warning("Local Kubernetes config failed", error=str(e2))

    def register_edge_node(self, node: EdgeNode) -> bool:
        """Register a new edge node"""
        try:
            node_data = node.dict()
            self.redis_client.hset(
                f"edge_nodes:{node.node_id}",
                mapping=node_data
            )
            self.redis_client.expire(f"edge_nodes:{node.node_id}", 300)  # 5 min TTL

            self.active_nodes[node.node_id] = node
            logger.info("Edge node registered", node_id=node.node_id, node_type=node.node_type)
            return True

        except Exception as e:
            logger.error("Failed to register edge node", node_id=node.node_id, error=str(e))
            return False

    def get_available_nodes(self, requirements: Dict[str, Any] = None) -> List[EdgeNode]:
        """Get list of available edge nodes matching requirements"""
        try:
            available_nodes = []
            node_keys = self.redis_client.keys("edge_nodes:*")

            for key in node_keys:
                node_data = self.redis_client.hgetall(key)
                if node_data:
                    node_data = {k.decode(): v.decode() for k, v in node_data.items()}
                    node = EdgeNode(**node_data)

                    # Check if node meets requirements
                    if self._node_meets_requirements(node, requirements):
                        available_nodes.append(node)

            return available_nodes

        except Exception as e:
            logger.error("Failed to get available nodes", error=str(e))
            return []

    def _node_meets_requirements(self, node: EdgeNode, requirements: Dict[str, Any]) -> bool:
        """Check if node meets task requirements"""
        if not requirements:
            return True

        try:
            # Check capabilities
            required_caps = requirements.get('capabilities', [])
            if not all(cap in node.capabilities for cap in required_caps):
                return False

            # Check resources
            required_resources = requirements.get('resources', {})
            for resource, min_value in required_resources.items():
                if node.resources.get(resource, 0) < min_value:
                    return False

            # Check node type
            required_type = requirements.get('node_type')
            if required_type and node.node_type != required_type:
                return False

            return True

        except Exception as e:
            logger.error("Error checking node requirements", node_id=node.node_id, error=str(e))
            return False

    async def distribute_task(self, task: EdgeTask) -> Dict[str, Any]:
        """Distribute task to appropriate edge nodes"""
        try:
            # Get available nodes
            available_nodes = self.get_available_nodes(task.requirements)

            if not available_nodes:
                raise Exception("No suitable edge nodes available")

            # Select optimal nodes
            selected_nodes = self._select_optimal_nodes(available_nodes, task)

            if not selected_nodes:
                raise Exception("No optimal nodes found for task")

            # Distribute task to selected nodes
            results = await self._execute_on_nodes(task, selected_nodes)

            return {
                "task_id": task.task_id,
                "status": "completed",
                "results": results,
                "nodes_used": [node.node_id for node in selected_nodes],
                "execution_time": time.time()
            }

        except Exception as e:
            logger.error("Task distribution failed", task_id=task.task_id, error=str(e))
            raise

    def _select_optimal_nodes(self, available_nodes: List[EdgeNode], task: EdgeTask) -> List[EdgeNode]:
        """Select optimal nodes for task execution"""
        try:
            # Simple selection strategy based on capabilities and resources
            scored_nodes = []

            for node in available_nodes:
                score = 0

                # Score based on capabilities
                matching_caps = sum(1 for cap in task.requirements.get('capabilities', [])
                                  if cap in node.capabilities)
                score += matching_caps * 10

                # Score based on available resources
                cpu_score = min(node.resources.get('cpu', 0) / 100, 1) * 5
                memory_score = min(node.resources.get('memory', 0) / 1000, 1) * 5
                score += cpu_score + memory_score

                scored_nodes.append((score, node))

            # Sort by score and select top nodes
            scored_nodes.sort(key=lambda x: x[0], reverse=True)

            # Select nodes based on task requirements
            max_nodes = task.requirements.get('max_nodes', 3)
            selected = [node for score, node in scored_nodes[:max_nodes]]

            return selected

        except Exception as e:
            logger.error("Node selection failed", task_id=task.task_id, error=str(e))
            return []

    async def _execute_on_nodes(self, task: EdgeTask, nodes: List[EdgeNode]) -> List[Dict]:
        """Execute task on selected edge nodes"""
        results = []

        try:
            async with httpx.AsyncClient(timeout=task.timeout) as client:
                tasks = []

                for node in nodes:
                    # Create execution task for each node
                    node_task = self._create_node_task(task, node, client)
                    tasks.append(node_task)

                # Execute tasks concurrently
                completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(completed_tasks):
                    if isinstance(result, Exception):
                        logger.error("Node execution failed",
                                   node_id=nodes[i].node_id,
                                   error=str(result))
                        results.append({
                            "node_id": nodes[i].node_id,
                            "status": "failed",
                            "error": str(result)
                        })
                    else:
                        results.append(result)

            return results

        except Exception as e:
            logger.error("Parallel execution failed", task_id=task.task_id, error=str(e))
            raise

    async def _create_node_task(self, task: EdgeTask, node: EdgeNode, client: httpx.AsyncClient):
        """Create execution task for specific node"""
        try:
            # Prepare task payload for node
            node_payload = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "payload": task.payload,
                "node_id": node.node_id
            }

            # Send task to node (assuming HTTP API endpoint)
            node_url = f"http://{node.node_id}:8080/execute"
            response = await client.post(node_url, json=node_payload)

            if response.status_code == 200:
                result = response.json()
                return {
                    "node_id": node.node_id,
                    "status": "completed",
                    "result": result
                }
            else:
                return {
                    "node_id": node.node_id,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            return {
                "node_id": node.node_id,
                "status": "failed",
                "error": str(e)
            }

    def deploy_edge_service(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy service to edge nodes using Kubernetes"""
        try:
            if not self.k8s_client:
                raise Exception("Kubernetes client not available")

            # Create deployment configuration
            deployment = self._create_deployment_config(service_config)

            # Deploy to Kubernetes
            api_response = self.k8s_client.create_namespaced_deployment(
                body=deployment,
                namespace=service_config.get('namespace', 'default')
            )

            logger.info("Edge service deployed",
                       service_name=service_config['name'],
                       deployment_id=api_response.metadata.uid)

            return {
                "status": "deployed",
                "deployment_id": api_response.metadata.uid,
                "service_name": service_config['name']
            }

        except Exception as e:
            logger.error("Edge service deployment failed",
                        service_name=service_config.get('name'),
                        error=str(e))
            raise

    def _create_deployment_config(self, service_config: Dict[str, Any]) -> Dict:
        """Create Kubernetes deployment configuration"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": service_config['name'],
                "labels": {
                    "app": service_config['name'],
                    "version": service_config.get('version', '1.0.0')
                }
            },
            "spec": {
                "replicas": service_config.get('replicas', 3),
                "selector": {
                    "matchLabels": {
                        "app": service_config['name']
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": service_config['name']
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": service_config['name'],
                            "image": service_config['image'],
                            "ports": [{
                                "containerPort": service_config.get('port', 8080)
                            }],
                            "env": service_config.get('environment', []),
                            "resources": service_config.get('resources', {})
                        }]
                    }
                }
            }
        }

# Initialize edge computing manager
edge_manager = EdgeComputingManager()

class EdgeComputingTask(Task):
    """Custom Celery task for edge computing"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("Edge computing task failed",
                    task_id=task_id,
                    exception=str(exc),
                    traceback=str(einfo))

@app.task(bind=True, base=EdgeComputingTask, queue='edge_computing')
def distribute_edge_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Distribute task to edge computing nodes"""
    try:
        logger.info("Processing edge computing task", task_id=self.request.id)

        task = EdgeTask(**task_data)

        # Run async task distribution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(edge_manager.distribute_task(task))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error("Edge task distribution failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=EdgeComputingTask, queue='edge_computing')
def register_edge_node(self, node_data: Dict[str, Any]) -> bool:
    """Register new edge node"""
    try:
        logger.info("Registering edge node", node_id=node_data.get('node_id'))

        node = EdgeNode(**node_data)
        result = edge_manager.register_edge_node(node)

        return result

    except Exception as e:
        logger.error("Edge node registration failed", error=str(e))
        raise

@app.task(bind=True, base=EdgeComputingTask, queue='edge_computing')
def deploy_edge_service(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deploy service to edge infrastructure"""
    try:
        logger.info("Deploying edge service", service_name=service_config.get('name'))

        result = edge_manager.deploy_edge_service(service_config)
        return result

    except Exception as e:
        logger.error("Edge service deployment failed", error=str(e))
        raise self.retry(exc=e, countdown=120, max_retries=2)

@app.task(bind=True, base=EdgeComputingTask, queue='edge_computing')
def monitor_edge_infrastructure(self) -> Dict[str, Any]:
    """Monitor edge infrastructure health and performance"""
    try:
        logger.info("Monitoring edge infrastructure")

        # Get all active nodes
        nodes = edge_manager.get_available_nodes()

        # Collect health metrics
        health_metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_nodes": len(nodes),
            "active_nodes": len([n for n in nodes if n.status == "active"]),
            "node_types": {},
            "total_resources": {"cpu": 0, "memory": 0},
            "nodes": []
        }

        for node in nodes:
            # Count node types
            health_metrics["node_types"][node.node_type] = \
                health_metrics["node_types"].get(node.node_type, 0) + 1

            # Sum resources
            health_metrics["total_resources"]["cpu"] += node.resources.get("cpu", 0)
            health_metrics["total_resources"]["memory"] += node.resources.get("memory", 0)

            # Add node details
            health_metrics["nodes"].append({
                "node_id": node.node_id,
                "node_type": node.node_type,
                "status": node.status,
                "capabilities": node.capabilities,
                "resources": node.resources
            })

        return health_metrics

    except Exception as e:
        logger.error("Infrastructure monitoring failed", error=str(e))
        raise

if __name__ == "__main__":
    # Test edge computing functionality
    test_task = EdgeTask(
        task_id="test_001",
        task_type="data_processing",
        payload={"data": [1, 2, 3, 4, 5]},
        requirements={"capabilities": ["python", "numpy"]}
    )

    print("Edge computing worker ready for task distribution")