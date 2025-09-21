#!/usr/bin/env python3
"""
BEV OSINT Framework - Cluster Coordinator

This service coordinates the distributed deployment of BEV OSINT nodes,
managing service discovery, health monitoring, and deployment orchestration.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configuration
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", "8080"))
COORDINATOR_HOST = os.getenv("COORDINATOR_HOST", "0.0.0.0")
CLUSTER_NAME = os.getenv("BEV_CLUSTER_NAME", "bev-osint-cluster")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
NODE_TIMEOUT = int(os.getenv("NODE_TIMEOUT", "300"))
DEPLOYMENT_TIMEOUT = int(os.getenv("DEPLOYMENT_TIMEOUT", "1800"))

# Node type deployment order (dependencies)
NODE_DEPLOYMENT_ORDER = [
    "data-core",
    "data-analytics", 
    "message-infrastructure",
    "infrastructure-monitor",
    "processing-core",
    "specialized-processing",
    "ml-intelligence",
    "frontend-api",
    "edge-computing"
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NodeInfo:
    """Information about a registered node"""
    node_id: str
    node_type: str
    hostname: str
    endpoints: Dict[str, str]
    status: str  # registering, healthy, unhealthy, offline
    last_heartbeat: datetime
    deployment_phase: str  # pending, deploying, deployed, failed
    metadata: Dict[str, any]

class NodeRegistration(BaseModel):
    """Node registration request model"""
    node_type: str
    node_id: str
    hostname: str = None
    endpoints: Dict[str, str] = {}
    metadata: Dict[str, any] = {}

class ClusterStatus(BaseModel):
    """Cluster status response model"""
    cluster_name: str
    total_nodes: int
    healthy_nodes: int
    deployment_phase: str
    nodes_by_type: Dict[str, int]
    last_updated: datetime

class ClusterCoordinator:
    """Main cluster coordinator service"""
    
    def __init__(self):
        self.app = FastAPI(
            title="BEV OSINT Cluster Coordinator",
            description="Coordinates distributed BEV OSINT deployment",
            version="1.0.0"
        )
        self.nodes: Dict[str, NodeInfo] = {}
        self.deployment_queue: List[str] = []
        self.deployment_in_progress: Set[str] = set()
        self.cluster_deployment_phase = "idle"
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/register")
        async def register_node(registration: NodeRegistration):
            """Register a new node with the cluster"""
            return await self.register_node_handler(registration)
        
        @self.app.post("/heartbeat/{node_id}")
        async def node_heartbeat(node_id: str):
            """Node heartbeat endpoint"""
            return await self.heartbeat_handler(node_id)
        
        @self.app.get("/status")
        async def cluster_status():
            """Get cluster status"""
            return await self.get_cluster_status()
        
        @self.app.get("/nodes")
        async def list_nodes():
            """List all registered nodes"""
            return {"nodes": [asdict(node) for node in self.nodes.values()]}
        
        @self.app.get("/nodes/{node_id}")
        async def get_node(node_id: str):
            """Get specific node information"""
            if node_id not in self.nodes:
                raise HTTPException(status_code=404, detail="Node not found")
            return asdict(self.nodes[node_id])
        
        @self.app.post("/deploy")
        async def deploy_cluster(background_tasks: BackgroundTasks):
            """Initiate cluster deployment"""
            background_tasks.add_task(self.deploy_cluster_task)
            return {"message": "Cluster deployment initiated"}
        
        @self.app.post("/deploy/{node_type}")
        async def deploy_node_type(node_type: str, background_tasks: BackgroundTasks):
            """Deploy specific node type"""
            background_tasks.add_task(self.deploy_node_type_task, node_type)
            return {"message": f"Deployment initiated for {node_type} nodes"}
        
        @self.app.get("/discovery/{service_type}")
        async def service_discovery(service_type: str):
            """Service discovery endpoint"""
            return await self.discover_services(service_type)
        
        @self.app.get("/health")
        async def health_check():
            """Coordinator health check"""
            return {
                "status": "healthy",
                "cluster_name": CLUSTER_NAME,
                "nodes_count": len(self.nodes),
                "deployment_phase": self.cluster_deployment_phase
            }
    
    async def register_node_handler(self, registration: NodeRegistration) -> JSONResponse:
        """Handle node registration"""
        try:
            # Generate hostname if not provided
            hostname = registration.hostname or f"{registration.node_type}-{registration.node_id}"
            
            # Create node info
            node_info = NodeInfo(
                node_id=registration.node_id,
                node_type=registration.node_type,
                hostname=hostname,
                endpoints=registration.endpoints,
                status="registering",
                last_heartbeat=datetime.now(),
                deployment_phase="pending",
                metadata=registration.metadata
            )
            
            # Register the node
            self.nodes[registration.node_id] = node_info
            
            logger.info(f"Registered node: {registration.node_id} ({registration.node_type})")
            
            # Add to deployment queue if cluster deployment is active
            if self.cluster_deployment_phase == "deploying":
                await self.schedule_node_deployment(registration.node_id)
            
            return JSONResponse({
                "status": "registered",
                "node_id": registration.node_id,
                "cluster_name": CLUSTER_NAME,
                "coordinator_endpoints": {
                    "heartbeat": f"/heartbeat/{registration.node_id}",
                    "discovery": "/discovery",
                    "status": "/status"
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to register node {registration.node_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def heartbeat_handler(self, node_id: str) -> JSONResponse:
        """Handle node heartbeat"""
        if node_id not in self.nodes:
            raise HTTPException(status_code=404, detail="Node not registered")
        
        node = self.nodes[node_id]
        node.last_heartbeat = datetime.now()
        
        # Update status based on heartbeat
        if node.status == "registering":
            node.status = "healthy"
        elif node.status == "unhealthy":
            node.status = "healthy"
            logger.info(f"Node {node_id} recovered")
        
        return JSONResponse({"status": "acknowledged"})
    
    async def get_cluster_status(self) -> ClusterStatus:
        """Get current cluster status"""
        healthy_nodes = sum(1 for node in self.nodes.values() if node.status == "healthy")
        
        nodes_by_type = {}
        for node in self.nodes.values():
            node_type = node.node_type
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1
        
        return ClusterStatus(
            cluster_name=CLUSTER_NAME,
            total_nodes=len(self.nodes),
            healthy_nodes=healthy_nodes,
            deployment_phase=self.cluster_deployment_phase,
            nodes_by_type=nodes_by_type,
            last_updated=datetime.now()
        )
    
    async def discover_services(self, service_type: str) -> JSONResponse:
        """Service discovery for specific service types"""
        services = []
        
        # Map service types to node types and endpoints
        service_mappings = {
            "postgres": ("data-core", "postgres"),
            "neo4j": ("data-core", "neo4j_bolt"),
            "redis": ("data-core", "redis_cluster"),
            "elasticsearch": ("data-analytics", "elasticsearch"),
            "rabbitmq": ("message-infrastructure", "rabbitmq"),
            "intelowl": ("processing-core", "intelowl"),
            "mcp-server": ("processing-core", "mcp_server"),
            "grafana": ("infrastructure-monitor", "grafana"),
            "prometheus": ("infrastructure-monitor", "prometheus")
        }
        
        if service_type in service_mappings:
            target_node_type, endpoint_key = service_mappings[service_type]
            
            for node in self.nodes.values():
                if (node.node_type == target_node_type and 
                    node.status == "healthy" and
                    endpoint_key in node.endpoints):
                    services.append({
                        "node_id": node.node_id,
                        "hostname": node.hostname,
                        "endpoint": node.endpoints[endpoint_key],
                        "status": node.status
                    })
        
        return JSONResponse({
            "service_type": service_type,
            "services": services,
            "count": len(services)
        })
    
    async def deploy_cluster_task(self):
        """Background task to deploy entire cluster"""
        try:
            self.cluster_deployment_phase = "deploying"
            logger.info("Starting cluster deployment")
            
            for node_type in NODE_DEPLOYMENT_ORDER:
                await self.deploy_node_type_task(node_type)
                
                # Wait for node type to be healthy before proceeding
                await self.wait_for_node_type_health(node_type)
            
            self.cluster_deployment_phase = "deployed"
            logger.info("Cluster deployment completed successfully")
            
        except Exception as e:
            self.cluster_deployment_phase = "failed"
            logger.error(f"Cluster deployment failed: {e}")
    
    async def deploy_node_type_task(self, node_type: str):
        """Deploy all nodes of a specific type"""
        try:
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == node_type and 
                          node.deployment_phase == "pending"]
            
            if not target_nodes:
                logger.info(f"No {node_type} nodes to deploy")
                return
            
            logger.info(f"Deploying {len(target_nodes)} {node_type} nodes")
            
            # Deploy nodes in parallel
            deployment_tasks = []
            for node in target_nodes:
                task = asyncio.create_task(self.deploy_node(node.node_id))
                deployment_tasks.append(task)
            
            # Wait for all deployments to complete
            await asyncio.gather(*deployment_tasks, return_exceptions=True)
            
            logger.info(f"Completed deployment of {node_type} nodes")
            
        except Exception as e:
            logger.error(f"Failed to deploy {node_type} nodes: {e}")
    
    async def deploy_node(self, node_id: str):
        """Deploy a specific node"""
        try:
            node = self.nodes[node_id]
            node.deployment_phase = "deploying"
            self.deployment_in_progress.add(node_id)
            
            logger.info(f"Deploying node {node_id}")
            
            # In a real implementation, this would trigger the deployment
            # For now, we simulate deployment time
            await asyncio.sleep(10)  # Simulate deployment time
            
            node.deployment_phase = "deployed"
            self.deployment_in_progress.discard(node_id)
            
            logger.info(f"Successfully deployed node {node_id}")
            
        except Exception as e:
            node = self.nodes.get(node_id)
            if node:
                node.deployment_phase = "failed"
            self.deployment_in_progress.discard(node_id)
            logger.error(f"Failed to deploy node {node_id}: {e}")
    
    async def wait_for_node_type_health(self, node_type: str, timeout: int = 300):
        """Wait for all nodes of a type to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            target_nodes = [node for node in self.nodes.values() 
                          if node.node_type == node_type]
            
            if not target_nodes:
                break
                
            healthy_nodes = [node for node in target_nodes 
                           if node.status == "healthy"]
            
            if len(healthy_nodes) == len(target_nodes):
                logger.info(f"All {node_type} nodes are healthy")
                break
            
            logger.info(f"Waiting for {node_type} nodes: {len(healthy_nodes)}/{len(target_nodes)} healthy")
            await asyncio.sleep(10)
        else:
            logger.warning(f"Timeout waiting for {node_type} nodes to become healthy")
    
    async def schedule_node_deployment(self, node_id: str):
        """Schedule a node for deployment based on dependencies"""
        node = self.nodes[node_id]
        
        # Check if dependencies are satisfied
        dependencies_met = await self.check_node_dependencies(node.node_type)
        
        if dependencies_met:
            await self.deploy_node(node_id)
        else:
            self.deployment_queue.append(node_id)
            logger.info(f"Node {node_id} queued for deployment (waiting for dependencies)")
    
    async def check_node_dependencies(self, node_type: str) -> bool:
        """Check if node type dependencies are satisfied"""
        try:
            node_index = NODE_DEPLOYMENT_ORDER.index(node_type)
        except ValueError:
            return True  # Unknown node type, assume no dependencies
        
        # Check if all previous node types are healthy
        for i in range(node_index):
            required_type = NODE_DEPLOYMENT_ORDER[i]
            required_nodes = [node for node in self.nodes.values() 
                            if node.node_type == required_type]
            
            if required_nodes:
                healthy_required = [node for node in required_nodes 
                                  if node.status == "healthy"]
                if len(healthy_required) != len(required_nodes):
                    return False
        
        return True
    
    async def monitor_node_health(self):
        """Background task to monitor node health"""
        while True:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=NODE_TIMEOUT)
                
                for node_id, node in self.nodes.items():
                    if node.last_heartbeat < timeout_threshold:
                        if node.status != "offline":
                            node.status = "offline"
                            logger.warning(f"Node {node_id} marked as offline (no heartbeat)")
                    elif node.status == "offline":
                        # Don't automatically mark as healthy, wait for heartbeat
                        pass
                
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    async def startup(self):
        """Startup tasks"""
        logger.info(f"Starting BEV OSINT Cluster Coordinator for cluster: {CLUSTER_NAME}")
        
        # Start health monitoring
        asyncio.create_task(self.monitor_node_health())
        
        logger.info(f"Coordinator running on {COORDINATOR_HOST}:{COORDINATOR_PORT}")
    
    async def shutdown(self):
        """Shutdown tasks"""
        logger.info("Shutting down cluster coordinator")

# Create coordinator instance
coordinator = ClusterCoordinator()

# Setup FastAPI events
@coordinator.app.on_event("startup")
async def startup_event():
    await coordinator.startup()

@coordinator.app.on_event("shutdown") 
async def shutdown_event():
    await coordinator.shutdown()

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "cluster-coordinator:coordinator.app",
        host=COORDINATOR_HOST,
        port=COORDINATOR_PORT,
        log_level="info",
        reload=False
    )