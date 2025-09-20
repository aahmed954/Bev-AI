#!/usr/bin/env python3
"""
Autonomous Enhancement Controller
Self-improvement and capability discovery system with auto-scaling
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import httpx
import psutil
import yaml
from prometheus_client import Counter, Gauge, Histogram, generate_latest
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
import pickle
from collections import deque
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
operation_counter = Counter('autonomous_operations_total', 'Total autonomous operations', ['type'])
performance_gauge = Gauge('autonomous_performance_score', 'Current performance score')
capability_gauge = Gauge('autonomous_capabilities_active', 'Active capabilities count')
improvement_histogram = Histogram('autonomous_improvement_duration', 'Time spent on improvements')
resource_gauge = Gauge('autonomous_resource_usage', 'Resource usage percentage', ['resource'])

class OperationMode(Enum):
    """Operation modes for the autonomous controller"""
    SUPERVISED = "supervised"
    SEMI_AUTONOMOUS = "semi_autonomous"
    FULLY_AUTONOMOUS = "fully_autonomous"
    LEARNING = "learning"
    OPTIMIZATION = "optimization"

class CapabilityStatus(Enum):
    """Status of discovered capabilities"""
    DISCOVERED = "discovered"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"

@dataclass
class Capability:
    """Represents a discovered or learned capability"""
    id: str
    name: str
    description: str
    type: str
    status: CapabilityStatus
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    discovery_time: datetime = field(default_factory=datetime.now)
    validation_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_throughput: float
    response_time: float
    error_rate: float
    throughput: float
    efficiency_score: float
    timestamp: datetime = field(default_factory=datetime.now)

class NeuralOptimizer(nn.Module):
    """Neural network for performance optimization"""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

class AutonomousController:
    """Main autonomous controller with self-improvement capabilities"""

    def __init__(self, config_path: str = "config/autonomous.yaml"):
        self.config = self._load_config(config_path)
        self.mode = OperationMode.SUPERVISED
        self.capabilities: Dict[str, Capability] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.learning_buffer: deque = deque(maxlen=500)
        self.optimizer = NeuralOptimizer()
        self.optimizer_trained = False
        self.redis_client = None
        self.neo4j_driver = None
        self.http_client = None
        self.scaling_policy = self.config.get('scaling_policy', {})
        self.improvement_strategies: List[str] = []
        self.current_performance = None
        self.target_performance = self.config.get('target_performance', {})

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'mode': 'supervised',
            'redis_url': 'redis://redis:6379',
            'neo4j_url': 'bolt://neo4j:7687',
            'neo4j_auth': ('neo4j', 'password'),
            'optimization_interval': 300,
            'capability_discovery_interval': 600,
            'performance_threshold': 0.8,
            'resource_limits': {
                'cpu': 80,
                'memory': 85,
                'disk': 90
            },
            'scaling_policy': {
                'min_instances': 1,
                'max_instances': 10,
                'scale_up_threshold': 0.8,
                'scale_down_threshold': 0.3,
                'cooldown_period': 300
            },
            'target_performance': {
                'response_time': 100,
                'error_rate': 0.01,
                'throughput': 1000,
                'efficiency': 0.9
            }
        }

    async def initialize(self):
        """Initialize connections and load state"""
        try:
            # Initialize Redis
            self.redis_client = await redis.from_url(
                self.config['redis_url'],
                encoding="utf-8",
                decode_responses=True
            )

            # Initialize Neo4j
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config['neo4j_url'],
                auth=self.config['neo4j_auth']
            )

            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)

            # Load saved capabilities
            await self._load_capabilities()

            # Load optimizer state if exists
            await self._load_optimizer_state()

            # Start background tasks
            asyncio.create_task(self._optimization_loop())
            asyncio.create_task(self._capability_discovery_loop())
            asyncio.create_task(self._performance_monitoring_loop())
            asyncio.create_task(self._auto_scaling_loop())

            logger.info("Autonomous controller initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            raise

    async def _load_capabilities(self):
        """Load saved capabilities from Redis"""
        try:
            capabilities_data = await self.redis_client.get("autonomous:capabilities")
            if capabilities_data:
                capabilities = json.loads(capabilities_data)
                for cap_data in capabilities:
                    capability = Capability(**cap_data)
                    self.capabilities[capability.id] = capability
                logger.info(f"Loaded {len(self.capabilities)} capabilities")
        except Exception as e:
            logger.error(f"Failed to load capabilities: {e}")

    async def _save_capabilities(self):
        """Save capabilities to Redis"""
        try:
            capabilities_data = [
                {
                    'id': cap.id,
                    'name': cap.name,
                    'description': cap.description,
                    'type': cap.type,
                    'status': cap.status.value,
                    'performance_metrics': cap.performance_metrics,
                    'dependencies': cap.dependencies,
                    'parameters': cap.parameters,
                    'discovery_time': cap.discovery_time.isoformat(),
                    'validation_score': cap.validation_score,
                    'usage_count': cap.usage_count,
                    'last_used': cap.last_used.isoformat() if cap.last_used else None
                }
                for cap in self.capabilities.values()
            ]
            await self.redis_client.set(
                "autonomous:capabilities",
                json.dumps(capabilities_data)
            )
        except Exception as e:
            logger.error(f"Failed to save capabilities: {e}")

    async def _load_optimizer_state(self):
        """Load saved optimizer state"""
        try:
            state_data = await self.redis_client.get("autonomous:optimizer_state")
            if state_data:
                state_dict = pickle.loads(state_data.encode('latin-1'))
                self.optimizer.load_state_dict(state_dict)
                self.optimizer_trained = True
                logger.info("Loaded optimizer state")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    async def _save_optimizer_state(self):
        """Save optimizer state"""
        try:
            state_dict = self.optimizer.state_dict()
            state_data = pickle.dumps(state_dict).decode('latin-1')
            await self.redis_client.set("autonomous:optimizer_state", state_data)
        except Exception as e:
            logger.error(f"Failed to save optimizer state: {e}")

    async def discover_capability(self, source: str, metadata: Dict[str, Any]) -> Optional[Capability]:
        """Discover a new capability from various sources"""
        try:
            # Generate capability ID
            cap_id = hashlib.md5(
                f"{source}:{json.dumps(metadata, sort_keys=True)}".encode()
            ).hexdigest()[:12]

            # Check if already discovered
            if cap_id in self.capabilities:
                return self.capabilities[cap_id]

            # Create new capability
            capability = Capability(
                id=cap_id,
                name=metadata.get('name', f'capability_{cap_id}'),
                description=metadata.get('description', 'Auto-discovered capability'),
                type=metadata.get('type', 'unknown'),
                status=CapabilityStatus.DISCOVERED,
                parameters=metadata.get('parameters', {}),
                dependencies=metadata.get('dependencies', [])
            )

            # Test the capability
            validation_score = await self._test_capability(capability)
            capability.validation_score = validation_score

            if validation_score > 0.7:
                capability.status = CapabilityStatus.VALIDATED
                self.capabilities[cap_id] = capability
                await self._save_capabilities()

                # Log to Neo4j
                await self._log_capability_discovery(capability)

                operation_counter.labels(type='capability_discovered').inc()
                capability_gauge.set(len(self.capabilities))

                logger.info(f"Discovered new capability: {capability.name} (score: {validation_score:.2f})")
                return capability

            return None

        except Exception as e:
            logger.error(f"Failed to discover capability: {e}")
            return None

    async def _test_capability(self, capability: Capability) -> float:
        """Test a discovered capability"""
        try:
            # Simulate capability testing
            # In production, this would actually test the capability
            test_score = np.random.random() * 0.3 + 0.6  # 0.6-0.9 range

            # Consider dependencies
            if capability.dependencies:
                dep_score = sum(
                    self.capabilities.get(dep, Capability(id=dep, name=dep, description='', type='', status=CapabilityStatus.DISCOVERED)).validation_score
                    for dep in capability.dependencies
                ) / len(capability.dependencies)
                test_score = (test_score + dep_score) / 2

            return min(test_score, 1.0)

        except Exception as e:
            logger.error(f"Failed to test capability: {e}")
            return 0.0

    async def _log_capability_discovery(self, capability: Capability):
        """Log capability discovery to Neo4j"""
        try:
            async with self.neo4j_driver.session() as session:
                await session.run(
                    """
                    CREATE (c:Capability {
                        id: $id,
                        name: $name,
                        type: $type,
                        status: $status,
                        validation_score: $score,
                        discovery_time: datetime($time)
                    })
                    """,
                    id=capability.id,
                    name=capability.name,
                    type=capability.type,
                    status=capability.status.value,
                    score=capability.validation_score,
                    time=capability.discovery_time.isoformat()
                )
        except Exception as e:
            logger.error(f"Failed to log capability discovery: {e}")

    async def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance using neural optimizer"""
        try:
            with improvement_histogram.time():
                # Collect current metrics
                metrics = await self._collect_performance_metrics()

                # Prepare input for optimizer
                input_tensor = self._metrics_to_tensor(metrics)

                if self.optimizer_trained:
                    # Get optimization suggestions
                    self.optimizer.eval()
                    with torch.no_grad():
                        optimization_score = self.optimizer(input_tensor).item()

                    # Apply optimizations based on score
                    optimizations = await self._apply_optimizations(optimization_score, metrics)
                else:
                    # Train the optimizer with collected data
                    if len(self.learning_buffer) >= 100:
                        await self._train_optimizer()
                    optimizations = await self._apply_heuristic_optimizations(metrics)

                # Update performance history
                self.performance_history.append({
                    'metrics': metrics.__dict__,
                    'optimizations': optimizations,
                    'timestamp': datetime.now().isoformat()
                })

                operation_counter.labels(type='optimization').inc()
                performance_gauge.set(metrics.efficiency_score)

                return {
                    'status': 'optimized',
                    'metrics': metrics.__dict__,
                    'optimizations': optimizations,
                    'efficiency_score': metrics.efficiency_score
                }

        except Exception as e:
            logger.error(f"Failed to optimize performance: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Calculate network throughput (simplified)
            net_io = psutil.net_io_counters()
            throughput = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB

            # Calculate response time (from Redis if available)
            response_time = 50.0  # Default 50ms
            try:
                rt_data = await self.redis_client.get("metrics:response_time")
                if rt_data:
                    response_time = float(rt_data)
            except:
                pass

            # Calculate error rate
            error_rate = 0.01  # Default 1%
            try:
                er_data = await self.redis_client.get("metrics:error_rate")
                if er_data:
                    error_rate = float(er_data)
            except:
                pass

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency(
                cpu_percent, memory.percent, response_time, error_rate
            )

            metrics = PerformanceMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_throughput=throughput,
                response_time=response_time,
                error_rate=error_rate,
                throughput=throughput * 10,  # Simplified
                efficiency_score=efficiency_score
            )

            # Update Prometheus metrics
            resource_gauge.labels(resource='cpu').set(cpu_percent)
            resource_gauge.labels(resource='memory').set(memory.percent)
            resource_gauge.labels(resource='disk').set(disk.percent)

            self.current_performance = metrics
            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return PerformanceMetrics(0, 0, 0, 0, 100, 0.1, 0, 0.5)

    def _calculate_efficiency(self, cpu: float, memory: float,
                             response_time: float, error_rate: float) -> float:
        """Calculate overall efficiency score"""
        # Normalize metrics
        cpu_score = max(0, 1 - cpu / 100)
        memory_score = max(0, 1 - memory / 100)
        response_score = max(0, 1 - response_time / 1000)  # Assuming 1000ms is worst
        error_score = max(0, 1 - error_rate)

        # Weighted average
        weights = [0.3, 0.3, 0.25, 0.15]
        scores = [cpu_score, memory_score, response_score, error_score]

        return sum(w * s for w, s in zip(weights, scores))

    def _metrics_to_tensor(self, metrics: PerformanceMetrics) -> torch.Tensor:
        """Convert metrics to tensor for neural network"""
        values = [
            metrics.cpu_usage / 100,
            metrics.memory_usage / 100,
            metrics.disk_usage / 100,
            min(metrics.network_throughput / 1000, 1),
            min(metrics.response_time / 1000, 1),
            metrics.error_rate,
            min(metrics.throughput / 10000, 1),
            metrics.efficiency_score,
            self.mode.value == 'fully_autonomous',
            len(self.capabilities) / 100
        ]
        return torch.tensor(values, dtype=torch.float32)

    async def _train_optimizer(self):
        """Train the neural optimizer"""
        try:
            if len(self.learning_buffer) < 100:
                return

            # Prepare training data
            X = []
            y = []

            for i in range(len(self.learning_buffer) - 1):
                current = self.learning_buffer[i]
                next_metrics = self.learning_buffer[i + 1]

                X.append(self._metrics_to_tensor(current['metrics']))
                y.append(next_metrics['metrics'].efficiency_score)

            X = torch.stack(X)
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            # Train the model
            optimizer = torch.optim.Adam(self.optimizer.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.optimizer.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = self.optimizer(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                if epoch % 20 == 0:
                    logger.debug(f"Training epoch {epoch}, loss: {loss.item():.4f}")

            self.optimizer_trained = True
            await self._save_optimizer_state()

            logger.info("Neural optimizer trained successfully")

        except Exception as e:
            logger.error(f"Failed to train optimizer: {e}")

    async def _apply_optimizations(self, score: float, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Apply optimizations based on neural network score"""
        optimizations = {}

        try:
            # High-level optimization decisions based on score
            if score < 0.5:
                # Major optimizations needed
                optimizations['cache_size'] = 'increased'
                optimizations['connection_pool'] = 'expanded'
                optimizations['query_optimization'] = 'aggressive'

                # Update Redis settings
                await self.redis_client.config_set('maxmemory', '2gb')

            elif score < 0.75:
                # Moderate optimizations
                optimizations['cache_ttl'] = 'extended'
                optimizations['batch_size'] = 'optimized'

            else:
                # Fine-tuning
                optimizations['monitoring'] = 'enhanced'
                optimizations['logging'] = 'optimized'

            # Specific optimizations based on metrics
            if metrics.cpu_usage > 80:
                optimizations['cpu_optimization'] = 'thread_pool_adjusted'

            if metrics.memory_usage > 85:
                optimizations['memory_optimization'] = 'garbage_collection_tuned'

            if metrics.response_time > 200:
                optimizations['response_optimization'] = 'caching_enhanced'

            return optimizations

        except Exception as e:
            logger.error(f"Failed to apply optimizations: {e}")
            return {}

    async def _apply_heuristic_optimizations(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Apply heuristic-based optimizations"""
        optimizations = {}

        try:
            # CPU optimization
            if metrics.cpu_usage > self.config['resource_limits']['cpu']:
                optimizations['cpu'] = {
                    'action': 'scale_out',
                    'reason': f'CPU usage {metrics.cpu_usage:.1f}% exceeds limit'
                }

            # Memory optimization
            if metrics.memory_usage > self.config['resource_limits']['memory']:
                optimizations['memory'] = {
                    'action': 'clear_cache',
                    'reason': f'Memory usage {metrics.memory_usage:.1f}% exceeds limit'
                }
                # Clear some Redis cache
                await self.redis_client.execute_command('MEMORY', 'PURGE')

            # Response time optimization
            if metrics.response_time > self.target_performance['response_time']:
                optimizations['response'] = {
                    'action': 'enable_caching',
                    'reason': f'Response time {metrics.response_time:.1f}ms exceeds target'
                }

            # Error rate optimization
            if metrics.error_rate > self.target_performance['error_rate']:
                optimizations['errors'] = {
                    'action': 'enable_retry_logic',
                    'reason': f'Error rate {metrics.error_rate:.2%} exceeds target'
                }

            return optimizations

        except Exception as e:
            logger.error(f"Failed to apply heuristic optimizations: {e}")
            return {}

    async def auto_scale(self) -> Dict[str, Any]:
        """Auto-scale resources based on performance"""
        try:
            if not self.current_performance:
                return {'status': 'no_metrics'}

            metrics = self.current_performance
            scaling_decision = None

            # Check if we should scale up
            if metrics.efficiency_score < self.scaling_policy['scale_up_threshold']:
                current_instances = await self._get_current_instances()
                if current_instances < self.scaling_policy['max_instances']:
                    scaling_decision = 'scale_up'
                    await self._scale_instances(current_instances + 1)

            # Check if we should scale down
            elif metrics.efficiency_score > self.scaling_policy['scale_down_threshold']:
                current_instances = await self._get_current_instances()
                if current_instances > self.scaling_policy['min_instances']:
                    scaling_decision = 'scale_down'
                    await self._scale_instances(current_instances - 1)

            if scaling_decision:
                operation_counter.labels(type=f'auto_scale_{scaling_decision}').inc()
                return {
                    'status': 'scaled',
                    'action': scaling_decision,
                    'efficiency_score': metrics.efficiency_score
                }

            return {'status': 'no_scaling_needed'}

        except Exception as e:
            logger.error(f"Failed to auto-scale: {e}")
            return {'status': 'error', 'message': str(e)}

    async def _get_current_instances(self) -> int:
        """Get current number of instances"""
        try:
            instances_data = await self.redis_client.get("autonomous:instances")
            return int(instances_data) if instances_data else 1
        except:
            return 1

    async def _scale_instances(self, target_count: int):
        """Scale to target number of instances"""
        try:
            await self.redis_client.set("autonomous:instances", str(target_count))
            logger.info(f"Scaled to {target_count} instances")
        except Exception as e:
            logger.error(f"Failed to scale instances: {e}")

    async def _optimization_loop(self):
        """Background loop for continuous optimization"""
        while True:
            try:
                await asyncio.sleep(self.config['optimization_interval'])

                if self.mode in [OperationMode.SEMI_AUTONOMOUS, OperationMode.FULLY_AUTONOMOUS]:
                    result = await self.optimize_performance()
                    logger.info(f"Optimization result: {result.get('efficiency_score', 'N/A')}")

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")

    async def _capability_discovery_loop(self):
        """Background loop for capability discovery"""
        while True:
            try:
                await asyncio.sleep(self.config['capability_discovery_interval'])

                if self.mode in [OperationMode.LEARNING, OperationMode.FULLY_AUTONOMOUS]:
                    # Attempt to discover new capabilities
                    sources = ['api_endpoints', 'system_apis', 'learned_patterns']
                    for source in sources:
                        metadata = await self._explore_capability_source(source)
                        if metadata:
                            await self.discover_capability(source, metadata)

            except Exception as e:
                logger.error(f"Error in capability discovery loop: {e}")

    async def _explore_capability_source(self, source: str) -> Optional[Dict[str, Any]]:
        """Explore a source for new capabilities"""
        try:
            # Simulate capability exploration
            # In production, this would actually probe APIs, analyze patterns, etc.
            if np.random.random() > 0.7:  # 30% chance of discovery
                return {
                    'name': f'{source}_capability_{int(time.time())}',
                    'description': f'Capability discovered from {source}',
                    'type': source,
                    'parameters': {'threshold': 0.5, 'enabled': True}
                }
            return None
        except Exception as e:
            logger.error(f"Failed to explore source {source}: {e}")
            return None

    async def _performance_monitoring_loop(self):
        """Background loop for performance monitoring"""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                metrics = await self._collect_performance_metrics()

                # Add to learning buffer
                self.learning_buffer.append({
                    'metrics': metrics,
                    'timestamp': datetime.now()
                })

                # Check for alerts
                if metrics.efficiency_score < self.config['performance_threshold']:
                    logger.warning(f"Performance below threshold: {metrics.efficiency_score:.2f}")

                    if self.mode == OperationMode.FULLY_AUTONOMOUS:
                        # Trigger immediate optimization
                        await self.optimize_performance()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    async def _auto_scaling_loop(self):
        """Background loop for auto-scaling"""
        while True:
            try:
                await asyncio.sleep(self.scaling_policy['cooldown_period'])

                if self.mode in [OperationMode.SEMI_AUTONOMOUS, OperationMode.FULLY_AUTONOMOUS]:
                    result = await self.auto_scale()
                    if result['status'] == 'scaled':
                        logger.info(f"Auto-scaling: {result['action']}")

            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")

    async def set_mode(self, mode: OperationMode) -> Dict[str, str]:
        """Set the operation mode"""
        self.mode = mode
        await self.redis_client.set("autonomous:mode", mode.value)
        logger.info(f"Operation mode set to: {mode.value}")
        return {'status': 'success', 'mode': mode.value}

    async def get_status(self) -> Dict[str, Any]:
        """Get current controller status"""
        return {
            'mode': self.mode.value,
            'capabilities_count': len(self.capabilities),
            'performance': self.current_performance.__dict__ if self.current_performance else None,
            'optimizer_trained': self.optimizer_trained,
            'active_capabilities': [
                cap.name for cap in self.capabilities.values()
                if cap.status == CapabilityStatus.DEPLOYED
            ]
        }

# FastAPI application
app = FastAPI(title="Autonomous Controller", version="1.0.0")
controller = None

@app.on_event("startup")
async def startup_event():
    global controller
    controller = AutonomousController()
    await controller.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    if controller:
        if controller.redis_client:
            await controller.redis_client.close()
        if controller.neo4j_driver:
            await controller.neo4j_driver.close()
        if controller.http_client:
            await controller.http_client.aclose()

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return await controller.get_status()

@app.post("/mode")
async def set_mode(mode: str):
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    try:
        op_mode = OperationMode(mode)
        return await controller.set_mode(op_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

@app.post("/optimize")
async def optimize():
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return await controller.optimize_performance()

@app.post("/discover")
async def discover_capability(source: str, metadata: Dict[str, Any]):
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    capability = await controller.discover_capability(source, metadata)
    if capability:
        return {
            'status': 'discovered',
            'capability': {
                'id': capability.id,
                'name': capability.name,
                'validation_score': capability.validation_score
            }
        }
    return {'status': 'failed'}

@app.get("/capabilities")
async def list_capabilities():
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    return [
        {
            'id': cap.id,
            'name': cap.name,
            'status': cap.status.value,
            'validation_score': cap.validation_score,
            'usage_count': cap.usage_count
        }
        for cap in controller.capabilities.values()
    ]

@app.get("/metrics")
async def get_metrics():
    return generate_latest()

@app.post("/scale")
async def trigger_scaling():
    if not controller:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    return await controller.auto_scale()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)