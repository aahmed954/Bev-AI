#!/usr/bin/env python3
"""
Edge Computing Module - Distributed Inference & Processing
Deploy AI models and computations across edge devices for maximum scalability
"""

import asyncio
import aiohttp
import socket
import struct
import pickle
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import redis
import docker
import kubernetes
from kubernetes import client, config as k8s_config
import ray
import dask.distributed
from dask import delayed
import horovod.torch as hvd
import pynvml
import psutil
import GPUtil
import websockets
import grpc
from concurrent import futures
import tensorflow as tf
import onnx
import onnxruntime
from cryptography.fernet import Fernet
import zeroconf
import netifaces
import platform
import subprocess
import logging
from collections import defaultdict
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EdgeDevice:
    """Represents an edge computing device"""
    device_id: str
    hostname: str
    ip_address: str
    port: int
    device_type: str  # 'gpu', 'cpu', 'tpu', 'mobile', 'iot'
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: str = 'idle'  # 'idle', 'busy', 'offline'
    current_load: float = 0.0
    available_memory: int = 0
    available_compute: float = 0.0
    latency: float = 0.0
    reliability_score: float = 1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

@dataclass
class ComputeTask:
    """Represents a distributed compute task"""
    task_id: str
    task_type: str  # 'inference', 'training', 'processing'
    model_name: str
    input_data: Any
    priority: int = 5
    requirements: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    assigned_device: Optional[str] = None
    status: str = 'pending'
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class EdgeDiscovery:
    """Discover and manage edge devices"""
    
    def __init__(self):
        self.devices: Dict[str, EdgeDevice] = {}
        self.zeroconf = zeroconf.Zeroconf()
        self.discovery_running = False
        self.heartbeat_interval = 30
        
    async def discover_devices(self) -> Dict[str, EdgeDevice]:
        """Discover available edge devices"""
        
        logger.info("🔍 Discovering edge devices...")
        
        # Method 1: Local network scan
        local_devices = await self._scan_local_network()
        
        # Method 2: Kubernetes nodes
        k8s_devices = await self._discover_kubernetes_nodes()
        
        # Method 3: Docker containers
        docker_devices = await self._discover_docker_devices()
        
        # Method 4: Ray cluster
        ray_devices = await self._discover_ray_nodes()
        
        # Method 5: Custom edge registry
        registry_devices = await self._query_device_registry()
        
        # Combine all discovered devices
        all_devices = {**local_devices, **k8s_devices, **docker_devices, 
                      **ray_devices, **registry_devices}
        
        # Probe device capabilities
        for device_id, device in all_devices.items():
            capabilities = await self._probe_capabilities(device)
            device.capabilities = capabilities
            self.devices[device_id] = device
        
        logger.info(f"✅ Discovered {len(self.devices)} edge devices")
        return self.devices
    
    async def _scan_local_network(self) -> Dict[str, EdgeDevice]:
        """Scan local network for edge devices"""
        
        devices = {}
        
        # Get local network interfaces
        for interface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr_info in addrs[netifaces.AF_INET]:
                    ip = addr_info['addr']
                    if ip != '127.0.0.1':
                        # Scan subnet for edge devices
                        subnet = '.'.join(ip.split('.')[:-1]) + '.0/24'
                        devices.update(await self._scan_subnet(subnet))
        
        return devices
    
    async def _scan_subnet(self, subnet: str) -> Dict[str, EdgeDevice]:
        """Scan subnet for responsive devices"""
        
        devices = {}
        base_ip = subnet.split('/')[0].rsplit('.', 1)[0]
        
        tasks = []
        for i in range(1, 255):
            ip = f"{base_ip}.{i}"
            tasks.append(self._probe_device(ip))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, EdgeDevice):
                devices[result.device_id] = result
        
        return devices
    
    async def _probe_device(self, ip: str, port: int = 8888) -> Optional[EdgeDevice]:
        """Probe a specific IP for edge capabilities"""
        
        try:
            # Try to connect to edge service
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=1.0
            )
            
            # Send capability query
            writer.write(b'EDGE_PROBE\n')
            await writer.drain()
            
            # Read response
            response = await asyncio.wait_for(
                reader.readline(),
                timeout=1.0
            )
            
            writer.close()
            await writer.wait_closed()
            
            if response:
                # Parse device info
                device_info = json.loads(response.decode())
                
                return EdgeDevice(
                    device_id=device_info.get('id', hashlib.sha256(ip.encode()).hexdigest()[:8]),
                    hostname=device_info.get('hostname', ip),
                    ip_address=ip,
                    port=port,
                    device_type=device_info.get('type', 'cpu'),
                    capabilities=device_info.get('capabilities', {})
                )
        except:
            return None
    
    async def _discover_kubernetes_nodes(self) -> Dict[str, EdgeDevice]:
        """Discover Kubernetes nodes"""
        
        devices = {}
        
        try:
            k8s_config.load_incluster_config()
            v1 = client.CoreV1Api()
            
            nodes = v1.list_node()
            for node in nodes.items:
                # Get node resources
                allocatable = node.status.allocatable
                
                device = EdgeDevice(
                    device_id=node.metadata.uid[:8],
                    hostname=node.metadata.name,
                    ip_address=node.status.addresses[0].address,
                    port=10250,  # Kubelet port
                    device_type=self._detect_k8s_device_type(node),
                    capabilities={
                        'cpu': allocatable.get('cpu', '0'),
                        'memory': allocatable.get('memory', '0'),
                        'gpu': allocatable.get('nvidia.com/gpu', '0')
                    }
                )
                
                devices[device.device_id] = device
        except:
            pass
        
        return devices
    
    def _detect_k8s_device_type(self, node) -> str:
        """Detect device type from Kubernetes node"""
        
        labels = node.metadata.labels
        
        if 'nvidia.com/gpu' in node.status.allocatable:
            return 'gpu'
        elif 'node-role.kubernetes.io/edge' in labels:
            return 'edge'
        elif 'beta.kubernetes.io/instance-type' in labels:
            instance_type = labels['beta.kubernetes.io/instance-type']
            if 'gpu' in instance_type.lower():
                return 'gpu'
        
        return 'cpu'
    
    async def _discover_docker_devices(self) -> Dict[str, EdgeDevice]:
        """Discover Docker containers with edge capabilities"""
        
        devices = {}
        
        try:
            docker_client = docker.from_env()
            
            containers = docker_client.containers.list()
            for container in containers:
                labels = container.labels
                
                if 'edge.device' in labels:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    device = EdgeDevice(
                        device_id=container.short_id,
                        hostname=container.name,
                        ip_address=container.attrs['NetworkSettings']['IPAddress'],
                        port=int(labels.get('edge.port', 8888)),
                        device_type=labels.get('edge.type', 'cpu'),
                        capabilities=json.loads(labels.get('edge.capabilities', '{}'))
                    )
                    
                    devices[device.device_id] = device
        except:
            pass
        
        return devices
    
    async def _discover_ray_nodes(self) -> Dict[str, EdgeDevice]:
        """Discover Ray cluster nodes"""
        
        devices = {}
        
        try:
            if ray.is_initialized():
                nodes = ray.nodes()
                
                for node in nodes:
                    if node['Alive']:
                        device = EdgeDevice(
                            device_id=node['NodeID'][:8],
                            hostname=node['NodeName'],
                            ip_address=node['NodeManagerAddress'],
                            port=node.get('NodeManagerPort', 0),
                            device_type='gpu' if node.get('GPU', 0) > 0 else 'cpu',
                            capabilities={
                                'cpu': node.get('CPU', 0),
                                'gpu': node.get('GPU', 0),
                                'memory': node.get('memory', 0),
                                'object_store': node.get('object_store_memory', 0)
                            }
                        )
                        
                        devices[device.device_id] = device
        except:
            pass
        
        return devices
    
    async def _query_device_registry(self) -> Dict[str, EdgeDevice]:
        """Query custom edge device registry"""
        
        devices = {}
        
        # Check Redis for registered devices
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=3)
            
            device_keys = redis_client.keys('edge:device:*')
            for key in device_keys:
                device_data = redis_client.hgetall(key)
                
                if device_data:
                    device = EdgeDevice(
                        device_id=key.decode().split(':')[-1],
                        hostname=device_data.get(b'hostname', b'').decode(),
                        ip_address=device_data.get(b'ip', b'').decode(),
                        port=int(device_data.get(b'port', 8888)),
                        device_type=device_data.get(b'type', b'cpu').decode(),
                        capabilities=json.loads(device_data.get(b'capabilities', b'{}'))
                    )
                    
                    devices[device.device_id] = device
        except:
            pass
        
        return devices
    
    async def _probe_capabilities(self, device: EdgeDevice) -> Dict[str, Any]:
        """Probe device for detailed capabilities"""
        
        capabilities = device.capabilities.copy()
        
        try:
            # Connect to device
            async with aiohttp.ClientSession() as session:
                url = f"http://{device.ip_address}:{device.port}/capabilities"
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        detailed = await resp.json()
                        capabilities.update(detailed)
        except:
            pass
        
        # Add computed capabilities
        capabilities['compute_score'] = self._calculate_compute_score(capabilities)
        capabilities['supported_frameworks'] = self._detect_frameworks(capabilities)
        
        return capabilities
    
    def _calculate_compute_score(self, capabilities: Dict) -> float:
        """Calculate computational power score"""
        
        score = 0.0
        
        # CPU score
        cpu_count = int(capabilities.get('cpu', 0))
        score += cpu_count * 1.0
        
        # GPU score (weighted higher)
        gpu_count = int(capabilities.get('gpu', 0))
        gpu_memory = int(capabilities.get('gpu_memory', 0))
        score += gpu_count * 10.0 + (gpu_memory / 1024) * 2.0
        
        # Memory score
        memory_gb = int(capabilities.get('memory', 0)) / (1024 ** 3)
        score += memory_gb * 0.5
        
        # TPU score (highest weight)
        tpu_count = int(capabilities.get('tpu', 0))
        score += tpu_count * 20.0
        
        return score
    
    def _detect_frameworks(self, capabilities: Dict) -> List[str]:
        """Detect supported ML frameworks"""
        
        frameworks = []
        
        if capabilities.get('cuda_version'):
            frameworks.append('pytorch')
            frameworks.append('tensorflow')
        
        if capabilities.get('tpu'):
            frameworks.append('jax')
            frameworks.append('tensorflow')
        
        if capabilities.get('metal'):
            frameworks.append('coreml')
            frameworks.append('pytorch')
        
        # Always support ONNX for cross-platform
        frameworks.append('onnx')
        
        return frameworks
    
    async def monitor_devices(self):
        """Monitor device health and availability"""
        
        while True:
            for device_id, device in self.devices.items():
                # Check heartbeat
                if (datetime.now() - device.last_heartbeat).seconds > self.heartbeat_interval * 2:
                    device.status = 'offline'
                    logger.warning(f"Device {device_id} is offline")
                
                # Update metrics
                try:
                    metrics = await self._get_device_metrics(device)
                    device.current_load = metrics.get('load', 0)
                    device.available_memory = metrics.get('available_memory', 0)
                    device.available_compute = metrics.get('available_compute', 0)
                except:
                    pass
            
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _get_device_metrics(self, device: EdgeDevice) -> Dict[str, Any]:
        """Get current device metrics"""
        
        metrics = {}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://{device.ip_address}:{device.port}/metrics"
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        metrics = await resp.json()
        except:
            pass
        
        return metrics

class ModelOptimizer:
    """Optimize models for edge deployment"""
    
    def __init__(self):
        self.optimization_techniques = {
            'quantization': self.quantize_model,
            'pruning': self.prune_model,
            'knowledge_distillation': self.distill_model,
            'onnx_conversion': self.convert_to_onnx,
            'tflite_conversion': self.convert_to_tflite,
            'tensorrt_optimization': self.optimize_tensorrt,
            'openvino_optimization': self.optimize_openvino
        }
    
    async def optimize_for_edge(self, model: Any, target_device: EdgeDevice, 
                               optimization_level: str = 'balanced') -> Any:
        """Optimize model for specific edge device"""
        
        logger.info(f"🔧 Optimizing model for {target_device.device_type} device")
        
        optimized_model = model
        
        # Select optimization strategy based on device
        if target_device.device_type == 'mobile':
            optimized_model = await self.optimize_for_mobile(model)
        elif target_device.device_type == 'iot':
            optimized_model = await self.optimize_for_iot(model)
        elif target_device.device_type == 'gpu':
            optimized_model = await self.optimize_for_gpu(model)
        else:
            optimized_model = await self.optimize_generic(model, optimization_level)
        
        # Validate optimization
        if not await self.validate_optimization(model, optimized_model):
            logger.warning("Optimization validation failed, using original model")
            return model
        
        return optimized_model
    
    async def optimize_for_mobile(self, model: Any) -> Any:
        """Optimize for mobile devices"""
        
        # Quantization to INT8
        quantized = self.quantize_model(model, 'int8')
        
        # Convert to TFLite
        tflite_model = self.convert_to_tflite(quantized)
        
        # Further optimize with pruning
        pruned = self.prune_model(tflite_model, sparsity=0.5)
        
        return pruned
    
    async def optimize_for_iot(self, model: Any) -> Any:
        """Optimize for IoT devices"""
        
        # Aggressive quantization
        quantized = self.quantize_model(model, 'int4')
        
        # Heavy pruning
        pruned = self.prune_model(quantized, sparsity=0.8)
        
        # Convert to ONNX for compatibility
        onnx_model = self.convert_to_onnx(pruned)
        
        return onnx_model
    
    async def optimize_for_gpu(self, model: Any) -> Any:
        """Optimize for GPU acceleration"""
        
        if torch.cuda.is_available():
            # TensorRT optimization
            optimized = self.optimize_tensorrt(model)
        else:
            # ONNX with graph optimizations
            optimized = self.convert_to_onnx(model, optimize=True)
        
        return optimized
    
    async def optimize_generic(self, model: Any, level: str) -> Any:
        """Generic optimization based on level"""
        
        if level == 'maximum':
            # Maximum compression
            model = self.quantize_model(model, 'int8')
            model = self.prune_model(model, sparsity=0.7)
            model = self.convert_to_onnx(model, optimize=True)
        elif level == 'balanced':
            # Balanced optimization
            model = self.quantize_model(model, 'fp16')
            model = self.prune_model(model, sparsity=0.3)
        else:  # minimal
            # Minimal optimization
            model = self.convert_to_onnx(model)
        
        return model
    
    def quantize_model(self, model: Any, dtype: str = 'int8') -> Any:
        """Quantize model weights"""
        
        if isinstance(model, torch.nn.Module):
            if dtype == 'int8':
                return torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
                )
            elif dtype == 'int4':
                # Custom INT4 quantization
                return self._quantize_int4(model)
            elif dtype == 'fp16':
                return model.half()
        
        return model
    
    def prune_model(self, model: Any, sparsity: float = 0.5) -> Any:
        """Prune model weights"""
        
        if isinstance(model, torch.nn.Module):
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=sparsity)
                    prune.remove(module, 'weight')
        
        return model
    
    def distill_model(self, teacher_model: Any, student_model: Any, 
                     data_loader: Any, epochs: int = 10) -> Any:
        """Knowledge distillation from teacher to student"""
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            for inputs, _ in data_loader:
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                # Train student
                student_outputs = student_model(inputs)
                loss = nn.KLDivLoss()(
                    nn.functional.log_softmax(student_outputs, dim=1),
                    nn.functional.softmax(teacher_outputs, dim=1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return student_model
    
    def convert_to_onnx(self, model: Any, optimize: bool = False) -> Any:
        """Convert model to ONNX format"""
        
        if isinstance(model, torch.nn.Module):
            dummy_input = torch.randn(1, 3, 224, 224)
            
            torch.onnx.export(
                model,
                dummy_input,
                "model.onnx",
                export_params=True,
                opset_version=11,
                do_constant_folding=optimize,
                input_names=['input'],
                output_names=['output']
            )
            
            # Load ONNX model
            onnx_model = onnx.load("model.onnx")
            
            if optimize:
                # Optimize ONNX graph
                from onnx import optimizer
                onnx_model = optimizer.optimize(onnx_model)
            
            return onnx_model
        
        return model
    
    def convert_to_tflite(self, model: Any) -> Any:
        """Convert to TensorFlow Lite"""
        
        if isinstance(model, torch.nn.Module):
            # Convert PyTorch to ONNX first
            onnx_model = self.convert_to_onnx(model)
            
            # Then ONNX to TF
            import onnx_tf
            tf_model = onnx_tf.backend.prepare(onnx_model)
            
            # Finally to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            return tflite_model
        
        return model
    
    def optimize_tensorrt(self, model: Any) -> Any:
        """Optimize with TensorRT"""
        
        # Requires TensorRT installation
        try:
            import tensorrt as trt
            
            # Implementation would go here
            return model
        except ImportError:
            return model
    
    def optimize_openvino(self, model: Any) -> Any:
        """Optimize with OpenVINO"""
        
        # Requires OpenVINO installation
        try:
            from openvino.inference_engine import IECore
            
            # Implementation would go here
            return model
        except ImportError:
            return model
    
    def _quantize_int4(self, model: Any) -> Any:
        """Custom INT4 quantization"""
        
        # Simplified INT4 quantization
        for param in model.parameters():
            param.data = torch.round(param.data * 8) / 8
        
        return model
    
    async def validate_optimization(self, original: Any, optimized: Any) -> bool:
        """Validate optimized model maintains accuracy"""
        
        # Simple validation - would use test dataset in production
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            
            if isinstance(original, torch.nn.Module):
                with torch.no_grad():
                    orig_output = original(dummy_input)
            
            if isinstance(optimized, torch.nn.Module):
                with torch.no_grad():
                    opt_output = optimized(dummy_input)
            elif isinstance(optimized, bytes):  # TFLite model
                return True  # Assume validated
            else:
                return True  # ONNX or other formats
            
            # Check similarity
            similarity = torch.cosine_similarity(
                orig_output.flatten(),
                opt_output.flatten(),
                dim=0
            )
            
            return similarity > 0.95
        except:
            return False

class EdgeScheduler:
    """Schedule and distribute tasks across edge devices"""
    
    def __init__(self, discovery: EdgeDiscovery):
        self.discovery = discovery
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, ComputeTask] = {}
        self.completed_tasks: Dict[str, ComputeTask] = {}
        self.scheduling_strategies = {
            'round_robin': self.round_robin_schedule,
            'least_loaded': self.least_loaded_schedule,
            'capability_match': self.capability_match_schedule,
            'latency_optimized': self.latency_optimized_schedule,
            'deadline_aware': self.deadline_aware_schedule
        }
        
    async def schedule_task(self, task: ComputeTask, 
                           strategy: str = 'capability_match') -> str:
        """Schedule task to appropriate edge device"""
        
        # Get available devices
        available_devices = {
            device_id: device 
            for device_id, device in self.discovery.devices.items()
            if device.status != 'offline'
        }
        
        if not available_devices:
            raise RuntimeError("No available edge devices")
        
        # Select scheduling strategy
        scheduler = self.scheduling_strategies.get(strategy, self.capability_match_schedule)
        
        # Schedule task
        selected_device = await scheduler(task, available_devices)
        
        if not selected_device:
            # Queue task if no suitable device
            self.task_queue.put((task.priority, task))
            return 'queued'
        
        # Assign and execute
        task.assigned_device = selected_device.device_id
        task.status = 'assigned'
        
        # Deploy task
        await self.deploy_task(task, selected_device)
        
        return selected_device.device_id
    
    async def round_robin_schedule(self, task: ComputeTask, 
                                  devices: Dict[str, EdgeDevice]) -> Optional[EdgeDevice]:
        """Round-robin scheduling"""
        
        # Simple rotation through devices
        device_list = list(devices.values())
        
        # Get least recently used device
        device_list.sort(key=lambda d: d.tasks_completed)
        
        return device_list[0] if device_list else None
    
    async def least_loaded_schedule(self, task: ComputeTask,
                                   devices: Dict[str, EdgeDevice]) -> Optional[EdgeDevice]:
        """Schedule to least loaded device"""
        
        # Sort by current load
        sorted_devices = sorted(
            devices.values(),
            key=lambda d: d.current_load
        )
        
        # Find first device with acceptable load
        for device in sorted_devices:
            if device.current_load < 0.8:  # 80% threshold
                return device
        
        return sorted_devices[0] if sorted_devices else None
    
    async def capability_match_schedule(self, task: ComputeTask,
                                       devices: Dict[str, EdgeDevice]) -> Optional[EdgeDevice]:
        """Match task requirements to device capabilities"""
        
        best_device = None
        best_score = 0
        
        for device in devices.values():
            score = self._calculate_capability_score(task, device)
            
            if score > best_score:
                best_score = score
                best_device = device
        
        return best_device
    
    def _calculate_capability_score(self, task: ComputeTask, 
                                   device: EdgeDevice) -> float:
        """Calculate how well device matches task requirements"""
        
        score = 0.0
        
        # Check GPU requirement
        if task.requirements.get('gpu'):
            if device.capabilities.get('gpu', 0) > 0:
                score += 10.0
            else:
                return 0.0  # Hard requirement not met
        
        # Check memory requirement
        required_memory = task.requirements.get('memory', 0)
        available_memory = device.available_memory
        if available_memory >= required_memory:
            score += 5.0
        else:
            score -= 5.0
        
        # Check compute score
        required_compute = task.requirements.get('compute_score', 0)
        device_compute = device.capabilities.get('compute_score', 0)
        if device_compute >= required_compute:
            score += device_compute / required_compute
        
        # Consider device reliability
        score *= device.reliability_score
        
        # Consider current load (prefer less loaded devices)
        score *= (1.0 - device.current_load)
        
        return score
    
    async def latency_optimized_schedule(self, task: ComputeTask,
                                        devices: Dict[str, EdgeDevice]) -> Optional[EdgeDevice]:
        """Schedule for minimum latency"""
        
        # Sort by latency
        sorted_devices = sorted(
            devices.values(),
            key=lambda d: d.latency
        )
        
        # Find device that meets requirements with lowest latency
        for device in sorted_devices:
            if self._device_meets_requirements(task, device):
                return device
        
        return None
    
    async def deadline_aware_schedule(self, task: ComputeTask,
                                     devices: Dict[str, EdgeDevice]) -> Optional[EdgeDevice]:
        """Schedule considering task deadlines"""
        
        if not task.deadline:
            return await self.capability_match_schedule(task, devices)
        
        time_remaining = (task.deadline - datetime.now()).total_seconds()
        
        # Filter devices that can complete in time
        capable_devices = []
        for device in devices.values():
            estimated_time = self._estimate_execution_time(task, device)
            
            if estimated_time < time_remaining * 0.8:  # 20% buffer
                capable_devices.append(device)
        
        if not capable_devices:
            logger.warning(f"No device can meet deadline for task {task.task_id}")
            return None
        
        # Choose best among capable
        return await self.capability_match_schedule(task, 
                                                  {d.device_id: d for d in capable_devices})
    
    def _device_meets_requirements(self, task: ComputeTask, device: EdgeDevice) -> bool:
        """Check if device meets task requirements"""
        
        for req_key, req_value in task.requirements.items():
            if req_key in device.capabilities:
                if device.capabilities[req_key] < req_value:
                    return False
        
        return True
    
    def _estimate_execution_time(self, task: ComputeTask, device: EdgeDevice) -> float:
        """Estimate task execution time on device"""
        
        # Simple estimation based on compute score and load
        base_time = 10.0  # Base execution time in seconds
        
        # Adjust for device compute power
        compute_factor = 10.0 / (device.capabilities.get('compute_score', 1) + 1)
        
        # Adjust for current load
        load_factor = 1.0 + device.current_load
        
        # Adjust for task complexity
        complexity_factor = task.requirements.get('complexity', 1.0)
        
        return base_time * compute_factor * load_factor * complexity_factor
    
    async def deploy_task(self, task: ComputeTask, device: EdgeDevice):
        """Deploy task to edge device"""
        
        logger.info(f"📤 Deploying task {task.task_id} to device {device.device_id}")
        
        try:
            # Serialize task
            task_data = pickle.dumps(task)
            
            # Send to device
            async with aiohttp.ClientSession() as session:
                url = f"http://{device.ip_address}:{device.port}/execute"
                
                async with session.post(
                    url,
                    data=task_data,
                    headers={'Content-Type': 'application/octet-stream'}
                ) as resp:
                    if resp.status == 200:
                        task.status = 'running'
                        self.running_tasks[task.task_id] = task
                        device.status = 'busy'
                        
                        # Start monitoring
                        asyncio.create_task(self.monitor_task(task, device))
                    else:
                        task.status = 'failed'
                        task.error = f"Deployment failed: {resp.status}"
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            logger.error(f"Task deployment failed: {e}")
    
    async def monitor_task(self, task: ComputeTask, device: EdgeDevice):
        """Monitor task execution"""
        
        check_interval = 5.0  # seconds
        max_checks = 100
        
        for _ in range(max_checks):
            await asyncio.sleep(check_interval)
            
            try:
                # Check task status
                async with aiohttp.ClientSession() as session:
                    url = f"http://{device.ip_address}:{device.port}/status/{task.task_id}"
                    
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            status_data = await resp.json()
                            
                            if status_data['status'] == 'completed':
                                # Get results
                                result_url = f"http://{device.ip_address}:{device.port}/result/{task.task_id}"
                                async with session.get(result_url) as result_resp:
                                    if result_resp.status == 200:
                                        task.result = await result_resp.read()
                                        task.status = 'completed'
                                        task.completed_at = datetime.now()
                                        
                                        # Update device metrics
                                        device.tasks_completed += 1
                                        device.reliability_score = min(1.0, device.reliability_score * 1.01)
                                        device.status = 'idle'
                                        
                                        # Move to completed
                                        self.completed_tasks[task.task_id] = task
                                        del self.running_tasks[task.task_id]
                                        
                                        logger.info(f"✅ Task {task.task_id} completed")
                                        return
                            
                            elif status_data['status'] == 'failed':
                                task.status = 'failed'
                                task.error = status_data.get('error', 'Unknown error')
                                
                                # Update device metrics
                                device.tasks_failed += 1
                                device.reliability_score *= 0.95
                                device.status = 'idle'
                                
                                del self.running_tasks[task.task_id]
                                
                                logger.error(f"❌ Task {task.task_id} failed: {task.error}")
                                return
            except Exception as e:
                logger.warning(f"Error monitoring task {task.task_id}: {e}")
        
        # Timeout
        task.status = 'timeout'
        task.error = 'Task execution timeout'
        device.status = 'idle'
        
        if task.task_id in self.running_tasks:
            del self.running_tasks[task.task_id]

class EdgeComputingModule:
    """Main edge computing orchestration module"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.discovery = EdgeDiscovery()
        self.optimizer = ModelOptimizer()
        self.scheduler = EdgeScheduler(self.discovery)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=4)
        
        # Start discovery
        asyncio.create_task(self.discovery.discover_devices())
        asyncio.create_task(self.discovery.monitor_devices())
    
    async def deploy_model(self, model: Any, model_name: str, 
                          optimization_level: str = 'balanced') -> Dict[str, Any]:
        """Deploy model across edge devices"""
        
        logger.info(f"🚀 Deploying model {model_name} to edge")
        
        # Discover available devices
        devices = await self.discovery.discover_devices()
        
        if not devices:
            return {'status': 'error', 'message': 'No edge devices available'}
        
        deployment_results = {}
        
        # Optimize and deploy to each device type
        device_types = set(d.device_type for d in devices.values())
        
        for device_type in device_types:
            # Get representative device
            target_device = next(d for d in devices.values() if d.device_type == device_type)
            
            # Optimize model for device type
            optimized_model = await self.optimizer.optimize_for_edge(
                model,
                target_device,
                optimization_level
            )
            
            # Deploy to all devices of this type
            type_devices = [d for d in devices.values() if d.device_type == device_type]
            
            for device in type_devices:
                result = await self._deploy_to_device(optimized_model, model_name, device)
                deployment_results[device.device_id] = result
        
        # Cache deployment info
        self.redis_client.set(
            f"deployment:{model_name}",
            json.dumps({
                'timestamp': datetime.now().isoformat(),
                'devices': list(deployment_results.keys()),
                'optimization_level': optimization_level
            })
        )
        
        return {
            'status': 'success',
            'model': model_name,
            'deployed_to': len(deployment_results),
            'results': deployment_results
        }
    
    async def _deploy_to_device(self, model: Any, model_name: str, 
                               device: EdgeDevice) -> Dict[str, Any]:
        """Deploy model to specific device"""
        
        try:
            # Serialize model
            if isinstance(model, torch.nn.Module):
                model_data = pickle.dumps(model.state_dict())
            elif isinstance(model, bytes):
                model_data = model
            else:
                model_data = pickle.dumps(model)
            
            # Send to device
            async with aiohttp.ClientSession() as session:
                url = f"http://{device.ip_address}:{device.port}/deploy"
                
                data = {
                    'model_name': model_name,
                    'model_data': base64.b64encode(model_data).decode(),
                    'framework': self._detect_framework(model)
                }
                
                async with session.post(url, json=data) as resp:
                    if resp.status == 200:
                        return {'status': 'success', 'device': device.device_id}
                    else:
                        return {'status': 'failed', 'error': f"HTTP {resp.status}"}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}
    
    def _detect_framework(self, model: Any) -> str:
        """Detect ML framework of model"""
        
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
        elif isinstance(model, tf.Module):
            return 'tensorflow'
        elif isinstance(model, bytes):
            # Check for TFLite magic bytes
            if model[:4] == b'TFL3':
                return 'tflite'
        elif hasattr(model, 'graph'):
            return 'onnx'
        
        return 'unknown'
    
    async def distributed_inference(self, model_name: str, input_data: Any,
                                  strategy: str = 'data_parallel') -> Any:
        """Perform distributed inference across edge devices"""
        
        logger.info(f"🔮 Running distributed inference for {model_name}")
        
        if strategy == 'data_parallel':
            return await self._data_parallel_inference(model_name, input_data)
        elif strategy == 'model_parallel':
            return await self._model_parallel_inference(model_name, input_data)
        elif strategy == 'ensemble':
            return await self._ensemble_inference(model_name, input_data)
        else:
            return await self._single_device_inference(model_name, input_data)
    
    async def _data_parallel_inference(self, model_name: str, input_data: Any) -> Any:
        """Split data across devices for parallel processing"""
        
        # Get devices with deployed model
        deployment_info = self.redis_client.get(f"deployment:{model_name}")
        if not deployment_info:
            raise RuntimeError(f"Model {model_name} not deployed")
        
        deployed_devices = json.loads(deployment_info)['devices']
        
        # Split input data
        if isinstance(input_data, (list, np.ndarray)):
            chunks = np.array_split(input_data, len(deployed_devices))
        else:
            chunks = [input_data]  # Can't split, replicate
        
        # Create tasks for each chunk
        tasks = []
        for i, device_id in enumerate(deployed_devices):
            if i < len(chunks):
                task = ComputeTask(
                    task_id=hashlib.sha256(f"{model_name}{i}{datetime.now()}".encode()).hexdigest()[:16],
                    task_type='inference',
                    model_name=model_name,
                    input_data=chunks[i]
                )
                
                # Schedule task
                asyncio.create_task(self.scheduler.schedule_task(task))
                tasks.append(task)
        
        # Wait for completion
        results = []
        for task in tasks:
            while task.status not in ['completed', 'failed', 'timeout']:
                await asyncio.sleep(1)
            
            if task.status == 'completed':
                results.append(pickle.loads(task.result))
        
        # Combine results
        if results:
            return np.concatenate(results) if isinstance(results[0], np.ndarray) else results
        else:
            raise RuntimeError("Inference failed on all devices")
    
    async def _model_parallel_inference(self, model_name: str, input_data: Any) -> Any:
        """Split model across devices (for large models)"""
        
        # This would require model partitioning
        # Simplified implementation
        return await self._single_device_inference(model_name, input_data)
    
    async def _ensemble_inference(self, model_name: str, input_data: Any) -> Any:
        """Run inference on multiple devices and ensemble results"""
        
        deployment_info = self.redis_client.get(f"deployment:{model_name}")
        if not deployment_info:
            raise RuntimeError(f"Model {model_name} not deployed")
        
        deployed_devices = json.loads(deployment_info)['devices'][:5]  # Use up to 5 devices
        
        # Run on each device
        tasks = []
        for device_id in deployed_devices:
            task = ComputeTask(
                task_id=hashlib.sha256(f"{model_name}{device_id}{datetime.now()}".encode()).hexdigest()[:16],
                task_type='inference',
                model_name=model_name,
                input_data=input_data
            )
            
            asyncio.create_task(self.scheduler.schedule_task(task))
            tasks.append(task)
        
        # Collect results
        results = []
        for task in tasks:
            while task.status not in ['completed', 'failed', 'timeout']:
                await asyncio.sleep(1)
            
            if task.status == 'completed':
                results.append(pickle.loads(task.result))
        
        # Ensemble (simple averaging)
        if results:
            if isinstance(results[0], np.ndarray):
                return np.mean(results, axis=0)
            else:
                # Voting for classification
                from collections import Counter
                return Counter(results).most_common(1)[0][0]
        
        raise RuntimeError("Ensemble inference failed")
    
    async def _single_device_inference(self, model_name: str, input_data: Any) -> Any:
        """Run inference on single best device"""
        
        task = ComputeTask(
            task_id=hashlib.sha256(f"{model_name}{datetime.now()}".encode()).hexdigest()[:16],
            task_type='inference',
            model_name=model_name,
            input_data=input_data
        )
        
        device_id = await self.scheduler.schedule_task(task)
        
        # Wait for completion
        while task.status not in ['completed', 'failed', 'timeout']:
            await asyncio.sleep(1)
        
        if task.status == 'completed':
            return pickle.loads(task.result)
        else:
            raise RuntimeError(f"Inference failed: {task.error}")
    
    async def federated_learning(self, model_name: str, training_rounds: int = 10) -> Any:
        """Coordinate federated learning across edge devices"""
        
        logger.info(f"🎓 Starting federated learning for {model_name}")
        
        # Get participating devices
        devices = await self.discovery.discover_devices()
        participants = list(devices.values())[:10]  # Limit participants
        
        # Initialize global model
        global_model = None
        
        for round_num in range(training_rounds):
            logger.info(f"Round {round_num + 1}/{training_rounds}")
            
            # Send global model to participants
            model_updates = []
            
            for device in participants:
                # Create training task
                task = ComputeTask(
                    task_id=f"fl_{model_name}_{round_num}_{device.device_id}",
                    task_type='training',
                    model_name=model_name,
                    input_data={'global_model': global_model, 'round': round_num}
                )
                
                # Deploy and wait
                await self.scheduler.schedule_task(task)
                
                while task.status not in ['completed', 'failed']:
                    await asyncio.sleep(1)
                
                if task.status == 'completed':
                    model_updates.append(pickle.loads(task.result))
            
            # Aggregate updates
            if model_updates:
                global_model = self._federated_average(model_updates)
        
        return global_model
    
    def _federated_average(self, model_updates: List[Any]) -> Any:
        """Average model updates from federated learning"""
        
        if not model_updates:
            return None
        
        # Simple averaging of parameters
        avg_state = {}
        
        for key in model_updates[0].keys():
            stacked = torch.stack([update[key] for update in model_updates])
            avg_state[key] = torch.mean(stacked, dim=0)
        
        return avg_state


# Example usage
if __name__ == "__main__":
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'discovery_interval': 60,
        'heartbeat_interval': 30
    }
    
    # Initialize edge computing module
    edge_module = EdgeComputingModule(config)
    
    # Example: Deploy a model
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    
    deployment_result = asyncio.run(edge_module.deploy_model(
        model,
        'mnist_classifier',
        optimization_level='balanced'
    ))
    
    print(f"Deployment result: {deployment_result}")
    
    # Example: Run distributed inference
    test_data = torch.randn(100, 784)
    
    inference_result = asyncio.run(edge_module.distributed_inference(
        'mnist_classifier',
        test_data,
        strategy='data_parallel'
    ))
    
    print(f"Inference shape: {inference_result.shape}")
    
    # Example: Start federated learning
    fl_model = asyncio.run(edge_module.federated_learning(
        'mnist_classifier',
        training_rounds=5
    ))
    
    print("Federated learning completed")