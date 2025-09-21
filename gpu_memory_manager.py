#!/usr/bin/env python3
"""
GPU Memory Manager for RTX 4090 Advanced AI Companion
Handles dynamic allocation, monitoring, and optimization of VRAM resources
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time
import psutil
import pynvml

# Initialize NVIDIA Management Library
pynvml.nvmlInit()

logger = logging.getLogger(__name__)


class Priority(Enum):
    """Resource allocation priority levels"""
    CRITICAL = 1      # Real-time features (avatar, emotion)
    INTERACTIVE = 2   # User-facing features (voice, desktop)
    BACKGROUND = 3    # Analysis and learning
    OPTIONAL = 4      # Enhancements and effects


class MemoryTier(Enum):
    """Memory tier classification"""
    CORE = "core"           # Always in VRAM
    DYNAMIC = "dynamic"     # Hot-swappable
    SWAP = "swap"          # Can be moved to CPU
    OFFLOAD = "offload"    # CPU-only processing


@dataclass
class ModelResource:
    """Represents a model or resource requiring GPU memory"""
    name: str
    size_mb: int
    priority: Priority
    tier: MemoryTier
    loaded: bool = False
    last_accessed: float = 0
    access_count: int = 0
    can_quantize: bool = True
    quantized_size_mb: Optional[int] = None


@dataclass
class GPUMetrics:
    """Current GPU state metrics"""
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    temperature_c: float
    power_watts: float
    utilization_percent: float
    memory_bandwidth_percent: float


class GPUMemoryManager:
    """
    Manages GPU memory allocation for AI companion features
    Optimized for RTX 4090 with 24GB VRAM
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

        # Memory configuration (in MB)
        self.total_vram = 24576    # 24GB
        self.core_reserved = 6144   # 6GB for critical features
        self.dynamic_pool = 8192    # 8GB for hot-swap
        self.swap_pool = 6144      # 6GB for cold storage
        self.osint_reserved = 4096  # 4GB for OSINT operations

        # Resource tracking
        self.resources: Dict[str, ModelResource] = {}
        self.loaded_resources: List[str] = []
        self.memory_map: Dict[MemoryTier, List[str]] = {
            MemoryTier.CORE: [],
            MemoryTier.DYNAMIC: [],
            MemoryTier.SWAP: [],
            MemoryTier.OFFLOAD: []
        }

        # Performance thresholds
        self.temp_throttle = 83
        self.temp_critical = 87
        self.vram_warning = 22000
        self.vram_critical = 23000

        # Degradation level
        self.degradation_level = 0

        # Initialize default resources
        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize default AI companion resources"""
        default_resources = [
            # Core features (Tier 1)
            ModelResource("avatar_renderer", 2048, Priority.CRITICAL, MemoryTier.CORE, can_quantize=False),
            ModelResource("emotion_processor", 1536, Priority.CRITICAL, MemoryTier.CORE, quantized_size_mb=384),
            ModelResource("voice_synthesizer", 1024, Priority.CRITICAL, MemoryTier.CORE, quantized_size_mb=512),
            ModelResource("ui_components", 512, Priority.CRITICAL, MemoryTier.CORE, can_quantize=False),

            # Dynamic features (Tier 2)
            ModelResource("personality_engine", 1024, Priority.INTERACTIVE, MemoryTier.DYNAMIC, quantized_size_mb=256),
            ModelResource("context_analyzer", 768, Priority.INTERACTIVE, MemoryTier.DYNAMIC, quantized_size_mb=192),
            ModelResource("memory_system", 1024, Priority.INTERACTIVE, MemoryTier.DYNAMIC),
            ModelResource("relationship_graph", 512, Priority.INTERACTIVE, MemoryTier.DYNAMIC),

            # Swappable features (Tier 3)
            ModelResource("creative_generator", 768, Priority.BACKGROUND, MemoryTier.SWAP),
            ModelResource("visual_enhancer", 1024, Priority.BACKGROUND, MemoryTier.SWAP),
            ModelResource("audio_processor", 512, Priority.BACKGROUND, MemoryTier.SWAP),
            ModelResource("learning_module", 768, Priority.BACKGROUND, MemoryTier.SWAP),

            # Optional features (Tier 4)
            ModelResource("biometric_analyzer", 512, Priority.OPTIONAL, MemoryTier.OFFLOAD),
            ModelResource("physiological_monitor", 384, Priority.OPTIONAL, MemoryTier.OFFLOAD),
            ModelResource("enhancement_effects", 256, Priority.OPTIONAL, MemoryTier.OFFLOAD),
        ]

        for resource in default_resources:
            self.register_resource(resource)

    def register_resource(self, resource: ModelResource):
        """Register a new resource with the memory manager"""
        self.resources[resource.name] = resource
        self.memory_map[resource.tier].append(resource.name)
        logger.info(f"Registered resource: {resource.name} ({resource.size_mb}MB, {resource.tier.value})")

    def get_gpu_metrics(self) -> GPUMetrics:
        """Get current GPU metrics"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000  # Convert to watts
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

        return GPUMetrics(
            total_vram_mb=mem_info.total // 1048576,
            used_vram_mb=mem_info.used // 1048576,
            free_vram_mb=mem_info.free // 1048576,
            temperature_c=temperature,
            power_watts=power,
            utilization_percent=utilization.gpu,
            memory_bandwidth_percent=utilization.memory
        )

    async def allocate_resource(self, resource_name: str, force: bool = False) -> bool:
        """
        Allocate GPU memory for a resource
        Returns True if successful, False otherwise
        """
        if resource_name not in self.resources:
            logger.error(f"Unknown resource: {resource_name}")
            return False

        resource = self.resources[resource_name]

        if resource.loaded and not force:
            resource.last_accessed = time.time()
            resource.access_count += 1
            return True

        # Check available memory
        metrics = self.get_gpu_metrics()
        required_mb = resource.quantized_size_mb if resource.can_quantize and self.degradation_level > 0 else resource.size_mb

        if metrics.free_vram_mb < required_mb:
            # Try to free memory
            if not await self._free_memory(required_mb):
                logger.warning(f"Insufficient memory for {resource_name} ({required_mb}MB needed)")
                return False

        # Allocate the resource
        resource.loaded = True
        resource.last_accessed = time.time()
        resource.access_count += 1
        self.loaded_resources.append(resource_name)

        logger.info(f"Allocated {resource_name} ({required_mb}MB)")
        return True

    async def _free_memory(self, required_mb: int) -> bool:
        """
        Free memory by evicting lowest priority resources
        Returns True if enough memory was freed
        """
        metrics = self.get_gpu_metrics()
        freed_mb = 0

        # Sort loaded resources by priority and last access time
        eviction_candidates = []
        for name in self.loaded_resources:
            resource = self.resources[name]
            if resource.tier != MemoryTier.CORE:  # Never evict core resources
                eviction_candidates.append((
                    resource.priority.value,
                    -resource.last_accessed,
                    name
                ))

        eviction_candidates.sort()

        # Evict resources until we have enough memory
        for _, _, name in eviction_candidates:
            if freed_mb >= required_mb:
                break

            resource = self.resources[name]
            if await self.deallocate_resource(name):
                freed_mb += resource.size_mb

        return freed_mb >= required_mb

    async def deallocate_resource(self, resource_name: str) -> bool:
        """Deallocate a resource from GPU memory"""
        if resource_name not in self.resources:
            return False

        resource = self.resources[resource_name]
        if not resource.loaded:
            return True

        resource.loaded = False
        if resource_name in self.loaded_resources:
            self.loaded_resources.remove(resource_name)

        logger.info(f"Deallocated {resource_name}")
        return True

    def determine_degradation_level(self) -> int:
        """
        Determine current degradation level based on resource usage
        0: No degradation
        1: Minor optimization (18-20GB)
        2: Moderate reduction (20-22GB)
        3: Significant scaling (22-23GB)
        4: Emergency mode (23-24GB)
        """
        metrics = self.get_gpu_metrics()
        used_gb = metrics.used_vram_mb / 1024

        if used_gb < 18:
            return 0
        elif used_gb < 20:
            return 1
        elif used_gb < 22:
            return 2
        elif used_gb < 23:
            return 3
        else:
            return 4

    async def apply_degradation(self, level: int):
        """Apply degradation strategies based on level"""
        self.degradation_level = level

        if level == 0:
            logger.info("Operating at full capacity")
            return

        logger.warning(f"Applying degradation level {level}")

        degradation_actions = {
            1: self._apply_minor_optimization,
            2: self._apply_moderate_reduction,
            3: self._apply_significant_scaling,
            4: self._apply_emergency_mode
        }

        if level in degradation_actions:
            await degradation_actions[level]()

    async def _apply_minor_optimization(self):
        """Level 1: Minor optimizations"""
        # Reduce texture resolution, lower voice quality
        logger.info("Applying minor optimizations")
        # Implementation would interact with rendering and audio systems

    async def _apply_moderate_reduction(self):
        """Level 2: Moderate feature reduction"""
        # Simplify models, disable effects
        logger.info("Applying moderate reductions")
        # Deallocate optional features
        for name in self.memory_map[MemoryTier.OFFLOAD]:
            await self.deallocate_resource(name)

    async def _apply_significant_scaling(self):
        """Level 3: Significant scaling back"""
        # Reduce FPS, use cached responses
        logger.info("Applying significant scaling")
        # Deallocate swap tier features
        for name in self.memory_map[MemoryTier.SWAP]:
            await self.deallocate_resource(name)

    async def _apply_emergency_mode(self):
        """Level 4: Emergency mode - core features only"""
        logger.critical("Entering emergency mode")
        # Keep only core features
        for tier in [MemoryTier.OFFLOAD, MemoryTier.SWAP, MemoryTier.DYNAMIC]:
            for name in self.memory_map[tier]:
                await self.deallocate_resource(name)

    async def optimize_for_thermal(self) -> bool:
        """Optimize based on thermal conditions"""
        metrics = self.get_gpu_metrics()

        if metrics.temperature_c > self.temp_critical:
            logger.critical(f"Critical temperature: {metrics.temperature_c}°C")
            await self.apply_degradation(4)
            return False
        elif metrics.temperature_c > self.temp_throttle:
            logger.warning(f"Throttle temperature: {metrics.temperature_c}°C")
            current_level = self.determine_degradation_level()
            if current_level < 3:
                await self.apply_degradation(3)
            return False

        return True

    async def monitor_and_optimize(self):
        """Continuous monitoring and optimization loop"""
        while True:
            try:
                # Get current metrics
                metrics = self.get_gpu_metrics()

                # Log current state
                logger.debug(f"GPU State - VRAM: {metrics.used_vram_mb}/{metrics.total_vram_mb}MB, "
                           f"Temp: {metrics.temperature_c}°C, Power: {metrics.power_watts}W")

                # Check thermal conditions
                await self.optimize_for_thermal()

                # Determine and apply degradation if needed
                new_level = self.determine_degradation_level()
                if new_level != self.degradation_level:
                    await self.apply_degradation(new_level)

                # Memory optimization
                if metrics.used_vram_mb > self.vram_warning:
                    logger.warning(f"High VRAM usage: {metrics.used_vram_mb}MB")
                    await self._free_memory(1024)  # Try to free 1GB

                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(10)

    def get_resource_status(self) -> Dict:
        """Get current resource allocation status"""
        status = {
            'metrics': self.get_gpu_metrics().__dict__,
            'degradation_level': self.degradation_level,
            'loaded_resources': [],
            'available_resources': []
        }

        for name, resource in self.resources.items():
            resource_info = {
                'name': name,
                'size_mb': resource.size_mb,
                'priority': resource.priority.name,
                'tier': resource.tier.value,
                'loaded': resource.loaded,
                'access_count': resource.access_count
            }

            if resource.loaded:
                status['loaded_resources'].append(resource_info)
            else:
                status['available_resources'].append(resource_info)

        return status

    def cleanup(self):
        """Cleanup NVML resources"""
        pynvml.nvmlShutdown()


class CUDAStreamManager:
    """
    Manages CUDA streams for parallel processing
    Optimizes concurrent execution of AI companion features
    """

    def __init__(self, num_streams: int = 16):
        self.num_streams = num_streams
        self.stream_allocation = {
            'avatar_rendering': list(range(0, 4)),      # Streams 0-3
            'emotion_processing': list(range(4, 8)),    # Streams 4-7
            'voice_audio': list(range(8, 12)),          # Streams 8-11
            'context_intelligence': list(range(12, 16))  # Streams 12-15
        }

        self.stream_status = {i: 'idle' for i in range(num_streams)}
        self.stream_metrics = {i: {'tasks': 0, 'total_time': 0} for i in range(num_streams)}

    def allocate_stream(self, task_type: str) -> Optional[int]:
        """Allocate a CUDA stream for a specific task type"""
        if task_type not in self.stream_allocation:
            logger.error(f"Unknown task type: {task_type}")
            return None

        available_streams = self.stream_allocation[task_type]
        for stream_id in available_streams:
            if self.stream_status[stream_id] == 'idle':
                self.stream_status[stream_id] = 'active'
                return stream_id

        logger.warning(f"No available streams for {task_type}")
        return None

    def release_stream(self, stream_id: int, execution_time: float = 0):
        """Release a CUDA stream after task completion"""
        if stream_id < 0 or stream_id >= self.num_streams:
            return

        self.stream_status[stream_id] = 'idle'
        self.stream_metrics[stream_id]['tasks'] += 1
        self.stream_metrics[stream_id]['total_time'] += execution_time

    def get_stream_utilization(self) -> Dict:
        """Get utilization statistics for all streams"""
        utilization = {}
        for task_type, streams in self.stream_allocation.items():
            active_count = sum(1 for s in streams if self.stream_status[s] == 'active')
            utilization[task_type] = {
                'total_streams': len(streams),
                'active_streams': active_count,
                'utilization_percent': (active_count / len(streams)) * 100
            }
        return utilization


async def main():
    """Example usage and testing"""
    # Initialize GPU memory manager
    manager = GPUMemoryManager()

    # Start monitoring in background
    monitor_task = asyncio.create_task(manager.monitor_and_optimize())

    try:
        # Allocate core resources
        await manager.allocate_resource("avatar_renderer")
        await manager.allocate_resource("emotion_processor")
        await manager.allocate_resource("voice_synthesizer")

        # Try to allocate dynamic resources
        await manager.allocate_resource("personality_engine")
        await manager.allocate_resource("context_analyzer")

        # Get status
        status = manager.get_resource_status()
        print(f"Current status: {status}")

        # Simulate running for a while
        await asyncio.sleep(30)

    finally:
        # Cleanup
        monitor_task.cancel()
        manager.cleanup()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())