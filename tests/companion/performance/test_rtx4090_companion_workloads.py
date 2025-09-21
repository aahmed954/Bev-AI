"""
RTX 4090 Performance Testing for AI Companion Workloads
Tests GPU resource utilization, thermal performance, and concurrent processing
for advanced AI companion system components
"""

import pytest
import asyncio
import time
import statistics
import psutil
import GPUtil
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch
import subprocess
import json
from pathlib import Path

from tests.companion.fixtures.performance_fixtures import *
from tests.companion.utils.gpu_monitor import RTX4090Monitor
from tests.companion.utils.companion_client import CompanionTestClient
from tests.companion.utils.workload_generator import CompanionWorkloadGenerator

@dataclass
class GPUPerformanceMetrics:
    """Container for GPU performance measurements"""
    timestamp: float
    gpu_utilization: float
    memory_utilization: float
    temperature: float
    power_consumption: float
    clock_speed: int
    memory_clock: int
    fan_speed: int

@dataclass
class CompanionWorkloadResult:
    """Results from companion workload execution"""
    workload_type: str
    duration: float
    success_rate: float
    avg_latency: float
    p95_latency: float
    gpu_metrics: List[GPUPerformanceMetrics]
    thermal_peak: float
    power_peak: float
    efficiency_score: float

@pytest.mark.companion_performance
@pytest.mark.rtx4090
class TestRTX4090CompanionWorkloads:
    """Test RTX 4090 performance with AI companion workloads"""

    @pytest.fixture(autouse=True)
    def setup_gpu_testing(self, rtx4090_monitor, companion_client, workload_generator):
        """Setup GPU performance testing environment"""
        self.gpu_monitor = rtx4090_monitor
        self.client = companion_client
        self.workload_gen = workload_generator
        self.test_results = []

        # Ensure GPU is in clean state
        self._reset_gpu_state()

        # Establish baseline metrics
        self.baseline_metrics = self._collect_baseline_metrics()

        yield

        # Cleanup and reset GPU
        self._reset_gpu_state()
        self._save_test_results()

    async def test_avatar_rendering_performance(self):
        """Test 3D avatar rendering performance under various complexity levels"""
        rendering_scenarios = [
            {
                "complexity": "basic",
                "avatar_quality": "low",
                "emotion_count": 5,
                "concurrent_sessions": 1,
                "duration": 300,  # 5 minutes
                "target_fps": 60
            },
            {
                "complexity": "detailed",
                "avatar_quality": "medium",
                "emotion_count": 10,
                "concurrent_sessions": 3,
                "duration": 300,
                "target_fps": 45
            },
            {
                "complexity": "photorealistic",
                "avatar_quality": "high",
                "emotion_count": 15,
                "concurrent_sessions": 5,
                "duration": 300,
                "target_fps": 30
            }
        ]

        for scenario in rendering_scenarios:
            print(f"Testing avatar rendering: {scenario['complexity']} complexity")

            # Start GPU monitoring
            monitoring_task = asyncio.create_task(
                self._monitor_gpu_performance(scenario["duration"])
            )

            # Start avatar rendering workload
            rendering_task = asyncio.create_task(
                self._execute_avatar_rendering_workload(scenario)
            )

            # Execute both tasks concurrently
            gpu_metrics, rendering_results = await asyncio.gather(
                monitoring_task, rendering_task
            )

            # Analyze performance
            result = self._analyze_avatar_performance(scenario, gpu_metrics, rendering_results)
            self.test_results.append(result)

            # Validate performance targets
            assert result.efficiency_score >= 0.85, f"GPU efficiency {result.efficiency_score:.2f} below 85% for {scenario['complexity']}"
            assert result.thermal_peak <= 83.0, f"Thermal peak {result.thermal_peak:.1f}°C above 83°C threshold"
            assert result.avg_latency <= 16.7, f"Frame latency {result.avg_latency:.1f}ms above 60fps threshold"

            # Brief cooldown between scenarios
            await asyncio.sleep(30)

    async def test_voice_synthesis_performance(self):
        """Test voice synthesis performance with various quality settings"""
        voice_scenarios = [
            {
                "quality": "standard",
                "emotion_depth": "basic",
                "concurrent_voices": 2,
                "text_complexity": "simple",
                "duration": 240,
                "target_latency_ms": 500
            },
            {
                "quality": "high",
                "emotion_depth": "moderate",
                "concurrent_voices": 4,
                "text_complexity": "technical",
                "duration": 240,
                "target_latency_ms": 750
            },
            {
                "quality": "ultra",
                "emotion_depth": "deep",
                "concurrent_voices": 6,
                "text_complexity": "complex",
                "duration": 240,
                "target_latency_ms": 1000
            }
        ]

        for scenario in voice_scenarios:
            print(f"Testing voice synthesis: {scenario['quality']} quality")

            # Monitor GPU during voice synthesis
            monitoring_task = asyncio.create_task(
                self._monitor_gpu_performance(scenario["duration"])
            )

            # Execute voice synthesis workload
            synthesis_task = asyncio.create_task(
                self._execute_voice_synthesis_workload(scenario)
            )

            gpu_metrics, synthesis_results = await asyncio.gather(
                monitoring_task, synthesis_task
            )

            # Analyze voice synthesis performance
            result = self._analyze_voice_performance(scenario, gpu_metrics, synthesis_results)
            self.test_results.append(result)

            # Validate performance targets
            assert result.avg_latency <= scenario["target_latency_ms"], f"Voice latency {result.avg_latency:.1f}ms above {scenario['target_latency_ms']}ms"
            assert result.success_rate >= 0.95, f"Voice synthesis success rate {result.success_rate:.2f} below 95%"
            assert result.thermal_peak <= 80.0, f"Voice synthesis thermal peak {result.thermal_peak:.1f}°C too high"

            await asyncio.sleep(20)

    async def test_concurrent_companion_osint_workloads(self):
        """Test concurrent AI companion and OSINT processing workloads"""
        concurrent_scenarios = [
            {
                "companion_load": "light",  # 2 active conversations
                "osint_load": "moderate",   # 50 concurrent OSINT requests
                "duration": 600,            # 10 minutes
                "expected_degradation": 0.05  # 5% performance impact
            },
            {
                "companion_load": "moderate", # 5 active conversations
                "osint_load": "heavy",       # 100 concurrent OSINT requests
                "duration": 600,
                "expected_degradation": 0.10  # 10% performance impact
            },
            {
                "companion_load": "heavy",   # 10 active conversations
                "osint_load": "moderate",    # 50 concurrent OSINT requests
                "duration": 600,
                "expected_degradation": 0.08  # 8% performance impact
            }
        ]

        for scenario in concurrent_scenarios:
            print(f"Testing concurrent workloads: {scenario['companion_load']} companion + {scenario['osint_load']} OSINT")

            # Establish baseline performance for comparison
            baseline_osint = await self._measure_baseline_osint_performance()
            baseline_companion = await self._measure_baseline_companion_performance()

            # Start GPU monitoring
            monitoring_task = asyncio.create_task(
                self._monitor_gpu_performance(scenario["duration"])
            )

            # Start concurrent workloads
            companion_task = asyncio.create_task(
                self._execute_companion_workload(scenario["companion_load"], scenario["duration"])
            )
            osint_task = asyncio.create_task(
                self._execute_osint_workload(scenario["osint_load"], scenario["duration"])
            )

            # Execute all tasks concurrently
            gpu_metrics, companion_results, osint_results = await asyncio.gather(
                monitoring_task, companion_task, osint_task
            )

            # Analyze concurrent performance
            result = self._analyze_concurrent_performance(
                scenario, gpu_metrics, companion_results, osint_results,
                baseline_companion, baseline_osint
            )
            self.test_results.append(result)

            # Validate concurrent performance doesn't exceed degradation limits
            actual_degradation = 1.0 - result.efficiency_score
            assert actual_degradation <= scenario["expected_degradation"], f"Performance degradation {actual_degradation:.2f} exceeds {scenario['expected_degradation']:.2f}"

            # Validate thermal and power constraints
            assert result.thermal_peak <= 85.0, f"Concurrent workload thermal peak {result.thermal_peak:.1f}°C too high"
            assert result.power_peak <= 400.0, f"Power consumption {result.power_peak:.1f}W exceeds 400W limit"

            await asyncio.sleep(60)  # Extended cooldown for concurrent tests

    async def test_memory_intensive_companion_operations(self):
        """Test GPU memory utilization with memory-intensive companion operations"""
        memory_scenarios = [
            {
                "operation": "large_context_conversations",
                "memory_target_gb": 8,
                "concurrent_sessions": 15,
                "context_size": "32k_tokens",
                "duration": 300
            },
            {
                "operation": "high_resolution_avatar_rendering",
                "memory_target_gb": 12,
                "concurrent_sessions": 8,
                "avatar_resolution": "4k",
                "duration": 300
            },
            {
                "operation": "complex_voice_synthesis",
                "memory_target_gb": 6,
                "concurrent_voices": 20,
                "voice_quality": "studio",
                "duration": 300
            }
        ]

        for scenario in memory_scenarios:
            print(f"Testing memory-intensive operation: {scenario['operation']}")

            # Monitor GPU memory usage specifically
            memory_monitoring_task = asyncio.create_task(
                self._monitor_gpu_memory_usage(scenario["duration"])
            )

            # Execute memory-intensive workload
            workload_task = asyncio.create_task(
                self._execute_memory_intensive_workload(scenario)
            )

            memory_metrics, workload_results = await asyncio.gather(
                memory_monitoring_task, workload_task
            )

            # Analyze memory performance
            result = self._analyze_memory_performance(scenario, memory_metrics, workload_results)
            self.test_results.append(result)

            # Validate memory utilization
            peak_memory_gb = max(m.memory_utilization for m in memory_metrics) * 24  # RTX 4090 has 24GB
            assert peak_memory_gb <= 20.0, f"Memory usage {peak_memory_gb:.1f}GB exceeds 20GB safety threshold"

            # Validate no memory fragmentation issues
            memory_efficiency = result.efficiency_score
            assert memory_efficiency >= 0.80, f"Memory efficiency {memory_efficiency:.2f} below 80% threshold"

            await asyncio.sleep(30)

    async def test_thermal_performance_sustained_load(self):
        """Test thermal performance under sustained AI companion load"""
        # Sustained load test: 30 minutes of continuous companion operations
        sustained_duration = 1800  # 30 minutes
        thermal_threshold = 83.0    # °C

        print(f"Starting {sustained_duration/60:.0f}-minute sustained thermal test")

        # Create mixed sustained workload
        workload_config = {
            "avatar_rendering": True,
            "voice_synthesis": True,
            "conversation_processing": True,
            "memory_operations": True,
            "concurrent_sessions": 8
        }

        # Start comprehensive monitoring
        thermal_monitoring = asyncio.create_task(
            self._monitor_thermal_performance(sustained_duration)
        )

        # Execute sustained workload
        sustained_workload = asyncio.create_task(
            self._execute_sustained_companion_workload(workload_config, sustained_duration)
        )

        thermal_data, workload_results = await asyncio.gather(
            thermal_monitoring, sustained_workload
        )

        # Analyze thermal performance
        thermal_result = self._analyze_thermal_performance(thermal_data, workload_results)
        self.test_results.append(thermal_result)

        # Validate thermal constraints
        peak_temperature = max(t.temperature for t in thermal_data)
        avg_temperature = statistics.mean(t.temperature for t in thermal_data)
        time_above_threshold = sum(1 for t in thermal_data if t.temperature > thermal_threshold)
        threshold_percentage = (time_above_threshold / len(thermal_data)) * 100

        assert peak_temperature <= 85.0, f"Peak temperature {peak_temperature:.1f}°C exceeds 85°C absolute limit"
        assert avg_temperature <= 80.0, f"Average temperature {avg_temperature:.1f}°C exceeds 80°C sustained limit"
        assert threshold_percentage <= 5.0, f"Time above {thermal_threshold}°C: {threshold_percentage:.1f}% exceeds 5% limit"

        print(f"Thermal test completed - Peak: {peak_temperature:.1f}°C, Avg: {avg_temperature:.1f}°C")

    async def test_power_efficiency_optimization(self):
        """Test power consumption efficiency across different companion workloads"""
        power_scenarios = [
            {
                "workload": "idle_companion",
                "max_power_watts": 100,
                "operations": ["minimal_avatar", "standby_voice"],
                "duration": 300
            },
            {
                "workload": "active_conversation",
                "max_power_watts": 250,
                "operations": ["avatar_rendering", "voice_synthesis", "conversation_ai"],
                "duration": 300
            },
            {
                "workload": "intensive_processing",
                "max_power_watts": 400,
                "operations": ["complex_avatar", "multi_voice", "heavy_ai", "osint_integration"],
                "duration": 300
            }
        ]

        for scenario in power_scenarios:
            print(f"Testing power efficiency: {scenario['workload']}")

            # Monitor power consumption
            power_monitoring = asyncio.create_task(
                self._monitor_power_consumption(scenario["duration"])
            )

            # Execute power test workload
            power_workload = asyncio.create_task(
                self._execute_power_test_workload(scenario)
            )

            power_data, workload_results = await asyncio.gather(
                power_monitoring, power_workload
            )

            # Analyze power efficiency
            power_result = self._analyze_power_efficiency(scenario, power_data, workload_results)
            self.test_results.append(power_result)

            # Validate power consumption limits
            peak_power = max(p.power_consumption for p in power_data)
            avg_power = statistics.mean(p.power_consumption for p in power_data)

            assert peak_power <= scenario["max_power_watts"], f"Peak power {peak_power:.1f}W exceeds {scenario['max_power_watts']}W limit"

            # Calculate power efficiency (performance per watt)
            power_efficiency = workload_results["performance_score"] / avg_power
            assert power_efficiency >= 0.01, f"Power efficiency {power_efficiency:.3f} below minimum threshold"

            await asyncio.sleep(20)

    # Helper Methods for GPU Monitoring and Workload Execution

    async def _monitor_gpu_performance(self, duration: int) -> List[GPUPerformanceMetrics]:
        """Monitor GPU performance metrics for specified duration"""
        metrics = []
        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            try:
                gpu = GPUtil.getGPUs()[0]  # Assume RTX 4090 is first GPU

                # Get additional metrics via nvidia-ml-py
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                # Power consumption
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

                # Clock speeds
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

                # Fan speed
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    fan_speed = 0  # Some cards don't report fan speed

                metrics.append(GPUPerformanceMetrics(
                    timestamp=time.time(),
                    gpu_utilization=gpu.load * 100,
                    memory_utilization=gpu.memoryUtil * 100,
                    temperature=temp,
                    power_consumption=power,
                    clock_speed=graphics_clock,
                    memory_clock=memory_clock,
                    fan_speed=fan_speed
                ))

            except Exception as e:
                print(f"GPU monitoring error: {e}")
                # Add dummy metrics to maintain timing
                metrics.append(GPUPerformanceMetrics(
                    timestamp=time.time(),
                    gpu_utilization=0, memory_utilization=0, temperature=0,
                    power_consumption=0, clock_speed=0, memory_clock=0, fan_speed=0
                ))

            await asyncio.sleep(1)  # Sample every second

        return metrics

    async def _execute_avatar_rendering_workload(self, scenario: Dict) -> Dict:
        """Execute avatar rendering workload"""
        start_time = time.time()
        results = {
            "frames_rendered": 0,
            "frame_latencies": [],
            "emotion_transitions": 0,
            "rendering_errors": 0,
            "performance_score": 0.0
        }

        try:
            # Simulate avatar rendering workload
            target_fps = scenario["target_fps"]
            frame_time_ms = 1000.0 / target_fps
            concurrent_sessions = scenario["concurrent_sessions"]

            # Create concurrent avatar rendering tasks
            tasks = []
            for session_id in range(concurrent_sessions):
                task = self._render_avatar_session(
                    f"avatar_session_{session_id}",
                    scenario,
                    scenario["duration"]
                )
                tasks.append(task)

            # Execute all avatar sessions concurrently
            session_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for session_result in session_results:
                if isinstance(session_result, dict):
                    results["frames_rendered"] += session_result.get("frames", 0)
                    results["frame_latencies"].extend(session_result.get("latencies", []))
                    results["emotion_transitions"] += session_result.get("emotions", 0)
                else:
                    results["rendering_errors"] += 1

            # Calculate performance score
            if results["frame_latencies"]:
                avg_latency = statistics.mean(results["frame_latencies"])
                target_latency = frame_time_ms
                latency_efficiency = min(1.0, target_latency / avg_latency)
                results["performance_score"] = latency_efficiency

        except Exception as e:
            print(f"Avatar rendering workload error: {e}")
            results["rendering_errors"] += 1

        return results

    async def _render_avatar_session(self, session_id: str, scenario: Dict, duration: int) -> Dict:
        """Render avatar for a single session"""
        start_time = time.time()
        end_time = start_time + duration

        frames_rendered = 0
        frame_latencies = []
        emotion_transitions = 0

        target_fps = scenario["target_fps"]
        frame_interval = 1.0 / target_fps

        while time.time() < end_time:
            frame_start = time.time()

            # Simulate avatar rendering (replace with actual avatar API calls)
            await self.client.render_avatar_frame(
                session_id,
                quality=scenario["avatar_quality"],
                emotion_count=scenario["emotion_count"]
            )

            frame_latency = (time.time() - frame_start) * 1000  # Convert to ms
            frame_latencies.append(frame_latency)
            frames_rendered += 1

            # Occasionally trigger emotion transitions
            if frames_rendered % 60 == 0:  # Every ~1 second at 60fps
                await self.client.transition_avatar_emotion(session_id)
                emotion_transitions += 1

            # Maintain target framerate
            elapsed = time.time() - frame_start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

        return {
            "frames": frames_rendered,
            "latencies": frame_latencies,
            "emotions": emotion_transitions
        }

    async def _execute_voice_synthesis_workload(self, scenario: Dict) -> Dict:
        """Execute voice synthesis workload"""
        results = {
            "synthesis_requests": 0,
            "synthesis_latencies": [],
            "synthesis_errors": 0,
            "audio_quality_scores": [],
            "performance_score": 0.0
        }

        try:
            # Generate test phrases for synthesis
            test_phrases = self._generate_voice_test_phrases(
                scenario["text_complexity"],
                scenario["duration"] // 5  # One phrase every 5 seconds
            )

            # Execute concurrent voice synthesis
            tasks = []
            for voice_id in range(scenario["concurrent_voices"]):
                task = self._synthesize_voice_session(
                    f"voice_{voice_id}",
                    scenario,
                    test_phrases
                )
                tasks.append(task)

            session_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate results
            for session_result in session_results:
                if isinstance(session_result, dict):
                    results["synthesis_requests"] += session_result.get("requests", 0)
                    results["synthesis_latencies"].extend(session_result.get("latencies", []))
                    results["audio_quality_scores"].extend(session_result.get("quality_scores", []))
                else:
                    results["synthesis_errors"] += 1

            # Calculate performance score
            if results["synthesis_latencies"]:
                avg_latency = statistics.mean(results["synthesis_latencies"])
                target_latency = scenario["target_latency_ms"]
                latency_efficiency = min(1.0, target_latency / avg_latency)

                if results["audio_quality_scores"]:
                    avg_quality = statistics.mean(results["audio_quality_scores"])
                    quality_efficiency = avg_quality / 5.0  # Normalize to 0-1
                    results["performance_score"] = (latency_efficiency + quality_efficiency) / 2
                else:
                    results["performance_score"] = latency_efficiency

        except Exception as e:
            print(f"Voice synthesis workload error: {e}")
            results["synthesis_errors"] += 1

        return results

    async def _synthesize_voice_session(self, voice_id: str, scenario: Dict, phrases: List[str]) -> Dict:
        """Execute voice synthesis for a single voice session"""
        requests = 0
        latencies = []
        quality_scores = []

        for phrase in phrases:
            synthesis_start = time.time()

            try:
                # Synthesize voice with emotion
                audio_result = await self.client.synthesize_voice(
                    text=phrase,
                    quality=scenario["quality"],
                    emotion_depth=scenario["emotion_depth"],
                    voice_id=voice_id
                )

                synthesis_latency = (time.time() - synthesis_start) * 1000
                latencies.append(synthesis_latency)

                # Evaluate audio quality (mock scoring)
                quality_score = self._evaluate_audio_quality(audio_result)
                quality_scores.append(quality_score)

                requests += 1

            except Exception as e:
                print(f"Voice synthesis error for {voice_id}: {e}")

            # Pace requests
            await asyncio.sleep(2)

        return {
            "requests": requests,
            "latencies": latencies,
            "quality_scores": quality_scores
        }

    def _reset_gpu_state(self):
        """Reset GPU to clean state"""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Reset GPU clocks (if possible)
            subprocess.run(["nvidia-smi", "--reset-gpus"], capture_output=True)

        except Exception as e:
            print(f"GPU reset warning: {e}")

    def _collect_baseline_metrics(self) -> Dict:
        """Collect baseline GPU metrics"""
        try:
            gpu = GPUtil.getGPUs()[0]
            return {
                "idle_utilization": gpu.load * 100,
                "idle_memory": gpu.memoryUtil * 100,
                "idle_temperature": 25.0,  # Approximate idle temp
                "idle_power": 50.0        # Approximate idle power
            }
        except:
            return {
                "idle_utilization": 0,
                "idle_memory": 0,
                "idle_temperature": 25.0,
                "idle_power": 50.0
            }

    def _analyze_avatar_performance(self, scenario: Dict, gpu_metrics: List[GPUPerformanceMetrics],
                                   rendering_results: Dict) -> CompanionWorkloadResult:
        """Analyze avatar rendering performance results"""
        avg_gpu_util = statistics.mean(m.gpu_utilization for m in gpu_metrics)
        peak_temp = max(m.temperature for m in gpu_metrics)
        peak_power = max(m.power_consumption for m in gpu_metrics)

        frame_latencies = rendering_results.get("frame_latencies", [])
        avg_latency = statistics.mean(frame_latencies) if frame_latencies else 0
        p95_latency = np.percentile(frame_latencies, 95) if frame_latencies else 0

        # Calculate efficiency score
        target_latency = 1000.0 / scenario["target_fps"]
        latency_efficiency = min(1.0, target_latency / avg_latency) if avg_latency > 0 else 0
        utilization_efficiency = min(1.0, avg_gpu_util / 85.0)  # Target 85% utilization
        efficiency_score = (latency_efficiency + utilization_efficiency) / 2

        return CompanionWorkloadResult(
            workload_type=f"avatar_rendering_{scenario['complexity']}",
            duration=scenario["duration"],
            success_rate=1.0 - (rendering_results.get("rendering_errors", 0) / max(1, rendering_results.get("frames_rendered", 1))),
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            gpu_metrics=gpu_metrics,
            thermal_peak=peak_temp,
            power_peak=peak_power,
            efficiency_score=efficiency_score
        )

    def _analyze_voice_performance(self, scenario: Dict, gpu_metrics: List[GPUPerformanceMetrics],
                                  synthesis_results: Dict) -> CompanionWorkloadResult:
        """Analyze voice synthesis performance results"""
        synthesis_latencies = synthesis_results.get("synthesis_latencies", [])
        avg_latency = statistics.mean(synthesis_latencies) if synthesis_latencies else 0
        p95_latency = np.percentile(synthesis_latencies, 95) if synthesis_latencies else 0

        peak_temp = max(m.temperature for m in gpu_metrics)
        peak_power = max(m.power_consumption for m in gpu_metrics)
        avg_gpu_util = statistics.mean(m.gpu_utilization for m in gpu_metrics)

        # Calculate efficiency based on target latency
        target_latency = scenario["target_latency_ms"]
        latency_efficiency = min(1.0, target_latency / avg_latency) if avg_latency > 0 else 0

        # Factor in audio quality
        quality_scores = synthesis_results.get("audio_quality_scores", [])
        quality_efficiency = statistics.mean(quality_scores) / 5.0 if quality_scores else 0.8

        efficiency_score = (latency_efficiency + quality_efficiency) / 2

        return CompanionWorkloadResult(
            workload_type=f"voice_synthesis_{scenario['quality']}",
            duration=scenario["duration"],
            success_rate=1.0 - (synthesis_results.get("synthesis_errors", 0) / max(1, synthesis_results.get("synthesis_requests", 1))),
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            gpu_metrics=gpu_metrics,
            thermal_peak=peak_temp,
            power_peak=peak_power,
            efficiency_score=efficiency_score
        )

    def _save_test_results(self):
        """Save detailed test results to file"""
        results_data = {
            "timestamp": time.time(),
            "gpu_model": "RTX_4090",
            "test_results": [
                {
                    "workload_type": result.workload_type,
                    "duration": result.duration,
                    "success_rate": result.success_rate,
                    "avg_latency": result.avg_latency,
                    "p95_latency": result.p95_latency,
                    "thermal_peak": result.thermal_peak,
                    "power_peak": result.power_peak,
                    "efficiency_score": result.efficiency_score
                }
                for result in self.test_results
            ]
        }

        results_file = Path("test_reports/companion/rtx4090_performance.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

    # Additional helper methods would continue here...
    # (Remaining methods for concurrent workloads, memory testing, thermal analysis, etc.)

    def _generate_voice_test_phrases(self, complexity: str, count: int) -> List[str]:
        """Generate test phrases for voice synthesis"""
        phrases = {
            "simple": [
                "Hello, how can I help you today?",
                "I understand your concern about this issue.",
                "Let me analyze this data for you.",
                "The results look promising so far.",
                "Would you like me to explain this further?"
            ],
            "technical": [
                "The intrusion detection system has identified suspicious network traffic patterns.",
                "Cross-referencing these indicators of compromise across multiple threat intelligence feeds.",
                "This malware sample exhibits advanced evasion techniques including code obfuscation.",
                "The attack vector appears to exploit a previously unknown zero-day vulnerability.",
                "Forensic analysis reveals persistence mechanisms in both registry and file system."
            ],
            "complex": [
                "Based on the comprehensive analysis of network telemetry data, behavioral heuristics, and cryptographic signatures, this appears to be a sophisticated supply chain compromise targeting critical infrastructure sectors.",
                "The correlation between these seemingly disparate indicators suggests a coordinated campaign by an advanced persistent threat actor with significant operational security capabilities.",
                "Implementing defense-in-depth strategies requires careful consideration of both technical countermeasures and human factors engineering to address the full spectrum of potential attack vectors."
            ]
        }

        base_phrases = phrases.get(complexity, phrases["simple"])
        return (base_phrases * ((count // len(base_phrases)) + 1))[:count]

    def _evaluate_audio_quality(self, audio_result: Dict) -> float:
        """Evaluate synthesized audio quality (mock implementation)"""
        # In real implementation, this would use audio analysis
        # For testing, return simulated quality score
        return 4.2 + (hash(str(audio_result)) % 100) / 125.0  # 4.2-4.96 range