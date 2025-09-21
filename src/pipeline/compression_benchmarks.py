#!/usr/bin/env python3
"""
Context Compression Performance Benchmarks and Optimization Metrics
Comprehensive benchmarking suite for the BEV OSINT Context Compression Engine
"""

import asyncio
import json
import logging
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import hashlib
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# Import compression components
from .context_compressor import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    ContextCompressionResult
)
from .quality_validator import QualityValidator, ValidationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkMetrics:
    """Comprehensive metrics for a single benchmark run"""
    # Identifiers
    test_id: str
    test_name: str
    timestamp: datetime

    # Test parameters
    strategy: str
    content_size: int
    content_type: str
    target_ratio: float

    # Performance metrics
    compression_time: float
    decompression_time: float
    total_time: float
    throughput_mbps: float

    # Compression metrics
    original_size: int
    compressed_size: int
    compression_ratio: float
    token_compression_ratio: float

    # Quality metrics
    information_loss: float
    semantic_similarity: float
    coherence_score: float
    bleu_score: float
    rouge_score: float
    reconstruction_accuracy: float

    # Resource metrics
    peak_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    gpu_memory_mb: float = 0.0

    # Additional metrics
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    quality_score: float = 0.0

@dataclass
class BenchmarkSuite:
    """Configuration for a benchmark suite"""
    name: str
    description: str
    test_cases: List[Dict[str, Any]]
    strategies: List[CompressionStrategy]
    metrics_to_collect: List[str]
    iterations: int = 3
    warmup_iterations: int = 1

@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report"""
    suite_name: str
    execution_time: float
    total_tests: int
    successful_tests: int
    failed_tests: int

    # Aggregated metrics
    avg_compression_ratio: float
    avg_information_loss: float
    avg_throughput: float
    avg_quality_score: float

    # Performance statistics
    performance_stats: Dict[str, Dict[str, float]]
    quality_stats: Dict[str, Dict[str, float]]
    resource_stats: Dict[str, Dict[str, float]]

    # Individual test results
    test_results: List[BenchmarkMetrics]

    # Recommendations
    recommendations: List[str]
    optimal_configurations: Dict[str, Any]

class ResourceMonitor:
    """Real-time resource monitoring during benchmarks"""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.monitoring = False
        self.metrics = {
            'cpu_percent': [],
            'memory_mb': [],
            'timestamps': []
        }
        self._monitor_thread = None

    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.metrics = {'cpu_percent': [], 'memory_mb': [], 'timestamps': []}
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()

        if not self.metrics['cpu_percent']:
            return {'peak_memory_mb': 0.0, 'avg_cpu_percent': 0.0, 'peak_cpu_percent': 0.0}

        return {
            'peak_memory_mb': max(self.metrics['memory_mb']),
            'avg_cpu_percent': statistics.mean(self.metrics['cpu_percent']),
            'peak_cpu_percent': max(self.metrics['cpu_percent'])
        }

    def _monitor_loop(self):
        """Monitoring loop"""
        process = psutil.Process()

        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024

                self.metrics['cpu_percent'].append(cpu_percent)
                self.metrics['memory_mb'].append(memory_mb)
                self.metrics['timestamps'].append(time.time())

                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break

class CompressionBenchmark:
    """
    Comprehensive benchmark suite for context compression engine
    """

    def __init__(self, infrastructure_config: Dict[str, Any]):
        self.infrastructure_config = infrastructure_config
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize test data
        self.test_datasets = {}
        self._load_test_datasets()

        # Results storage
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.baseline_metrics = {}

    def _load_test_datasets(self):
        """Load or generate test datasets of various types and sizes"""

        # Small text dataset (1KB - 10KB)
        self.test_datasets['small_text'] = [
            "This is a small text sample for compression testing. " * 20,
            "Another small text with different content for variety. " * 15,
            "Technical documentation excerpt with specific terminology. " * 25
        ]

        # Medium text dataset (10KB - 100KB)
        medium_base = ("In the field of artificial intelligence and machine learning, " +
                      "context compression represents a critical optimization technique " +
                      "for processing large volumes of textual information efficiently. " +
                      "This comprehensive analysis examines various methodologies and " +
                      "their comparative effectiveness in real-world applications. ")

        self.test_datasets['medium_text'] = [medium_base * 50, medium_base * 75, medium_base * 100]

        # Large text dataset (100KB - 1MB)
        large_base = medium_base * 200
        self.test_datasets['large_text'] = [large_base * 2, large_base * 3, large_base * 5]

        # Code dataset
        code_sample = '''
def process_data(input_data):
    """Process input data with error handling"""
    try:
        result = []
        for item in input_data:
            if validate_item(item):
                processed = transform_item(item)
                result.append(processed)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return None

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.validators = []

    def add_validator(self, validator):
        self.validators.append(validator)
'''

        self.test_datasets['code'] = [code_sample * 10, code_sample * 25, code_sample * 50]

        # Structured data dataset (JSON-like)
        structured_sample = '''
{
    "user_id": "12345",
    "profile": {
        "name": "John Doe",
        "email": "john@example.com",
        "preferences": {
            "notifications": true,
            "theme": "dark",
            "language": "en"
        }
    },
    "activity": [
        {"timestamp": "2024-01-15T10:30:00Z", "action": "login"},
        {"timestamp": "2024-01-15T10:35:00Z", "action": "view_dashboard"},
        {"timestamp": "2024-01-15T10:45:00Z", "action": "update_profile"}
    ]
}
'''

        self.test_datasets['structured'] = [structured_sample * 5, structured_sample * 15, structured_sample * 30]

        # Repetitive content (high redundancy)
        repetitive_base = "This is a repeated sentence for testing compression efficiency. "
        self.test_datasets['repetitive'] = [
            repetitive_base * 100,
            repetitive_base * 200,
            repetitive_base * 500
        ]

        # Mixed content
        mixed_content = (
            "# Research Report\n\n" +
            "## Introduction\n" +
            medium_base +
            "\n\n## Code Examples\n" +
            code_sample +
            "\n\n## Data Analysis\n" +
            structured_sample
        )

        self.test_datasets['mixed'] = [mixed_content * 2, mixed_content * 5, mixed_content * 10]

    async def run_comprehensive_benchmark(self,
                                        strategies: List[CompressionStrategy] = None,
                                        test_types: List[str] = None) -> BenchmarkReport:
        """Run comprehensive benchmark across all test scenarios"""

        if strategies is None:
            strategies = [
                CompressionStrategy.CONSERVATIVE,
                CompressionStrategy.BALANCED,
                CompressionStrategy.AGGRESSIVE,
                CompressionStrategy.SEMANTIC_ONLY,
                CompressionStrategy.ENTROPY_ONLY
            ]

        if test_types is None:
            test_types = list(self.test_datasets.keys())

        logger.info(f"Starting comprehensive benchmark: {len(strategies)} strategies × {len(test_types)} content types")

        start_time = time.time()
        all_results = []
        failed_tests = 0

        for strategy in strategies:
            for content_type in test_types:
                for i, content in enumerate(self.test_datasets[content_type]):
                    test_id = f"{strategy.value}_{content_type}_{i}"

                    try:
                        result = await self._run_single_benchmark(
                            test_id=test_id,
                            content=content,
                            strategy=strategy,
                            content_type=content_type
                        )
                        all_results.append(result)

                    except Exception as e:
                        logger.error(f"Benchmark failed for {test_id}: {e}")
                        failed_tests += 1

        execution_time = time.time() - start_time

        # Generate comprehensive report
        report = await self._generate_benchmark_report(
            suite_name="Comprehensive Benchmark",
            execution_time=execution_time,
            test_results=all_results,
            failed_tests=failed_tests
        )

        # Save report
        await self._save_benchmark_report(report)

        return report

    async def _run_single_benchmark(self,
                                   test_id: str,
                                   content: str,
                                   strategy: CompressionStrategy,
                                   content_type: str) -> BenchmarkMetrics:
        """Run a single benchmark test"""

        logger.info(f"Running benchmark: {test_id}")

        # Initialize compressor
        config = CompressionConfig(
            strategy=strategy,
            target_compression_ratio=0.4,
            max_information_loss=0.05,
            enable_caching=False,  # Disable caching for accurate benchmarks
            quality_validation=True
        )

        compressor = ContextCompressor(config, self.infrastructure_config)

        # Prepare content
        original_size = len(content.encode('utf-8'))
        token_count = len(self.encoding.encode(content))

        # Start resource monitoring
        monitor = ResourceMonitor()
        monitor.start()

        try:
            # Compression benchmark
            compression_start = time.time()
            compression_result = await compressor.compress_context([content])
            compression_time = time.time() - compression_start

            # Decompression benchmark
            decompression_start = time.time()
            decompressed = await compressor.decompress_context(compression_result)
            decompression_time = time.time() - decompression_start

            total_time = compression_time + decompression_time

            # Calculate throughput
            throughput_mbps = (original_size / (1024 * 1024)) / total_time

            # Quality validation
            validator = QualityValidator(ValidationConfig())
            validation_result = await validator.validate_compression(
                [content],
                compression_result.compressed_content
            )

            # Stop monitoring
            resource_stats = monitor.stop()

            # Create metrics object
            metrics = BenchmarkMetrics(
                test_id=test_id,
                test_name=f"{strategy.value}_{content_type}",
                timestamp=datetime.now(),
                strategy=strategy.value,
                content_size=original_size,
                content_type=content_type,
                target_ratio=config.target_compression_ratio,
                compression_time=compression_time,
                decompression_time=decompression_time,
                total_time=total_time,
                throughput_mbps=throughput_mbps,
                original_size=original_size,
                compressed_size=compression_result.metrics.compressed_size,
                compression_ratio=compression_result.metrics.compression_ratio,
                token_compression_ratio=compression_result.metrics.token_compression_ratio,
                information_loss=compression_result.metrics.information_loss_score,
                semantic_similarity=compression_result.metrics.semantic_similarity_score,
                coherence_score=compression_result.metrics.coherence_score,
                bleu_score=validation_result.individual_metrics[4].score if len(validation_result.individual_metrics) > 4 else 0.0,
                rouge_score=validation_result.individual_metrics[5].score if len(validation_result.individual_metrics) > 5 else 0.0,
                reconstruction_accuracy=validation_result.reconstruction_accuracy,
                quality_score=validation_result.overall_score,
                **resource_stats
            )

            return metrics

        except Exception as e:
            monitor.stop()
            logger.error(f"Benchmark {test_id} failed: {e}")
            raise

    async def _generate_benchmark_report(self,
                                       suite_name: str,
                                       execution_time: float,
                                       test_results: List[BenchmarkMetrics],
                                       failed_tests: int) -> BenchmarkReport:
        """Generate comprehensive benchmark report"""

        if not test_results:
            raise ValueError("No successful test results to generate report")

        # Calculate aggregated metrics
        avg_compression_ratio = statistics.mean([r.compression_ratio for r in test_results])
        avg_information_loss = statistics.mean([r.information_loss for r in test_results])
        avg_throughput = statistics.mean([r.throughput_mbps for r in test_results])
        avg_quality_score = statistics.mean([r.quality_score for r in test_results])

        # Generate statistics by category
        performance_stats = self._calculate_performance_stats(test_results)
        quality_stats = self._calculate_quality_stats(test_results)
        resource_stats = self._calculate_resource_stats(test_results)

        # Generate recommendations
        recommendations = self._generate_recommendations(test_results)

        # Find optimal configurations
        optimal_configs = self._find_optimal_configurations(test_results)

        report = BenchmarkReport(
            suite_name=suite_name,
            execution_time=execution_time,
            total_tests=len(test_results) + failed_tests,
            successful_tests=len(test_results),
            failed_tests=failed_tests,
            avg_compression_ratio=avg_compression_ratio,
            avg_information_loss=avg_information_loss,
            avg_throughput=avg_throughput,
            avg_quality_score=avg_quality_score,
            performance_stats=performance_stats,
            quality_stats=quality_stats,
            resource_stats=resource_stats,
            test_results=test_results,
            recommendations=recommendations,
            optimal_configurations=optimal_configs
        )

        return report

    def _calculate_performance_stats(self, results: List[BenchmarkMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate performance statistics"""

        stats = {}

        # Compression time statistics
        compression_times = [r.compression_time for r in results]
        stats['compression_time'] = {
            'mean': statistics.mean(compression_times),
            'median': statistics.median(compression_times),
            'std': statistics.stdev(compression_times) if len(compression_times) > 1 else 0.0,
            'min': min(compression_times),
            'max': max(compression_times)
        }

        # Throughput statistics
        throughputs = [r.throughput_mbps for r in results]
        stats['throughput'] = {
            'mean': statistics.mean(throughputs),
            'median': statistics.median(throughputs),
            'std': statistics.stdev(throughputs) if len(throughputs) > 1 else 0.0,
            'min': min(throughputs),
            'max': max(throughputs)
        }

        # Compression ratio statistics
        ratios = [r.compression_ratio for r in results]
        stats['compression_ratio'] = {
            'mean': statistics.mean(ratios),
            'median': statistics.median(ratios),
            'std': statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            'min': min(ratios),
            'max': max(ratios)
        }

        return stats

    def _calculate_quality_stats(self, results: List[BenchmarkMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate quality statistics"""

        stats = {}

        # Information loss statistics
        info_loss = [r.information_loss for r in results]
        stats['information_loss'] = {
            'mean': statistics.mean(info_loss),
            'median': statistics.median(info_loss),
            'std': statistics.stdev(info_loss) if len(info_loss) > 1 else 0.0,
            'min': min(info_loss),
            'max': max(info_loss)
        }

        # Semantic similarity statistics
        similarities = [r.semantic_similarity for r in results]
        stats['semantic_similarity'] = {
            'mean': statistics.mean(similarities),
            'median': statistics.median(similarities),
            'std': statistics.stdev(similarities) if len(similarities) > 1 else 0.0,
            'min': min(similarities),
            'max': max(similarities)
        }

        # Overall quality statistics
        quality_scores = [r.quality_score for r in results]
        stats['quality_score'] = {
            'mean': statistics.mean(quality_scores),
            'median': statistics.median(quality_scores),
            'std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0,
            'min': min(quality_scores),
            'max': max(quality_scores)
        }

        return stats

    def _calculate_resource_stats(self, results: List[BenchmarkMetrics]) -> Dict[str, Dict[str, float]]:
        """Calculate resource usage statistics"""

        stats = {}

        # Memory usage statistics
        memory_usage = [r.peak_memory_mb for r in results]
        stats['memory_usage'] = {
            'mean': statistics.mean(memory_usage),
            'median': statistics.median(memory_usage),
            'std': statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0.0,
            'min': min(memory_usage),
            'max': max(memory_usage)
        }

        # CPU usage statistics
        cpu_usage = [r.avg_cpu_percent for r in results]
        stats['cpu_usage'] = {
            'mean': statistics.mean(cpu_usage),
            'median': statistics.median(cpu_usage),
            'std': statistics.stdev(cpu_usage) if len(cpu_usage) > 1 else 0.0,
            'min': min(cpu_usage),
            'max': max(cpu_usage)
        }

        return stats

    def _generate_recommendations(self, results: List[BenchmarkMetrics]) -> List[str]:
        """Generate performance and optimization recommendations"""

        recommendations = []

        # Analyze strategy performance
        strategy_performance = {}
        for result in results:
            if result.strategy not in strategy_performance:
                strategy_performance[result.strategy] = []

            # Combined score: compression efficiency + quality + speed
            score = (
                result.compression_ratio * 0.4 +
                (1 - result.information_loss) * 0.3 +
                result.quality_score * 0.2 +
                min(1.0, result.throughput_mbps / 10) * 0.1
            )
            strategy_performance[result.strategy].append(score)

        # Find best strategy
        avg_scores = {
            strategy: statistics.mean(scores)
            for strategy, scores in strategy_performance.items()
        }

        best_strategy = max(avg_scores, key=avg_scores.get)
        recommendations.append(f"Best overall strategy: {best_strategy} (score: {avg_scores[best_strategy]:.3f})")

        # Memory usage recommendations
        avg_memory = statistics.mean([r.peak_memory_mb for r in results])
        if avg_memory > 2000:  # > 2GB
            recommendations.append(f"High memory usage detected ({avg_memory:.0f}MB). Consider batch processing for large content.")

        # Performance recommendations
        slow_operations = [r for r in results if r.compression_time > 10.0]
        if slow_operations:
            recommendations.append(f"{len(slow_operations)} slow operations detected. Consider optimizing for content types: {set([r.content_type for r in slow_operations])}")

        # Quality recommendations
        low_quality = [r for r in results if r.quality_score < 0.8]
        if low_quality:
            recommendations.append(f"{len(low_quality)} low-quality results. Review compression parameters for content types: {set([r.content_type for r in low_quality])}")

        return recommendations

    def _find_optimal_configurations(self, results: List[BenchmarkMetrics]) -> Dict[str, Any]:
        """Find optimal configurations for different use cases"""

        configs = {}

        # Best compression ratio
        best_compression = max(results, key=lambda r: r.compression_ratio)
        configs['best_compression'] = {
            'strategy': best_compression.strategy,
            'compression_ratio': best_compression.compression_ratio,
            'information_loss': best_compression.information_loss,
            'content_type': best_compression.content_type
        }

        # Best quality preservation
        best_quality = min(results, key=lambda r: r.information_loss)
        configs['best_quality'] = {
            'strategy': best_quality.strategy,
            'compression_ratio': best_quality.compression_ratio,
            'information_loss': best_quality.information_loss,
            'content_type': best_quality.content_type
        }

        # Best performance
        best_performance = max(results, key=lambda r: r.throughput_mbps)
        configs['best_performance'] = {
            'strategy': best_performance.strategy,
            'throughput_mbps': best_performance.throughput_mbps,
            'compression_time': best_performance.compression_time,
            'content_type': best_performance.content_type
        }

        # Best balanced (compression + quality + speed)
        best_balanced = max(results, key=lambda r: (
            r.compression_ratio * 0.4 +
            (1 - r.information_loss) * 0.4 +
            min(1.0, r.throughput_mbps / 10) * 0.2
        ))
        configs['best_balanced'] = {
            'strategy': best_balanced.strategy,
            'compression_ratio': best_balanced.compression_ratio,
            'information_loss': best_balanced.information_loss,
            'throughput_mbps': best_balanced.throughput_mbps,
            'content_type': best_balanced.content_type
        }

        return configs

    async def _save_benchmark_report(self, report: BenchmarkReport):
        """Save benchmark report to files"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = self.results_dir / f"benchmark_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            # Convert to dict for JSON serialization
            report_dict = asdict(report)
            # Convert datetime objects to strings
            for result in report_dict['test_results']:
                result['timestamp'] = result['timestamp'].isoformat()

            json.dump(report_dict, f, indent=2, default=str)

        # Generate and save visualizations
        await self._generate_visualizations(report, timestamp)

        # Save CSV for detailed analysis
        csv_path = self.results_dir / f"benchmark_details_{timestamp}.csv"
        df = pd.DataFrame([asdict(result) for result in report.test_results])
        df.to_csv(csv_path, index=False)

        logger.info(f"Benchmark report saved: {json_path}")

    async def _generate_visualizations(self, report: BenchmarkReport, timestamp: str):
        """Generate visualization charts for benchmark results"""

        try:
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Context Compression Benchmark Results - {timestamp}', fontsize=16)

            # Prepare data
            df = pd.DataFrame([asdict(result) for result in report.test_results])

            # 1. Compression Ratio by Strategy
            sns.boxplot(data=df, x='strategy', y='compression_ratio', ax=axes[0, 0])
            axes[0, 0].set_title('Compression Ratio by Strategy')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # 2. Information Loss by Strategy
            sns.boxplot(data=df, x='strategy', y='information_loss', ax=axes[0, 1])
            axes[0, 1].set_title('Information Loss by Strategy')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Throughput by Content Type
            sns.barplot(data=df, x='content_type', y='throughput_mbps', ax=axes[0, 2])
            axes[0, 2].set_title('Throughput by Content Type')
            axes[0, 2].tick_params(axis='x', rotation=45)

            # 4. Compression Time vs File Size
            axes[1, 0].scatter(df['content_size'], df['compression_time'],
                             c=df['strategy'].astype('category').cat.codes, alpha=0.6)
            axes[1, 0].set_xlabel('Content Size (bytes)')
            axes[1, 0].set_ylabel('Compression Time (seconds)')
            axes[1, 0].set_title('Compression Time vs Content Size')

            # 5. Quality Score by Strategy
            sns.violinplot(data=df, x='strategy', y='quality_score', ax=axes[1, 1])
            axes[1, 1].set_title('Quality Score Distribution by Strategy')
            axes[1, 1].tick_params(axis='x', rotation=45)

            # 6. Memory Usage by Strategy
            sns.boxplot(data=df, x='strategy', y='peak_memory_mb', ax=axes[1, 2])
            axes[1, 2].set_title('Peak Memory Usage by Strategy')
            axes[1, 2].tick_params(axis='x', rotation=45)

            plt.tight_layout()

            # Save plot
            plot_path = self.results_dir / f"benchmark_charts_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Generate performance heatmap
            self._generate_performance_heatmap(df, timestamp)

            logger.info(f"Visualizations saved: {plot_path}")

        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")

    def _generate_performance_heatmap(self, df: pd.DataFrame, timestamp: str):
        """Generate performance heatmap"""

        try:
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values='compression_ratio',
                index='strategy',
                columns='content_type',
                aggfunc='mean'
            )

            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, cmap='viridis', fmt='.3f')
            plt.title('Average Compression Ratio by Strategy and Content Type')
            plt.tight_layout()

            heatmap_path = self.results_dir / f"performance_heatmap_{timestamp}.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")

    async def run_stress_test(self,
                            strategy: CompressionStrategy = CompressionStrategy.BALANCED,
                            max_content_size: int = 10 * 1024 * 1024,  # 10MB
                            concurrent_operations: int = 10) -> Dict[str, Any]:
        """Run stress test with large content and concurrent operations"""

        logger.info(f"Starting stress test: {max_content_size/1024/1024:.1f}MB content, {concurrent_operations} concurrent ops")

        # Generate large test content
        base_content = "This is stress test content for large-scale compression testing. " * 100
        large_content = base_content * (max_content_size // len(base_content.encode('utf-8')))

        # Initialize compressor
        config = CompressionConfig(strategy=strategy)
        compressor = ContextCompressor(config, self.infrastructure_config)

        # Run concurrent operations
        start_time = time.time()

        async def single_compression():
            return await compressor.compress_context([large_content])

        tasks = [single_compression() for _ in range(concurrent_operations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # Analyze results
        successful = [r for r in results if not isinstance(r, Exception)]
        failed = [r for r in results if isinstance(r, Exception)]

        stress_report = {
            'test_duration': end_time - start_time,
            'content_size_mb': max_content_size / 1024 / 1024,
            'concurrent_operations': concurrent_operations,
            'successful_operations': len(successful),
            'failed_operations': len(failed),
            'success_rate': len(successful) / concurrent_operations,
            'avg_compression_ratio': statistics.mean([r.metrics.compression_ratio for r in successful]) if successful else 0,
            'total_throughput_mbps': (max_content_size * len(successful) / 1024 / 1024) / (end_time - start_time),
            'errors': [str(e) for e in failed]
        }

        return stress_report

    async def run_regression_test(self, baseline_report_path: str) -> Dict[str, Any]:
        """Run regression test against baseline performance"""

        # Load baseline report
        with open(baseline_report_path, 'r') as f:
            baseline_data = json.load(f)

        # Run current benchmark
        current_report = await self.run_comprehensive_benchmark()

        # Compare key metrics
        regression_analysis = {
            'baseline_date': baseline_data.get('timestamp', 'unknown'),
            'current_date': datetime.now().isoformat(),
            'performance_changes': {},
            'quality_changes': {},
            'regressions_detected': []
        }

        # Compare average metrics
        baseline_compression = baseline_data.get('avg_compression_ratio', 0)
        current_compression = current_report.avg_compression_ratio
        compression_change = (current_compression - baseline_compression) / baseline_compression * 100

        baseline_loss = baseline_data.get('avg_information_loss', 0)
        current_loss = current_report.avg_information_loss
        loss_change = (current_loss - baseline_loss) / baseline_loss * 100 if baseline_loss > 0 else 0

        baseline_throughput = baseline_data.get('avg_throughput', 0)
        current_throughput = current_report.avg_throughput
        throughput_change = (current_throughput - baseline_throughput) / baseline_throughput * 100 if baseline_throughput > 0 else 0

        regression_analysis['performance_changes'] = {
            'compression_ratio_change_pct': compression_change,
            'information_loss_change_pct': loss_change,
            'throughput_change_pct': throughput_change
        }

        # Detect regressions (> 5% degradation)
        if compression_change < -5:
            regression_analysis['regressions_detected'].append(f"Compression ratio decreased by {abs(compression_change):.1f}%")

        if loss_change > 5:
            regression_analysis['regressions_detected'].append(f"Information loss increased by {loss_change:.1f}%")

        if throughput_change < -5:
            regression_analysis['regressions_detected'].append(f"Throughput decreased by {abs(throughput_change):.1f}%")

        return regression_analysis

# Example usage and CLI interface
async def main():
    """Main function for running benchmarks"""

    # Configuration
    infrastructure_config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'mongodb_url': 'mongodb://localhost:27017/',
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'weaviate_host': 'localhost',
        'weaviate_port': 8080
    }

    # Initialize benchmark suite
    benchmark = CompressionBenchmark(infrastructure_config)

    # Run comprehensive benchmark
    print("Running comprehensive benchmark...")
    report = await benchmark.run_comprehensive_benchmark()

    print(f"\nBenchmark completed!")
    print(f"Total tests: {report.total_tests}")
    print(f"Successful: {report.successful_tests}")
    print(f"Failed: {report.failed_tests}")
    print(f"Average compression ratio: {report.avg_compression_ratio:.3f}")
    print(f"Average information loss: {report.avg_information_loss:.3f}")
    print(f"Average throughput: {report.avg_throughput:.2f} MB/s")
    print(f"Average quality score: {report.avg_quality_score:.3f}")

    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")

    # Run stress test
    print("\nRunning stress test...")
    stress_result = await benchmark.run_stress_test()
    print(f"Stress test: {stress_result['success_rate']:.1%} success rate, "
          f"{stress_result['total_throughput_mbps']:.2f} MB/s total throughput")

if __name__ == "__main__":
    asyncio.run(main())