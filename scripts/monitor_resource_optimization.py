#!/usr/bin/env python3
"""
BEV Platform Resource Optimization Monitor
Real-time monitoring and alerting for multi-node resource utilization
"""

import time
import json
import docker
import psutil
import subprocess
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple
import requests
import yaml

class ResourceMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.nodes = {
            'STARLORD': {'host': 'localhost', 'ssh': None},
            'THANOS': {'host': 'thanos', 'ssh': 'thanos'},
            'ORACLE1': {'host': 'oracle1', 'ssh': 'oracle1'}
        }
        self.thresholds = {
            'memory_warning': 80,  # %
            'memory_critical': 90,
            'cpu_warning': 80,
            'cpu_critical': 90,
            'gpu_warning': 85,
            'gpu_critical': 95
        }
        self.log_file = f'/var/log/bev_resource_monitor_{datetime.now().strftime("%Y%m%d")}.log'

    def log(self, message: str, level: str = 'INFO'):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)

        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Failed to write to log file: {e}")

    def get_local_stats(self) -> Dict:
        """Get local system statistics"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')

            stats = {
                'memory': {
                    'total': round(memory.total / (1024**3), 2),  # GB
                    'used': round(memory.used / (1024**3), 2),   # GB
                    'percent': memory.percent
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'percent': cpu_percent
                },
                'disk': {
                    'total': round(disk.total / (1024**3), 2),   # GB
                    'used': round(disk.used / (1024**3), 2),    # GB
                    'percent': (disk.used / disk.total) * 100
                }
            }

            # GPU stats if available
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu',
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_data = result.stdout.strip().split('\n')[0].split(', ')
                    stats['gpu'] = {
                        'memory_total': int(gpu_data[0]) / 1024,  # GB
                        'memory_used': int(gpu_data[1]) / 1024,   # GB
                        'memory_percent': (int(gpu_data[1]) / int(gpu_data[0])) * 100,
                        'utilization': int(gpu_data[2]),
                        'temperature': int(gpu_data[3])
                    }
            except Exception as e:
                stats['gpu'] = None

            return stats
        except Exception as e:
            self.log(f"Error getting local stats: {e}", 'ERROR')
            return {}

    def get_remote_stats(self, host: str) -> Dict:
        """Get remote system statistics via SSH"""
        try:
            # Memory stats
            cmd = f"ssh {host} 'free -m && cat /proc/cpuinfo | grep processor | wc -l && top -bn1 | grep \"Cpu(s)\"'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            if result.returncode != 0:
                return {}

            lines = result.stdout.strip().split('\n')

            # Parse free output
            memory_line = lines[1].split()
            total_mem = int(memory_line[1]) / 1024  # GB
            used_mem = int(memory_line[2]) / 1024   # GB

            # CPU count and usage
            cpu_count = int(lines[3])
            cpu_usage_line = [line for line in lines if 'Cpu(s)' in line][0]
            cpu_percent = float(cpu_usage_line.split()[1].replace('%us,', ''))

            stats = {
                'memory': {
                    'total': round(total_mem, 2),
                    'used': round(used_mem, 2),
                    'percent': round((used_mem / total_mem) * 100, 2)
                },
                'cpu': {
                    'count': cpu_count,
                    'percent': cpu_percent
                },
                'gpu': None  # Remote GPU monitoring needs special setup
            }

            return stats
        except Exception as e:
            self.log(f"Error getting stats from {host}: {e}", 'ERROR')
            return {}

    def get_docker_stats(self, host: str = None) -> List[Dict]:
        """Get Docker container statistics"""
        try:
            if host:
                # Remote Docker stats
                cmd = f"ssh {host} 'docker stats --no-stream --format \"table {{{{.Container}}}}\\t{{{{.CPUPerc}}}}\\t{{{{.MemUsage}}}}\\t{{{{.MemPerc}}}}\"'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    return []

                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                containers = []
                for line in lines:
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        containers.append({
                            'name': parts[0],
                            'cpu_percent': parts[1].replace('%', ''),
                            'memory_usage': parts[2],
                            'memory_percent': parts[3].replace('%', '')
                        })
                return containers
            else:
                # Local Docker stats
                containers = []
                for container in self.docker_client.containers.list():
                    try:
                        stats = container.stats(stream=False)

                        # Calculate CPU percentage
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                   stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                      stats['precpu_stats']['system_cpu_usage']
                        cpu_percent = (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100

                        # Memory stats
                        mem_usage = stats['memory_stats']['usage']
                        mem_limit = stats['memory_stats']['limit']
                        mem_percent = (mem_usage / mem_limit) * 100

                        containers.append({
                            'name': container.name,
                            'cpu_percent': round(cpu_percent, 2),
                            'memory_usage': f"{mem_usage / (1024**3):.2f}GB",
                            'memory_percent': round(mem_percent, 2)
                        })
                    except Exception as e:
                        self.log(f"Error getting stats for container {container.name}: {e}", 'WARNING')

                return containers
        except Exception as e:
            self.log(f"Error getting Docker stats: {e}", 'ERROR')
            return []

    def check_thresholds(self, node_name: str, stats: Dict) -> List[str]:
        """Check if any thresholds are exceeded"""
        alerts = []

        if 'memory' in stats:
            mem_percent = stats['memory']['percent']
            if mem_percent >= self.thresholds['memory_critical']:
                alerts.append(f"🚨 CRITICAL: {node_name} memory at {mem_percent:.1f}%")
            elif mem_percent >= self.thresholds['memory_warning']:
                alerts.append(f"⚠️ WARNING: {node_name} memory at {mem_percent:.1f}%")

        if 'cpu' in stats:
            cpu_percent = stats['cpu']['percent']
            if cpu_percent >= self.thresholds['cpu_critical']:
                alerts.append(f"🚨 CRITICAL: {node_name} CPU at {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds['cpu_warning']:
                alerts.append(f"⚠️ WARNING: {node_name} CPU at {cpu_percent:.1f}%")

        if stats.get('gpu'):
            gpu_mem_percent = stats['gpu']['memory_percent']
            gpu_util = stats['gpu']['utilization']

            if gpu_mem_percent >= self.thresholds['gpu_critical']:
                alerts.append(f"🚨 CRITICAL: {node_name} GPU memory at {gpu_mem_percent:.1f}%")
            elif gpu_mem_percent >= self.thresholds['gpu_warning']:
                alerts.append(f"⚠️ WARNING: {node_name} GPU memory at {gpu_mem_percent:.1f}%")

            if gpu_util >= self.thresholds['gpu_critical']:
                alerts.append(f"🚨 CRITICAL: {node_name} GPU utilization at {gpu_util}%")

        return alerts

    def recommend_actions(self, node_name: str, stats: Dict, containers: List[Dict]) -> List[str]:
        """Recommend optimization actions"""
        recommendations = []

        if 'memory' in stats and stats['memory']['percent'] > 80:
            # Find memory-hungry containers
            high_mem_containers = [c for c in containers if float(c['memory_percent']) > 10]
            if high_mem_containers:
                recommendations.append(f"Consider reducing memory limits for: {', '.join([c['name'] for c in high_mem_containers[:3]])}")

        if node_name == 'THANOS' and stats.get('memory', {}).get('percent', 0) > 85:
            recommendations.append("Migrate GPU-intensive services to STARLORD")
            recommendations.append("Consider stopping non-essential services")

        if node_name == 'STARLORD' and stats.get('gpu'):
            gpu_util = stats['gpu']['utilization']
            if gpu_util < 20:
                recommendations.append("GPU underutilized - consider stopping GPU services to save power")
            elif gpu_util > 90:
                recommendations.append("GPU overutilized - consider load balancing")

        if node_name == 'ORACLE1' and stats.get('memory', {}).get('percent', 0) > 90:
            recommendations.append("ORACLE1 near capacity - avoid adding more services")

        return recommendations

    def generate_report(self) -> Dict:
        """Generate comprehensive resource utilization report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'nodes': {},
            'alerts': [],
            'recommendations': [],
            'summary': {}
        }

        total_memory_used = 0
        total_memory_available = 0
        total_containers = 0

        for node_name, node_config in self.nodes.items():
            self.log(f"Collecting stats for {node_name}...")

            if node_name == 'STARLORD':
                stats = self.get_local_stats()
                containers = self.get_docker_stats()
            else:
                stats = self.get_remote_stats(node_config['ssh'])
                containers = self.get_docker_stats(node_config['ssh'])

            if stats:
                # Check thresholds
                node_alerts = self.check_thresholds(node_name, stats)
                report['alerts'].extend(node_alerts)

                # Get recommendations
                node_recommendations = self.recommend_actions(node_name, stats, containers)
                report['recommendations'].extend([f"{node_name}: {rec}" for rec in node_recommendations])

                # Add to totals
                if 'memory' in stats:
                    total_memory_used += stats['memory']['used']
                    total_memory_available += stats['memory']['total']

                total_containers += len(containers)

                report['nodes'][node_name] = {
                    'stats': stats,
                    'containers': containers,
                    'container_count': len(containers),
                    'alerts': node_alerts,
                    'recommendations': node_recommendations
                }

        # Generate summary
        report['summary'] = {
            'total_nodes': len(self.nodes),
            'total_containers': total_containers,
            'total_memory_used': round(total_memory_used, 2),
            'total_memory_available': round(total_memory_available, 2),
            'overall_memory_utilization': round((total_memory_used / total_memory_available) * 100, 2) if total_memory_available > 0 else 0,
            'alert_count': len(report['alerts']),
            'recommendation_count': len(report['recommendations'])
        }

        return report

    def display_report(self, report: Dict):
        """Display formatted report"""
        print("\n" + "="*80)
        print("BEV PLATFORM RESOURCE UTILIZATION REPORT")
        print("="*80)
        print(f"Generated: {report['timestamp']}")
        print(f"Overall Memory Utilization: {report['summary']['overall_memory_utilization']:.1f}%")
        print(f"Total Containers: {report['summary']['total_containers']}")
        print(f"Active Alerts: {report['summary']['alert_count']}")

        # Node details
        for node_name, node_data in report['nodes'].items():
            print(f"\n{'─'*60}")
            print(f"NODE: {node_name}")
            print(f"{'─'*60}")

            stats = node_data['stats']
            if 'memory' in stats:
                print(f"Memory: {stats['memory']['used']:.1f}/{stats['memory']['total']:.1f} GB ({stats['memory']['percent']:.1f}%)")
            if 'cpu' in stats:
                print(f"CPU: {stats['cpu']['percent']:.1f}% ({stats['cpu']['count']} cores)")
            if stats.get('gpu'):
                gpu = stats['gpu']
                print(f"GPU: {gpu['memory_used']:.1f}/{gpu['memory_total']:.1f} GB ({gpu['memory_percent']:.1f}%), {gpu['utilization']}% util, {gpu['temperature']}°C")

            print(f"Containers: {node_data['container_count']}")

            # Top containers by memory
            if node_data['containers']:
                sorted_containers = sorted(node_data['containers'],
                                         key=lambda x: float(x['memory_percent']),
                                         reverse=True)[:3]
                print("Top memory consumers:")
                for container in sorted_containers:
                    print(f"  - {container['name']}: {container['memory_percent']}%")

        # Alerts
        if report['alerts']:
            print(f"\n{'─'*60}")
            print("🚨 ALERTS")
            print(f"{'─'*60}")
            for alert in report['alerts']:
                print(f"  {alert}")

        # Recommendations
        if report['recommendations']:
            print(f"\n{'─'*60}")
            print("💡 RECOMMENDATIONS")
            print(f"{'─'*60}")
            for rec in report['recommendations'][:10]:  # Limit to top 10
                print(f"  • {rec}")

    def save_report(self, report: Dict, filename: str = None):
        """Save report to JSON file"""
        if not filename:
            filename = f"/var/log/bev_resource_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"Report saved to {filename}")
        except Exception as e:
            self.log(f"Failed to save report: {e}", 'ERROR')

    def monitor_continuous(self, interval: int = 60):
        """Continuous monitoring mode"""
        self.log("Starting continuous monitoring...")
        print(f"Monitoring every {interval} seconds. Press Ctrl+C to stop.")

        try:
            while True:
                report = self.generate_report()

                # Only display if there are alerts
                if report['alerts']:
                    self.display_report(report)

                # Save critical alerts
                critical_alerts = [a for a in report['alerts'] if 'CRITICAL' in a]
                if critical_alerts:
                    self.save_report(report)
                    self.log(f"Critical alerts detected: {len(critical_alerts)}", 'CRITICAL')

                time.sleep(interval)

        except KeyboardInterrupt:
            self.log("Monitoring stopped by user")

def main():
    """Main function"""
    monitor = ResourceMonitor()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'continuous':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            monitor.monitor_continuous(interval)
        elif sys.argv[1] == 'report':
            report = monitor.generate_report()
            monitor.display_report(report)
            if len(sys.argv) > 2:
                monitor.save_report(report, sys.argv[2])
            else:
                monitor.save_report(report)
        else:
            print("Usage: python3 monitor_resource_optimization.py [continuous [interval]|report [filename]]")
    else:
        # Single report by default
        report = monitor.generate_report()
        monitor.display_report(report)

if __name__ == "__main__":
    main()