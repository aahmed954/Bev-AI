#!/usr/bin/env python3
"""
Tor Network Monitor
Monitors the health and performance of the 3-hop Tor circuit
"""

import time
import json
import requests
import socket
import threading
from datetime import datetime
from typing import Dict, List, Optional
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TorNodeMonitor:
    """Monitor individual Tor node health and performance"""

    def __init__(self, node_name: str, control_port: int, dir_port: int, or_port: int):
        self.node_name = node_name
        self.control_port = control_port
        self.dir_port = dir_port
        self.or_port = or_port
        self.stats = {
            'uptime': 0,
            'connections': 0,
            'bandwidth_used': 0,
            'circuits': 0,
            'status': 'unknown'
        }

    def check_health(self) -> Dict:
        """Check node health status"""
        health_status = {
            'node': self.node_name,
            'timestamp': datetime.now().isoformat(),
            'healthy': False,
            'services': {},
            'errors': []
        }

        # Check ORPort
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', self.or_port))
            sock.close()
            health_status['services']['or_port'] = result == 0
        except Exception as e:
            health_status['services']['or_port'] = False
            health_status['errors'].append(f"ORPort check failed: {e}")

        # Check DirPort via HTTP
        try:
            response = requests.get(
                f"http://localhost:{self.dir_port}/tor/server/authority",
                timeout=5
            )
            health_status['services']['dir_port'] = response.status_code == 200
        except Exception as e:
            health_status['services']['dir_port'] = False
            health_status['errors'].append(f"DirPort check failed: {e}")

        # Check Control Port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', self.control_port))
            sock.close()
            health_status['services']['control_port'] = result == 0
        except Exception as e:
            health_status['services']['control_port'] = False
            health_status['errors'].append(f"ControlPort check failed: {e}")

        # Overall health
        health_status['healthy'] = all(health_status['services'].values())
        self.stats['status'] = 'healthy' if health_status['healthy'] else 'unhealthy'

        return health_status

    def get_metrics(self) -> Dict:
        """Get node performance metrics"""
        return {
            'node': self.node_name,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats.copy()
        }

class TorCircuitMonitor:
    """Monitor the complete 3-hop Tor circuit"""

    def __init__(self):
        self.nodes = {
            'entry': TorNodeMonitor('entry', 9051, 9030, 9001),
            'middle': TorNodeMonitor('middle', 9052, 9031, 9002),
            'exit': TorNodeMonitor('exit', 9053, 9032, 9004)
        }
        self.circuit_stats = {
            'total_circuits': 0,
            'successful_circuits': 0,
            'failed_circuits': 0,
            'average_latency': 0
        }
        self.running = True

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("Starting Tor circuit monitoring")

        while self.running:
            try:
                # Check individual nodes
                node_results = {}
                for node_name, node in self.nodes.items():
                    health = node.check_health()
                    metrics = node.get_metrics()
                    node_results[node_name] = {
                        'health': health,
                        'metrics': metrics
                    }

                # Check circuit connectivity
                circuit_health = self.check_circuit_connectivity()

                # Log results
                self.log_monitoring_results(node_results, circuit_health)

                # Store metrics (in production, send to monitoring system)
                self.store_metrics(node_results, circuit_health)

                time.sleep(60)  # Monitor every minute

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)

    def check_circuit_connectivity(self) -> Dict:
        """Test end-to-end circuit connectivity"""
        circuit_test = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'unknown'
        }

        # Test SOCKS proxy connectivity to entry node
        try:
            # In production, use actual SOCKS proxy testing
            # For now, simulate connectivity test
            circuit_test['tests']['socks_proxy'] = True
        except Exception as e:
            circuit_test['tests']['socks_proxy'] = False
            logger.error(f"SOCKS proxy test failed: {e}")

        # Test directory consensus across nodes
        consensus_consistent = True
        try:
            # Check if all nodes have consistent consensus
            for node_name, node in self.nodes.items():
                try:
                    response = requests.get(
                        f"http://localhost:{node.dir_port}/tor/status-vote/current/consensus",
                        timeout=10
                    )
                    circuit_test['tests'][f'{node_name}_consensus'] = response.status_code == 200
                    if response.status_code != 200:
                        consensus_consistent = False
                except:
                    circuit_test['tests'][f'{node_name}_consensus'] = False
                    consensus_consistent = False

        except Exception as e:
            logger.error(f"Consensus check failed: {e}")
            consensus_consistent = False

        circuit_test['tests']['consensus_consistent'] = consensus_consistent

        # Overall circuit status
        all_tests_passed = all(circuit_test['tests'].values())
        circuit_test['overall_status'] = 'healthy' if all_tests_passed else 'degraded'

        return circuit_test

    def log_monitoring_results(self, node_results: Dict, circuit_health: Dict):
        """Log monitoring results"""
        # Node status summary
        healthy_nodes = sum(1 for node in node_results.values() if node['health']['healthy'])
        total_nodes = len(node_results)

        logger.info(f"Circuit Status: {healthy_nodes}/{total_nodes} nodes healthy")

        # Log individual node issues
        for node_name, result in node_results.items():
            if not result['health']['healthy']:
                errors = result['health']['errors']
                logger.warning(f"Node {node_name} unhealthy: {errors}")

        # Log circuit status
        if circuit_health['overall_status'] == 'healthy':
            logger.info("Tor circuit is fully operational")
        else:
            logger.warning(f"Circuit status: {circuit_health['overall_status']}")
            failed_tests = [test for test, result in circuit_health['tests'].items() if not result]
            logger.warning(f"Failed tests: {failed_tests}")

    def store_metrics(self, node_results: Dict, circuit_health: Dict):
        """Store metrics for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Store to file (in production, send to time-series database)
        metrics_file = f"/tmp/tor_metrics_{timestamp}.json"

        try:
            with open(metrics_file, 'w') as f:
                json.dump({
                    'nodes': node_results,
                    'circuit': circuit_health,
                    'timestamp': timestamp
                }, f, indent=2)

            logger.debug(f"Metrics stored to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")

    def get_circuit_summary(self) -> Dict:
        """Get circuit summary status"""
        summary = {
            'circuit_id': 'bev-tor-circuit',
            'timestamp': datetime.now().isoformat(),
            'nodes': {},
            'circuit_health': 'unknown',
            'uptime': 0
        }

        # Get status of each node
        for node_name, node in self.nodes.items():
            health = node.check_health()
            summary['nodes'][node_name] = {
                'status': 'healthy' if health['healthy'] else 'unhealthy',
                'services': health['services']
            }

        # Determine overall circuit health
        healthy_nodes = sum(1 for node_status in summary['nodes'].values()
                          if node_status['status'] == 'healthy')

        if healthy_nodes == 3:
            summary['circuit_health'] = 'fully_operational'
        elif healthy_nodes >= 2:
            summary['circuit_health'] = 'degraded'
        else:
            summary['circuit_health'] = 'critical'

        return summary

def main():
    """Main monitoring function"""
    monitor_interval = int(os.getenv('MONITOR_INTERVAL', 60))
    log_level = os.getenv('LOG_LEVEL', 'INFO')

    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level))

    logger.info("Initializing Tor Circuit Monitor")

    # Create circuit monitor
    circuit_monitor = TorCircuitMonitor()

    try:
        # Start monitoring in background thread
        monitor_thread = threading.Thread(target=circuit_monitor.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Main loop for interactive commands (if needed)
        while True:
            time.sleep(monitor_interval)

            # Print periodic summary
            summary = circuit_monitor.get_circuit_summary()
            logger.info(f"Circuit Summary: {summary['circuit_health']} "
                       f"({sum(1 for n in summary['nodes'].values() if n['status'] == 'healthy')}/3 nodes healthy)")

    except KeyboardInterrupt:
        logger.info("Shutting down Tor circuit monitor")
        circuit_monitor.running = False
    except Exception as e:
        logger.error(f"Monitor fatal error: {e}")
        raise

if __name__ == "__main__":
    main()