#!/usr/bin/env python3
"""
Guardian Security Enforcer - High-Availability Security Framework
Military-grade security enforcement with automatic threat response
"""

import asyncio
import json
import time
import hashlib
import socket
import struct
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import signal
import sys
import os

import redis
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP
import iptables
import psutil
import requests
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/guardian.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    BLOCK = "BLOCK"
    QUARANTINE = "QUARANTINE"
    MONITOR = "MONITOR"
    ALERT = "ALERT"

@dataclass
class ThreatEvent:
    timestamp: datetime
    source_ip: str
    destination_ip: str
    protocol: str
    port: int
    threat_type: str
    threat_level: ThreatLevel
    payload_hash: str
    action_taken: ActionType
    confidence: float

@dataclass
class GuardianNode:
    node_id: str
    ip_address: str
    port: int
    status: str
    last_heartbeat: datetime
    is_primary: bool

class GuardianCluster:
    """High-Availability Guardian Cluster Manager"""

    def __init__(self, node_id: str, redis_host: str = "localhost"):
        self.node_id = node_id
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        self.is_primary = False
        self.cluster_nodes: Dict[str, GuardianNode] = {}
        self.heartbeat_interval = 5
        self.failover_timeout = 15
        self.running = True

    async def register_node(self):
        """Register this node in the cluster"""
        node = GuardianNode(
            node_id=self.node_id,
            ip_address=socket.gethostbyname(socket.gethostname()),
            port=8080,
            status="ACTIVE",
            last_heartbeat=datetime.now(),
            is_primary=False
        )

        await self._set_node_data(node)
        logger.info(f"Node {self.node_id} registered in cluster")

    async def _set_node_data(self, node: GuardianNode):
        """Store node data in Redis"""
        key = f"guardian:node:{node.node_id}"
        data = asdict(node)
        data['last_heartbeat'] = node.last_heartbeat.isoformat()
        self.redis_client.hset(key, mapping=data)
        self.redis_client.expire(key, self.failover_timeout * 2)

    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                node = GuardianNode(
                    node_id=self.node_id,
                    ip_address=socket.gethostbyname(socket.gethostname()),
                    port=8080,
                    status="ACTIVE",
                    last_heartbeat=datetime.now(),
                    is_primary=self.is_primary
                )

                await self._set_node_data(node)
                await self._check_cluster_health()
                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.heartbeat_interval)

    async def _check_cluster_health(self):
        """Monitor cluster health and handle failover"""
        try:
            # Get all cluster nodes
            keys = self.redis_client.keys("guardian:node:*")
            active_nodes = []

            for key in keys:
                node_data = self.redis_client.hgetall(key)
                if node_data:
                    node = GuardianNode(
                        node_id=node_data['node_id'],
                        ip_address=node_data['ip_address'],
                        port=int(node_data['port']),
                        status=node_data['status'],
                        last_heartbeat=datetime.fromisoformat(node_data['last_heartbeat']),
                        is_primary=node_data.get('is_primary', 'False') == 'True'
                    )

                    # Check if node is still active
                    if (datetime.now() - node.last_heartbeat).seconds < self.failover_timeout:
                        active_nodes.append(node)

            self.cluster_nodes = {node.node_id: node for node in active_nodes}

            # Check if we need to elect a new primary
            primary_nodes = [n for n in active_nodes if n.is_primary]

            if not primary_nodes and active_nodes:
                # Elect this node as primary if it has the lowest node_id
                sorted_nodes = sorted(active_nodes, key=lambda x: x.node_id)
                if sorted_nodes[0].node_id == self.node_id:
                    await self._become_primary()

        except Exception as e:
            logger.error(f"Cluster health check error: {e}")

    async def _become_primary(self):
        """Promote this node to primary"""
        self.is_primary = True
        logger.info(f"Node {self.node_id} promoted to PRIMARY")

class ThreatDetector:
    """Advanced threat detection engine"""

    def __init__(self):
        self.known_threats: Set[str] = set()
        self.suspicious_ips: Dict[str, int] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self.load_threat_signatures()

    def load_threat_signatures(self):
        """Load known threat signatures"""
        # Common malicious payloads and patterns
        self.threat_patterns = [
            b'../../../etc/passwd',
            b'<script>alert',
            b'UNION SELECT',
            b'DROP TABLE',
            b'\x90\x90\x90\x90',  # NOP sled
            b'\xff\xff\xff\xff',  # Buffer overflow
            b'cmd.exe',
            b'/bin/sh',
            b'nc -l -p',  # Netcat reverse shell
        ]

        # Port scan patterns
        self.scan_ports = {21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995}

    def analyze_packet(self, packet) -> Optional[ThreatEvent]:
        """Analyze network packet for threats"""
        try:
            if not packet.haslayer(IP):
                return None

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

            # Check for port scanning
            if packet.haslayer(TCP):
                dst_port = packet[TCP].dport
                if self._is_port_scan(src_ip, dst_port):
                    return ThreatEvent(
                        timestamp=datetime.now(),
                        source_ip=src_ip,
                        destination_ip=dst_ip,
                        protocol="TCP",
                        port=dst_port,
                        threat_type="PORT_SCAN",
                        threat_level=ThreatLevel.MEDIUM,
                        payload_hash=self._hash_payload(bytes(packet)),
                        action_taken=ActionType.BLOCK,
                        confidence=0.8
                    )

            # Check for DDoS patterns
            if self._is_ddos_pattern(src_ip):
                return ThreatEvent(
                    timestamp=datetime.now(),
                    source_ip=src_ip,
                    destination_ip=dst_ip,
                    protocol=packet[IP].proto,
                    port=0,
                    threat_type="DDOS",
                    threat_level=ThreatLevel.HIGH,
                    payload_hash=self._hash_payload(bytes(packet)),
                    action_taken=ActionType.BLOCK,
                    confidence=0.9
                )

            # Check payload for malicious content
            if packet.haslayer(scapy.Raw):
                payload = bytes(packet[scapy.Raw].load)
                threat = self._analyze_payload(payload, src_ip, dst_ip, packet)
                if threat:
                    return threat

        except Exception as e:
            logger.error(f"Packet analysis error: {e}")

        return None

    def _is_port_scan(self, src_ip: str, dst_port: int) -> bool:
        """Detect port scanning behavior"""
        current_time = time.time()
        key = f"portscan:{src_ip}"

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old entries
        self.rate_limits[key] = [t for t in self.rate_limits[key] if current_time - t < 60]

        # Add current scan
        if dst_port in self.scan_ports:
            self.rate_limits[key].append(current_time)

        # Detect if too many ports scanned in short time
        return len(self.rate_limits[key]) > 10

    def _is_ddos_pattern(self, src_ip: str) -> bool:
        """Detect DDoS patterns"""
        current_time = time.time()
        key = f"ddos:{src_ip}"

        if key not in self.rate_limits:
            self.rate_limits[key] = []

        # Clean old entries (1 minute window)
        self.rate_limits[key] = [t for t in self.rate_limits[key] if current_time - t < 60]

        # Add current request
        self.rate_limits[key].append(current_time)

        # Detect if too many requests in short time
        return len(self.rate_limits[key]) > 100

    def _analyze_payload(self, payload: bytes, src_ip: str, dst_ip: str, packet) -> Optional[ThreatEvent]:
        """Analyze packet payload for threats"""
        for pattern in self.threat_patterns:
            if pattern in payload:
                threat_level = ThreatLevel.HIGH if pattern in [
                    b'../../../etc/passwd', b'UNION SELECT', b'DROP TABLE'
                ] else ThreatLevel.MEDIUM

                return ThreatEvent(
                    timestamp=datetime.now(),
                    source_ip=src_ip,
                    destination_ip=dst_ip,
                    protocol="TCP" if packet.haslayer(TCP) else "UDP",
                    port=packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport,
                    threat_type="MALICIOUS_PAYLOAD",
                    threat_level=threat_level,
                    payload_hash=self._hash_payload(payload),
                    action_taken=ActionType.BLOCK,
                    confidence=0.95
                )

        return None

    def _hash_payload(self, payload: bytes) -> str:
        """Generate SHA-256 hash of payload"""
        return hashlib.sha256(payload).hexdigest()

class SecurityEnforcer:
    """Security policy enforcement engine"""

    def __init__(self):
        self.blocked_ips: Set[str] = set()
        self.quarantine_ips: Set[str] = set()
        self.iptables_chain = "GUARDIAN_CHAIN"
        self._initialize_iptables()

    def _initialize_iptables(self):
        """Initialize iptables chain for security rules"""
        try:
            # Create custom chain if it doesn't exist
            import subprocess
            subprocess.run(['iptables', '-N', self.iptables_chain],
                         capture_output=True, check=False)
            subprocess.run(['iptables', '-I', 'INPUT', '-j', self.iptables_chain],
                         capture_output=True, check=False)
        except Exception as e:
            logger.error(f"Failed to initialize iptables: {e}")

    def enforce_action(self, threat: ThreatEvent) -> bool:
        """Enforce security action based on threat"""
        try:
            if threat.action_taken == ActionType.BLOCK:
                return self._block_ip(threat.source_ip)
            elif threat.action_taken == ActionType.QUARANTINE:
                return self._quarantine_ip(threat.source_ip)
            elif threat.action_taken == ActionType.ALERT:
                return self._send_alert(threat)
            return True
        except Exception as e:
            logger.error(f"Failed to enforce action: {e}")
            return False

    def _block_ip(self, ip_address: str) -> bool:
        """Block IP address using iptables"""
        if ip_address in self.blocked_ips:
            return True

        try:
            import subprocess
            result = subprocess.run([
                'iptables', '-A', self.iptables_chain,
                '-s', ip_address, '-j', 'DROP'
            ], capture_output=True, check=True)

            self.blocked_ips.add(ip_address)
            logger.info(f"Blocked IP: {ip_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to block IP {ip_address}: {e}")
            return False

    def _quarantine_ip(self, ip_address: str) -> bool:
        """Quarantine IP address with limited access"""
        if ip_address in self.quarantine_ips:
            return True

        try:
            import subprocess
            # Allow only essential ports (DNS, etc.)
            result = subprocess.run([
                'iptables', '-A', self.iptables_chain,
                '-s', ip_address, '-p', 'tcp', '--dport', '53', '-j', 'ACCEPT'
            ], capture_output=True, check=True)

            result = subprocess.run([
                'iptables', '-A', self.iptables_chain,
                '-s', ip_address, '-j', 'REJECT'
            ], capture_output=True, check=True)

            self.quarantine_ips.add(ip_address)
            logger.info(f"Quarantined IP: {ip_address}")
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine IP {ip_address}: {e}")
            return False

    def _send_alert(self, threat: ThreatEvent) -> bool:
        """Send security alert"""
        try:
            alert_data = {
                'timestamp': threat.timestamp.isoformat(),
                'threat_type': threat.threat_type,
                'source_ip': threat.source_ip,
                'threat_level': threat.threat_level.name,
                'confidence': threat.confidence
            }

            # In production, send to SIEM/monitoring system
            logger.warning(f"SECURITY ALERT: {json.dumps(alert_data)}")
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

class GuardianSecurityEnforcer:
    """Main Guardian Security Enforcer class"""

    def __init__(self, node_id: str = None):
        self.node_id = node_id or f"guardian-{int(time.time())}"
        self.cluster = GuardianCluster(self.node_id)
        self.threat_detector = ThreatDetector()
        self.security_enforcer = SecurityEnforcer()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.running = True

        # Performance metrics
        self.metrics = {
            'packets_processed': 0,
            'threats_detected': 0,
            'threats_blocked': 0,
            'start_time': time.time()
        }

    async def start(self):
        """Start the Guardian Security Enforcer"""
        logger.info(f"Starting Guardian Security Enforcer - Node: {self.node_id}")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Start cluster management
        await self.cluster.register_node()

        # Start background tasks
        tasks = [
            asyncio.create_task(self.cluster.heartbeat_loop()),
            asyncio.create_task(self._packet_capture_loop()),
            asyncio.create_task(self._metrics_reporter()),
            asyncio.create_task(self._health_check_server())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Guardian error: {e}")
        finally:
            await self._cleanup()

    async def _packet_capture_loop(self):
        """Capture and analyze network packets"""
        def packet_handler(packet):
            try:
                self.metrics['packets_processed'] += 1

                threat = self.threat_detector.analyze_packet(packet)
                if threat:
                    self.metrics['threats_detected'] += 1

                    # Only primary node enforces actions
                    if self.cluster.is_primary:
                        if self.security_enforcer.enforce_action(threat):
                            self.metrics['threats_blocked'] += 1

                        # Store threat data
                        self._store_threat_event(threat)

            except Exception as e:
                logger.error(f"Packet handler error: {e}")

        # Start packet capture
        try:
            scapy.sniff(iface="eth0", prn=packet_handler, store=0)
        except Exception as e:
            logger.error(f"Packet capture error: {e}")

    def _store_threat_event(self, threat: ThreatEvent):
        """Store threat event in Redis"""
        try:
            key = f"threat:{int(time.time())}"
            data = asdict(threat)
            data['timestamp'] = threat.timestamp.isoformat()
            data['threat_level'] = threat.threat_level.name
            data['action_taken'] = threat.action_taken.name

            self.redis_client.hset(key, mapping=data)
            self.redis_client.expire(key, 86400 * 7)  # Keep for 7 days
        except Exception as e:
            logger.error(f"Failed to store threat event: {e}")

    async def _metrics_reporter(self):
        """Report performance metrics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']
                pps = self.metrics['packets_processed'] / max(uptime, 1)

                metrics_data = {
                    'node_id': self.node_id,
                    'uptime': uptime,
                    'packets_per_second': pps,
                    'total_packets': self.metrics['packets_processed'],
                    'threats_detected': self.metrics['threats_detected'],
                    'threats_blocked': self.metrics['threats_blocked'],
                    'is_primary': self.cluster.is_primary
                }

                key = f"metrics:{self.node_id}"
                self.redis_client.hset(key, mapping=metrics_data)
                self.redis_client.expire(key, 300)  # 5 minutes

                logger.info(f"Metrics: {json.dumps(metrics_data)}")
                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")
                await asyncio.sleep(60)

    async def _health_check_server(self):
        """Simple HTTP health check server"""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import threading

        class HealthHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()

                    health_data = {
                        'status': 'healthy',
                        'node_id': self.server.guardian.node_id,
                        'uptime': time.time() - self.server.guardian.metrics['start_time'],
                        'is_primary': self.server.guardian.cluster.is_primary
                    }

                    self.wfile.write(json.dumps(health_data).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                # Suppress access logs
                pass

        def run_server():
            server = HTTPServer(('0.0.0.0', 8080), HealthHandler)
            server.guardian = self
            server.serve_forever()

        health_thread = threading.Thread(target=run_server, daemon=True)
        health_thread.start()

        # Keep the coroutine running
        while self.running:
            await asyncio.sleep(1)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Guardian Security Enforcer...")
        self.cluster.running = False

if __name__ == "__main__":
    node_id = os.environ.get('GUARDIAN_NODE_ID', f"guardian-{int(time.time())}")
    guardian = GuardianSecurityEnforcer(node_id)

    try:
        asyncio.run(guardian.start())
    except KeyboardInterrupt:
        logger.info("Guardian Security Enforcer stopped by user")
    except Exception as e:
        logger.error(f"Guardian fatal error: {e}")
        sys.exit(1)