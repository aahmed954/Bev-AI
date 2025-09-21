#!/usr/bin/env python3
"""
Advanced Defense Automation Engine
Real-time intrusion prevention, malware analysis, and automated response
"""

import asyncio
import json
import os
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import docker
import subprocess
import tempfile
import shutil
import psutil
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
import iptables
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import logging
from .security_framework import OperationalSecurityFramework

# Metrics
INTRUSION_ATTEMPTS = Counter('intrusion_attempts_total', 'Total intrusion attempts', ['source', 'type'])
BLOCKED_CONNECTIONS = Counter('blocked_connections_total', 'Total blocked connections', ['reason'])
MALWARE_DETECTIONS = Counter('malware_detections_total', 'Total malware detections', ['family'])
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time for security events')
ACTIVE_HONEYPOTS = Gauge('active_honeypots', 'Number of active honeypots')
SANDBOX_EXECUTIONS = Counter('sandbox_executions_total', 'Total sandbox executions', ['result'])

logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    """Threat level classifications"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BENIGN = "benign"

class ResponseAction(Enum):
    """Automated response actions"""
    BLOCK_IP = "block_ip"
    QUARANTINE_FILE = "quarantine_file"
    ISOLATE_HOST = "isolate_host"
    ALERT_SOC = "alert_soc"
    LOG_ONLY = "log_only"
    HONEYPOT_REDIRECT = "honeypot_redirect"

class SandboxResult(Enum):
    """Sandbox analysis results"""
    MALICIOUS = "malicious"
    SUSPICIOUS = "suspicious"
    BENIGN = "benign"
    ERROR = "error"
    TIMEOUT = "timeout"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    id: str
    timestamp: datetime
    event_type: str
    source_ip: str
    destination_ip: str
    source_port: int
    destination_port: int
    protocol: str
    payload: bytes
    threat_level: ThreatLevel
    confidence: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class MalwareAnalysis:
    """Malware analysis results"""
    file_hash: str
    file_name: str
    file_size: int
    file_type: str
    analysis_timestamp: datetime
    sandbox_result: SandboxResult
    threat_score: float
    family: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    network_behavior: Dict = field(default_factory=dict)
    file_operations: List[str] = field(default_factory=list)
    registry_operations: List[str] = field(default_factory=list)
    process_behavior: Dict = field(default_factory=dict)
    yara_matches: List[str] = field(default_factory=list)

@dataclass
class HoneypotConfig:
    """Honeypot configuration"""
    id: str
    name: str
    service_type: str
    port: int
    interface: str
    is_active: bool
    interaction_level: str  # low, medium, high
    logging_enabled: bool
    deception_techniques: List[str] = field(default_factory=list)

class MLIntrusionDetector:
    """Machine learning-based intrusion detection"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.feature_extractor = NetworkFeatureExtractor()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        try:
            # Initialize deep learning model for packet analysis
            self.model = IntrusionDetectionNN().to(self.device)

            # Load pre-trained weights if available
            model_path = "/opt/bev/models/intrusion_detection.pth"
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pre-trained intrusion detection model")
            else:
                logger.info("Using untrained intrusion detection model")

            self.model.eval()

        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")

    async def analyze_network_traffic(self, packet_data: List[Dict]) -> List[SecurityEvent]:
        """Analyze network traffic for intrusions"""
        try:
            security_events = []

            # Extract features
            features = []
            for packet in packet_data:
                feature_vector = self.feature_extractor.extract_features(packet)
                features.append(feature_vector)

            if not features:
                return security_events

            # Normalize features
            features_array = np.array(features)
            features_normalized = self.scaler.fit_transform(features_array)

            # Detect anomalies
            anomaly_predictions = self.anomaly_detector.fit_predict(features_normalized)

            # Deep learning analysis
            features_tensor = torch.FloatTensor(features_normalized).to(self.device)

            with torch.no_grad():
                predictions = self.model(features_tensor)
                threat_scores = F.softmax(predictions, dim=1)

            # Process predictions
            for i, (packet, anomaly, threat_score) in enumerate(
                zip(packet_data, anomaly_predictions, threat_scores)
            ):
                if anomaly == -1 or threat_score.max().item() > 0.7:
                    event = self._create_security_event(packet, threat_score.cpu().numpy())
                    security_events.append(event)

            return security_events

        except Exception as e:
            logger.error(f"Error analyzing network traffic: {e}")
            return []

    def _create_security_event(self, packet: Dict, threat_scores: np.ndarray) -> SecurityEvent:
        """Create security event from packet analysis"""
        # Determine threat level
        max_score = threat_scores.max()
        if max_score > 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif max_score > 0.7:
            threat_level = ThreatLevel.HIGH
        elif max_score > 0.5:
            threat_level = ThreatLevel.MEDIUM
        else:
            threat_level = ThreatLevel.LOW

        event = SecurityEvent(
            id=hashlib.sha256(str(packet).encode()).hexdigest()[:16],
            timestamp=datetime.now(),
            event_type="intrusion_attempt",
            source_ip=packet.get('src_ip', ''),
            destination_ip=packet.get('dst_ip', ''),
            source_port=packet.get('src_port', 0),
            destination_port=packet.get('dst_port', 0),
            protocol=packet.get('protocol', ''),
            payload=packet.get('payload', b''),
            threat_level=threat_level,
            confidence=float(max_score),
            metadata={
                'threat_scores': threat_scores.tolist(),
                'packet_size': packet.get('size', 0),
                'flags': packet.get('flags', [])
            }
        )

        return event

class IntrusionDetectionNN(nn.Module):
    """Neural network for intrusion detection"""

    def __init__(self, input_size=50, hidden_size=128, num_classes=6):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class NetworkFeatureExtractor:
    """Extract features from network packets"""

    def extract_features(self, packet: Dict) -> List[float]:
        """Extract numerical features from packet"""
        features = []

        # Basic packet features
        features.append(float(packet.get('size', 0)))
        features.append(float(packet.get('src_port', 0)))
        features.append(float(packet.get('dst_port', 0)))

        # Protocol encoding
        protocol = packet.get('protocol', '').lower()
        features.extend([
            1.0 if protocol == 'tcp' else 0.0,
            1.0 if protocol == 'udp' else 0.0,
            1.0 if protocol == 'icmp' else 0.0
        ])

        # TCP flags (if applicable)
        flags = packet.get('flags', [])
        tcp_flags = ['syn', 'ack', 'fin', 'rst', 'psh', 'urg']
        for flag in tcp_flags:
            features.append(1.0 if flag in flags else 0.0)

        # Payload analysis
        payload = packet.get('payload', b'')
        if payload:
            features.extend([
                len(payload),
                payload.count(b'\x00') / len(payload) if payload else 0.0,  # Null byte ratio
                len(set(payload)) / 256.0,  # Byte entropy approximation
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Time-based features
        hour = datetime.now().hour
        features.extend([
            np.sin(2 * np.pi * hour / 24),  # Hour of day (cyclical)
            np.cos(2 * np.pi * hour / 24)
        ])

        # Statistical features
        if len(payload) > 0:
            payload_array = np.frombuffer(payload, dtype=np.uint8)
            features.extend([
                float(np.mean(payload_array)),
                float(np.std(payload_array)),
                float(np.median(payload_array))
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Pad or truncate to fixed size
        target_size = 50
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        elif len(features) > target_size:
            features = features[:target_size]

        return features

class MalwareSandbox:
    """Automated malware detonation and analysis sandbox"""

    def __init__(self):
        self.docker_client = None
        self.sandbox_configs = {
            'windows': {
                'image': 'malware-sandbox-windows:latest',
                'timeout': 300,  # 5 minutes
                'memory_limit': '2g',
                'cpu_limit': 1
            },
            'linux': {
                'image': 'malware-sandbox-linux:latest',
                'timeout': 300,
                'memory_limit': '2g',
                'cpu_limit': 1
            }
        }
        self.yara_rules_path = "/opt/bev/yara_rules"
        self._initialize_sandbox()

    def _initialize_sandbox(self):
        """Initialize sandbox environment"""
        try:
            self.docker_client = docker.from_env()

            # Verify sandbox images exist
            for platform, config in self.sandbox_configs.items():
                try:
                    self.docker_client.images.get(config['image'])
                    logger.info(f"Sandbox image available: {config['image']}")
                except docker.errors.ImageNotFound:
                    logger.warning(f"Sandbox image not found: {config['image']}")

        except Exception as e:
            logger.error(f"Failed to initialize sandbox: {e}")

    async def analyze_file(self, file_path: str, file_hash: str) -> MalwareAnalysis:
        """Analyze file in sandbox"""
        try:
            # Get file info
            file_info = self._get_file_info(file_path)

            # Determine platform for analysis
            platform = self._determine_platform(file_path)

            # Run YARA scan first
            yara_matches = await self._run_yara_scan(file_path)

            # Static analysis
            static_analysis = await self._static_analysis(file_path)

            # Dynamic analysis in sandbox
            dynamic_analysis = await self._dynamic_analysis(file_path, platform)

            # Calculate threat score
            threat_score = self._calculate_threat_score(
                yara_matches, static_analysis, dynamic_analysis
            )

            # Determine sandbox result
            sandbox_result = self._determine_sandbox_result(threat_score, dynamic_analysis)

            analysis = MalwareAnalysis(
                file_hash=file_hash,
                file_name=os.path.basename(file_path),
                file_size=file_info['size'],
                file_type=file_info['type'],
                analysis_timestamp=datetime.now(),
                sandbox_result=sandbox_result,
                threat_score=threat_score,
                family=dynamic_analysis.get('family'),
                capabilities=dynamic_analysis.get('capabilities', []),
                network_behavior=dynamic_analysis.get('network', {}),
                file_operations=dynamic_analysis.get('file_ops', []),
                registry_operations=dynamic_analysis.get('registry_ops', []),
                process_behavior=dynamic_analysis.get('processes', {}),
                yara_matches=yara_matches
            )

            SANDBOX_EXECUTIONS.labels(result=sandbox_result.value).inc()

            if sandbox_result in [SandboxResult.MALICIOUS, SandboxResult.SUSPICIOUS]:
                family = analysis.family or 'unknown'
                MALWARE_DETECTIONS.labels(family=family).inc()

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return MalwareAnalysis(
                file_hash=file_hash,
                file_name=os.path.basename(file_path) if file_path else "unknown",
                file_size=0,
                file_type="unknown",
                analysis_timestamp=datetime.now(),
                sandbox_result=SandboxResult.ERROR,
                threat_score=0.0
            )

    def _get_file_info(self, file_path: str) -> Dict:
        """Get basic file information"""
        try:
            stat = os.stat(file_path)

            # Determine file type
            result = subprocess.run(['file', '-b', '--mime-type', file_path],
                                  capture_output=True, text=True)
            file_type = result.stdout.strip() if result.returncode == 0 else 'unknown'

            return {
                'size': stat.st_size,
                'type': file_type,
                'modified': datetime.fromtimestamp(stat.st_mtime)
            }
        except Exception as e:
            logger.error(f"Error getting file info: {e}")
            return {'size': 0, 'type': 'unknown', 'modified': datetime.now()}

    def _determine_platform(self, file_path: str) -> str:
        """Determine best platform for analysis"""
        try:
            result = subprocess.run(['file', '-b', file_path], capture_output=True, text=True)
            file_desc = result.stdout.lower()

            if 'pe32' in file_desc or 'ms-dos' in file_desc or '.exe' in file_desc:
                return 'windows'
            elif 'elf' in file_desc or 'linux' in file_desc:
                return 'linux'
            else:
                return 'linux'  # Default to Linux

        except Exception:
            return 'linux'

    async def _run_yara_scan(self, file_path: str) -> List[str]:
        """Run YARA rules against file"""
        try:
            if not os.path.exists(self.yara_rules_path):
                return []

            # Compile and run YARA rules
            result = subprocess.run([
                'yara', '-r', self.yara_rules_path, file_path
            ], capture_output=True, text=True)

            if result.returncode == 0:
                matches = [line.split()[0] for line in result.stdout.strip().split('\n') if line]
                return matches

            return []

        except Exception as e:
            logger.error(f"Error running YARA scan: {e}")
            return []

    async def _static_analysis(self, file_path: str) -> Dict:
        """Perform static analysis"""
        analysis = {
            'entropy': 0.0,
            'suspicious_strings': [],
            'imports': [],
            'sections': [],
            'metadata': {}
        }

        try:
            # Calculate entropy
            with open(file_path, 'rb') as f:
                data = f.read()
                if data:
                    analysis['entropy'] = self._calculate_entropy(data)

            # Extract strings
            result = subprocess.run(['strings', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                strings = result.stdout.split('\n')
                analysis['suspicious_strings'] = self._find_suspicious_strings(strings)

            return analysis

        except Exception as e:
            logger.error(f"Error in static analysis: {e}")
            return analysis

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0

        # Count byte frequencies
        frequencies = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        frequencies = frequencies / len(data)

        # Calculate entropy
        entropy = 0.0
        for freq in frequencies:
            if freq > 0:
                entropy -= freq * np.log2(freq)

        return entropy

    def _find_suspicious_strings(self, strings: List[str]) -> List[str]:
        """Find suspicious strings in file"""
        suspicious_patterns = [
            'shell32.dll', 'kernel32.dll', 'ntdll.dll',
            'CreateProcess', 'VirtualAlloc', 'WriteProcessMemory',
            'RegOpenKey', 'RegSetValue', 'RegDeleteKey',
            'InternetOpen', 'HttpSendRequest', 'URLDownloadToFile',
            'GetSystemDirectory', 'GetWindowsDirectory',
            'cmd.exe', 'powershell.exe', 'rundll32.exe'
        ]

        suspicious = []
        for string in strings:
            if len(string) > 4:  # Filter out short strings
                for pattern in suspicious_patterns:
                    if pattern.lower() in string.lower():
                        suspicious.append(string)
                        break

        return suspicious[:20]  # Limit to top 20

    async def _dynamic_analysis(self, file_path: str, platform: str) -> Dict:
        """Perform dynamic analysis in sandbox"""
        if not self.docker_client:
            return {'error': 'Docker not available'}

        config = self.sandbox_configs.get(platform, self.sandbox_configs['linux'])

        try:
            # Create temporary directory for analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy file to temp directory
                temp_file = os.path.join(temp_dir, 'sample')
                shutil.copy2(file_path, temp_file)

                # Run sandbox container
                container = self.docker_client.containers.run(
                    config['image'],
                    command=f'/analyze.sh /sample',
                    volumes={temp_dir: {'bind': '/workdir', 'mode': 'rw'}},
                    working_dir='/workdir',
                    mem_limit=config['memory_limit'],
                    cpu_period=100000,
                    cpu_quota=int(100000 * config['cpu_limit']),
                    network_mode='none',  # Isolated network
                    detach=True,
                    remove=True
                )

                # Wait for completion with timeout
                try:
                    result = container.wait(timeout=config['timeout'])
                    logs = container.logs().decode('utf-8')

                    # Parse analysis results
                    return self._parse_sandbox_results(logs)

                except Exception as e:
                    # Kill container on timeout
                    try:
                        container.kill()
                    except:
                        pass

                    return {'error': f'Sandbox timeout: {e}'}

        except Exception as e:
            logger.error(f"Error in dynamic analysis: {e}")
            return {'error': str(e)}

    def _parse_sandbox_results(self, logs: str) -> Dict:
        """Parse sandbox analysis results"""
        results = {
            'capabilities': [],
            'network': {},
            'file_ops': [],
            'registry_ops': [],
            'processes': {},
            'family': None
        }

        try:
            # Parse logs for analysis results
            lines = logs.split('\n')

            for line in lines:
                if line.startswith('CAPABILITY:'):
                    capability = line.split(':', 1)[1].strip()
                    results['capabilities'].append(capability)
                elif line.startswith('NETWORK:'):
                    network_data = line.split(':', 1)[1].strip()
                    # Parse network connections
                    results['network'][network_data] = True
                elif line.startswith('FILE:'):
                    file_op = line.split(':', 1)[1].strip()
                    results['file_ops'].append(file_op)
                elif line.startswith('REGISTRY:'):
                    reg_op = line.split(':', 1)[1].strip()
                    results['registry_ops'].append(reg_op)
                elif line.startswith('PROCESS:'):
                    proc_data = line.split(':', 1)[1].strip()
                    results['processes'][proc_data] = True
                elif line.startswith('FAMILY:'):
                    results['family'] = line.split(':', 1)[1].strip()

            return results

        except Exception as e:
            logger.error(f"Error parsing sandbox results: {e}")
            return results

    def _calculate_threat_score(self, yara_matches: List[str],
                               static_analysis: Dict, dynamic_analysis: Dict) -> float:
        """Calculate overall threat score"""
        score = 0.0

        # YARA matches contribute significantly
        if yara_matches:
            score += min(len(yara_matches) * 0.2, 0.8)

        # High entropy suggests packing/encryption
        entropy = static_analysis.get('entropy', 0.0)
        if entropy > 7.5:
            score += 0.3
        elif entropy > 6.5:
            score += 0.2

        # Suspicious strings
        suspicious_strings = static_analysis.get('suspicious_strings', [])
        if suspicious_strings:
            score += min(len(suspicious_strings) * 0.05, 0.3)

        # Dynamic analysis indicators
        capabilities = dynamic_analysis.get('capabilities', [])
        if capabilities:
            score += min(len(capabilities) * 0.1, 0.4)

        # Network activity
        if dynamic_analysis.get('network'):
            score += 0.2

        # File operations
        file_ops = dynamic_analysis.get('file_ops', [])
        if file_ops:
            score += min(len(file_ops) * 0.02, 0.2)

        return min(score, 1.0)

    def _determine_sandbox_result(self, threat_score: float, dynamic_analysis: Dict) -> SandboxResult:
        """Determine final sandbox result"""
        if 'error' in dynamic_analysis:
            return SandboxResult.ERROR

        if threat_score >= 0.8:
            return SandboxResult.MALICIOUS
        elif threat_score >= 0.5:
            return SandboxResult.SUSPICIOUS
        else:
            return SandboxResult.BENIGN

class NetworkTrafficManipulator:
    """Manipulate network traffic for containment and analysis"""

    def __init__(self):
        self.iptables_rules = []
        self.traffic_capture = None
        self.blocked_ips = set()
        self.redirected_ports = {}

    async def block_ip_address(self, ip_address: str, duration: Optional[int] = None) -> bool:
        """Block IP address using iptables"""
        try:
            # Add to iptables
            rule = f"-A INPUT -s {ip_address} -j DROP"
            result = subprocess.run(['iptables'] + rule.split()[1:], capture_output=True)

            if result.returncode == 0:
                self.iptables_rules.append(rule)
                self.blocked_ips.add(ip_address)

                # Schedule removal if duration specified
                if duration:
                    asyncio.create_task(self._unblock_ip_after_delay(ip_address, duration))

                BLOCKED_CONNECTIONS.labels(reason='ip_block').inc()
                logger.info(f"Blocked IP address: {ip_address}")
                return True
            else:
                logger.error(f"Failed to block IP {ip_address}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error blocking IP {ip_address}: {e}")
            return False

    async def _unblock_ip_after_delay(self, ip_address: str, delay: int):
        """Unblock IP after delay"""
        await asyncio.sleep(delay)
        await self.unblock_ip_address(ip_address)

    async def unblock_ip_address(self, ip_address: str) -> bool:
        """Unblock IP address"""
        try:
            # Remove from iptables
            rule = f"-D INPUT -s {ip_address} -j DROP"
            result = subprocess.run(['iptables'] + rule.split()[1:], capture_output=True)

            if result.returncode == 0:
                self.blocked_ips.discard(ip_address)
                logger.info(f"Unblocked IP address: {ip_address}")
                return True
            else:
                logger.warning(f"Failed to unblock IP {ip_address}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error unblocking IP {ip_address}: {e}")
            return False

    async def redirect_port_to_honeypot(self, port: int, honeypot_ip: str, honeypot_port: int) -> bool:
        """Redirect traffic to honeypot"""
        try:
            # Use iptables DNAT to redirect traffic
            rule = f"-t nat -A PREROUTING -p tcp --dport {port} -j DNAT --to-destination {honeypot_ip}:{honeypot_port}"
            result = subprocess.run(['iptables'] + rule.split()[1:], capture_output=True)

            if result.returncode == 0:
                self.redirected_ports[port] = (honeypot_ip, honeypot_port)
                logger.info(f"Redirected port {port} to honeypot {honeypot_ip}:{honeypot_port}")
                return True
            else:
                logger.error(f"Failed to redirect port {port}: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error redirecting port {port}: {e}")
            return False

    async def start_packet_capture(self, interface: str = "eth0", filter_expr: str = "") -> bool:
        """Start packet capture for analysis"""
        try:
            def packet_handler(packet):
                asyncio.create_task(self._process_captured_packet(packet))

            # Start capture in background
            self.traffic_capture = asyncio.create_task(
                self._capture_packets(interface, filter_expr, packet_handler)
            )

            logger.info(f"Started packet capture on {interface}")
            return True

        except Exception as e:
            logger.error(f"Error starting packet capture: {e}")
            return False

    async def _capture_packets(self, interface: str, filter_expr: str, handler):
        """Capture packets using scapy"""
        try:
            scapy.sniff(iface=interface, filter=filter_expr, prn=handler, store=0)
        except Exception as e:
            logger.error(f"Error in packet capture: {e}")

    async def _process_captured_packet(self, packet):
        """Process captured packet"""
        try:
            # Extract packet information
            packet_info = self._extract_packet_info(packet)

            # Analyze packet (this would integrate with ML intrusion detector)
            # For now, just log suspicious packets
            if self._is_suspicious_packet(packet_info):
                logger.warning(f"Suspicious packet detected: {packet_info}")

        except Exception as e:
            logger.error(f"Error processing packet: {e}")

    def _extract_packet_info(self, packet) -> Dict:
        """Extract information from packet"""
        info = {
            'timestamp': datetime.now(),
            'size': len(packet),
            'protocol': 'unknown',
            'src_ip': '',
            'dst_ip': '',
            'src_port': 0,
            'dst_port': 0,
            'flags': [],
            'payload': b''
        }

        try:
            if IP in packet:
                info['src_ip'] = packet[IP].src
                info['dst_ip'] = packet[IP].dst

                if TCP in packet:
                    info['protocol'] = 'tcp'
                    info['src_port'] = packet[TCP].sport
                    info['dst_port'] = packet[TCP].dport
                    info['flags'] = self._parse_tcp_flags(packet[TCP].flags)
                    if packet[TCP].payload:
                        info['payload'] = bytes(packet[TCP].payload)

                elif UDP in packet:
                    info['protocol'] = 'udp'
                    info['src_port'] = packet[UDP].sport
                    info['dst_port'] = packet[UDP].dport
                    if packet[UDP].payload:
                        info['payload'] = bytes(packet[UDP].payload)

                elif ICMP in packet:
                    info['protocol'] = 'icmp'

        except Exception as e:
            logger.error(f"Error extracting packet info: {e}")

        return info

    def _parse_tcp_flags(self, flags: int) -> List[str]:
        """Parse TCP flags"""
        flag_names = []
        if flags & 0x01: flag_names.append('fin')
        if flags & 0x02: flag_names.append('syn')
        if flags & 0x04: flag_names.append('rst')
        if flags & 0x08: flag_names.append('psh')
        if flags & 0x10: flag_names.append('ack')
        if flags & 0x20: flag_names.append('urg')
        return flag_names

    def _is_suspicious_packet(self, packet_info: Dict) -> bool:
        """Simple heuristics for suspicious packets"""
        # Check for common attack patterns
        dst_port = packet_info.get('dst_port', 0)
        src_ip = packet_info.get('src_ip', '')
        payload = packet_info.get('payload', b'')

        # Common exploit ports
        suspicious_ports = [135, 139, 445, 1433, 3389, 5985, 5986]
        if dst_port in suspicious_ports:
            return True

        # Known bad IPs
        if src_ip in self.blocked_ips:
            return True

        # Suspicious payload patterns
        if payload:
            suspicious_patterns = [b'cmd.exe', b'powershell', b'/bin/sh', b'wget', b'curl']
            for pattern in suspicious_patterns:
                if pattern in payload:
                    return True

        return False

class HoneypotManager:
    """Manage deception honeypots"""

    def __init__(self):
        self.honeypots: Dict[str, HoneypotConfig] = {}
        self.honeypot_processes = {}

    async def deploy_honeypot(self, config: HoneypotConfig) -> bool:
        """Deploy a honeypot service"""
        try:
            # Store configuration
            self.honeypots[config.id] = config

            # Start honeypot service based on type
            if config.service_type == 'ssh':
                success = await self._deploy_ssh_honeypot(config)
            elif config.service_type == 'http':
                success = await self._deploy_http_honeypot(config)
            elif config.service_type == 'ftp':
                success = await self._deploy_ftp_honeypot(config)
            elif config.service_type == 'telnet':
                success = await self._deploy_telnet_honeypot(config)
            else:
                logger.error(f"Unknown honeypot service type: {config.service_type}")
                return False

            if success:
                config.is_active = True
                ACTIVE_HONEYPOTS.inc()
                logger.info(f"Deployed honeypot: {config.name} on port {config.port}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Error deploying honeypot {config.name}: {e}")
            return False

    async def _deploy_ssh_honeypot(self, config: HoneypotConfig) -> bool:
        """Deploy SSH honeypot"""
        try:
            # Use cowrie SSH honeypot
            cmd = [
                'docker', 'run', '-d',
                '--name', f'honeypot_{config.id}',
                '-p', f'{config.port}:2222',
                '-v', '/opt/bev/honeypot/ssh:/cowrie/var',
                'cowrie/cowrie:latest'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                container_id = result.stdout.strip()
                self.honeypot_processes[config.id] = container_id
                return True
            else:
                logger.error(f"Failed to deploy SSH honeypot: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error deploying SSH honeypot: {e}")
            return False

    async def _deploy_http_honeypot(self, config: HoneypotConfig) -> bool:
        """Deploy HTTP honeypot"""
        try:
            # Simple HTTP honeypot using nginx
            cmd = [
                'docker', 'run', '-d',
                '--name', f'honeypot_{config.id}',
                '-p', f'{config.port}:80',
                '-v', '/opt/bev/honeypot/http:/usr/share/nginx/html',
                'nginx:alpine'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                container_id = result.stdout.strip()
                self.honeypot_processes[config.id] = container_id
                return True
            else:
                logger.error(f"Failed to deploy HTTP honeypot: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error deploying HTTP honeypot: {e}")
            return False

    async def _deploy_ftp_honeypot(self, config: HoneypotConfig) -> bool:
        """Deploy FTP honeypot"""
        # Placeholder implementation
        return True

    async def _deploy_telnet_honeypot(self, config: HoneypotConfig) -> bool:
        """Deploy Telnet honeypot"""
        # Placeholder implementation
        return True

    async def shutdown_honeypot(self, honeypot_id: str) -> bool:
        """Shutdown a honeypot"""
        try:
            if honeypot_id not in self.honeypots:
                return False

            # Stop container if exists
            if honeypot_id in self.honeypot_processes:
                container_id = self.honeypot_processes[honeypot_id]

                result = subprocess.run(['docker', 'stop', container_id], capture_output=True)
                if result.returncode == 0:
                    subprocess.run(['docker', 'rm', container_id], capture_output=True)

                del self.honeypot_processes[honeypot_id]

            # Update configuration
            self.honeypots[honeypot_id].is_active = False
            ACTIVE_HONEYPOTS.dec()

            logger.info(f"Shutdown honeypot: {honeypot_id}")
            return True

        except Exception as e:
            logger.error(f"Error shutting down honeypot {honeypot_id}: {e}")
            return False

    async def get_honeypot_logs(self, honeypot_id: str) -> List[Dict]:
        """Get logs from honeypot"""
        try:
            if honeypot_id not in self.honeypot_processes:
                return []

            container_id = self.honeypot_processes[honeypot_id]

            result = subprocess.run(['docker', 'logs', container_id], capture_output=True, text=True)

            if result.returncode == 0:
                logs = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logs.append({
                            'timestamp': datetime.now().isoformat(),
                            'message': line.strip(),
                            'honeypot_id': honeypot_id
                        })
                return logs

            return []

        except Exception as e:
            logger.error(f"Error getting honeypot logs: {e}")
            return []

class DefenseAutomationEngine:
    """Main defense automation orchestrator"""

    def __init__(self, security_framework: OperationalSecurityFramework):
        self.security_framework = security_framework
        self.ml_detector = MLIntrusionDetector()
        self.malware_sandbox = MalwareSandbox()
        self.traffic_manipulator = NetworkTrafficManipulator()
        self.honeypot_manager = HoneypotManager()

        self.db_pool = None
        self.redis_client = None

        self.response_rules = {
            ThreatLevel.CRITICAL: [ResponseAction.BLOCK_IP, ResponseAction.ISOLATE_HOST, ResponseAction.ALERT_SOC],
            ThreatLevel.HIGH: [ResponseAction.BLOCK_IP, ResponseAction.ALERT_SOC],
            ThreatLevel.MEDIUM: [ResponseAction.HONEYPOT_REDIRECT, ResponseAction.LOG_ONLY],
            ThreatLevel.LOW: [ResponseAction.LOG_ONLY]
        }

    async def initialize(self, redis_url: str = "redis://localhost:6379",
                        db_url: str = "postgresql://user:pass@localhost/bev"):
        """Initialize the defense automation engine"""
        try:
            # Initialize database connections
            self.redis_client = redis.from_url(redis_url)
            self.db_pool = await asyncpg.create_pool(db_url)

            # Deploy default honeypots
            await self._deploy_default_honeypots()

            # Start network monitoring
            await self.traffic_manipulator.start_packet_capture()

            logger.info("Defense Automation Engine initialized")
            print("🛡️ Defense Automation Engine Ready")

        except Exception as e:
            logger.error(f"Failed to initialize defense automation engine: {e}")
            raise

    async def _deploy_default_honeypots(self):
        """Deploy default honeypot configuration"""
        default_honeypots = [
            HoneypotConfig(
                id="ssh_honeypot_001",
                name="SSH Honeypot",
                service_type="ssh",
                port=2222,
                interface="eth0",
                is_active=False,
                interaction_level="medium",
                logging_enabled=True,
                deception_techniques=["fake_filesystem", "fake_users"]
            ),
            HoneypotConfig(
                id="http_honeypot_001",
                name="HTTP Honeypot",
                service_type="http",
                port=8080,
                interface="eth0",
                is_active=False,
                interaction_level="low",
                logging_enabled=True,
                deception_techniques=["fake_webapp", "tracking_pixels"]
            )
        ]

        for config in default_honeypots:
            await self.honeypot_manager.deploy_honeypot(config)

    async def process_security_event(self, event: SecurityEvent) -> List[ResponseAction]:
        """Process security event and execute automated response"""
        try:
            with RESPONSE_TIME.time():
                # Log the event
                await self._log_security_event(event)

                # Record metrics
                INTRUSION_ATTEMPTS.labels(
                    source=event.source_ip,
                    type=event.event_type
                ).inc()

                # Determine response actions
                actions = self.response_rules.get(event.threat_level, [ResponseAction.LOG_ONLY])

                # Execute response actions
                executed_actions = []
                for action in actions:
                    success = await self._execute_response_action(action, event)
                    if success:
                        executed_actions.append(action)

                logger.info(f"Processed security event {event.id}, executed {len(executed_actions)} actions")
                return executed_actions

        except Exception as e:
            logger.error(f"Error processing security event {event.id}: {e}")
            return []

    async def _execute_response_action(self, action: ResponseAction, event: SecurityEvent) -> bool:
        """Execute a specific response action"""
        try:
            if action == ResponseAction.BLOCK_IP:
                return await self.traffic_manipulator.block_ip_address(
                    event.source_ip, duration=3600  # 1 hour
                )

            elif action == ResponseAction.QUARANTINE_FILE:
                # This would be implemented for file-based events
                return True

            elif action == ResponseAction.ISOLATE_HOST:
                # Implement host isolation
                return await self._isolate_host(event.source_ip)

            elif action == ResponseAction.ALERT_SOC:
                return await self._alert_soc(event)

            elif action == ResponseAction.HONEYPOT_REDIRECT:
                # Redirect to appropriate honeypot
                return await self._redirect_to_honeypot(event)

            elif action == ResponseAction.LOG_ONLY:
                return True  # Already logged

            return False

        except Exception as e:
            logger.error(f"Error executing response action {action}: {e}")
            return False

    async def _isolate_host(self, ip_address: str) -> bool:
        """Isolate host from network"""
        try:
            # Block all traffic to/from host
            rules = [
                f"-A INPUT -s {ip_address} -j DROP",
                f"-A OUTPUT -d {ip_address} -j DROP",
                f"-A FORWARD -s {ip_address} -j DROP",
                f"-A FORWARD -d {ip_address} -j DROP"
            ]

            for rule in rules:
                subprocess.run(['iptables'] + rule.split()[1:], capture_output=True)

            logger.info(f"Isolated host: {ip_address}")
            return True

        except Exception as e:
            logger.error(f"Error isolating host {ip_address}: {e}")
            return False

    async def _alert_soc(self, event: SecurityEvent) -> bool:
        """Send alert to Security Operations Center"""
        try:
            alert = {
                'event_id': event.id,
                'timestamp': event.timestamp.isoformat(),
                'threat_level': event.threat_level.value,
                'source_ip': event.source_ip,
                'destination_ip': event.destination_ip,
                'event_type': event.event_type,
                'confidence': event.confidence,
                'metadata': event.metadata
            }

            # Store in database
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO soc_alerts (event_id, alert_data, created_at)
                        VALUES ($1, $2, $3)
                    """, event.id, json.dumps(alert), datetime.now())

            # Send to alerting system (webhook, email, etc.)
            # This would integrate with your SOC platform

            logger.info(f"SOC alert sent for event {event.id}")
            return True

        except Exception as e:
            logger.error(f"Error sending SOC alert: {e}")
            return False

    async def _redirect_to_honeypot(self, event: SecurityEvent) -> bool:
        """Redirect traffic to appropriate honeypot"""
        try:
            # Find appropriate honeypot based on destination port
            target_port = event.destination_port

            # Map common ports to honeypot services
            port_mapping = {
                22: 'ssh',
                80: 'http',
                443: 'http',
                21: 'ftp',
                23: 'telnet'
            }

            service_type = port_mapping.get(target_port)
            if not service_type:
                return False

            # Find active honeypot of this type
            for honeypot in self.honeypot_manager.honeypots.values():
                if honeypot.service_type == service_type and honeypot.is_active:
                    # Redirect traffic to honeypot
                    return await self.traffic_manipulator.redirect_port_to_honeypot(
                        target_port, "127.0.0.1", honeypot.port
                    )

            return False

        except Exception as e:
            logger.error(f"Error redirecting to honeypot: {e}")
            return False

    async def _log_security_event(self, event: SecurityEvent):
        """Log security event to database"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO security_events (
                            id, timestamp, event_type, source_ip, destination_ip,
                            source_port, destination_port, protocol, threat_level,
                            confidence, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    event.id, event.timestamp, event.event_type, event.source_ip,
                    event.destination_ip, event.source_port, event.destination_port,
                    event.protocol, event.threat_level.value, event.confidence,
                    json.dumps(event.metadata)
                    )

        except Exception as e:
            logger.error(f"Error logging security event: {e}")

    async def analyze_file_sample(self, file_path: str) -> MalwareAnalysis:
        """Analyze suspicious file in sandbox"""
        try:
            # Calculate file hash
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            # Run sandbox analysis
            analysis = await self.malware_sandbox.analyze_file(file_path, file_hash)

            # Store results
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO malware_analysis (
                            file_hash, file_name, file_size, file_type,
                            analysis_timestamp, sandbox_result, threat_score,
                            family, capabilities, yara_matches
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (file_hash) DO UPDATE SET
                            analysis_timestamp = $5,
                            threat_score = $7
                    """,
                    analysis.file_hash, analysis.file_name, analysis.file_size,
                    analysis.file_type, analysis.analysis_timestamp,
                    analysis.sandbox_result.value, analysis.threat_score,
                    analysis.family, json.dumps(analysis.capabilities),
                    json.dumps(analysis.yara_matches)
                    )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            raise

    async def get_defense_status(self) -> Dict:
        """Get current defense system status"""
        try:
            return {
                'active_honeypots': len([h for h in self.honeypot_manager.honeypots.values() if h.is_active]),
                'blocked_ips': len(self.traffic_manipulator.blocked_ips),
                'redirected_ports': len(self.traffic_manipulator.redirected_ports),
                'system_load': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            }

        except Exception as e:
            logger.error(f"Error getting defense status: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Shutdown defense automation engine"""
        try:
            # Shutdown all honeypots
            for honeypot_id in list(self.honeypot_manager.honeypots.keys()):
                await self.honeypot_manager.shutdown_honeypot(honeypot_id)

            # Stop traffic capture
            if self.traffic_manipulator.traffic_capture:
                self.traffic_manipulator.traffic_capture.cancel()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            if self.redis_client:
                await self.redis_client.close()

            logger.info("Defense Automation Engine shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage
async def main():
    """Example usage of the Defense Automation Engine"""
    try:
        # Initialize security framework
        security = OperationalSecurityFramework()
        await security.initialize_security()

        # Initialize defense automation engine
        engine = DefenseAutomationEngine(security)
        await engine.initialize()

        # Example security event
        event = SecurityEvent(
            id="evt_001",
            timestamp=datetime.now(),
            event_type="port_scan",
            source_ip="192.168.1.100",
            destination_ip="10.0.0.1",
            source_port=12345,
            destination_port=22,
            protocol="tcp",
            payload=b'',
            threat_level=ThreatLevel.MEDIUM,
            confidence=0.8
        )

        # Process security event
        actions = await engine.process_security_event(event)
        print(f"✅ Executed response actions: {[a.value for a in actions]}")

        # Get defense status
        status = await engine.get_defense_status()
        print(f"📊 Defense Status: {status}")

        # Shutdown
        await engine.shutdown()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())