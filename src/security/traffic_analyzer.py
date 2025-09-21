#!/usr/bin/env python3
"""
Real-time Network Traffic Analyzer
Advanced network analysis with InfluxDB integration and anomaly detection
"""

import asyncio
import json
import time
import struct
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import logging
import threading
import geoip2.database
import geoip2.errors

# Network analysis libraries
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.dns import DNS, DNSQR
import dpkt
import psutil

# Data storage and processing
import redis
import pandas as pd
import numpy as np
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Logging
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/app/logs/traffic_analyzer.log", rotation="100 MB", retention="7 days")

@dataclass
class ConnectionInfo:
    """Network connection information"""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: datetime
    last_seen: datetime
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    flags: Set[str]
    duration: float

@dataclass
class TrafficStats:
    """Traffic statistics for analysis"""
    timestamp: datetime
    total_packets: int
    total_bytes: int
    tcp_packets: int
    udp_packets: int
    icmp_packets: int
    unique_sources: int
    unique_destinations: int
    top_protocols: Dict[str, int]
    top_ports: Dict[int, int]
    bandwidth_mbps: float
    packet_rate: float

@dataclass
class GeoLocation:
    """Geographic location information"""
    country: str
    city: str
    latitude: float
    longitude: float
    asn: Optional[int]
    organization: Optional[str]

@dataclass
class TrafficAnomaly:
    """Traffic anomaly detection result"""
    timestamp: datetime
    anomaly_type: str
    source_ip: str
    destination_ip: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    metrics: Dict[str, float]
    confidence: float

class GeoIPAnalyzer:
    """Geographic IP analysis"""

    def __init__(self, geoip_db_path: str = "/app/geoip/GeoLite2-City.mmdb"):
        self.geoip_db_path = geoip_db_path
        self.reader = None
        self._initialize_geoip()

    def _initialize_geoip(self):
        """Initialize GeoIP database"""
        try:
            self.reader = geoip2.database.Reader(self.geoip_db_path)
            logger.info("GeoIP database initialized")
        except Exception as e:
            logger.warning(f"GeoIP database not available: {e}")

    def get_location(self, ip_address: str) -> Optional[GeoLocation]:
        """Get geographic location for IP address"""
        if not self.reader:
            return None

        try:
            response = self.reader.city(ip_address)

            return GeoLocation(
                country=response.country.name or "Unknown",
                city=response.city.name or "Unknown",
                latitude=float(response.location.latitude or 0.0),
                longitude=float(response.location.longitude or 0.0),
                asn=response.traits.autonomous_system_number,
                organization=response.traits.autonomous_system_organization
            )

        except geoip2.errors.AddressNotFoundError:
            return None
        except Exception as e:
            logger.error(f"GeoIP lookup error for {ip_address}: {e}")
            return None

class ConnectionTracker:
    """Track network connections and their statistics"""

    def __init__(self, connection_timeout: int = 300):
        self.connections: Dict[str, ConnectionInfo] = {}
        self.connection_timeout = connection_timeout
        self.last_cleanup = time.time()

    def get_connection_key(self, packet) -> Optional[str]:
        """Generate connection key from packet"""
        try:
            if not packet.haslayer(IP):
                return None

            src_ip = packet[IP].src
            dst_ip = packet[IP].dst

            if packet.haslayer(TCP):
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                protocol = "TCP"
            elif packet.haslayer(UDP):
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport
                protocol = "UDP"
            else:
                src_port = 0
                dst_port = 0
                protocol = "ICMP"

            return f"{src_ip}:{src_port}->{dst_ip}:{dst_port}:{protocol}"

        except Exception as e:
            logger.error(f"Connection key generation error: {e}")
            return None

    def update_connection(self, packet):
        """Update connection information from packet"""
        try:
            conn_key = self.get_connection_key(packet)
            if not conn_key:
                return

            current_time = datetime.now()
            packet_size = len(packet)

            if conn_key not in self.connections:
                # New connection
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst

                if packet.haslayer(TCP):
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    protocol = "TCP"
                    flags = self._get_tcp_flags(packet[TCP])
                elif packet.haslayer(UDP):
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport
                    protocol = "UDP"
                    flags = set()
                else:
                    src_port = 0
                    dst_port = 0
                    protocol = "ICMP"
                    flags = set()

                self.connections[conn_key] = ConnectionInfo(
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=src_port,
                    dst_port=dst_port,
                    protocol=protocol,
                    start_time=current_time,
                    last_seen=current_time,
                    bytes_sent=packet_size,
                    bytes_received=0,
                    packets_sent=1,
                    packets_received=0,
                    flags=flags,
                    duration=0.0
                )
            else:
                # Update existing connection
                conn = self.connections[conn_key]
                conn.last_seen = current_time
                conn.duration = (current_time - conn.start_time).total_seconds()

                # Determine direction and update accordingly
                if packet[IP].src == conn.src_ip:
                    conn.bytes_sent += packet_size
                    conn.packets_sent += 1
                else:
                    conn.bytes_received += packet_size
                    conn.packets_received += 1

                # Update TCP flags if applicable
                if packet.haslayer(TCP):
                    conn.flags.update(self._get_tcp_flags(packet[TCP]))

            # Periodic cleanup
            if current_time.timestamp() - self.last_cleanup > 60:  # Every minute
                self._cleanup_old_connections()

        except Exception as e:
            logger.error(f"Connection update error: {e}")

    def _get_tcp_flags(self, tcp_layer) -> Set[str]:
        """Extract TCP flags"""
        flags = set()
        tcp_flags = tcp_layer.flags

        if tcp_flags & 0x01: flags.add("FIN")
        if tcp_flags & 0x02: flags.add("SYN")
        if tcp_flags & 0x04: flags.add("RST")
        if tcp_flags & 0x08: flags.add("PSH")
        if tcp_flags & 0x10: flags.add("ACK")
        if tcp_flags & 0x20: flags.add("URG")

        return flags

    def _cleanup_old_connections(self):
        """Remove old/expired connections"""
        current_time = datetime.now()
        expired_keys = []

        for key, conn in self.connections.items():
            if (current_time - conn.last_seen).seconds > self.connection_timeout:
                expired_keys.append(key)

        for key in expired_keys:
            del self.connections[key]

        self.last_cleanup = current_time.timestamp()
        logger.debug(f"Cleaned up {len(expired_keys)} expired connections")

    def get_active_connections(self) -> List[ConnectionInfo]:
        """Get list of currently active connections"""
        return list(self.connections.values())

class AnomalyDetector:
    """Detect traffic anomalies"""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.traffic_history: deque = deque(maxlen=1440)  # 24 hours of minutes
        self.ip_rate_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=300))  # 5 minutes
        self.port_scan_tracker: Dict[str, Set[int]] = defaultdict(set)
        self.baseline_stats = None

    def analyze_traffic_stats(self, stats: TrafficStats) -> List[TrafficAnomaly]:
        """Analyze traffic statistics for anomalies"""
        anomalies = []
        current_time = datetime.now()

        # Store current stats
        self.traffic_history.append(stats)

        # Update baseline after sufficient data
        if len(self.traffic_history) > 60:  # 1 hour minimum
            self._update_baseline()

        if self.baseline_stats:
            # Check for volume anomalies
            anomalies.extend(self._detect_volume_anomalies(stats, current_time))

            # Check for rate anomalies
            anomalies.extend(self._detect_rate_anomalies(stats, current_time))

        return anomalies

    def analyze_connection(self, conn: ConnectionInfo) -> List[TrafficAnomaly]:
        """Analyze individual connection for anomalies"""
        anomalies = []
        current_time = datetime.now()

        # Check for high-rate connections
        if conn.packet_rate > 1000:  # packets per second
            anomalies.append(TrafficAnomaly(
                timestamp=current_time,
                anomaly_type="HIGH_PACKET_RATE",
                source_ip=conn.src_ip,
                destination_ip=conn.dst_ip,
                severity="HIGH",
                description=f"High packet rate: {conn.packet_rate:.2f} pps",
                metrics={"packet_rate": conn.packet_rate},
                confidence=0.8
            ))

        # Check for unusual port connections
        if self._is_unusual_port(conn.dst_port):
            anomalies.append(TrafficAnomaly(
                timestamp=current_time,
                anomaly_type="UNUSUAL_PORT",
                source_ip=conn.src_ip,
                destination_ip=conn.dst_ip,
                severity="MEDIUM",
                description=f"Connection to unusual port: {conn.dst_port}",
                metrics={"destination_port": conn.dst_port},
                confidence=0.6
            ))

        return anomalies

    def detect_port_scan(self, src_ip: str, dst_port: int) -> Optional[TrafficAnomaly]:
        """Detect port scanning behavior"""
        self.port_scan_tracker[src_ip].add(dst_port)

        # Clean old entries (ports accessed in last 5 minutes)
        # In a real implementation, you'd track timestamps
        unique_ports = len(self.port_scan_tracker[src_ip])

        if unique_ports > 20:  # Threshold for port scan
            return TrafficAnomaly(
                timestamp=datetime.now(),
                anomaly_type="PORT_SCAN",
                source_ip=src_ip,
                destination_ip="multiple",
                severity="HIGH",
                description=f"Port scan detected: {unique_ports} unique ports",
                metrics={"unique_ports_scanned": unique_ports},
                confidence=0.9
            )

        return None

    def detect_ddos_pattern(self, src_ip: str) -> Optional[TrafficAnomaly]:
        """Detect DDoS patterns"""
        current_time = time.time()
        rate_tracker = self.ip_rate_tracker[src_ip]

        # Clean old entries (5 minute window)
        while rate_tracker and current_time - rate_tracker[0] > 300:
            rate_tracker.popleft()

        rate_tracker.append(current_time)

        # Check request rate
        requests_per_minute = len(rate_tracker) / 5.0  # 5-minute window

        if requests_per_minute > 100:  # Threshold
            return TrafficAnomaly(
                timestamp=datetime.now(),
                anomaly_type="DDOS_PATTERN",
                source_ip=src_ip,
                destination_ip="multiple",
                severity="CRITICAL",
                description=f"Potential DDoS: {requests_per_minute:.1f} req/min",
                metrics={"requests_per_minute": requests_per_minute},
                confidence=0.85
            )

        return None

    def _update_baseline(self):
        """Update baseline traffic statistics"""
        if len(self.traffic_history) < 60:
            return

        recent_stats = list(self.traffic_history)[-60:]  # Last hour

        self.baseline_stats = {
            'avg_packets': statistics.mean([s.total_packets for s in recent_stats]),
            'std_packets': statistics.stdev([s.total_packets for s in recent_stats]),
            'avg_bytes': statistics.mean([s.total_bytes for s in recent_stats]),
            'std_bytes': statistics.stdev([s.total_bytes for s in recent_stats]),
            'avg_bandwidth': statistics.mean([s.bandwidth_mbps for s in recent_stats]),
            'std_bandwidth': statistics.stdev([s.bandwidth_mbps for s in recent_stats]),
        }

    def _detect_volume_anomalies(self, stats: TrafficStats, timestamp: datetime) -> List[TrafficAnomaly]:
        """Detect volume-based anomalies"""
        anomalies = []

        # Packet volume anomaly
        if stats.total_packets > self.baseline_stats['avg_packets'] + 3 * self.baseline_stats['std_packets']:
            anomalies.append(TrafficAnomaly(
                timestamp=timestamp,
                anomaly_type="HIGH_PACKET_VOLUME",
                source_ip="multiple",
                destination_ip="multiple",
                severity="HIGH",
                description=f"High packet volume: {stats.total_packets} packets",
                metrics={"packet_volume": stats.total_packets},
                confidence=0.85
            ))

        # Bandwidth anomaly
        if stats.bandwidth_mbps > self.baseline_stats['avg_bandwidth'] + 3 * self.baseline_stats['std_bandwidth']:
            anomalies.append(TrafficAnomaly(
                timestamp=timestamp,
                anomaly_type="HIGH_BANDWIDTH",
                source_ip="multiple",
                destination_ip="multiple",
                severity="HIGH",
                description=f"High bandwidth usage: {stats.bandwidth_mbps:.2f} Mbps",
                metrics={"bandwidth_mbps": stats.bandwidth_mbps},
                confidence=0.8
            ))

        return anomalies

    def _detect_rate_anomalies(self, stats: TrafficStats, timestamp: datetime) -> List[TrafficAnomaly]:
        """Detect rate-based anomalies"""
        anomalies = []

        # High packet rate
        if stats.packet_rate > 10000:  # packets per second
            anomalies.append(TrafficAnomaly(
                timestamp=timestamp,
                anomaly_type="HIGH_PACKET_RATE",
                source_ip="multiple",
                destination_ip="multiple",
                severity="MEDIUM",
                description=f"High packet rate: {stats.packet_rate:.2f} pps",
                metrics={"packet_rate": stats.packet_rate},
                confidence=0.7
            ))

        return anomalies

    def _is_unusual_port(self, port: int) -> bool:
        """Check if port is unusual/suspicious"""
        common_ports = {21, 22, 23, 25, 53, 67, 68, 69, 80, 110, 123, 143, 161, 389, 443, 993, 995}
        high_ports = range(49152, 65536)  # Dynamic/private ports

        return port not in common_ports and port not in high_ports

class TrafficAnalyzer:
    """Main traffic analyzer class"""

    def __init__(self):
        self.connection_tracker = ConnectionTracker()
        self.anomaly_detector = AnomalyDetector()
        self.geoip_analyzer = GeoIPAnalyzer()

        # Data storage
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="your-influxdb-token",
            org="bev-security"
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # Statistics tracking
        self.stats_window = deque(maxlen=60)  # 1-minute stats
        self.last_stats_time = time.time()

        self.running = True
        self.metrics = {
            'packets_processed': 0,
            'connections_tracked': 0,
            'anomalies_detected': 0,
            'start_time': time.time()
        }

    async def start(self):
        """Start the traffic analyzer"""
        logger.info("Starting Real-time Traffic Analyzer")

        # Start analysis tasks
        tasks = [
            asyncio.create_task(self._packet_capture_loop()),
            asyncio.create_task(self._statistics_reporter()),
            asyncio.create_task(self._anomaly_analysis_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Traffic analyzer error: {e}")

    async def _packet_capture_loop(self):
        """Capture and process network packets"""
        def packet_handler(packet):
            try:
                self.metrics['packets_processed'] += 1

                # Update connection tracking
                self.connection_tracker.update_connection(packet)

                # Check for immediate anomalies
                if packet.haslayer(IP):
                    src_ip = packet[IP].src

                    # Port scan detection
                    if packet.haslayer(TCP):
                        dst_port = packet[TCP].dport
                        port_scan_anomaly = self.anomaly_detector.detect_port_scan(src_ip, dst_port)
                        if port_scan_anomaly:
                            asyncio.create_task(self._handle_anomaly(port_scan_anomaly))

                    # DDoS pattern detection
                    ddos_anomaly = self.anomaly_detector.detect_ddos_pattern(src_ip)
                    if ddos_anomaly:
                        asyncio.create_task(self._handle_anomaly(ddos_anomaly))

                # Update statistics
                self._update_statistics(packet)

            except Exception as e:
                logger.error(f"Packet processing error: {e}")

        # Start packet capture
        try:
            scapy.sniff(iface="eth0", prn=packet_handler, store=0)
        except Exception as e:
            logger.error(f"Packet capture failed: {e}")

    def _update_statistics(self, packet):
        """Update traffic statistics"""
        current_time = time.time()

        # Initialize stats if needed
        if not hasattr(self, '_current_stats'):
            self._reset_current_stats()

        # Update packet counts
        self._current_stats['total_packets'] += 1
        self._current_stats['total_bytes'] += len(packet)

        # Protocol breakdown
        if packet.haslayer(TCP):
            self._current_stats['tcp_packets'] += 1
            port = packet[TCP].dport
            self._current_stats['top_ports'][port] += 1
        elif packet.haslayer(UDP):
            self._current_stats['udp_packets'] += 1
            port = packet[UDP].dport
            self._current_stats['top_ports'][port] += 1
        elif packet.haslayer(ICMP):
            self._current_stats['icmp_packets'] += 1

        # Unique IPs
        if packet.haslayer(IP):
            self._current_stats['unique_sources'].add(packet[IP].src)
            self._current_stats['unique_destinations'].add(packet[IP].dst)

        # Generate stats every minute
        if current_time - self.last_stats_time >= 60:
            self._finalize_current_stats()

    def _reset_current_stats(self):
        """Reset current statistics counters"""
        self._current_stats = {
            'total_packets': 0,
            'total_bytes': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'unique_sources': set(),
            'unique_destinations': set(),
            'top_ports': defaultdict(int),
            'start_time': time.time()
        }

    def _finalize_current_stats(self):
        """Finalize and store current statistics"""
        current_time = datetime.now()
        duration = time.time() - self._current_stats['start_time']

        # Calculate rates
        packet_rate = self._current_stats['total_packets'] / max(duration, 1)
        bandwidth_mbps = (self._current_stats['total_bytes'] * 8) / (max(duration, 1) * 1_000_000)

        # Create TrafficStats object
        stats = TrafficStats(
            timestamp=current_time,
            total_packets=self._current_stats['total_packets'],
            total_bytes=self._current_stats['total_bytes'],
            tcp_packets=self._current_stats['tcp_packets'],
            udp_packets=self._current_stats['udp_packets'],
            icmp_packets=self._current_stats['icmp_packets'],
            unique_sources=len(self._current_stats['unique_sources']),
            unique_destinations=len(self._current_stats['unique_destinations']),
            top_protocols={
                'TCP': self._current_stats['tcp_packets'],
                'UDP': self._current_stats['udp_packets'],
                'ICMP': self._current_stats['icmp_packets']
            },
            top_ports=dict(self._current_stats['top_ports'].most_common(10)),
            bandwidth_mbps=bandwidth_mbps,
            packet_rate=packet_rate
        )

        # Store stats
        asyncio.create_task(self._store_traffic_stats(stats))

        # Check for anomalies
        anomalies = self.anomaly_detector.analyze_traffic_stats(stats)
        for anomaly in anomalies:
            asyncio.create_task(self._handle_anomaly(anomaly))

        # Reset for next period
        self._reset_current_stats()
        self.last_stats_time = time.time()

    async def _store_traffic_stats(self, stats: TrafficStats):
        """Store traffic statistics in InfluxDB"""
        try:
            # Create InfluxDB point
            point = Point("traffic_stats") \
                .tag("source", "traffic_analyzer") \
                .field("total_packets", stats.total_packets) \
                .field("total_bytes", stats.total_bytes) \
                .field("tcp_packets", stats.tcp_packets) \
                .field("udp_packets", stats.udp_packets) \
                .field("icmp_packets", stats.icmp_packets) \
                .field("unique_sources", stats.unique_sources) \
                .field("unique_destinations", stats.unique_destinations) \
                .field("bandwidth_mbps", stats.bandwidth_mbps) \
                .field("packet_rate", stats.packet_rate) \
                .time(stats.timestamp, WritePrecision.NS)

            self.write_api.write(bucket="security", org="bev-security", record=point)

            # Store in Redis for real-time access
            stats_data = asdict(stats)
            stats_data['timestamp'] = stats.timestamp.isoformat()
            self.redis_client.hset("traffic:current_stats", mapping=stats_data)
            self.redis_client.expire("traffic:current_stats", 300)  # 5 minutes

        except Exception as e:
            logger.error(f"Failed to store traffic stats: {e}")

    async def _handle_anomaly(self, anomaly: TrafficAnomaly):
        """Handle detected traffic anomaly"""
        try:
            self.metrics['anomalies_detected'] += 1

            # Log anomaly
            anomaly_data = asdict(anomaly)
            anomaly_data['timestamp'] = anomaly.timestamp.isoformat()

            logger.warning(f"TRAFFIC ANOMALY: {json.dumps(anomaly_data)}")

            # Store in Redis
            key = f"anomaly:traffic:{int(time.time())}"
            self.redis_client.hset(key, mapping=anomaly_data)
            self.redis_client.expire(key, 86400)  # 24 hours

            # Store in InfluxDB
            point = Point("traffic_anomalies") \
                .tag("anomaly_type", anomaly.anomaly_type) \
                .tag("severity", anomaly.severity) \
                .tag("source_ip", anomaly.source_ip) \
                .field("confidence", anomaly.confidence) \
                .field("description", anomaly.description) \
                .time(anomaly.timestamp, WritePrecision.NS)

            for metric_name, metric_value in anomaly.metrics.items():
                point = point.field(f"metric_{metric_name}", metric_value)

            self.write_api.write(bucket="security", org="bev-security", record=point)

            # Send alerts for high-severity anomalies
            if anomaly.severity in ['HIGH', 'CRITICAL']:
                await self._send_anomaly_alert(anomaly)

        except Exception as e:
            logger.error(f"Anomaly handling error: {e}")

    async def _send_anomaly_alert(self, anomaly: TrafficAnomaly):
        """Send alert for high-severity anomalies"""
        alert_data = {
            'type': 'TRAFFIC_ANOMALY',
            'severity': anomaly.severity,
            'description': anomaly.description,
            'source_ip': anomaly.source_ip,
            'confidence': anomaly.confidence,
            'timestamp': anomaly.timestamp.isoformat()
        }

        logger.critical(f"HIGH SEVERITY TRAFFIC ALERT: {json.dumps(alert_data)}")

    async def _statistics_reporter(self):
        """Report traffic analyzer statistics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']

                metrics = {
                    'uptime_seconds': uptime,
                    'packets_processed': self.metrics['packets_processed'],
                    'active_connections': len(self.connection_tracker.connections),
                    'anomalies_detected': self.metrics['anomalies_detected'],
                    'packets_per_second': self.metrics['packets_processed'] / max(uptime, 1)
                }

                logger.info(f"Traffic Analyzer Metrics: {json.dumps(metrics)}")

                # Store metrics in InfluxDB
                point = Point("traffic_analyzer_metrics") \
                    .field("packets_processed", self.metrics['packets_processed']) \
                    .field("active_connections", len(self.connection_tracker.connections)) \
                    .field("anomalies_detected", self.metrics['anomalies_detected']) \
                    .field("packets_per_second", metrics['packets_per_second']) \
                    .time(datetime.now(), WritePrecision.NS)

                self.write_api.write(bucket="security", org="bev-security", record=point)

                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                logger.error(f"Statistics reporting error: {e}")
                await asyncio.sleep(60)

    async def _anomaly_analysis_loop(self):
        """Periodic anomaly analysis of connections"""
        while self.running:
            try:
                active_connections = self.connection_tracker.get_active_connections()

                for conn in active_connections:
                    # Calculate additional metrics
                    if conn.duration > 0:
                        conn.packet_rate = (conn.packets_sent + conn.packets_received) / conn.duration
                        conn.byte_rate = (conn.bytes_sent + conn.bytes_received) / conn.duration
                    else:
                        conn.packet_rate = 0
                        conn.byte_rate = 0

                    # Analyze for anomalies
                    anomalies = self.anomaly_detector.analyze_connection(conn)
                    for anomaly in anomalies:
                        await self._handle_anomaly(anomaly)

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Anomaly analysis loop error: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Cleanup is handled by individual components
                # This loop can be used for additional maintenance tasks
                await asyncio.sleep(300)  # 5 minutes
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(300)

if __name__ == "__main__":
    analyzer = TrafficAnalyzer()

    try:
        asyncio.run(analyzer.start())
    except KeyboardInterrupt:
        logger.info("Traffic analyzer stopped by user")
    except Exception as e:
        logger.error(f"Traffic analyzer fatal error: {e}")
        sys.exit(1)