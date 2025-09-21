#!/usr/bin/env python3
"""
Isolation Forest Anomaly Detector
Advanced anomaly detection with continuous learning and adaptive thresholds
"""

import asyncio
import json
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import logging
import threading
import os

# ML libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import joblib

# Data visualization (for model analysis)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Data storage
import redis
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Web API
from flask import Flask, jsonify, request
import psutil

# Logging
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/app/logs/anomaly_detector.log", rotation="100 MB", retention="7 days")

@dataclass
class AnomalyFeatures:
    """Features for anomaly detection"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    disk_read_bytes: int
    disk_write_bytes: int
    connections_count: int
    processes_count: int
    load_avg_1: float
    load_avg_5: float
    load_avg_15: float
    context_switches: int
    interrupts: int
    tcp_connections: int
    udp_connections: int
    unique_source_ips: int
    unique_dest_ips: int
    failed_login_attempts: int
    suspicious_processes: int
    port_scan_indicators: int
    ddos_indicators: int
    malware_indicators: int

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    timestamp: datetime
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    features_contribution: Dict[str, float]
    cluster_id: Optional[int]
    explanation: str
    recommended_action: str

class FeatureCollector:
    """Collect system and network features for anomaly detection"""

    def __init__(self):
        self.baseline_metrics = {}
        self.last_network_stats = None
        self.last_disk_stats = None
        self.suspicious_processes = [
            'nc', 'nmap', 'nikto', 'sqlmap', 'metasploit',
            'john', 'hashcat', 'aircrack', 'wireshark'
        ]

    def collect_features(self) -> AnomalyFeatures:
        """Collect current system features"""
        current_time = datetime.now()

        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            load_avg = os.getloadavg()

            # Network stats
            network_stats = psutil.net_io_counters()
            if self.last_network_stats:
                net_bytes_sent = network_stats.bytes_sent - self.last_network_stats.bytes_sent
                net_bytes_recv = network_stats.bytes_recv - self.last_network_stats.bytes_recv
                net_packets_sent = network_stats.packets_sent - self.last_network_stats.packets_sent
                net_packets_recv = network_stats.packets_recv - self.last_network_stats.packets_recv
            else:
                net_bytes_sent = network_stats.bytes_sent
                net_bytes_recv = network_stats.bytes_recv
                net_packets_sent = network_stats.packets_sent
                net_packets_recv = network_stats.packets_recv

            self.last_network_stats = network_stats

            # Disk stats
            disk_stats = psutil.disk_io_counters()
            if self.last_disk_stats:
                disk_read = disk_stats.read_bytes - self.last_disk_stats.read_bytes
                disk_write = disk_stats.write_bytes - self.last_disk_stats.write_bytes
            else:
                disk_read = disk_stats.read_bytes
                disk_write = disk_stats.write_bytes

            self.last_disk_stats = disk_stats

            # Connection stats
            connections = psutil.net_connections()
            tcp_count = len([c for c in connections if c.type == 1])  # SOCK_STREAM
            udp_count = len([c for c in connections if c.type == 2])  # SOCK_DGRAM

            # Process analysis
            processes = list(psutil.process_iter(['pid', 'name', 'cmdline']))
            suspicious_proc_count = sum(1 for p in processes
                                      if any(susp in ' '.join(p.info['cmdline'] or []).lower()
                                           for susp in self.suspicious_processes))

            # Network security indicators (simplified)
            unique_source_ips = self._count_unique_source_ips()
            unique_dest_ips = self._count_unique_dest_ips()
            failed_logins = self._count_failed_logins()
            port_scan_indicators = self._detect_port_scan_indicators()
            ddos_indicators = self._detect_ddos_indicators()
            malware_indicators = self._detect_malware_indicators()

            return AnomalyFeatures(
                timestamp=current_time,
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                network_bytes_sent=net_bytes_sent,
                network_bytes_recv=net_bytes_recv,
                network_packets_sent=net_packets_sent,
                network_packets_recv=net_packets_recv,
                disk_read_bytes=disk_read,
                disk_write_bytes=disk_write,
                connections_count=len(connections),
                processes_count=len(processes),
                load_avg_1=load_avg[0],
                load_avg_5=load_avg[1],
                load_avg_15=load_avg[2],
                context_switches=psutil.cpu_stats().ctx_switches,
                interrupts=psutil.cpu_stats().interrupts,
                tcp_connections=tcp_count,
                udp_connections=udp_count,
                unique_source_ips=unique_source_ips,
                unique_dest_ips=unique_dest_ips,
                failed_login_attempts=failed_logins,
                suspicious_processes=suspicious_proc_count,
                port_scan_indicators=port_scan_indicators,
                ddos_indicators=ddos_indicators,
                malware_indicators=malware_indicators
            )

        except Exception as e:
            logger.error(f"Feature collection error: {e}")
            return None

    def _count_unique_source_ips(self) -> int:
        """Count unique source IPs from recent connections"""
        try:
            connections = psutil.net_connections(kind='inet')
            source_ips = set()
            for conn in connections:
                if conn.raddr and conn.raddr.ip:
                    source_ips.add(conn.raddr.ip)
            return len(source_ips)
        except:
            return 0

    def _count_unique_dest_ips(self) -> int:
        """Count unique destination IPs from recent connections"""
        try:
            connections = psutil.net_connections(kind='inet')
            dest_ips = set()
            for conn in connections:
                if conn.laddr and conn.laddr.ip:
                    dest_ips.add(conn.laddr.ip)
            return len(dest_ips)
        except:
            return 0

    def _count_failed_logins(self) -> int:
        """Count failed login attempts (simplified)"""
        try:
            # In a real implementation, parse /var/log/auth.log or similar
            # For now, return a placeholder
            return 0
        except:
            return 0

    def _detect_port_scan_indicators(self) -> int:
        """Detect port scanning indicators"""
        try:
            # Simplified detection based on connection patterns
            connections = psutil.net_connections()
            syn_connections = len([c for c in connections if c.status == 'SYN_SENT'])
            return min(syn_connections, 100)  # Cap at 100
        except:
            return 0

    def _detect_ddos_indicators(self) -> int:
        """Detect DDoS indicators"""
        try:
            # High number of connections from single IP (simplified)
            connections = psutil.net_connections()
            ip_count = {}
            for conn in connections:
                if conn.raddr:
                    ip = conn.raddr.ip
                    ip_count[ip] = ip_count.get(ip, 0) + 1

            max_connections = max(ip_count.values()) if ip_count else 0
            return min(max_connections, 1000)  # Cap at 1000
        except:
            return 0

    def _detect_malware_indicators(self) -> int:
        """Detect malware indicators"""
        try:
            indicators = 0
            processes = list(psutil.process_iter(['pid', 'name', 'cmdline', 'connections']))

            for proc in processes:
                try:
                    # Check for suspicious process names
                    proc_name = proc.info['name'].lower()
                    if any(sus in proc_name for sus in ['tmp', 'var', 'dev', 'boot']):
                        if proc_name.startswith('.') or len(proc_name) > 15:
                            indicators += 1

                    # Check for processes with many network connections
                    conn_count = len(proc.info.get('connections', []))
                    if conn_count > 50:
                        indicators += 1

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            return min(indicators, 50)  # Cap at 50
        except:
            return 0

class IsolationForestDetector:
    """Isolation Forest-based anomaly detector with continuous learning"""

    def __init__(self, model_path: str = "/app/models/isolation_forest.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        # Model components
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=15)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

        # Training data storage
        self.training_data: deque = deque(maxlen=10000)  # Keep last 10k samples
        self.feature_columns = []

        # Adaptive thresholds
        self.threshold_percentile = 95
        self.contamination_rate = 0.1
        self.dynamic_threshold = -0.5

        # Model performance tracking
        self.model_version = 1
        self.last_retrain_time = None
        self.performance_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'false_positive_rate': 0.0
        }

        # Initialize model
        self._initialize_model()
        self._load_model()

    def _initialize_model(self):
        """Initialize the Isolation Forest model"""
        self.isolation_forest = IsolationForest(
            contamination=self.contamination_rate,
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            max_samples=1024,
            max_features=1.0
        )

        logger.info("Isolation Forest model initialized")

    def _load_model(self):
        """Load trained model from disk"""
        try:
            if self.model_path.exists():
                model_data = joblib.load(self.model_path)

                self.isolation_forest = model_data['isolation_forest']
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.feature_columns = model_data['feature_columns']
                self.model_version = model_data.get('version', 1)
                self.dynamic_threshold = model_data.get('threshold', -0.5)

                logger.info(f"Loaded model version {self.model_version}")
            else:
                logger.info("No existing model found, will train with incoming data")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._initialize_model()

    def _save_model(self):
        """Save trained model to disk"""
        try:
            model_data = {
                'isolation_forest': self.isolation_forest,
                'scaler': self.scaler,
                'pca': self.pca,
                'feature_columns': self.feature_columns,
                'version': self.model_version,
                'threshold': self.dynamic_threshold,
                'timestamp': datetime.now().isoformat()
            }

            joblib.dump(model_data, self.model_path)
            logger.info(f"Saved model version {self.model_version}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def add_training_sample(self, features: AnomalyFeatures, is_anomaly: bool = None):
        """Add sample to training data"""
        try:
            feature_dict = asdict(features)
            feature_dict.pop('timestamp')  # Remove timestamp
            feature_dict['is_anomaly'] = is_anomaly

            self.training_data.append(feature_dict)

            # Retrain if enough new data accumulated
            if len(self.training_data) >= 100 and len(self.training_data) % 50 == 0:
                asyncio.create_task(self._retrain_model())

        except Exception as e:
            logger.error(f"Failed to add training sample: {e}")

    def predict(self, features: AnomalyFeatures) -> AnomalyResult:
        """Detect anomalies in the given features"""
        try:
            if not self.isolation_forest:
                # No trained model yet
                return self._create_default_result(features)

            # Convert features to DataFrame
            feature_dict = asdict(features)
            feature_dict.pop('timestamp')

            if not self.feature_columns:
                self.feature_columns = list(feature_dict.keys())

            # Ensure consistent feature order
            feature_vector = np.array([[feature_dict[col] for col in self.feature_columns]])

            # Scale features
            scaled_features = self.scaler.transform(feature_vector)

            # Apply PCA if trained
            if hasattr(self.pca, 'components_'):
                scaled_features = self.pca.transform(scaled_features)

            # Predict anomaly
            anomaly_prediction = self.isolation_forest.predict(scaled_features)[0]
            anomaly_score = self.isolation_forest.score_samples(scaled_features)[0]

            # Adaptive threshold
            is_anomaly = anomaly_score < self.dynamic_threshold

            # Calculate confidence
            confidence = abs(anomaly_score - self.dynamic_threshold)
            confidence = min(confidence * 2, 1.0)  # Normalize to 0-1

            # Determine severity
            severity = self._calculate_severity(anomaly_score, confidence)

            # Analyze feature contributions
            feature_contributions = self._analyze_feature_contributions(
                scaled_features[0], feature_dict
            )

            # Cluster analysis
            cluster_id = self._get_cluster_id(scaled_features[0])

            # Generate explanation and recommendations
            explanation = self._generate_explanation(
                anomaly_score, feature_contributions, cluster_id
            )
            recommended_action = self._get_recommended_action(severity, explanation)

            return AnomalyResult(
                timestamp=features.timestamp,
                anomaly_score=float(anomaly_score),
                is_anomaly=is_anomaly,
                anomaly_type=self._classify_anomaly_type(feature_contributions),
                severity=severity,
                confidence=float(confidence),
                features_contribution=feature_contributions,
                cluster_id=cluster_id,
                explanation=explanation,
                recommended_action=recommended_action
            )

        except Exception as e:
            logger.error(f"Anomaly prediction error: {e}")
            return self._create_default_result(features)

    def _create_default_result(self, features: AnomalyFeatures) -> AnomalyResult:
        """Create default result when model is not available"""
        return AnomalyResult(
            timestamp=features.timestamp,
            anomaly_score=0.0,
            is_anomaly=False,
            anomaly_type="UNKNOWN",
            severity="LOW",
            confidence=0.0,
            features_contribution={},
            cluster_id=None,
            explanation="Model not yet trained",
            recommended_action="Continue monitoring"
        )

    def _calculate_severity(self, anomaly_score: float, confidence: float) -> str:
        """Calculate anomaly severity"""
        if anomaly_score < -0.8 and confidence > 0.8:
            return "CRITICAL"
        elif anomaly_score < -0.6 and confidence > 0.6:
            return "HIGH"
        elif anomaly_score < -0.4 and confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _analyze_feature_contributions(self, scaled_features: np.ndarray,
                                     original_features: Dict) -> Dict[str, float]:
        """Analyze which features contribute most to the anomaly score"""
        try:
            contributions = {}

            # Simple approach: use feature magnitude in scaled space
            for i, feature_name in enumerate(self.feature_columns[:len(scaled_features)]):
                if i < len(scaled_features):
                    contributions[feature_name] = abs(float(scaled_features[i]))

            # Normalize contributions
            total_contrib = sum(contributions.values())
            if total_contrib > 0:
                contributions = {k: v / total_contrib for k, v in contributions.items()}

            # Return top 5 contributors
            sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_contrib[:5])

        except Exception as e:
            logger.error(f"Feature contribution analysis error: {e}")
            return {}

    def _get_cluster_id(self, scaled_features: np.ndarray) -> Optional[int]:
        """Get cluster ID for the anomaly (if clustering is available)"""
        try:
            if hasattr(self.dbscan, 'labels_'):
                # Predict cluster for new sample (simplified)
                cluster_prediction = self.dbscan.fit_predict([scaled_features])
                return int(cluster_prediction[0]) if cluster_prediction[0] != -1 else None
        except:
            pass
        return None

    def _classify_anomaly_type(self, feature_contributions: Dict[str, float]) -> str:
        """Classify the type of anomaly based on feature contributions"""
        if not feature_contributions:
            return "UNKNOWN"

        top_feature = max(feature_contributions.keys(), key=lambda k: feature_contributions[k])

        if 'cpu_usage' in top_feature or 'load_avg' in top_feature:
            return "PERFORMANCE"
        elif 'memory_usage' in top_feature:
            return "MEMORY"
        elif 'network' in top_feature or 'connections' in top_feature:
            return "NETWORK"
        elif 'disk' in top_feature:
            return "STORAGE"
        elif 'suspicious' in top_feature or 'malware' in top_feature:
            return "SECURITY"
        elif 'port_scan' in top_feature or 'ddos' in top_feature:
            return "ATTACK"
        else:
            return "SYSTEM"

    def _generate_explanation(self, anomaly_score: float,
                            feature_contributions: Dict[str, float],
                            cluster_id: Optional[int]) -> str:
        """Generate human-readable explanation for the anomaly"""
        if anomaly_score >= self.dynamic_threshold:
            return "Normal system behavior detected"

        explanation_parts = [f"Anomaly detected with score {anomaly_score:.3f}"]

        if feature_contributions:
            top_contributor = max(feature_contributions.keys(),
                                key=lambda k: feature_contributions[k])
            explanation_parts.append(f"Primary factor: {top_contributor}")

        if cluster_id is not None:
            explanation_parts.append(f"Belongs to anomaly cluster {cluster_id}")

        return ". ".join(explanation_parts)

    def _get_recommended_action(self, severity: str, explanation: str) -> str:
        """Get recommended action based on severity and explanation"""
        if severity == "CRITICAL":
            return "Immediate investigation required - potential security incident"
        elif severity == "HIGH":
            return "Investigate within 1 hour - significant anomaly detected"
        elif severity == "MEDIUM":
            return "Review within 4 hours - moderate anomaly detected"
        else:
            return "Monitor for patterns - minor anomaly detected"

    async def _retrain_model(self):
        """Retrain the model with accumulated data"""
        if len(self.training_data) < 100:
            return

        try:
            logger.info("Starting model retraining")

            # Convert training data to DataFrame
            df = pd.DataFrame(list(self.training_data))
            df = df.dropna()

            if len(df) < 50:
                logger.warning("Insufficient clean data for retraining")
                return

            # Prepare features
            feature_cols = [col for col in df.columns if col != 'is_anomaly']
            X = df[feature_cols].values

            # Update feature columns
            self.feature_columns = feature_cols

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Apply PCA
            X_pca = self.pca.fit_transform(X_scaled)

            # Retrain Isolation Forest
            self.isolation_forest.fit(X_pca)

            # Update dynamic threshold based on training data
            scores = self.isolation_forest.score_samples(X_pca)
            self.dynamic_threshold = np.percentile(scores, self.threshold_percentile)

            # Cluster analysis for better anomaly classification
            self.dbscan.fit(X_pca)

            # Update model version
            self.model_version += 1
            self.last_retrain_time = datetime.now()

            # Save updated model
            self._save_model()

            # Log retraining results
            silhouette_avg = silhouette_score(X_pca, self.dbscan.labels_) if len(set(self.dbscan.labels_)) > 1 else 0
            logger.info(f"Model retrained - Version: {self.model_version}, "
                       f"Samples: {len(df)}, Silhouette Score: {silhouette_avg:.3f}")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    def generate_model_report(self) -> Dict:
        """Generate model performance report"""
        try:
            report = {
                'model_version': self.model_version,
                'last_retrain_time': self.last_retrain_time.isoformat() if self.last_retrain_time else None,
                'training_samples': len(self.training_data),
                'dynamic_threshold': self.dynamic_threshold,
                'contamination_rate': self.contamination_rate,
                'feature_count': len(self.feature_columns),
                'performance_metrics': self.performance_metrics.copy()
            }

            if self.isolation_forest and hasattr(self.isolation_forest, 'estimators_'):
                report['n_estimators'] = len(self.isolation_forest.estimators_)

            return report

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}

class AnomalyDetectionSystem:
    """Main anomaly detection system"""

    def __init__(self):
        self.feature_collector = FeatureCollector()
        self.detector = IsolationForestDetector()

        # Data storage
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="your-influxdb-token",
            org="bev-security"
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        # Flask app for API
        self.app = Flask(__name__)
        self._setup_api_routes()

        self.running = True
        self.metrics = {
            'samples_processed': 0,
            'anomalies_detected': 0,
            'model_retrains': 0,
            'start_time': time.time()
        }

    def _setup_api_routes(self):
        """Setup Flask API routes"""

        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'uptime': time.time() - self.metrics['start_time'],
                'samples_processed': self.metrics['samples_processed'],
                'anomalies_detected': self.metrics['anomalies_detected']
            })

        @self.app.route('/model/info', methods=['GET'])
        def model_info():
            return jsonify(self.detector.generate_model_report())

        @self.app.route('/model/retrain', methods=['POST'])
        def trigger_retrain():
            asyncio.create_task(self.detector._retrain_model())
            return jsonify({'message': 'Retraining triggered'})

        @self.app.route('/anomaly/current', methods=['GET'])
        def current_anomalies():
            try:
                # Get recent anomalies from Redis
                keys = self.redis_client.keys("anomaly:detector:*")
                recent_keys = sorted(keys, reverse=True)[:10]

                anomalies = []
                for key in recent_keys:
                    anomaly_data = self.redis_client.hgetall(key)
                    if anomaly_data:
                        anomalies.append(anomaly_data)

                return jsonify(anomalies)
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    async def start(self):
        """Start the anomaly detection system"""
        logger.info("Starting Isolation Forest Anomaly Detection System")

        # Start Flask API in background thread
        flask_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=8083, debug=False),
            daemon=True
        )
        flask_thread.start()

        # Start detection tasks
        tasks = [
            asyncio.create_task(self._detection_loop()),
            asyncio.create_task(self._metrics_reporter()),
            asyncio.create_task(self._maintenance_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Anomaly detection system error: {e}")

    async def _detection_loop(self):
        """Main detection loop"""
        while self.running:
            try:
                # Collect features
                features = self.feature_collector.collect_features()
                if not features:
                    await asyncio.sleep(10)
                    continue

                # Detect anomalies
                result = self.detector.predict(features)

                self.metrics['samples_processed'] += 1

                if result.is_anomaly:
                    self.metrics['anomalies_detected'] += 1
                    await self._handle_anomaly(result)

                # Add to training data
                self.detector.add_training_sample(features, result.is_anomaly)

                # Store results
                await self._store_result(result, features)

                # Wait before next detection
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                await asyncio.sleep(30)

    async def _handle_anomaly(self, result: AnomalyResult):
        """Handle detected anomaly"""
        try:
            # Log anomaly
            anomaly_data = asdict(result)
            anomaly_data['timestamp'] = result.timestamp.isoformat()

            logger.warning(f"ANOMALY DETECTED: {json.dumps(anomaly_data, default=str)}")

            # Store in Redis
            key = f"anomaly:detector:{int(time.time())}"
            self.redis_client.hset(key, mapping={k: str(v) for k, v in anomaly_data.items()})
            self.redis_client.expire(key, 86400)  # 24 hours

            # Send alerts for high-severity anomalies
            if result.severity in ['HIGH', 'CRITICAL']:
                await self._send_anomaly_alert(result)

        except Exception as e:
            logger.error(f"Anomaly handling error: {e}")

    async def _send_anomaly_alert(self, result: AnomalyResult):
        """Send alert for high-severity anomalies"""
        alert_data = {
            'type': 'SYSTEM_ANOMALY',
            'severity': result.severity,
            'anomaly_type': result.anomaly_type,
            'confidence': result.confidence,
            'explanation': result.explanation,
            'recommended_action': result.recommended_action,
            'timestamp': result.timestamp.isoformat()
        }

        logger.critical(f"HIGH SEVERITY ANOMALY ALERT: {json.dumps(alert_data)}")

    async def _store_result(self, result: AnomalyResult, features: AnomalyFeatures):
        """Store detection result in time-series database"""
        try:
            # Store in InfluxDB
            point = Point("anomaly_detection") \
                .tag("anomaly_type", result.anomaly_type) \
                .tag("severity", result.severity) \
                .tag("is_anomaly", str(result.is_anomaly)) \
                .field("anomaly_score", result.anomaly_score) \
                .field("confidence", result.confidence) \
                .time(result.timestamp, WritePrecision.NS)

            # Add feature contributions as fields
            for feature, contribution in result.features_contribution.items():
                point = point.field(f"contrib_{feature}", contribution)

            self.write_api.write(bucket="security", org="bev-security", record=point)

            # Store features for analysis
            feature_point = Point("system_features") \
                .field("cpu_usage", features.cpu_usage) \
                .field("memory_usage", features.memory_usage) \
                .field("network_bytes_sent", features.network_bytes_sent) \
                .field("network_bytes_recv", features.network_bytes_recv) \
                .field("connections_count", features.connections_count) \
                .field("processes_count", features.processes_count) \
                .field("load_avg_1", features.load_avg_1) \
                .field("suspicious_processes", features.suspicious_processes) \
                .time(features.timestamp, WritePrecision.NS)

            self.write_api.write(bucket="security", org="bev-security", record=feature_point)

        except Exception as e:
            logger.error(f"Failed to store detection result: {e}")

    async def _metrics_reporter(self):
        """Report system metrics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']

                metrics = {
                    'uptime_seconds': uptime,
                    'samples_processed': self.metrics['samples_processed'],
                    'anomalies_detected': self.metrics['anomalies_detected'],
                    'detection_rate': self.metrics['anomalies_detected'] / max(self.metrics['samples_processed'], 1),
                    'model_version': self.detector.model_version,
                    'training_samples': len(self.detector.training_data)
                }

                logger.info(f"Anomaly Detection Metrics: {json.dumps(metrics)}")

                # Store metrics in InfluxDB
                point = Point("anomaly_detector_metrics") \
                    .field("samples_processed", self.metrics['samples_processed']) \
                    .field("anomalies_detected", self.metrics['anomalies_detected']) \
                    .field("detection_rate", metrics['detection_rate']) \
                    .field("model_version", self.detector.model_version) \
                    .field("training_samples", len(self.detector.training_data)) \
                    .time(datetime.now(), WritePrecision.NS)

                self.write_api.write(bucket="security", org="bev-security", record=point)

                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")
                await asyncio.sleep(60)

    async def _maintenance_loop(self):
        """Periodic maintenance tasks"""
        while self.running:
            try:
                # Cleanup old Redis keys
                old_keys = []
                for key in self.redis_client.keys("anomaly:detector:*"):
                    try:
                        timestamp = int(key.split(":")[-1])
                        if time.time() - timestamp > 86400 * 7:  # 7 days
                            old_keys.append(key)
                    except:
                        pass

                if old_keys:
                    self.redis_client.delete(*old_keys)
                    logger.info(f"Cleaned up {len(old_keys)} old anomaly records")

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(3600)

if __name__ == "__main__":
    detector_system = AnomalyDetectionSystem()

    try:
        asyncio.run(detector_system.start())
    except KeyboardInterrupt:
        logger.info("Anomaly Detection System stopped by user")
    except Exception as e:
        logger.error(f"Anomaly Detection System fatal error: {e}")
        sys.exit(1)