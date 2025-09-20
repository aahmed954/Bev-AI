#!/usr/bin/env python3
"""
ML-Based Intrusion Detection System
Advanced threat detection using machine learning algorithms
"""

import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import threading
import pickle
import os

# ML libraries
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib

# Network libraries
import scapy.all as scapy
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import ARP, Ether
import dpkt
import psutil

# Data storage
import redis
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# Logging
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/app/logs/ids.log", rotation="100 MB", retention="7 days")

class AttackType(Enum):
    NORMAL = 0
    DOS = 1
    PROBE = 2
    R2L = 3  # Remote to Local
    U2R = 4  # User to Root

@dataclass
class NetworkFeatures:
    """Network packet features for ML analysis"""
    duration: float
    protocol_type: int  # TCP=0, UDP=1, ICMP=2
    service: int  # Encoded service type
    flag: int  # Connection flag
    src_bytes: int
    dst_bytes: int
    land: int  # 1 if connection is from/to same host/port
    wrong_fragment: int
    urgent: int
    hot: int  # Number of "hot" indicators
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int  # Connections in past 2 seconds
    srv_count: int  # Connections to same service
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: int
    dst_host_srv_count: int
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float

@dataclass
class DetectionResult:
    timestamp: datetime
    source_ip: str
    destination_ip: str
    attack_type: AttackType
    confidence: float
    features: Dict[str, float]
    model_name: str
    raw_score: float

class FeatureExtractor:
    """Extract features from network packets for ML analysis"""

    def __init__(self):
        self.connection_cache: Dict[str, Dict] = {}
        self.service_encoder = LabelEncoder()
        self.flag_encoder = LabelEncoder()
        self.protocol_map = {'tcp': 0, 'udp': 1, 'icmp': 2}

        # Initialize encoders with common values
        self.service_encoder.fit(['http', 'ftp', 'smtp', 'ssh', 'telnet', 'dns', 'pop3', 'imap4'])
        self.flag_encoder.fit(['S0', 'S1', 'SF', 'REJ', 'RSTO', 'RSTR', 'SH', 'OTH'])

    def extract_features(self, packet) -> Optional[NetworkFeatures]:
        """Extract features from a network packet"""
        try:
            if not packet.haslayer(IP):
                return None

            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            protocol = ip_layer.proto

            # Initialize features
            features = {
                'duration': 0.0,
                'protocol_type': self._get_protocol_type(protocol),
                'service': self._get_service(packet),
                'flag': self._get_connection_flag(packet),
                'src_bytes': len(packet),
                'dst_bytes': 0,
                'land': 1 if src_ip == dst_ip else 0,
                'wrong_fragment': 1 if ip_layer.flags & 0x2 else 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 0,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0
            }

            # Calculate connection-based features
            conn_key = f"{src_ip}:{dst_ip}"
            current_time = time.time()

            # Update connection cache
            if conn_key not in self.connection_cache:
                self.connection_cache[conn_key] = {
                    'connections': [],
                    'services': [],
                    'flags': []
                }

            self.connection_cache[conn_key]['connections'].append(current_time)
            self.connection_cache[conn_key]['services'].append(features['service'])
            self.connection_cache[conn_key]['flags'].append(features['flag'])

            # Clean old connections (older than 2 seconds)
            recent_connections = [t for t in self.connection_cache[conn_key]['connections']
                                if current_time - t <= 2.0]
            self.connection_cache[conn_key]['connections'] = recent_connections

            # Calculate statistical features
            features.update(self._calculate_connection_features(conn_key, current_time))

            return NetworkFeatures(**features)

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _get_protocol_type(self, protocol: int) -> int:
        """Map protocol number to encoded value"""
        if protocol == 6:  # TCP
            return 0
        elif protocol == 17:  # UDP
            return 1
        elif protocol == 1:  # ICMP
            return 2
        else:
            return 3  # Other

    def _get_service(self, packet) -> int:
        """Extract and encode service type"""
        try:
            if packet.haslayer(TCP):
                port = packet[TCP].dport
                service = self._port_to_service(port)
                return self._safe_encode(self.service_encoder, service)
            elif packet.haslayer(UDP):
                port = packet[UDP].dport
                service = self._port_to_service(port)
                return self._safe_encode(self.service_encoder, service)
            return 0
        except:
            return 0

    def _port_to_service(self, port: int) -> str:
        """Map port number to service name"""
        port_map = {
            80: 'http', 443: 'https', 21: 'ftp', 22: 'ssh',
            23: 'telnet', 25: 'smtp', 53: 'dns', 110: 'pop3',
            143: 'imap4', 993: 'imaps', 995: 'pop3s'
        }
        return port_map.get(port, 'other')

    def _get_connection_flag(self, packet) -> int:
        """Extract connection flag"""
        try:
            if packet.haslayer(TCP):
                tcp_flags = packet[TCP].flags
                if tcp_flags & 0x02:  # SYN
                    return self._safe_encode(self.flag_encoder, 'S0')
                elif tcp_flags & 0x10:  # ACK
                    return self._safe_encode(self.flag_encoder, 'SF')
                elif tcp_flags & 0x04:  # RST
                    return self._safe_encode(self.flag_encoder, 'REJ')
            return self._safe_encode(self.flag_encoder, 'OTH')
        except:
            return 0

    def _safe_encode(self, encoder, value: str) -> int:
        """Safely encode value, handling unknown categories"""
        try:
            return encoder.transform([value])[0]
        except:
            return 0

    def _calculate_connection_features(self, conn_key: str, current_time: float) -> Dict[str, float]:
        """Calculate connection-based statistical features"""
        cache = self.connection_cache[conn_key]

        # Count connections in past 2 seconds
        recent_connections = [t for t in cache['connections'] if current_time - t <= 2.0]
        count = len(recent_connections)

        # Service-related features
        recent_services = cache['services'][-count:] if count > 0 else []
        srv_count = len([s for s in recent_services if s == recent_services[-1]]) if recent_services else 0

        # Error rate calculations (simplified)
        serror_rate = 0.0
        srv_serror_rate = 0.0
        rerror_rate = 0.0
        srv_rerror_rate = 0.0
        same_srv_rate = srv_count / max(count, 1)
        diff_srv_rate = 1.0 - same_srv_rate
        srv_diff_host_rate = 0.0

        # Destination host features (simplified)
        dst_host_count = min(count, 255)
        dst_host_srv_count = min(srv_count, 255)
        dst_host_same_srv_rate = same_srv_rate
        dst_host_diff_srv_rate = diff_srv_rate
        dst_host_same_src_port_rate = 0.0
        dst_host_srv_diff_host_rate = 0.0
        dst_host_serror_rate = serror_rate
        dst_host_srv_serror_rate = srv_serror_rate
        dst_host_rerror_rate = rerror_rate
        dst_host_srv_rerror_rate = srv_rerror_rate

        return {
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_rate,
            'srv_serror_rate': srv_serror_rate,
            'rerror_rate': rerror_rate,
            'srv_rerror_rate': srv_rerror_rate,
            'same_srv_rate': same_srv_rate,
            'diff_srv_rate': diff_srv_rate,
            'srv_diff_host_rate': srv_diff_host_rate,
            'dst_host_count': dst_host_count,
            'dst_host_srv_count': dst_host_srv_count,
            'dst_host_same_srv_rate': dst_host_same_srv_rate,
            'dst_host_diff_srv_rate': dst_host_diff_srv_rate,
            'dst_host_same_src_port_rate': dst_host_same_src_port_rate,
            'dst_host_srv_diff_host_rate': dst_host_srv_diff_host_rate,
            'dst_host_serror_rate': dst_host_serror_rate,
            'dst_host_srv_serror_rate': dst_host_srv_serror_rate,
            'dst_host_rerror_rate': dst_host_rerror_rate,
            'dst_host_srv_rerror_rate': dst_host_srv_rerror_rate
        }

class MLModelManager:
    """Manage multiple ML models for intrusion detection"""

    def __init__(self, model_dir: str = "/app/models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []

        os.makedirs(model_dir, exist_ok=True)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models"""
        logger.info("Initializing ML models...")

        # Random Forest Classifier
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        # Isolation Forest for anomaly detection
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )

        # Neural Network
        self.models['neural_network'] = self._create_neural_network()

        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()

    def _create_neural_network(self) -> keras.Model:
        """Create neural network model"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(41,)),  # 41 features
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(5, activation='softmax')  # 5 attack types
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_models(self, training_data: pd.DataFrame):
        """Train all ML models"""
        logger.info("Training ML models...")

        # Prepare training data
        X = training_data.drop(['label'], axis=1)
        y = training_data['label']

        self.feature_columns = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train each model
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")

            try:
                # Scale features
                scaler = self.scalers[model_name]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                if model_name == 'neural_network':
                    # Train neural network
                    model.fit(
                        X_train_scaled, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0
                    )

                    # Evaluate
                    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                    logger.info(f"{model_name} accuracy: {accuracy:.4f}")

                elif model_name == 'isolation_forest':
                    # Train anomaly detector (unsupervised)
                    normal_data = X_train_scaled[y_train == 0]  # Assuming 0 is normal
                    model.fit(normal_data)

                    # Test on all data
                    predictions = model.predict(X_test_scaled)
                    anomaly_rate = (predictions == -1).mean()
                    logger.info(f"{model_name} anomaly detection rate: {anomaly_rate:.4f}")

                else:
                    # Train supervised models
                    model.fit(X_train_scaled, y_train)

                    # Evaluate
                    score = model.score(X_test_scaled, y_test)
                    logger.info(f"{model_name} accuracy: {score:.4f}")

                # Save model and scaler
                self._save_model(model_name, model, scaler)

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")

    def predict(self, features: NetworkFeatures, model_name: str = 'random_forest') -> DetectionResult:
        """Make prediction using specified model"""
        try:
            # Convert features to DataFrame
            feature_dict = asdict(features)
            feature_vector = np.array([list(feature_dict.values())])

            # Scale features
            if model_name in self.scalers:
                feature_vector = self.scalers[model_name].transform(feature_vector)

            model = self.models[model_name]

            if model_name == 'neural_network':
                predictions = model.predict(feature_vector, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                raw_score = float(confidence)

            elif model_name == 'isolation_forest':
                prediction = model.predict(feature_vector)[0]
                score = model.score_samples(feature_vector)[0]
                predicted_class = 1 if prediction == -1 else 0  # Anomaly or normal
                confidence = abs(score)
                raw_score = float(score)

            else:
                prediction = model.predict(feature_vector)[0]
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(feature_vector)[0]
                    confidence = np.max(probabilities)
                else:
                    confidence = 0.8  # Default confidence
                predicted_class = prediction
                raw_score = float(confidence)

            return DetectionResult(
                timestamp=datetime.now(),
                source_ip="unknown",  # Will be filled by caller
                destination_ip="unknown",
                attack_type=AttackType(predicted_class),
                confidence=float(confidence),
                features=feature_dict,
                model_name=model_name,
                raw_score=raw_score
            )

        except Exception as e:
            logger.error(f"Prediction error with {model_name}: {e}")
            return None

    def _save_model(self, model_name: str, model: Any, scaler: StandardScaler):
        """Save trained model and scaler"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")

            if model_name == 'neural_network':
                model.save(os.path.join(self.model_dir, f"{model_name}.h5"))
            else:
                joblib.dump(model, model_path)

            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {model_name} model and scaler")

        except Exception as e:
            logger.error(f"Failed to save {model_name}: {e}")

    def load_models(self):
        """Load trained models from disk"""
        try:
            for model_name in self.models.keys():
                model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
                scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")

                if model_name == 'neural_network':
                    nn_path = os.path.join(self.model_dir, f"{model_name}.h5")
                    if os.path.exists(nn_path):
                        self.models[model_name] = keras.models.load_model(nn_path)
                else:
                    if os.path.exists(model_path):
                        self.models[model_name] = joblib.load(model_path)

                if os.path.exists(scaler_path):
                    self.scalers[model_name] = joblib.load(scaler_path)

                logger.info(f"Loaded {model_name} model")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

class IntrusionDetectionSystem:
    """Main IDS class"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_manager = MLModelManager()
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

        # InfluxDB for time series data
        self.influx_client = InfluxDBClient(
            url="http://localhost:8086",
            token="your-influxdb-token",
            org="bev-security"
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

        self.running = True
        self.metrics = {
            'packets_analyzed': 0,
            'threats_detected': 0,
            'false_positives': 0,
            'start_time': time.time()
        }

        # Load existing models
        self.model_manager.load_models()

    async def start(self):
        """Start the IDS"""
        logger.info("Starting ML-based Intrusion Detection System")

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._packet_analysis_loop()),
            asyncio.create_task(self._metrics_reporter()),
            asyncio.create_task(self._model_retraining_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"IDS error: {e}")

    async def _packet_analysis_loop(self):
        """Analyze network packets in real-time"""
        def packet_handler(packet):
            try:
                self.metrics['packets_analyzed'] += 1

                # Extract features
                features = self.feature_extractor.extract_features(packet)
                if not features:
                    return

                # Get source and destination IPs
                src_ip = packet[IP].src if packet.haslayer(IP) else "unknown"
                dst_ip = packet[IP].dst if packet.haslayer(IP) else "unknown"

                # Run through multiple models for ensemble detection
                detections = []
                for model_name in ['random_forest', 'isolation_forest', 'neural_network']:
                    result = self.model_manager.predict(features, model_name)
                    if result:
                        result.source_ip = src_ip
                        result.destination_ip = dst_ip
                        detections.append(result)

                # Ensemble decision
                if detections:
                    final_result = self._ensemble_decision(detections)

                    if final_result.attack_type != AttackType.NORMAL:
                        self.metrics['threats_detected'] += 1
                        self._handle_threat_detection(final_result)

                    # Store results
                    await self._store_detection_result(final_result)

            except Exception as e:
                logger.error(f"Packet analysis error: {e}")

        # Start packet capture
        try:
            scapy.sniff(iface="eth0", prn=packet_handler, store=0)
        except Exception as e:
            logger.error(f"Packet capture failed: {e}")

    def _ensemble_decision(self, detections: List[DetectionResult]) -> DetectionResult:
        """Make ensemble decision from multiple model results"""
        if not detections:
            return None

        # Weighted voting based on model confidence
        model_weights = {
            'random_forest': 0.4,
            'neural_network': 0.4,
            'isolation_forest': 0.2
        }

        # Calculate weighted predictions
        attack_scores = {}
        total_confidence = 0

        for detection in detections:
            weight = model_weights.get(detection.model_name, 0.33)
            attack_type = detection.attack_type

            if attack_type not in attack_scores:
                attack_scores[attack_type] = 0

            attack_scores[attack_type] += weight * detection.confidence
            total_confidence += weight * detection.confidence

        # Find the attack type with highest weighted score
        final_attack_type = max(attack_scores.items(), key=lambda x: x[1])[0]
        final_confidence = attack_scores[final_attack_type] / len(detections)

        # Use the detection from the most confident model as base
        base_detection = max(detections, key=lambda x: x.confidence)
        base_detection.attack_type = final_attack_type
        base_detection.confidence = final_confidence
        base_detection.model_name = "ensemble"

        return base_detection

    async def _handle_threat_detection(self, detection: DetectionResult):
        """Handle detected threats"""
        try:
            # Log threat
            threat_data = {
                'timestamp': detection.timestamp.isoformat(),
                'source_ip': detection.source_ip,
                'destination_ip': detection.destination_ip,
                'attack_type': detection.attack_type.name,
                'confidence': detection.confidence,
                'model': detection.model_name
            }

            logger.warning(f"THREAT DETECTED: {json.dumps(threat_data)}")

            # Store in Redis for other security components
            key = f"threat:ids:{int(time.time())}"
            self.redis_client.hset(key, mapping=threat_data)
            self.redis_client.expire(key, 86400)  # 24 hours

            # Send alert based on threat level
            if detection.confidence > 0.8:
                await self._send_high_priority_alert(detection)

        except Exception as e:
            logger.error(f"Threat handling error: {e}")

    async def _send_high_priority_alert(self, detection: DetectionResult):
        """Send high priority security alert"""
        # In production, integrate with SIEM, email, Slack, etc.
        alert_data = {
            'severity': 'HIGH',
            'detection': asdict(detection),
            'recommended_action': self._get_recommended_action(detection.attack_type)
        }

        logger.critical(f"HIGH PRIORITY ALERT: {json.dumps(alert_data, default=str)}")

    def _get_recommended_action(self, attack_type: AttackType) -> str:
        """Get recommended response action"""
        actions = {
            AttackType.DOS: "Block source IP immediately",
            AttackType.PROBE: "Monitor and rate-limit source IP",
            AttackType.R2L: "Block connection and audit logs",
            AttackType.U2R: "Immediate investigation required"
        }
        return actions.get(attack_type, "Monitor and investigate")

    async def _store_detection_result(self, detection: DetectionResult):
        """Store detection result in time-series database"""
        try:
            # InfluxDB point
            point = Point("intrusion_detection") \
                .tag("source_ip", detection.source_ip) \
                .tag("attack_type", detection.attack_type.name) \
                .tag("model", detection.model_name) \
                .field("confidence", detection.confidence) \
                .field("raw_score", detection.raw_score) \
                .time(detection.timestamp, WritePrecision.NS)

            self.write_api.write(bucket="security", org="bev-security", record=point)

        except Exception as e:
            logger.error(f"Failed to store detection result: {e}")

    async def _metrics_reporter(self):
        """Report IDS performance metrics"""
        while self.running:
            try:
                uptime = time.time() - self.metrics['start_time']
                pps = self.metrics['packets_analyzed'] / max(uptime, 1)

                metrics = {
                    'uptime': uptime,
                    'packets_per_second': pps,
                    'total_packets': self.metrics['packets_analyzed'],
                    'threats_detected': self.metrics['threats_detected'],
                    'detection_rate': self.metrics['threats_detected'] / max(self.metrics['packets_analyzed'], 1)
                }

                logger.info(f"IDS Metrics: {json.dumps(metrics)}")

                # Store metrics in InfluxDB
                point = Point("ids_metrics") \
                    .field("packets_per_second", pps) \
                    .field("total_packets", self.metrics['packets_analyzed']) \
                    .field("threats_detected", self.metrics['threats_detected']) \
                    .time(datetime.now(), WritePrecision.NS)

                self.write_api.write(bucket="security", org="bev-security", record=point)

                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                logger.error(f"Metrics reporting error: {e}")
                await asyncio.sleep(60)

    async def _model_retraining_loop(self):
        """Periodic model retraining with new data"""
        while self.running:
            try:
                # Wait 24 hours between retraining cycles
                await asyncio.sleep(86400)

                logger.info("Starting model retraining cycle")
                await self._retrain_models()

            except Exception as e:
                logger.error(f"Model retraining error: {e}")

    async def _retrain_models(self):
        """Retrain models with accumulated data"""
        try:
            # In production, gather labeled training data
            # For now, use synthetic data or existing datasets
            logger.info("Model retraining completed (placeholder)")

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

if __name__ == "__main__":
    ids = IntrusionDetectionSystem()

    try:
        asyncio.run(ids.start())
    except KeyboardInterrupt:
        logger.info("IDS stopped by user")
    except Exception as e:
        logger.error(f"IDS fatal error: {e}")
        sys.exit(1)