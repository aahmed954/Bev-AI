#!/usr/bin/env python3
"""
Operational Security Framework (OPSEC Enforcer)
Insider threat detection, data protection, and security policy enforcement
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
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
import psutil
import socket
import subprocess
import re
import mimetypes
import asyncpg
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge
import logging
from .security_framework import OperationalSecurityFramework

# Metrics
INSIDER_THREATS = Counter('insider_threats_total', 'Total insider threat incidents', ['user', 'risk_level'])
DATA_EXFILTRATION_ATTEMPTS = Counter('data_exfiltration_attempts_total', 'Data exfiltration attempts', ['method'])
POLICY_VIOLATIONS = Counter('policy_violations_total', 'Security policy violations', ['policy', 'severity'])
COMMUNICATION_INTERCEPTS = Counter('communication_intercepts_total', 'Intercepted communications', ['type'])
ASSET_ACCESS_VIOLATIONS = Counter('asset_access_violations_total', 'Asset access violations', ['asset_type'])
SUPPLY_CHAIN_ALERTS = Counter('supply_chain_alerts_total', 'Supply chain security alerts', ['vendor'])

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class ThreatType(Enum):
    """Insider threat types"""
    DATA_THEFT = "data_theft"
    SABOTAGE = "sabotage"
    FRAUD = "fraud"
    ESPIONAGE = "espionage"
    POLICY_VIOLATION = "policy_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"

class AssetClassification(Enum):
    """Asset classification levels"""
    TOP_SECRET = "top_secret"
    SECRET = "secret"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PUBLIC = "public"

class DataCategory(Enum):
    """Data category types"""
    PII = "personally_identifiable_information"
    PHI = "protected_health_information"
    FINANCIAL = "financial_data"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    TRADE_SECRETS = "trade_secrets"
    OPERATIONAL = "operational_data"

@dataclass
class UserProfile:
    """User behavioral profile"""
    user_id: str
    username: str
    role: str
    department: str
    clearance_level: str
    risk_score: float
    baseline_behavior: Dict = field(default_factory=dict)
    current_behavior: Dict = field(default_factory=dict)
    anomaly_history: List[Dict] = field(default_factory=list)
    access_patterns: Dict = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class InsiderThreatEvent:
    """Insider threat event"""
    id: str
    user_id: str
    timestamp: datetime
    threat_type: ThreatType
    risk_level: RiskLevel
    confidence: float
    description: str
    evidence: List[str] = field(default_factory=list)
    indicators: Dict = field(default_factory=dict)
    response_actions: List[str] = field(default_factory=list)

@dataclass
class DataAccessEvent:
    """Data access event"""
    id: str
    user_id: str
    timestamp: datetime
    asset_id: str
    asset_type: str
    classification: AssetClassification
    action: str  # read, write, delete, copy, print
    location: str
    device_id: str
    authorized: bool
    context: Dict = field(default_factory=dict)

@dataclass
class CommunicationEvent:
    """Communication monitoring event"""
    id: str
    user_id: str
    timestamp: datetime
    communication_type: str  # email, chat, file_transfer
    participants: List[str]
    content_summary: str
    risk_indicators: List[str] = field(default_factory=list)
    classification: str = "unclassified"
    intercepted: bool = False

@dataclass
class SupplyChainAsset:
    """Supply chain asset tracking"""
    id: str
    vendor: str
    asset_type: str
    version: str
    installation_date: datetime
    last_updated: datetime
    risk_score: float
    vulnerabilities: List[str] = field(default_factory=list)
    compliance_status: str = "unknown"
    monitoring_enabled: bool = True

class BehavioralAnalysisEngine:
    """Analyze user behavior for insider threat detection"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.behavior_model = None
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()
        self.user_profiles: Dict[str, UserProfile] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize behavioral analysis models"""
        try:
            # Initialize deep learning model for behavior analysis
            self.behavior_model = BehaviorAnalysisNN().to(self.device)

            # Load pre-trained weights if available
            model_path = "/opt/bev/models/behavior_analysis.pth"
            if os.path.exists(model_path):
                self.behavior_model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pre-trained behavior analysis model")
            else:
                logger.info("Using untrained behavior analysis model")

            self.behavior_model.eval()

        except Exception as e:
            logger.error(f"Failed to initialize behavioral analysis models: {e}")

    async def analyze_user_behavior(self, user_id: str, activities: List[Dict]) -> Dict:
        """Analyze user behavior patterns"""
        try:
            # Get or create user profile
            if user_id not in self.user_profiles:
                await self._create_user_profile(user_id)

            profile = self.user_profiles[user_id]

            # Extract behavioral features
            features = self._extract_behavioral_features(activities)

            # Update current behavior
            profile.current_behavior = features

            # Compare with baseline
            anomaly_score = await self._calculate_anomaly_score(profile)

            # Update risk score
            profile.risk_score = self._update_risk_score(profile, anomaly_score)

            # Detect specific threat patterns
            threat_indicators = await self._detect_threat_patterns(user_id, activities, features)

            analysis_result = {
                'user_id': user_id,
                'risk_score': profile.risk_score,
                'anomaly_score': anomaly_score,
                'threat_indicators': threat_indicators,
                'behavioral_changes': self._identify_behavioral_changes(profile),
                'recommendations': self._generate_recommendations(profile, threat_indicators)
            }

            # Update profile
            profile.last_updated = datetime.now()
            if anomaly_score > 0.7:
                profile.anomaly_history.append({
                    'timestamp': datetime.now(),
                    'anomaly_score': anomaly_score,
                    'threat_indicators': threat_indicators
                })

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing user behavior for {user_id}: {e}")
            return {'error': str(e)}

    async def _create_user_profile(self, user_id: str):
        """Create new user profile"""
        try:
            # Get user information from database or directory service
            user_info = await self._get_user_info(user_id)

            profile = UserProfile(
                user_id=user_id,
                username=user_info.get('username', user_id),
                role=user_info.get('role', 'unknown'),
                department=user_info.get('department', 'unknown'),
                clearance_level=user_info.get('clearance_level', 'restricted'),
                risk_score=0.5  # Start with neutral risk
            )

            self.user_profiles[user_id] = profile
            logger.info(f"Created user profile for {user_id}")

        except Exception as e:
            logger.error(f"Error creating user profile for {user_id}: {e}")

    async def _get_user_info(self, user_id: str) -> Dict:
        """Get user information from directory service"""
        # Placeholder implementation
        # In production, integrate with LDAP/AD or HR system
        return {
            'username': user_id,
            'role': 'employee',
            'department': 'general',
            'clearance_level': 'restricted'
        }

    def _extract_behavioral_features(self, activities: List[Dict]) -> Dict:
        """Extract behavioral features from user activities"""
        features = {
            'login_patterns': {},
            'access_patterns': {},
            'communication_patterns': {},
            'data_patterns': {},
            'temporal_patterns': {}
        }

        try:
            # Analyze login patterns
            login_times = [act['timestamp'] for act in activities if act.get('type') == 'login']
            if login_times:
                features['login_patterns'] = {
                    'frequency': len(login_times),
                    'avg_hour': np.mean([t.hour for t in login_times]),
                    'weekend_logins': sum(1 for t in login_times if t.weekday() >= 5),
                    'off_hours_logins': sum(1 for t in login_times if t.hour < 7 or t.hour > 19)
                }

            # Analyze access patterns
            access_events = [act for act in activities if act.get('type') == 'file_access']
            if access_events:
                features['access_patterns'] = {
                    'files_accessed': len(set(act.get('file_path', '') for act in access_events)),
                    'sensitive_files': sum(1 for act in access_events if act.get('sensitive', False)),
                    'download_volume': sum(act.get('size', 0) for act in access_events),
                    'external_transfers': sum(1 for act in access_events if act.get('external', False))
                }

            # Analyze communication patterns
            comm_events = [act for act in activities if act.get('type') == 'communication']
            if comm_events:
                features['communication_patterns'] = {
                    'emails_sent': len([act for act in comm_events if act.get('method') == 'email']),
                    'external_contacts': len(set(act.get('recipient', '') for act in comm_events if act.get('external', False))),
                    'suspicious_keywords': sum(1 for act in comm_events if self._has_suspicious_keywords(act.get('content', '')))
                }

            # Analyze temporal patterns
            if activities:
                timestamps = [act['timestamp'] for act in activities]
                features['temporal_patterns'] = {
                    'activity_hours': list(set(t.hour for t in timestamps)),
                    'activity_days': list(set(t.weekday() for t in timestamps)),
                    'burst_activity': self._detect_burst_activity(timestamps)
                }

        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")

        return features

    def _has_suspicious_keywords(self, content: str) -> bool:
        """Check for suspicious keywords in communication"""
        suspicious_keywords = [
            'confidential', 'secret', 'classified', 'proprietary',
            'password', 'credentials', 'salary', 'compensation',
            'competitor', 'resignation', 'termination', 'lawsuit'
        ]

        content_lower = content.lower()
        return any(keyword in content_lower for keyword in suspicious_keywords)

    def _detect_burst_activity(self, timestamps: List[datetime]) -> bool:
        """Detect burst activity patterns"""
        if len(timestamps) < 5:
            return False

        # Sort timestamps
        sorted_times = sorted(timestamps)

        # Check for high activity in short time windows
        for i in range(len(sorted_times) - 4):
            window_start = sorted_times[i]
            window_end = sorted_times[i + 4]
            window_duration = (window_end - window_start).total_seconds()

            # If 5 activities in less than 5 minutes, consider burst
            if window_duration < 300:
                return True

        return False

    async def _calculate_anomaly_score(self, profile: UserProfile) -> float:
        """Calculate anomaly score based on behavioral changes"""
        try:
            if not profile.baseline_behavior or not profile.current_behavior:
                return 0.0

            # Calculate differences in behavioral features
            anomaly_scores = []

            # Compare login patterns
            baseline_login = profile.baseline_behavior.get('login_patterns', {})
            current_login = profile.current_behavior.get('login_patterns', {})

            if baseline_login and current_login:
                login_anomaly = self._compare_patterns(baseline_login, current_login)
                anomaly_scores.append(login_anomaly)

            # Compare access patterns
            baseline_access = profile.baseline_behavior.get('access_patterns', {})
            current_access = profile.current_behavior.get('access_patterns', {})

            if baseline_access and current_access:
                access_anomaly = self._compare_patterns(baseline_access, current_access)
                anomaly_scores.append(access_anomaly)

            # Compare communication patterns
            baseline_comm = profile.baseline_behavior.get('communication_patterns', {})
            current_comm = profile.current_behavior.get('communication_patterns', {})

            if baseline_comm and current_comm:
                comm_anomaly = self._compare_patterns(baseline_comm, current_comm)
                anomaly_scores.append(comm_anomaly)

            # Return average anomaly score
            return np.mean(anomaly_scores) if anomaly_scores else 0.0

        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0

    def _compare_patterns(self, baseline: Dict, current: Dict) -> float:
        """Compare behavioral patterns and return anomaly score"""
        anomaly_score = 0.0
        compared_features = 0

        for key in baseline:
            if key in current:
                baseline_val = baseline[key]
                current_val = current[key]

                if isinstance(baseline_val, (int, float)) and isinstance(current_val, (int, float)):
                    if baseline_val > 0:
                        # Calculate relative change
                        change = abs(current_val - baseline_val) / baseline_val
                        anomaly_score += min(change, 1.0)  # Cap at 1.0
                        compared_features += 1

        return anomaly_score / compared_features if compared_features > 0 else 0.0

    async def _detect_threat_patterns(self, user_id: str, activities: List[Dict], features: Dict) -> List[str]:
        """Detect specific insider threat patterns"""
        threat_indicators = []

        try:
            # Data exfiltration patterns
            access_patterns = features.get('access_patterns', {})
            if access_patterns.get('download_volume', 0) > 1e9:  # > 1GB
                threat_indicators.append('large_data_download')

            if access_patterns.get('external_transfers', 0) > 10:
                threat_indicators.append('excessive_external_transfers')

            # Off-hours activity
            login_patterns = features.get('login_patterns', {})
            if login_patterns.get('off_hours_logins', 0) > 5:
                threat_indicators.append('unusual_login_times')

            # Suspicious communication
            comm_patterns = features.get('communication_patterns', {})
            if comm_patterns.get('suspicious_keywords', 0) > 3:
                threat_indicators.append('suspicious_communications')

            # Access to sensitive files
            if access_patterns.get('sensitive_files', 0) > 20:
                threat_indicators.append('excessive_sensitive_access')

            # Burst activity
            temporal_patterns = features.get('temporal_patterns', {})
            if temporal_patterns.get('burst_activity', False):
                threat_indicators.append('burst_activity_pattern')

            # Check for privilege escalation attempts
            escalation_attempts = [act for act in activities if act.get('type') == 'privilege_escalation']
            if len(escalation_attempts) > 0:
                threat_indicators.append('privilege_escalation_attempts')

            # Check for unusual file access patterns
            file_accesses = [act for act in activities if act.get('type') == 'file_access']
            unique_paths = set(act.get('file_path', '') for act in file_accesses)
            if len(unique_paths) > 100:  # Accessing many different files
                threat_indicators.append('widespread_file_access')

        except Exception as e:
            logger.error(f"Error detecting threat patterns: {e}")

        return threat_indicators

    def _identify_behavioral_changes(self, profile: UserProfile) -> List[str]:
        """Identify significant behavioral changes"""
        changes = []

        try:
            if not profile.baseline_behavior or not profile.current_behavior:
                return changes

            # Check for significant changes in each pattern category
            categories = ['login_patterns', 'access_patterns', 'communication_patterns']

            for category in categories:
                baseline = profile.baseline_behavior.get(category, {})
                current = profile.current_behavior.get(category, {})

                if baseline and current:
                    change_score = self._compare_patterns(baseline, current)
                    if change_score > 0.5:  # Significant change threshold
                        changes.append(f'significant_change_in_{category}')

        except Exception as e:
            logger.error(f"Error identifying behavioral changes: {e}")

        return changes

    def _update_risk_score(self, profile: UserProfile, anomaly_score: float) -> float:
        """Update user risk score based on anomaly score and history"""
        try:
            # Base risk from current anomaly
            new_risk = anomaly_score

            # Factor in historical anomalies
            recent_anomalies = [
                a['anomaly_score'] for a in profile.anomaly_history
                if (datetime.now() - a['timestamp']).days <= 30
            ]

            if recent_anomalies:
                historical_risk = np.mean(recent_anomalies)
                new_risk = 0.7 * new_risk + 0.3 * historical_risk

            # Factor in role-based risk
            role_risk_multipliers = {
                'admin': 1.5,
                'privileged_user': 1.3,
                'contractor': 1.2,
                'employee': 1.0,
                'guest': 0.8
            }

            role_multiplier = role_risk_multipliers.get(profile.role, 1.0)
            new_risk *= role_multiplier

            # Ensure risk score is between 0 and 1
            return max(0.0, min(1.0, new_risk))

        except Exception as e:
            logger.error(f"Error updating risk score: {e}")
            return profile.risk_score

    def _generate_recommendations(self, profile: UserProfile, threat_indicators: List[str]) -> List[str]:
        """Generate security recommendations based on analysis"""
        recommendations = []

        try:
            # High risk user recommendations
            if profile.risk_score > 0.8:
                recommendations.append('immediate_security_review')
                recommendations.append('enhanced_monitoring')

            # Specific threat indicator recommendations
            if 'large_data_download' in threat_indicators:
                recommendations.append('investigate_data_access')
                recommendations.append('review_data_classification')

            if 'unusual_login_times' in threat_indicators:
                recommendations.append('verify_login_authenticity')
                recommendations.append('implement_additional_authentication')

            if 'suspicious_communications' in threat_indicators:
                recommendations.append('review_communication_content')
                recommendations.append('interview_user')

            if 'privilege_escalation_attempts' in threat_indicators:
                recommendations.append('urgent_privilege_review')
                recommendations.append('potential_account_compromise')

            # General recommendations based on risk level
            if profile.risk_score > 0.6:
                recommendations.append('increased_audit_frequency')

            if profile.risk_score > 0.4:
                recommendations.append('periodic_security_training')

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")

        return recommendations

class BehaviorAnalysisNN(nn.Module):
    """Neural network for behavioral analysis"""

    def __init__(self, input_size=100, hidden_size=64, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Output probability
        return x

class DataExfiltrationPrevention:
    """Prevent and detect data exfiltration attempts"""

    def __init__(self):
        self.ml_model = None
        self.scaler = StandardScaler()
        self.blocked_transfers = set()
        self.monitored_extensions = {
            '.doc', '.docx', '.pdf', '.xls', '.xlsx', '.ppt', '.pptx',
            '.txt', '.csv', '.sql', '.zip', '.rar', '.7z'
        }
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:password|pwd|pass)\s*[:=]\s*\S+\b'  # Passwords
        ]
        self._initialize_models()

    def _initialize_models(self):
        """Initialize data exfiltration detection models"""
        try:
            # Initialize ML model for transfer analysis
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)

            # In production, load pre-trained model
            logger.info("Data exfiltration prevention models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize DLP models: {e}")

    async def analyze_data_transfer(self, transfer_event: Dict) -> Dict:
        """Analyze data transfer for potential exfiltration"""
        try:
            analysis_result = {
                'transfer_id': transfer_event.get('id'),
                'risk_score': 0.0,
                'blocked': False,
                'reasons': [],
                'recommendations': []
            }

            # Extract features
            features = self._extract_transfer_features(transfer_event)

            # Calculate risk score
            risk_score = await self._calculate_transfer_risk(features, transfer_event)
            analysis_result['risk_score'] = risk_score

            # Check if transfer should be blocked
            should_block, reasons = await self._should_block_transfer(transfer_event, risk_score)
            analysis_result['blocked'] = should_block
            analysis_result['reasons'] = reasons

            # Block transfer if necessary
            if should_block:
                await self._block_transfer(transfer_event)
                DATA_EXFILTRATION_ATTEMPTS.labels(method=transfer_event.get('method', 'unknown')).inc()

            # Generate recommendations
            analysis_result['recommendations'] = self._generate_dlp_recommendations(transfer_event, risk_score)

            return analysis_result

        except Exception as e:
            logger.error(f"Error analyzing data transfer: {e}")
            return {'error': str(e)}

    def _extract_transfer_features(self, transfer_event: Dict) -> np.ndarray:
        """Extract features for transfer analysis"""
        features = []

        try:
            # File size (normalized)
            file_size = transfer_event.get('file_size', 0)
            features.append(min(file_size / 1e9, 1.0))  # Normalize to GB

            # Time of day (0-1)
            timestamp = transfer_event.get('timestamp', datetime.now())
            features.append(timestamp.hour / 24.0)

            # Day of week (0-1)
            features.append(timestamp.weekday() / 6.0)

            # File type risk (0-1)
            file_path = transfer_event.get('file_path', '')
            file_ext = os.path.splitext(file_path)[1].lower()
            features.append(1.0 if file_ext in self.monitored_extensions else 0.0)

            # Destination type (0-1)
            destination = transfer_event.get('destination', '')
            features.append(1.0 if self._is_external_destination(destination) else 0.0)

            # User risk score
            user_risk = transfer_event.get('user_risk_score', 0.5)
            features.append(user_risk)

            # Volume in time window
            recent_volume = transfer_event.get('recent_transfer_volume', 0)
            features.append(min(recent_volume / 1e8, 1.0))  # Normalize to 100MB

            # Sensitive content indicators
            content_risk = transfer_event.get('content_risk_score', 0.0)
            features.append(content_risk)

            # Encryption status
            is_encrypted = transfer_event.get('encrypted', False)
            features.append(0.0 if is_encrypted else 1.0)

            # Access method risk
            method_risk = self._get_method_risk(transfer_event.get('method', ''))
            features.append(method_risk)

            # Pad to fixed size
            while len(features) < 20:
                features.append(0.0)

            return np.array(features[:20])

        except Exception as e:
            logger.error(f"Error extracting transfer features: {e}")
            return np.zeros(20)

    def _is_external_destination(self, destination: str) -> bool:
        """Check if destination is external"""
        external_indicators = [
            'gmail.com', 'yahoo.com', 'hotmail.com',
            'dropbox.com', 'drive.google.com', 'onedrive.com',
            'ftp://', 'sftp://', 'http://', 'https://'
        ]

        destination_lower = destination.lower()
        return any(indicator in destination_lower for indicator in external_indicators)

    def _get_method_risk(self, method: str) -> float:
        """Get risk score for transfer method"""
        method_risks = {
            'email': 0.8,
            'usb': 0.9,
            'cloud_storage': 0.7,
            'ftp': 0.6,
            'network_share': 0.3,
            'print': 0.5,
            'unknown': 0.5
        }

        return method_risks.get(method.lower(), 0.5)

    async def _calculate_transfer_risk(self, features: np.ndarray, transfer_event: Dict) -> float:
        """Calculate transfer risk score"""
        try:
            # Base risk from features
            base_risk = np.mean(features)

            # Content analysis risk
            content_risk = await self._analyze_content_risk(transfer_event)

            # Behavioral risk
            behavioral_risk = await self._analyze_behavioral_risk(transfer_event)

            # Combined risk (weighted average)
            combined_risk = (
                base_risk * 0.4 +
                content_risk * 0.4 +
                behavioral_risk * 0.2
            )

            return min(combined_risk, 1.0)

        except Exception as e:
            logger.error(f"Error calculating transfer risk: {e}")
            return 0.5

    async def _analyze_content_risk(self, transfer_event: Dict) -> float:
        """Analyze content for sensitive information"""
        try:
            file_path = transfer_event.get('file_path', '')

            if not os.path.exists(file_path):
                return 0.0

            # Read file content (sample)
            content_sample = await self._get_file_sample(file_path)

            if not content_sample:
                return 0.0

            # Check for sensitive patterns
            risk_score = 0.0
            content_str = content_sample.decode('utf-8', errors='ignore')

            for pattern in self.sensitive_patterns:
                matches = re.findall(pattern, content_str, re.IGNORECASE)
                if matches:
                    risk_score += min(len(matches) * 0.1, 0.3)

            # Check for classification markings
            classification_markers = ['confidential', 'secret', 'restricted', 'proprietary']
            for marker in classification_markers:
                if marker.lower() in content_str.lower():
                    risk_score += 0.2

            return min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Error analyzing content risk: {e}")
            return 0.0

    async def _get_file_sample(self, file_path: str, sample_size: int = 8192) -> bytes:
        """Get sample of file content"""
        try:
            with open(file_path, 'rb') as f:
                return f.read(sample_size)
        except Exception as e:
            logger.error(f"Error reading file sample: {e}")
            return b''

    async def _analyze_behavioral_risk(self, transfer_event: Dict) -> float:
        """Analyze behavioral risk factors"""
        try:
            risk_score = 0.0

            # Time-based risk
            timestamp = transfer_event.get('timestamp', datetime.now())
            if timestamp.hour < 7 or timestamp.hour > 19:
                risk_score += 0.3  # Off-hours transfer

            if timestamp.weekday() >= 5:
                risk_score += 0.2  # Weekend transfer

            # Volume-based risk
            file_size = transfer_event.get('file_size', 0)
            if file_size > 1e8:  # > 100MB
                risk_score += 0.3

            # Frequency-based risk
            recent_transfers = transfer_event.get('recent_transfer_count', 0)
            if recent_transfers > 10:
                risk_score += 0.4

            # User context risk
            user_department = transfer_event.get('user_department', '')
            if user_department.lower() in ['it', 'admin', 'security']:
                risk_score += 0.1  # Privileged users have higher risk

            return min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Error analyzing behavioral risk: {e}")
            return 0.0

    async def _should_block_transfer(self, transfer_event: Dict, risk_score: float) -> Tuple[bool, List[str]]:
        """Determine if transfer should be blocked"""
        reasons = []
        should_block = False

        try:
            # High risk threshold
            if risk_score > 0.8:
                should_block = True
                reasons.append('high_risk_score')

            # File size threshold
            file_size = transfer_event.get('file_size', 0)
            if file_size > 5e8:  # > 500MB
                should_block = True
                reasons.append('large_file_size')

            # Sensitive content threshold
            content_risk = transfer_event.get('content_risk_score', 0.0)
            if content_risk > 0.7:
                should_block = True
                reasons.append('sensitive_content_detected')

            # External destination with sensitive data
            destination = transfer_event.get('destination', '')
            if self._is_external_destination(destination) and content_risk > 0.4:
                should_block = True
                reasons.append('external_transfer_with_sensitive_data')

            # User risk threshold
            user_risk = transfer_event.get('user_risk_score', 0.5)
            if user_risk > 0.8 and risk_score > 0.5:
                should_block = True
                reasons.append('high_risk_user')

            # Policy violations
            if self._violates_data_policy(transfer_event):
                should_block = True
                reasons.append('policy_violation')

        except Exception as e:
            logger.error(f"Error determining transfer block: {e}")

        return should_block, reasons

    def _violates_data_policy(self, transfer_event: Dict) -> bool:
        """Check if transfer violates data policies"""
        try:
            # Check classification policies
            classification = transfer_event.get('file_classification', 'public')
            destination = transfer_event.get('destination', '')

            # Secret/confidential data cannot go to external destinations
            if classification in ['secret', 'confidential'] and self._is_external_destination(destination):
                return True

            # Check time-based restrictions
            timestamp = transfer_event.get('timestamp', datetime.now())
            if classification == 'secret' and (timestamp.hour < 8 or timestamp.hour > 17):
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking policy violations: {e}")
            return False

    async def _block_transfer(self, transfer_event: Dict):
        """Block the data transfer"""
        try:
            transfer_id = transfer_event.get('id')
            self.blocked_transfers.add(transfer_id)

            # Log the block event
            logger.warning(f"Blocked data transfer: {transfer_id}")

            # Notify security team
            await self._notify_security_team(transfer_event)

        except Exception as e:
            logger.error(f"Error blocking transfer: {e}")

    async def _notify_security_team(self, transfer_event: Dict):
        """Notify security team of blocked transfer"""
        try:
            notification = {
                'type': 'data_exfiltration_blocked',
                'transfer_id': transfer_event.get('id'),
                'user_id': transfer_event.get('user_id'),
                'file_path': transfer_event.get('file_path'),
                'destination': transfer_event.get('destination'),
                'timestamp': datetime.now().isoformat(),
                'risk_score': transfer_event.get('risk_score', 0.0)
            }

            # Send notification (integrate with alerting system)
            logger.info(f"Security notification sent: {notification}")

        except Exception as e:
            logger.error(f"Error sending security notification: {e}")

    def _generate_dlp_recommendations(self, transfer_event: Dict, risk_score: float) -> List[str]:
        """Generate DLP recommendations"""
        recommendations = []

        try:
            if risk_score > 0.7:
                recommendations.append('investigate_user_intent')
                recommendations.append('review_file_classification')

            if transfer_event.get('file_size', 0) > 1e8:
                recommendations.append('verify_business_justification')

            if self._is_external_destination(transfer_event.get('destination', '')):
                recommendations.append('review_external_sharing_policy')

            if transfer_event.get('content_risk_score', 0.0) > 0.5:
                recommendations.append('data_classification_review')

        except Exception as e:
            logger.error(f"Error generating DLP recommendations: {e}")

        return recommendations

class CommunicationSecurityEnforcer:
    """Monitor and secure organizational communications"""

    def __init__(self):
        self.monitored_channels = ['email', 'chat', 'video_call', 'file_share']
        self.keywords_db = self._load_security_keywords()
        self.encryption_enforced = True

    def _load_security_keywords(self) -> Dict[str, List[str]]:
        """Load security-relevant keywords for monitoring"""
        return {
            'confidential': ['confidential', 'secret', 'classified', 'proprietary', 'restricted'],
            'threat': ['threat', 'attack', 'breach', 'hack', 'exploit', 'vulnerability'],
            'financial': ['salary', 'budget', 'revenue', 'profit', 'cost', 'price'],
            'legal': ['lawsuit', 'legal', 'litigation', 'contract', 'agreement'],
            'personnel': ['resignation', 'termination', 'firing', 'layoff', 'promotion']
        }

    async def monitor_communication(self, comm_event: CommunicationEvent) -> Dict:
        """Monitor communication for security concerns"""
        try:
            analysis_result = {
                'communication_id': comm_event.id,
                'risk_score': 0.0,
                'flagged': False,
                'categories': [],
                'recommendations': []
            }

            # Analyze content
            content_analysis = await self._analyze_communication_content(comm_event)
            analysis_result.update(content_analysis)

            # Check participants
            participant_risk = await self._analyze_participants(comm_event)
            analysis_result['participant_risk'] = participant_risk

            # Calculate overall risk
            overall_risk = (content_analysis['risk_score'] + participant_risk) / 2
            analysis_result['risk_score'] = overall_risk

            # Flag if necessary
            if overall_risk > 0.6:
                analysis_result['flagged'] = True
                COMMUNICATION_INTERCEPTS.labels(type=comm_event.communication_type).inc()

            # Generate recommendations
            analysis_result['recommendations'] = self._generate_comm_recommendations(analysis_result)

            return analysis_result

        except Exception as e:
            logger.error(f"Error monitoring communication: {e}")
            return {'error': str(e)}

    async def _analyze_communication_content(self, comm_event: CommunicationEvent) -> Dict:
        """Analyze communication content for security risks"""
        try:
            content = comm_event.content_summary
            risk_score = 0.0
            categories = []

            # Check for security keywords
            for category, keywords in self.keywords_db.items():
                for keyword in keywords:
                    if keyword.lower() in content.lower():
                        categories.append(category)
                        risk_score += 0.2
                        break

            # Check for data patterns
            if self._contains_sensitive_data(content):
                categories.append('sensitive_data')
                risk_score += 0.3

            # Check for external domains
            if self._contains_external_domains(content):
                categories.append('external_communication')
                risk_score += 0.1

            # Normalize risk score
            risk_score = min(risk_score, 1.0)

            return {
                'risk_score': risk_score,
                'categories': categories
            }

        except Exception as e:
            logger.error(f"Error analyzing communication content: {e}")
            return {'risk_score': 0.0, 'categories': []}

    def _contains_sensitive_data(self, content: str) -> bool:
        """Check if content contains sensitive data patterns"""
        patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Z]{2}\d{8}\b',  # Passport-like
            r'\b(?:password|pwd|pass|secret|key)\s*[:=]\s*\S+\b'  # Credentials
        ]

        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _contains_external_domains(self, content: str) -> bool:
        """Check if content contains external email domains"""
        external_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'protonmail.com'
        ]

        content_lower = content.lower()
        return any(domain in content_lower for domain in external_domains)

    async def _analyze_participants(self, comm_event: CommunicationEvent) -> float:
        """Analyze communication participants for risk"""
        try:
            risk_score = 0.0

            # Check for external participants
            external_participants = [
                p for p in comm_event.participants
                if self._is_external_participant(p)
            ]

            if external_participants:
                risk_score += 0.3

            # Check for high-risk participants
            # This would integrate with user risk scoring
            high_risk_participants = [
                p for p in comm_event.participants
                if await self._is_high_risk_user(p)
            ]

            if high_risk_participants:
                risk_score += 0.4

            # Check for unusual participant combinations
            if len(comm_event.participants) > 10:
                risk_score += 0.1  # Large group communications

            return min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Error analyzing participants: {e}")
            return 0.0

    def _is_external_participant(self, participant: str) -> bool:
        """Check if participant is external"""
        # Simple check - in production, integrate with directory service
        return '@' in participant and not participant.endswith('@company.com')

    async def _is_high_risk_user(self, user_id: str) -> bool:
        """Check if user is high risk"""
        # Placeholder - integrate with behavioral analysis
        return False

    def _generate_comm_recommendations(self, analysis_result: Dict) -> List[str]:
        """Generate communication security recommendations"""
        recommendations = []

        try:
            risk_score = analysis_result.get('risk_score', 0.0)
            categories = analysis_result.get('categories', [])

            if risk_score > 0.8:
                recommendations.append('immediate_review_required')

            if 'sensitive_data' in categories:
                recommendations.append('encrypt_communication')
                recommendations.append('review_data_sharing_policy')

            if 'external_communication' in categories:
                recommendations.append('verify_external_recipient')

            if 'confidential' in categories:
                recommendations.append('check_classification_handling')

            if risk_score > 0.5:
                recommendations.append('enhanced_monitoring')

        except Exception as e:
            logger.error(f"Error generating communication recommendations: {e}")

        return recommendations

class AssetTrackingProtection:
    """Track and protect organizational assets"""

    def __init__(self):
        self.assets: Dict[str, Dict] = {}
        self.access_logs: List[DataAccessEvent] = []

    async def track_asset_access(self, access_event: DataAccessEvent) -> Dict:
        """Track access to organizational assets"""
        try:
            # Store access event
            self.access_logs.append(access_event)

            # Analyze access pattern
            analysis = await self._analyze_asset_access(access_event)

            # Check for violations
            violations = await self._check_access_violations(access_event)

            # Update asset metadata
            await self._update_asset_metadata(access_event)

            if violations:
                ASSET_ACCESS_VIOLATIONS.labels(asset_type=access_event.asset_type).inc()

            return {
                'access_id': access_event.id,
                'authorized': access_event.authorized,
                'violations': violations,
                'risk_score': analysis.get('risk_score', 0.0),
                'recommendations': analysis.get('recommendations', [])
            }

        except Exception as e:
            logger.error(f"Error tracking asset access: {e}")
            return {'error': str(e)}

    async def _analyze_asset_access(self, access_event: DataAccessEvent) -> Dict:
        """Analyze asset access patterns"""
        try:
            risk_score = 0.0
            recommendations = []

            # Check classification vs clearance
            asset_level = access_event.classification.value
            user_clearance = access_event.context.get('user_clearance', 'public')

            classification_hierarchy = {
                'public': 0,
                'restricted': 1,
                'confidential': 2,
                'secret': 3,
                'top_secret': 4
            }

            asset_level_num = classification_hierarchy.get(asset_level, 0)
            user_level_num = classification_hierarchy.get(user_clearance, 0)

            if asset_level_num > user_level_num:
                risk_score += 0.8
                recommendations.append('clearance_violation')

            # Check access time
            if access_event.timestamp.hour < 7 or access_event.timestamp.hour > 19:
                risk_score += 0.2
                recommendations.append('off_hours_access')

            # Check location
            if access_event.location not in ['office', 'secure_facility']:
                risk_score += 0.3
                recommendations.append('remote_access_review')

            # Check action appropriateness
            if access_event.action in ['delete', 'modify'] and asset_level in ['secret', 'top_secret']:
                risk_score += 0.4
                recommendations.append('high_impact_action')

            return {
                'risk_score': min(risk_score, 1.0),
                'recommendations': recommendations
            }

        except Exception as e:
            logger.error(f"Error analyzing asset access: {e}")
            return {'risk_score': 0.0, 'recommendations': []}

    async def _check_access_violations(self, access_event: DataAccessEvent) -> List[str]:
        """Check for access policy violations"""
        violations = []

        try:
            # Unauthorized access
            if not access_event.authorized:
                violations.append('unauthorized_access')

            # Classification violations
            asset_class = access_event.classification
            user_clearance = access_event.context.get('user_clearance', 'public')

            if asset_class == AssetClassification.SECRET and user_clearance not in ['secret', 'top_secret']:
                violations.append('insufficient_clearance')

            # Time-based violations
            if asset_class in [AssetClassification.SECRET, AssetClassification.TOP_SECRET]:
                if access_event.timestamp.hour < 8 or access_event.timestamp.hour > 17:
                    violations.append('restricted_hours_violation')

            # Location violations
            if asset_class == AssetClassification.TOP_SECRET and access_event.location != 'secure_facility':
                violations.append('location_violation')

            # Device violations
            approved_devices = access_event.context.get('approved_devices', [])
            if access_event.device_id not in approved_devices and asset_class != AssetClassification.PUBLIC:
                violations.append('unapproved_device')

        except Exception as e:
            logger.error(f"Error checking access violations: {e}")

        return violations

    async def _update_asset_metadata(self, access_event: DataAccessEvent):
        """Update asset access metadata"""
        try:
            asset_id = access_event.asset_id

            if asset_id not in self.assets:
                self.assets[asset_id] = {
                    'id': asset_id,
                    'type': access_event.asset_type,
                    'classification': access_event.classification.value,
                    'access_count': 0,
                    'last_accessed': None,
                    'access_history': []
                }

            asset = self.assets[asset_id]
            asset['access_count'] += 1
            asset['last_accessed'] = access_event.timestamp
            asset['access_history'].append({
                'user_id': access_event.user_id,
                'timestamp': access_event.timestamp,
                'action': access_event.action,
                'authorized': access_event.authorized
            })

            # Keep only recent history
            if len(asset['access_history']) > 100:
                asset['access_history'] = asset['access_history'][-100:]

        except Exception as e:
            logger.error(f"Error updating asset metadata: {e}")

class SupplyChainSecurityMonitor:
    """Monitor supply chain security"""

    def __init__(self):
        self.supply_chain_assets: Dict[str, SupplyChainAsset] = {}
        self.vulnerability_db = {}

    async def monitor_supply_chain_asset(self, asset: SupplyChainAsset) -> Dict:
        """Monitor supply chain asset for security risks"""
        try:
            # Store asset
            self.supply_chain_assets[asset.id] = asset

            # Check for vulnerabilities
            vulnerabilities = await self._check_vulnerabilities(asset)

            # Update risk score
            risk_score = await self._calculate_supply_chain_risk(asset, vulnerabilities)

            # Check compliance
            compliance_status = await self._check_compliance(asset)

            # Generate alerts if necessary
            if risk_score > 0.7:
                SUPPLY_CHAIN_ALERTS.labels(vendor=asset.vendor).inc()

            return {
                'asset_id': asset.id,
                'risk_score': risk_score,
                'vulnerabilities': vulnerabilities,
                'compliance_status': compliance_status,
                'recommendations': self._generate_supply_chain_recommendations(asset, risk_score)
            }

        except Exception as e:
            logger.error(f"Error monitoring supply chain asset: {e}")
            return {'error': str(e)}

    async def _check_vulnerabilities(self, asset: SupplyChainAsset) -> List[str]:
        """Check asset for known vulnerabilities"""
        # Placeholder - integrate with CVE databases
        vulnerabilities = []

        try:
            # Check version against known vulnerabilities
            if asset.version == "1.0.0":  # Example vulnerable version
                vulnerabilities.append("CVE-2023-12345")

            # Check vendor security advisories
            # This would integrate with vendor APIs or feeds

        except Exception as e:
            logger.error(f"Error checking vulnerabilities: {e}")

        return vulnerabilities

    async def _calculate_supply_chain_risk(self, asset: SupplyChainAsset, vulnerabilities: List[str]) -> float:
        """Calculate supply chain risk score"""
        try:
            risk_score = 0.0

            # Vulnerability risk
            if vulnerabilities:
                risk_score += min(len(vulnerabilities) * 0.2, 0.6)

            # Age risk
            age_days = (datetime.now() - asset.installation_date).days
            if age_days > 365:  # > 1 year old
                risk_score += 0.2

            # Update risk
            update_days = (datetime.now() - asset.last_updated).days
            if update_days > 90:  # > 3 months since update
                risk_score += 0.3

            # Vendor risk (based on reputation)
            vendor_risk = self._get_vendor_risk(asset.vendor)
            risk_score += vendor_risk * 0.2

            return min(risk_score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating supply chain risk: {e}")
            return 0.5

    def _get_vendor_risk(self, vendor: str) -> float:
        """Get vendor risk score"""
        # Placeholder - in production, integrate with vendor risk database
        high_risk_vendors = ['unknown_vendor', 'compromised_vendor']

        if vendor.lower() in high_risk_vendors:
            return 1.0

        return 0.3  # Default moderate risk

    async def _check_compliance(self, asset: SupplyChainAsset) -> str:
        """Check asset compliance status"""
        try:
            # Check if asset meets compliance requirements
            compliance_checks = [
                self._check_security_standards(asset),
                self._check_update_policy(asset),
                self._check_vulnerability_management(asset)
            ]

            if all(compliance_checks):
                return "compliant"
            elif any(compliance_checks):
                return "partial_compliance"
            else:
                return "non_compliant"

        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return "unknown"

    def _check_security_standards(self, asset: SupplyChainAsset) -> bool:
        """Check if asset meets security standards"""
        # Placeholder compliance check
        return asset.monitoring_enabled

    def _check_update_policy(self, asset: SupplyChainAsset) -> bool:
        """Check if asset follows update policy"""
        days_since_update = (datetime.now() - asset.last_updated).days
        return days_since_update <= 90  # Must be updated within 90 days

    def _check_vulnerability_management(self, asset: SupplyChainAsset) -> bool:
        """Check vulnerability management compliance"""
        return len(asset.vulnerabilities) == 0  # No unpatched vulnerabilities

    def _generate_supply_chain_recommendations(self, asset: SupplyChainAsset, risk_score: float) -> List[str]:
        """Generate supply chain security recommendations"""
        recommendations = []

        try:
            if risk_score > 0.8:
                recommendations.append('immediate_security_review')
                recommendations.append('consider_replacement')

            if asset.vulnerabilities:
                recommendations.append('patch_vulnerabilities')

            days_since_update = (datetime.now() - asset.last_updated).days
            if days_since_update > 90:
                recommendations.append('update_asset')

            if not asset.monitoring_enabled:
                recommendations.append('enable_monitoring')

            if asset.compliance_status != "compliant":
                recommendations.append('address_compliance_gaps')

        except Exception as e:
            logger.error(f"Error generating supply chain recommendations: {e}")

        return recommendations

class OpsecEnforcer:
    """Main OPSEC enforcement orchestrator"""

    def __init__(self, security_framework: OperationalSecurityFramework):
        self.security_framework = security_framework
        self.behavioral_analyzer = BehavioralAnalysisEngine()
        self.dlp_system = DataExfiltrationPrevention()
        self.comm_enforcer = CommunicationSecurityEnforcer()
        self.asset_tracker = AssetTrackingProtection()
        self.supply_chain_monitor = SupplyChainSecurityMonitor()

        self.db_pool = None
        self.redis_client = None

    async def initialize(self, redis_url: str = "redis://localhost:6379",
                        db_url: str = "postgresql://user:pass@localhost/bev"):
        """Initialize the OPSEC enforcer"""
        try:
            # Initialize database connections
            self.redis_client = redis.from_url(redis_url)
            self.db_pool = await asyncpg.create_pool(db_url)

            logger.info("OPSEC Enforcer initialized")
            print("🔐 OPSEC Enforcer Ready")

        except Exception as e:
            logger.error(f"Failed to initialize OPSEC enforcer: {e}")
            raise

    async def process_user_activity(self, user_id: str, activities: List[Dict]) -> Dict:
        """Process user activities for insider threat detection"""
        try:
            # Analyze behavior
            behavior_analysis = await self.behavioral_analyzer.analyze_user_behavior(user_id, activities)

            # Check for insider threats
            if behavior_analysis.get('risk_score', 0.0) > 0.7:
                threat_event = InsiderThreatEvent(
                    id=hashlib.sha256(f"{user_id}_{datetime.now()}".encode()).hexdigest()[:16],
                    user_id=user_id,
                    timestamp=datetime.now(),
                    threat_type=ThreatType.POLICY_VIOLATION,  # Would be determined by analysis
                    risk_level=RiskLevel.HIGH,
                    confidence=behavior_analysis.get('risk_score', 0.0),
                    description="Unusual behavioral patterns detected",
                    evidence=behavior_analysis.get('threat_indicators', []),
                    indicators=behavior_analysis,
                    response_actions=behavior_analysis.get('recommendations', [])
                )

                # Store threat event
                await self._store_insider_threat_event(threat_event)

                # Record metrics
                INSIDER_THREATS.labels(
                    user=user_id,
                    risk_level=threat_event.risk_level.value
                ).inc()

            return behavior_analysis

        except Exception as e:
            logger.error(f"Error processing user activity: {e}")
            return {'error': str(e)}

    async def _store_insider_threat_event(self, event: InsiderThreatEvent):
        """Store insider threat event in database"""
        try:
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO insider_threat_events (
                            id, user_id, timestamp, threat_type, risk_level,
                            confidence, description, evidence, indicators, response_actions
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    event.id, event.user_id, event.timestamp, event.threat_type.value,
                    event.risk_level.value, event.confidence, event.description,
                    json.dumps(event.evidence), json.dumps(event.indicators),
                    json.dumps(event.response_actions)
                    )

        except Exception as e:
            logger.error(f"Error storing insider threat event: {e}")

    async def get_security_status(self) -> Dict:
        """Get overall security status"""
        try:
            return {
                'active_monitoring': True,
                'user_profiles': len(self.behavioral_analyzer.user_profiles),
                'blocked_transfers': len(self.dlp_system.blocked_transfers),
                'tracked_assets': len(self.asset_tracker.assets),
                'supply_chain_assets': len(self.supply_chain_monitor.supply_chain_assets),
                'system_health': 'operational'
            }

        except Exception as e:
            logger.error(f"Error getting security status: {e}")
            return {'error': str(e)}

    async def shutdown(self):
        """Shutdown OPSEC enforcer"""
        try:
            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            if self.redis_client:
                await self.redis_client.close()

            logger.info("OPSEC Enforcer shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Example usage
async def main():
    """Example usage of the OPSEC Enforcer"""
    try:
        # Initialize security framework
        security = OperationalSecurityFramework()
        await security.initialize_security()

        # Initialize OPSEC enforcer
        enforcer = OpsecEnforcer(security)
        await enforcer.initialize()

        # Example user activities
        activities = [
            {
                'type': 'login',
                'timestamp': datetime.now(),
                'location': 'office'
            },
            {
                'type': 'file_access',
                'timestamp': datetime.now(),
                'file_path': '/confidential/project_x.doc',
                'sensitive': True,
                'size': 1024000
            }
        ]

        # Process user activity
        result = await enforcer.process_user_activity('user123', activities)
        print(f"✅ User activity analysis: {result}")

        # Get security status
        status = await enforcer.get_security_status()
        print(f"📊 Security Status: {status}")

        # Shutdown
        await enforcer.shutdown()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())