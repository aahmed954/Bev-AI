#!/usr/bin/env python3
"""
Guardian - Security Enforcement and Anomaly Detection
PII protection, behavioral baselines, access control, audit logging
"""

import asyncio
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import spacy
from cryptography.fernet import Fernet
import jwt
import redis.asyncio as redis
import asyncpg
from collections import defaultdict, deque
import yaml
import pickle

@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    rules: List[Dict[str, Any]]
    actions: List[str]  # 'block', 'alert', 'log', 'sanitize'
    severity: str  # 'critical', 'high', 'medium', 'low'
    enabled: bool = True

@dataclass
class AccessRequest:
    """Access control request"""
    user_id: str
    resource: str
    action: str  # 'read', 'write', 'execute', 'delete'
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class AnomalyEvent:
    """Detected anomaly event"""
    event_type: str
    severity: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: datetime
    user_id: Optional[str] = None
    remediation: Optional[List[str]] = None

class PIIDetector:
    """Advanced PII detection with NER and pattern matching"""
    
    def __init__(self):
        # Load spaCy model for named entity recognition
        self.nlp = spacy.load('en_core_web_trf')
        
        # Enhanced PII patterns
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            'passport': r'\b[A-Z][0-9]{8}\b',
            'driver_license': r'\b[A-Z]\d{7}|\b[A-Z]\d{12}\b',
            'bank_account': r'\b\d{8,17}\b',
            'routing_number': r'\b\d{9}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'ipv6': r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',
            'mac_address': r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
            'bitcoin': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'ethereum': r'\b0x[a-fA-F0-9]{40}\b',
            'api_key': r'\b[A-Za-z0-9]{32,}\b',
            'jwt_token': r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b',
            'coordinates': r'\b[-]?\d{1,3}\.\d+,\s*[-]?\d{1,3}\.\d+\b'
        }
        
        # Sensitive keywords that might indicate PII context
        self.sensitive_keywords = {
            'financial': ['bank', 'account', 'credit', 'debit', 'payment', 'salary', 'income'],
            'medical': ['diagnosis', 'prescription', 'medical', 'health', 'patient', 'treatment'],
            'personal': ['birth', 'age', 'gender', 'race', 'religion', 'political', 'sexual'],
            'location': ['address', 'residence', 'home', 'location', 'coordinates', 'gps'],
            'identity': ['passport', 'license', 'identification', 'citizen', 'immigration']
        }
        
    def detect_pii(self, text: str) -> Dict[str, List[Tuple[str, str, int, int]]]:
        """
        Detect PII in text
        Returns: Dict mapping PII type to list of (value, context, start, end)
        """
        detected_pii = defaultdict(list)
        
        # Pattern-based detection
        for pii_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detected_pii[pii_type].append((
                    match.group(),
                    text[max(0, match.start()-20):min(len(text), match.end()+20)],
                    match.start(),
                    match.end()
                ))
        
        # NER-based detection
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC']:
                detected_pii[f'ner_{ent.label_.lower()}'].append((
                    ent.text,
                    text[max(0, ent.start_char-20):min(len(text), ent.end_char+20)],
                    ent.start_char,
                    ent.end_char
                ))
        
        # Context-based detection
        text_lower = text.lower()
        for category, keywords in self.sensitive_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Flag potential sensitive content based on context
                    idx = text_lower.find(keyword)
                    context = text[max(0, idx-50):min(len(text), idx+50)]
                    detected_pii[f'context_{category}'].append((
                        keyword,
                        context,
                        idx,
                        idx + len(keyword)
                    ))
        
        return dict(detected_pii)
    
    def sanitize_text(self, text: str, pii_data: Dict[str, List]) -> str:
        """Sanitize text by replacing PII with tokens"""
        sanitized = text
        
        # Sort by position to replace from end to beginning
        all_pii = []
        for pii_type, items in pii_data.items():
            for value, context, start, end in items:
                all_pii.append((start, end, pii_type, value))
        
        all_pii.sort(reverse=True)
        
        for start, end, pii_type, value in all_pii:
            token = f"[{pii_type.upper()}_REDACTED_{hashlib.md5(value.encode()).hexdigest()[:8]}]"
            sanitized = sanitized[:start] + token + sanitized[end:]
        
        return sanitized

class AnomalyDetector:
    """Behavioral anomaly detection system"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.behavior_history = defaultdict(lambda: deque(maxlen=window_size))
        self.models = {}
        self.baselines = {}
        self.scaler = StandardScaler()
        
    def extract_features(self, event: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from event"""
        features = []
        
        # Time-based features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        features.extend([
            hour,
            day_of_week,
            1 if hour < 6 or hour > 22 else 0,  # Unusual time flag
            1 if day_of_week >= 5 else 0  # Weekend flag
        ])
        
        # Event-specific features
        features.append(len(event.get('data', '')))
        features.append(event.get('request_rate', 0))
        features.append(event.get('error_count', 0))
        features.append(event.get('unique_ips', 1))
        features.append(event.get('data_volume', 0))
        
        # Access pattern features
        features.append(event.get('failed_attempts', 0))
        features.append(event.get('privilege_escalations', 0))
        features.append(event.get('resource_access_diversity', 0))
        
        return np.array(features).reshape(1, -1)
    
    def update_baseline(self, user_id: str, event: Dict[str, Any]):
        """Update behavioral baseline for user"""
        features = self.extract_features(event)
        self.behavior_history[user_id].append(features[0])
        
        # Retrain model if enough data
        if len(self.behavior_history[user_id]) >= 100:
            history_matrix = np.vstack(list(self.behavior_history[user_id]))
            
            # Fit Isolation Forest for anomaly detection
            model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Scale features
            scaled_features = self.scaler.fit_transform(history_matrix)
            model.fit(scaled_features)
            
            self.models[user_id] = model
            self.baselines[user_id] = {
                'mean': history_matrix.mean(axis=0),
                'std': history_matrix.std(axis=0),
                'median': np.median(history_matrix, axis=0)
            }
    
    def detect_anomaly(self, user_id: str, event: Dict[str, Any]) -> Optional[AnomalyEvent]:
        """Detect if event is anomalous"""
        features = self.extract_features(event)
        
        if user_id not in self.models:
            # Not enough data for baseline
            self.update_baseline(user_id, event)
            return None
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict using Isolation Forest
        anomaly_score = self.models[user_id].decision_function(scaled_features)[0]
        is_anomaly = self.models[user_id].predict(scaled_features)[0] == -1
        
        if is_anomaly:
            # Calculate deviation from baseline
            baseline = self.baselines[user_id]
            deviations = np.abs(features[0] - baseline['mean']) / (baseline['std'] + 1e-6)
            
            # Identify specific anomalous features
            anomalous_features = []
            feature_names = [
                'hour', 'day_of_week', 'unusual_time', 'weekend',
                'data_length', 'request_rate', 'error_count', 
                'unique_ips', 'data_volume', 'failed_attempts',
                'privilege_escalations', 'resource_diversity'
            ]
            
            for i, (feat_name, deviation) in enumerate(zip(feature_names, deviations)):
                if deviation > 3:  # 3 standard deviations
                    anomalous_features.append({
                        'feature': feat_name,
                        'deviation': float(deviation),
                        'value': float(features[0][i]),
                        'baseline': float(baseline['mean'][i])
                    })
            
            return AnomalyEvent(
                event_type='behavioral_anomaly',
                severity=min(1.0, abs(anomaly_score) * 2),
                details={
                    'anomaly_score': float(anomaly_score),
                    'anomalous_features': anomalous_features,
                    'event': event
                },
                timestamp=datetime.now(),
                user_id=user_id,
                remediation=[
                    'Review user activity logs',
                    'Verify user identity',
                    'Check for account compromise',
                    'Enable additional monitoring'
                ]
            )
        
        # Update baseline with normal behavior
        self.update_baseline(user_id, event)
        return None

class RBACManager:
    """Role-Based Access Control system"""
    
    def __init__(self):
        self.roles = {}
        self.permissions = {}
        self.user_roles = defaultdict(set)
        self.resource_permissions = defaultdict(dict)
        
        # Initialize default roles
        self._initialize_default_roles()
        
    def _initialize_default_roles(self):
        """Setup default security roles"""
        self.roles = {
            'admin': {
                'permissions': ['*'],  # All permissions
                'priority': 100
            },
            'researcher': {
                'permissions': [
                    'research:read', 'research:write',
                    'osint:read', 'osint:execute',
                    'memory:read', 'memory:write'
                ],
                'priority': 50
            },
            'analyst': {
                'permissions': [
                    'research:read', 'osint:read',
                    'memory:read', 'report:write'
                ],
                'priority': 30
            },
            'viewer': {
                'permissions': [
                    'research:read', 'report:read'
                ],
                'priority': 10
            },
            'blocked': {
                'permissions': [],
                'priority': 0
            }
        }
        
        # Define resource-specific permissions
        self.resource_permissions = {
            'osint_tools': {
                'breach_databases': ['admin', 'researcher'],
                'darknet_access': ['admin'],
                'social_scraping': ['admin', 'researcher'],
                'infrastructure_scan': ['admin', 'researcher', 'analyst']
            },
            'memory_systems': {
                'write': ['admin', 'researcher'],
                'read': ['admin', 'researcher', 'analyst', 'viewer'],
                'delete': ['admin']
            },
            'agent_control': {
                'start': ['admin', 'researcher'],
                'stop': ['admin'],
                'configure': ['admin']
            }
        }
    
    def assign_role(self, user_id: str, role: str):
        """Assign role to user"""
        if role in self.roles:
            self.user_roles[user_id].add(role)
    
    def revoke_role(self, user_id: str, role: str):
        """Revoke role from user"""
        self.user_roles[user_id].discard(role)
    
    def check_permission(self, request: AccessRequest) -> Tuple[bool, str]:
        """Check if access request is permitted"""
        user_roles = self.user_roles.get(request.user_id, set())
        
        if 'blocked' in user_roles:
            return False, "User is blocked"
        
        if 'admin' in user_roles:
            return True, "Admin access granted"
        
        # Check specific permission
        permission_string = f"{request.resource}:{request.action}"
        
        for role in user_roles:
            role_perms = self.roles.get(role, {}).get('permissions', [])
            
            if '*' in role_perms or permission_string in role_perms:
                return True, f"Access granted via {role} role"
            
            # Check wildcard permissions
            resource_wildcard = f"{request.resource}:*"
            if resource_wildcard in role_perms:
                return True, f"Access granted via {role} role (wildcard)"
        
        # Check resource-specific permissions
        if request.resource in self.resource_permissions:
            resource_perms = self.resource_permissions[request.resource]
            if request.action in resource_perms:
                allowed_roles = resource_perms[request.action]
                if any(role in user_roles for role in allowed_roles):
                    return True, "Resource-specific access granted"
        
        return False, f"Permission denied for {permission_string}"

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, db_url: str, encryption_key: bytes = None):
        self.db_url = db_url
        self.pool = None
        
        # Setup encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())
        
        # In-memory buffer for performance
        self.log_buffer = deque(maxlen=1000)
        self.flush_interval = 10  # seconds
        
    async def initialize(self):
        """Initialize database connection and schema"""
        self.pool = await asyncpg.create_pool(self.db_url)
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    user_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    result TEXT,
                    severity TEXT,
                    details_encrypted BYTEA,
                    ip_address INET,
                    session_id TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_logs(timestamp DESC)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_logs(user_id)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_severity 
                ON audit_logs(severity)
            """)
    
    async def log_event(self, event: Dict[str, Any]):
        """Log security event"""
        # Encrypt sensitive details
        details_json = json.dumps(event.get('details', {}))
        encrypted_details = self.cipher.encrypt(details_json.encode())
        
        log_entry = {
            'timestamp': event.get('timestamp', datetime.now()),
            'user_id': event.get('user_id'),
            'action': event.get('action'),
            'resource': event.get('resource'),
            'result': event.get('result'),
            'severity': event.get('severity', 'info'),
            'details_encrypted': encrypted_details,
            'ip_address': event.get('ip_address'),
            'session_id': event.get('session_id'),
            'metadata': event.get('metadata', {})
        }
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Flush if buffer is full
        if len(self.log_buffer) >= 100:
            await self.flush_logs()
    
    async def flush_logs(self):
        """Flush buffered logs to database"""
        if not self.log_buffer:
            return
        
        logs_to_write = list(self.log_buffer)
        self.log_buffer.clear()
        
        async with self.pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO audit_logs (
                    timestamp, user_id, action, resource, result,
                    severity, details_encrypted, ip_address, session_id, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, [
                (
                    log['timestamp'], log['user_id'], log['action'],
                    log['resource'], log['result'], log['severity'],
                    log['details_encrypted'], log['ip_address'],
                    log['session_id'], json.dumps(log['metadata'])
                )
                for log in logs_to_write
            ])
    
    async def query_logs(self, filters: Dict[str, Any], 
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        param_count = 0
        
        if 'user_id' in filters:
            param_count += 1
            query += f" AND user_id = ${param_count}"
            params.append(filters['user_id'])
        
        if 'severity' in filters:
            param_count += 1
            query += f" AND severity = ${param_count}"
            params.append(filters['severity'])
        
        if 'start_time' in filters:
            param_count += 1
            query += f" AND timestamp >= ${param_count}"
            params.append(filters['start_time'])
        
        if 'end_time' in filters:
            param_count += 1
            query += f" AND timestamp <= ${param_count}"
            params.append(filters['end_time'])
        
        query += f" ORDER BY timestamp DESC LIMIT ${param_count + 1}"
        params.append(limit)
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch(query, *params)
            
            # Decrypt details
            logs = []
            for row in results:
                log = dict(row)
                if log['details_encrypted']:
                    decrypted = self.cipher.decrypt(log['details_encrypted'])
                    log['details'] = json.loads(decrypted.decode())
                    del log['details_encrypted']
                logs.append(log)
            
            return logs

class Guardian:
    """Main security enforcement orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pii_detector = PIIDetector()
        self.anomaly_detector = AnomalyDetector()
        self.rbac = RBACManager()
        self.audit_logger = AuditLogger(
            config.get('audit_db_url'),
            config.get('encryption_key')
        )
        
        # Security policies
        self.policies = self._load_security_policies()
        
        # Alert thresholds
        self.alert_thresholds = {
            'pii_detection': 5,  # PII items per request
            'anomaly_score': 0.7,  # Anomaly severity
            'failed_auth': 3,  # Failed attempts
            'rate_limit': 100  # Requests per minute
        }
        
        # Rate limiting
        self.rate_limiter = defaultdict(lambda: deque(maxlen=1000))
        
        # Active threats tracking
        self.active_threats = {}
        
    def _load_security_policies(self) -> List[SecurityPolicy]:
        """Load security policies from configuration"""
        policies = []
        
        # Default policies
        policies.append(SecurityPolicy(
            name='pii_protection',
            rules=[
                {'type': 'pii_detection', 'threshold': 5},
                {'type': 'pii_sanitization', 'enabled': True}
            ],
            actions=['sanitize', 'log', 'alert'],
            severity='high'
        ))
        
        policies.append(SecurityPolicy(
            name='anomaly_response',
            rules=[
                {'type': 'anomaly_detection', 'threshold': 0.7},
                {'type': 'behavioral_analysis', 'window': 1000}
            ],
            actions=['log', 'alert', 'block'],
            severity='critical'
        ))
        
        policies.append(SecurityPolicy(
            name='access_control',
            rules=[
                {'type': 'rbac_check', 'strict': True},
                {'type': 'session_validation', 'timeout': 3600}
            ],
            actions=['block', 'log'],
            severity='high'
        ))
        
        return policies
    
    async def initialize(self):
        """Initialize Guardian systems"""
        await self.audit_logger.initialize()
        
        # Start background tasks
        asyncio.create_task(self._audit_flush_loop())
        asyncio.create_task(self._threat_monitor_loop())
        
        print("🛡️ Guardian security systems activated")
    
    async def enforce_security(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main security enforcement pipeline"""
        response = {
            'allowed': True,
            'warnings': [],
            'sanitized_data': None,
            'security_metadata': {}
        }
        
        # Extract request details
        user_id = request.get('user_id', 'anonymous')
        data = request.get('data', '')
        action = request.get('action')
        resource = request.get('resource')
        
        # 1. Rate limiting check
        if not self._check_rate_limit(user_id):
            response['allowed'] = False
            response['reason'] = 'Rate limit exceeded'
            await self._log_security_event('rate_limit_exceeded', user_id, request)
            return response
        
        # 2. RBAC check
        if action and resource:
            access_request = AccessRequest(
                user_id=user_id,
                resource=resource,
                action=action,
                context=request
            )
            
            allowed, reason = self.rbac.check_permission(access_request)
            if not allowed:
                response['allowed'] = False
                response['reason'] = reason
                await self._log_security_event('access_denied', user_id, request)
                return response
        
        # 3. PII detection and sanitization
        if isinstance(data, str) and data:
            pii_data = self.pii_detector.detect_pii(data)
            
            if pii_data:
                pii_count = sum(len(items) for items in pii_data.values())
                response['warnings'].append(f"Detected {pii_count} PII items")
                response['security_metadata']['pii_detected'] = list(pii_data.keys())
                
                # Sanitize if threshold exceeded
                if pii_count > self.alert_thresholds['pii_detection']:
                    response['sanitized_data'] = self.pii_detector.sanitize_text(data, pii_data)
                    response['warnings'].append("Data sanitized due to PII")
                    await self._log_security_event('pii_sanitized', user_id, {
                        'pii_types': list(pii_data.keys()),
                        'count': pii_count
                    })
        
        # 4. Anomaly detection
        event = {
            'data': str(data)[:1000],  # Truncate for analysis
            'request_rate': self._calculate_request_rate(user_id),
            'unique_ips': len(set(r.get('ip') for r in self.rate_limiter[user_id])),
            'action': action,
            'resource': resource
        }
        
        anomaly = self.anomaly_detector.detect_anomaly(user_id, event)
        
        if anomaly and anomaly.severity > self.alert_thresholds['anomaly_score']:
            response['warnings'].append(f"Anomalous behavior detected (severity: {anomaly.severity:.2f})")
            response['security_metadata']['anomaly'] = {
                'type': anomaly.event_type,
                'severity': anomaly.severity,
                'details': anomaly.details
            }
            
            # Block if critical
            if anomaly.severity > 0.9:
                response['allowed'] = False
                response['reason'] = 'Critical security anomaly detected'
                
            await self._log_security_event('anomaly_detected', user_id, anomaly.__dict__)
        
        # 5. Log the security check
        await self.audit_logger.log_event({
            'user_id': user_id,
            'action': 'security_check',
            'resource': resource,
            'result': 'allowed' if response['allowed'] else 'blocked',
            'severity': 'info',
            'details': response,
            'timestamp': datetime.now()
        })
        
        return response
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user exceeds rate limit"""
        now = datetime.now()
        
        # Add current request
        self.rate_limiter[user_id].append({
            'timestamp': now,
            'ip': '127.0.0.1'  # Would get real IP in production
        })
        
        # Count requests in last minute
        one_minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            r for r in self.rate_limiter[user_id]
            if r['timestamp'] > one_minute_ago
        ]
        
        return len(recent_requests) <= self.alert_thresholds['rate_limit']
    
    def _calculate_request_rate(self, user_id: str) -> float:
        """Calculate request rate for user"""
        if not self.rate_limiter[user_id]:
            return 0.0
        
        timestamps = [r['timestamp'] for r in self.rate_limiter[user_id]]
        if len(timestamps) < 2:
            return 0.0
        
        time_span = (timestamps[-1] - timestamps[0]).total_seconds()
        if time_span == 0:
            return 0.0
        
        return len(timestamps) / time_span
    
    async def _log_security_event(self, event_type: str, user_id: str, details: Any):
        """Log security event"""
        await self.audit_logger.log_event({
            'user_id': user_id,
            'action': event_type,
            'severity': 'high' if 'anomaly' in event_type or 'denied' in event_type else 'medium',
            'details': details if isinstance(details, dict) else {'data': str(details)},
            'timestamp': datetime.now()
        })
    
    async def _audit_flush_loop(self):
        """Periodically flush audit logs"""
        while True:
            await asyncio.sleep(self.audit_logger.flush_interval)
            await self.audit_logger.flush_logs()
    
    async def _threat_monitor_loop(self):
        """Monitor for active threats"""
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Query recent security events
            recent_logs = await self.audit_logger.query_logs({
                'start_time': datetime.now() - timedelta(minutes=5),
                'severity': 'high'
            }, limit=100)
            
            # Analyze for patterns
            threat_patterns = defaultdict(int)
            for log in recent_logs:
                if log['action'] in ['anomaly_detected', 'access_denied', 'pii_sanitized']:
                    threat_patterns[log['user_id']] += 1
            
            # Identify active threats
            for user_id, count in threat_patterns.items():
                if count > 10:  # Threshold for active threat
                    self.active_threats[user_id] = {
                        'threat_level': 'high',
                        'event_count': count,
                        'timestamp': datetime.now()
                    }
                    
                    # Auto-block if severe
                    if count > 20:
                        self.rbac.assign_role(user_id, 'blocked')
                        await self._log_security_event('user_blocked', user_id, {
                            'reason': 'Excessive security violations',
                            'event_count': count
                        })

# Example usage
async def main():
    config = {
        'audit_db_url': 'postgresql://security:secure@localhost:5432/audit_db',
        'encryption_key': Fernet.generate_key()
    }
    
    guardian = Guardian(config)
    await guardian.initialize()
    
    # Test security enforcement
    request = {
        'user_id': 'researcher_001',
        'data': 'Analyzing data for john.doe@example.com with SSN 123-45-6789',
        'action': 'read',
        'resource': 'research'
    }
    
    # Assign role
    guardian.rbac.assign_role('researcher_001', 'researcher')
    
    # Enforce security
    result = await guardian.enforce_security(request)
    print(f"Security result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
