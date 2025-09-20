# BEV OSINT Framework - Operational Security Guide

## Overview

This guide provides comprehensive operational security (OpSec) guidelines for the BEV OSINT Framework. Given the sensitive nature of OSINT operations and the framework's research focus, strict adherence to these guidelines is essential for legal compliance, data protection, and operational safety.

## Table of Contents

1. [Security Architecture Overview](#security-architecture-overview)
2. [Network Security](#network-security)
3. [Data Protection & Privacy](#data-protection--privacy)
4. [Access Control & Authentication](#access-control--authentication)
5. [Operational Procedures](#operational-procedures)
6. [Legal & Compliance Framework](#legal--compliance-framework)
7. [Incident Response](#incident-response)
8. [Security Monitoring](#security-monitoring)
9. [Threat Model](#threat-model)

---

## Security Architecture Overview

### Defense in Depth Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL THREATS                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Surveillance│  │ Legal Action│  │ Data Breach │  │ Attribution │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                   PERIMETER SECURITY                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Firewall  │  │     VPN     │  │     IDS     │  │   Traffic   │   │
│  │   Rules     │  │   Gateway   │  │   System    │  │ Filtering   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                   NETWORK SECURITY                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Network     │  │ Container   │  │   Service   │  │   Traffic   │   │
│  │ Isolation   │  │ Networks    │  │   Mesh      │  │ Encryption  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                 APPLICATION SECURITY                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │    HTTPS    │  │  API Keys   │  │   Input     │  │   Output    │   │
│  │  Everywhere │  │ Management  │  │ Validation  │ │ Sanitization│   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                    DATA SECURITY                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Encryption  │  │   Access    │  │    Audit    │  │   Backup    │   │
│  │ at Rest     │  │  Controls   │  │   Logging   │  │ Protection  │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Security Principles

1. **Zero Trust Architecture**: Verify every request and connection
2. **Principle of Least Privilege**: Minimum necessary access rights
3. **Defense in Depth**: Multiple security layers
4. **Data Minimization**: Collect and retain only necessary data
5. **Operational Anonymity**: Protect operator identity and location

---

## Network Security

### Network Isolation

#### Container Network Segmentation
```yaml
Networks:
  bev_frontend:        # Web interfaces only
    subnet: 172.30.0.0/26
    isolation: External access allowed
    
  bev_database:        # Database tier
    subnet: 172.30.0.64/26
    isolation: No external access
    
  bev_processing:      # Analysis engines
    subnet: 172.30.0.128/26
    isolation: Limited external access via proxy
    
  bev_monitoring:      # Monitoring stack
    subnet: 172.30.0.192/26
    isolation: Read-only access to other networks
```

#### Firewall Rules
```bash
# External access rules
iptables -A INPUT -p tcp --dport 80 -j ACCEPT     # HTTP
iptables -A INPUT -p tcp --dport 443 -j ACCEPT    # HTTPS
iptables -A INPUT -p tcp --dport 22 -j ACCEPT     # SSH (admin only)

# Block all other external access
iptables -A INPUT -j DROP

# Inter-container communication
iptables -A FORWARD -s 172.30.0.0/24 -d 172.30.0.0/24 -j ACCEPT

# Tor proxy rules
iptables -A OUTPUT -d 127.0.0.1 --dport 9050 -j ACCEPT  # SOCKS5
iptables -A OUTPUT -d 127.0.0.1 --dport 9051 -j ACCEPT  # Control
```

### Proxy Security

#### Tor Configuration
```ini
# /etc/tor/torrc
SocksPort 0.0.0.0:9050
ControlPort 0.0.0.0:9051
HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C
ExitPolicy reject *:*
MaxCircuitDirtiness 300
NewCircuitPeriod 30
NumEntryGuards 8
EnforceDistinctSubnets 1
```

#### Circuit Management
```python
# Automatic circuit rotation
import stem.control
import time

def rotate_circuit():
    with stem.control.Controller.from_port(port=9051) as controller:
        controller.authenticate()
        controller.signal(stem.Signal.NEWNYM)
        time.sleep(10)  # Wait for new circuit

# Rotate every 5 minutes during active operations
schedule.every(5).minutes.do(rotate_circuit)
```

#### User Agent Rotation
```python
# Dynamic user agent rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)
```

---

## Data Protection & Privacy

### Data Classification

#### Classification Levels
```yaml
PUBLIC:
  description: "General configuration, documentation"
  retention: "Indefinite"
  encryption: "Optional"
  access: "All operators"

INTERNAL:
  description: "System logs, metrics, operational data"
  retention: "90 days"
  encryption: "Required in transit"
  access: "Operators and administrators"

CONFIDENTIAL:
  description: "Analysis results, OSINT data"
  retention: "30 days (configurable)"
  encryption: "Required at rest and in transit"
  access: "Authorized analysts only"

RESTRICTED:
  description: "Personal data, breach records, sensitive intelligence"
  retention: "7 days (configurable)"
  encryption: "AES-256 at rest, TLS 1.3 in transit"
  access: "Designated personnel only"
```

### Encryption Standards

#### Data at Rest
```yaml
Database Encryption:
  PostgreSQL: 
    method: "Transparent Data Encryption (TDE)"
    algorithm: "AES-256-GCM"
    key_rotation: "30 days"
    
  Neo4j:
    method: "Database encryption"
    algorithm: "AES-256-CBC"
    key_management: "External key store"
    
  Elasticsearch:
    method: "Index encryption"
    algorithm: "AES-256-GCM"
    per_index: "Enabled"

File System:
  logs: "AES-256-XTS (LUKS)"
  backups: "AES-256-GCM + GPG"
  temp_files: "Automatic encryption + deletion"
```

#### Data in Transit
```yaml
Internal Communication:
  container_to_container: "TLS 1.3 with mTLS"
  api_calls: "HTTPS with certificate pinning"
  database_connections: "SSL/TLS required"

External Communication:
  tor_proxy: "Layered encryption (TLS over Tor)"
  api_endpoints: "TLS 1.3 minimum"
  webhook_callbacks: "Signed and encrypted payloads"
```

### Data Retention Policies

#### Automated Retention Management
```python
# Data retention automation
RETENTION_POLICIES = {
    'analysis_results': timedelta(days=30),
    'breach_data': timedelta(days=7),
    'system_logs': timedelta(days=90),
    'audit_logs': timedelta(days=365),
    'temp_files': timedelta(hours=24)
}

def enforce_retention_policy():
    for data_type, retention_period in RETENTION_POLICIES.items():
        cutoff_date = datetime.now() - retention_period
        delete_data_older_than(data_type, cutoff_date)
```

#### Secure Deletion
```bash
#!/bin/bash
# Secure file deletion script
secure_delete() {
    local file="$1"
    if [ -f "$file" ]; then
        # Multiple pass overwrite
        shred -vfz -n 3 "$file"
        rm -f "$file"
    fi
}

# Secure directory deletion
secure_delete_dir() {
    local dir="$1"
    if [ -d "$dir" ]; then
        find "$dir" -type f -exec shred -vfz -n 3 {} \;
        rm -rf "$dir"
    fi
}
```

---

## Access Control & Authentication

### Authentication Architecture

#### Multi-Factor Authentication (Optional)
```yaml
Authentication Methods:
  primary: "No authentication (single-user system)"
  optional_mfa: "TOTP for administrative functions"
  api_access: "API key authentication"
  service_to_service: "mTLS certificates"
```

#### API Key Management
```python
# API key generation and validation
import secrets
import hashlib
from datetime import datetime, timedelta

class APIKeyManager:
    def generate_key(self, purpose: str, expires_in_days: int = 30):
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        expiry = datetime.now() + timedelta(days=expires_in_days)
        
        # Store key metadata (not the key itself)
        store_key_metadata(key_hash, purpose, expiry)
        return key
    
    def validate_key(self, key: str) -> bool:
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return is_valid_key_hash(key_hash)
```

### Role-Based Access Control

#### Access Roles
```yaml
Roles:
  analyst:
    permissions:
      - "execute_analysis"
      - "view_results"
      - "export_data"
    restrictions:
      - "no_system_config"
      - "no_user_management"
      
  operator:
    permissions:
      - "system_monitoring"
      - "service_management"
      - "backup_restore"
    restrictions:
      - "no_data_export"
      - "read_only_analysis"
      
  administrator:
    permissions:
      - "full_system_access"
      - "user_management"
      - "security_config"
    restrictions:
      - "audit_logged"
      - "requires_mfa"
```

---

## Operational Procedures

### Secure Deployment Procedures

#### Pre-Deployment Security Checklist
```bash
#!/bin/bash
# Pre-deployment security validation

echo "🔒 BEV Security Deployment Checklist"
echo "=================================="

# 1. Verify secret generation
if [ ! -f ".env" ]; then
    echo "❌ Missing .env file"
    exit 1
fi

# 2. Check strong passwords
check_password_strength() {
    local password="$1"
    if [[ ${#password} -lt 20 ]]; then
        echo "❌ Password too short (minimum 20 characters)"
        return 1
    fi
    return 0
}

# 3. Verify network isolation
check_network_config() {
    docker network inspect bev_network >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Missing security network configuration"
        return 1
    fi
}

# 4. Validate TLS certificates
check_tls_config() {
    if [ ! -d "certs/" ]; then
        echo "❌ Missing TLS certificate directory"
        return 1
    fi
}

echo "✅ Security checks passed"
```

#### Operational Security Commands
```bash
# Monitor for security events
monitor_security_events() {
    docker logs bev_traefik 2>&1 | grep -E "(401|403|429|500)"
    docker logs bev_postgres 2>&1 | grep -E "(FATAL|ERROR)"
    docker logs bev_tor 2>&1 | grep -E "(WARNING|ERROR)"
}

# Check for suspicious network activity
monitor_network_activity() {
    netstat -an | grep :80 | wc -l
    netstat -an | grep :443 | wc -l
    ss -tuln | grep -E ":(9050|9051)"
}

# Validate Tor circuit health
check_tor_health() {
    curl -x socks5://127.0.0.1:9050 http://check.torproject.org/ \
        --silent | grep -q "Congratulations"
    if [ $? -eq 0 ]; then
        echo "✅ Tor proxy working"
    else
        echo "❌ Tor proxy issues"
    fi
}
```

### Incident Response Procedures

#### Security Incident Classification
```yaml
Severity Levels:
  CRITICAL:
    examples: ["Data breach", "System compromise", "Legal issues"]
    response_time: "Immediate (< 15 minutes)"
    actions: ["Isolate system", "Preserve evidence", "Legal notification"]
    
  HIGH:
    examples: ["Unauthorized access", "Service disruption", "Data exposure"]
    response_time: "1 hour"
    actions: ["Investigate", "Contain threat", "Update security"]
    
  MEDIUM:
    examples: ["Performance degradation", "Configuration issues"]
    response_time: "4 hours"
    actions: ["Monitor", "Plan remediation", "Update procedures"]
    
  LOW:
    examples: ["Minor anomalies", "Information gathering"]
    response_time: "24 hours"
    actions: ["Document", "Monitor trends", "Scheduled review"]
```

#### Incident Response Playbooks

##### Data Breach Response
```bash
#!/bin/bash
# Data breach incident response

echo "🚨 DATA BREACH INCIDENT RESPONSE"
echo "================================"

# 1. Immediate containment
isolate_system() {
    echo "🔒 Isolating affected systems..."
    docker-compose stop
    iptables -A INPUT -j DROP
    iptables -A OUTPUT -j DROP
}

# 2. Evidence preservation
preserve_evidence() {
    echo "📸 Preserving evidence..."
    timestamp=$(date +"%Y%m%d_%H%M%S")
    
    # Create forensic images
    dd if=/dev/sda of="/backup/forensic_$timestamp.img" bs=4M
    
    # Preserve logs
    cp -r logs/ "/backup/logs_$timestamp/"
    
    # Database snapshots
    docker exec bev_postgres pg_dump osint > "/backup/db_$timestamp.sql"
}

# 3. Legal notification (if required)
legal_notification() {
    echo "⚖️ Legal notification checklist:"
    echo "- [ ] Notify legal team"
    echo "- [ ] Document timeline"
    echo "- [ ] Preserve attorney-client privilege"
    echo "- [ ] Consider law enforcement notification"
}
```

##### System Compromise Response
```bash
#!/bin/bash
# System compromise response

compromise_response() {
    echo "🛡️ SYSTEM COMPROMISE RESPONSE"
    echo "============================="
    
    # 1. Assess scope
    check_running_processes() {
        ps aux > "/tmp/processes_$timestamp.txt"
        netstat -tuln > "/tmp/network_$timestamp.txt"
        lsof > "/tmp/open_files_$timestamp.txt"
    }
    
    # 2. Check for indicators of compromise
    check_ioc() {
        # Check for unusual network connections
        netstat -an | grep -E ":(1337|4444|5555|6666|7777)"
        
        # Check for suspicious processes
        ps aux | grep -E "(nc|ncat|netcat|socat)"
        
        # Check modified system files
        find /etc /usr/bin /usr/sbin -mtime -1 -type f
    }
    
    # 3. Remediation
    remediate() {
        echo "🔧 Starting remediation..."
        # Reset all passwords
        ./generate_secrets.sh
        
        # Restart all services with new credentials
        docker-compose down
        docker-compose up -d
        
        # Update security configurations
        ./fix_security_critical.sh
    }
}
```

---

## Security Monitoring

### Continuous Monitoring Strategy

#### Security Metrics Collection
```yaml
Prometheus Metrics:
  failed_login_attempts:
    type: "counter"
    labels: ["source_ip", "username"]
    alert_threshold: 5
    
  tor_circuit_failures:
    type: "counter"
    labels: ["exit_node", "failure_type"]
    alert_threshold: 3
    
  api_request_anomalies:
    type: "histogram"
    labels: ["endpoint", "method", "status"]
    alert_condition: "rate > 100/min"
    
  data_access_violations:
    type: "counter"
    labels: ["user", "resource", "action"]
    alert_threshold: 1
```

#### Automated Security Alerts
```python
# Security alerting system
class SecurityAlerting:
    def __init__(self):
        self.alert_channels = ['email', 'webhook', 'log']
    
    def check_failed_logins(self):
        failed_count = count_failed_logins(last_hour=True)
        if failed_count > 5:
            self.send_alert(
                severity='HIGH',
                message=f'Multiple failed login attempts: {failed_count}',
                recommended_action='Check source IPs and consider blocking'
            )
    
    def check_tor_health(self):
        if not is_tor_circuit_healthy():
            self.send_alert(
                severity='MEDIUM',
                message='Tor circuit health degraded',
                recommended_action='Restart Tor service and check network'
            )
    
    def check_data_access_patterns(self):
        anomalies = detect_access_anomalies()
        if anomalies:
            self.send_alert(
                severity='HIGH',
                message=f'Unusual data access patterns detected: {anomalies}',
                recommended_action='Review access logs and user behavior'
            )
```

#### Log Analysis and SIEM Integration
```python
# Security log analysis
import re
from collections import defaultdict

class SecurityLogAnalyzer:
    def __init__(self):
        self.threat_patterns = {
            'sql_injection': r'(\bUNION\b.*\bSELECT\b|\bOR\b.*=.*)',
            'xss_attempt': r'<script|javascript:|onload=|onerror=',
            'directory_traversal': r'\.\./|\.\.\\',
            'command_injection': r'[;&|`$()]'
        }
    
    def analyze_web_logs(self, log_file):
        threats = defaultdict(list)
        
        with open(log_file, 'r') as f:
            for line in f:
                for threat_type, pattern in self.threat_patterns.items():
                    if re.search(pattern, line, re.IGNORECASE):
                        threats[threat_type].append(line.strip())
        
        return dict(threats)
    
    def generate_security_report(self):
        report = {
            'timestamp': datetime.now().isoformat(),
            'threats_detected': self.analyze_web_logs('/var/log/access.log'),
            'tor_usage': self.analyze_tor_logs(),
            'system_events': self.analyze_system_logs()
        }
        return report
```

---

## Threat Model

### Threat Categories

#### External Threats
```yaml
Nation-State Actors:
  capabilities: ["Advanced persistent threats", "Zero-day exploits", "Traffic analysis"]
  motivations: ["Intelligence gathering", "Disruption", "Attribution"]
  mitigations: ["Tor anonymity", "Encryption", "Operational security"]
  
Cybercriminals:
  capabilities: ["Automated attacks", "Credential stuffing", "Ransomware"]
  motivations: ["Financial gain", "Data theft", "System access"]
  mitigations: ["Access controls", "Monitoring", "Backup procedures"]
  
Law Enforcement:
  capabilities: ["Legal orders", "Traffic analysis", "Network monitoring"]
  motivations: ["Investigation", "Evidence collection", "Compliance"]
  mitigations: ["Legal framework", "Data minimization", "Retention policies"]
```

#### Internal Threats
```yaml
Insider Threats:
  capabilities: ["Privileged access", "System knowledge", "Data export"]
  motivations: ["Malicious intent", "Negligence", "Coercion"]
  mitigations: ["Principle of least privilege", "Audit logging", "Data controls"]
  
Supply Chain:
  capabilities: ["Malicious dependencies", "Compromised updates", "Backdoors"]
  motivations: ["Espionage", "Disruption", "Access"]
  mitigations: ["Dependency scanning", "Update validation", "Container security"]
```

### Risk Assessment Matrix

| Threat | Likelihood | Impact | Risk Level | Mitigation Priority |
|--------|------------|--------|------------|-------------------|
| Data breach via web vulnerability | Medium | High | High | 1 - Critical |
| Tor network compromise | Low | High | Medium | 2 - High |
| Insider data exfiltration | Low | High | Medium | 3 - High |
| DDoS attack | High | Medium | Medium | 4 - Medium |
| Supply chain attack | Medium | Medium | Medium | 5 - Medium |
| Physical access | Low | Medium | Low | 6 - Low |

### Mitigation Strategies

#### Preventive Controls
```yaml
Technical Controls:
  - "Network segmentation and firewall rules"
  - "Encryption at rest and in transit"
  - "Tor proxy for anonymity"
  - "Container isolation"
  - "Input validation and sanitization"
  
Administrative Controls:
  - "Security policies and procedures"
  - "Regular security training"
  - "Incident response planning"
  - "Vendor security assessment"
  - "Data classification and handling"
  
Physical Controls:
  - "Secure hosting environment"
  - "Access control to systems"
  - "Environmental monitoring"
  - "Backup and recovery procedures"
```

#### Detective Controls
```yaml
Monitoring Systems:
  - "Real-time security alerting"
  - "Log analysis and correlation"
  - "Network traffic monitoring"
  - "Behavioral anomaly detection"
  - "Integrity monitoring"
  
Audit Procedures:
  - "Regular security assessments"
  - "Penetration testing"
  - "Compliance audits"
  - "Code security reviews"
  - "Configuration audits"
```

#### Corrective Controls
```yaml
Incident Response:
  - "Automated incident detection"
  - "Containment procedures"
  - "Evidence preservation"
  - "Recovery procedures"
  - "Lessons learned documentation"
  
Business Continuity:
  - "Backup and restore procedures"
  - "Disaster recovery planning"
  - "Alternative site operations"
  - "Data recovery procedures"
  - "Service continuity planning"
```

---

## Compliance Requirements

### Legal Framework Alignment
- **GDPR**: Data protection and privacy requirements
- **CCPA**: California consumer privacy regulations
- **SOC 2**: Security operational controls
- **NIST Cybersecurity Framework**: Risk management
- **ISO 27001**: Information security management

### Audit Requirements
- Security control testing
- Data handling verification
- Access control validation
- Incident response testing
- Business continuity validation

---

*Last Updated: 2025-09-19*
*Framework Version: BEV OSINT v2.0*
*Security Classification: INTERNAL*
*Document Version: 1.0*