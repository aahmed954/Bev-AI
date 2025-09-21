# BEV OSINT Framework - Integration & API Workflow Guide

## Overview

This guide provides comprehensive documentation for integrating with the BEV OSINT Framework APIs, including workflow patterns, authentication methods, data formats, and best practices for developers and system integrators.

## Table of Contents

1. [API Architecture Overview](#api-architecture-overview)
2. [Authentication & Authorization](#authentication--authorization)
3. [Core API Workflows](#core-api-workflows)
4. [Analysis Pipeline Integration](#analysis-pipeline-integration)
5. [Data Export & Import Workflows](#data-export--import-workflows)
6. [Real-time Integration Patterns](#real-time-integration-patterns)
7. [SDK and Client Libraries](#sdk-and-client-libraries)
8. [Error Handling & Retry Logic](#error-handling--retry-logic)
9. [Rate Limiting & Performance](#rate-limiting--performance)
10. [Integration Examples](#integration-examples)

---

## API Architecture Overview

### Service Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT APPLICATIONS                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Web Dashboard│  │  Mobile App │  │  CLI Tools  │  │ Third-party │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                        API GATEWAY                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ Rate Limiting│  │Authentication│  │  Load Bal.  │  │   Routing   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│  CORE API      │ │ ANALYSIS API│ │  WORKFLOW API   │
│  /api/v1/      │ │ /analyzer/  │ │   /workflow/    │
│                │ │             │ │                 │
│ • Users        │ │ • Breach DB │ │ • Airflow DAGs  │
│ • Projects     │ │ • Social    │ │ • n8n Workflows │
│ • Data Mgmt    │ │ • Crypto    │ │ • Automation    │
│ • Search       │ │ • Darknet   │ │ • Scheduling    │
└────────────────┘ └─────────────┘ └─────────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────┐
│                     DATA LAYER                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │ PostgreSQL  │  │    Neo4j    │  │    Redis    │  │Elasticsearch│   │
│  │ Relational  │  │ Graph Data  │  │   Cache     │  │ Full-text   │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### API Endpoints Structure
```yaml
Base URL: https://your-bev-instance/

Core API:
  /api/v1/projects          # Project management
  /api/v1/investigations    # Investigation lifecycle
  /api/v1/search           # Unified search interface
  /api/v1/export           # Data export functionality
  /api/v1/health           # System health checks

Analysis APIs:
  /analyzer/v1/breach      # Breach database analysis
  /analyzer/v1/social      # Social media analysis
  /analyzer/v1/crypto      # Cryptocurrency analysis
  /analyzer/v1/darknet     # Darknet monitoring
  /analyzer/v1/batch       # Batch analysis operations

Workflow APIs:
  /workflow/v1/airflow     # Airflow DAG management
  /workflow/v1/n8n         # n8n workflow automation
  /workflow/v1/schedule    # Scheduled operations

Real-time APIs:
  /ws/v1/analysis          # WebSocket analysis updates
  /ws/v1/monitoring        # Real-time monitoring
  /events/v1/              # Server-sent events

Graph APIs:
  /graph/v1/cypher         # Neo4j Cypher queries
  /graph/v1/visualize      # Graph visualization data
  /graph/v1/relationships  # Relationship analysis
```

---

## Authentication & Authorization

### API Key Authentication
```python
# API Key generation and usage
import requests
import hashlib
import hmac
from datetime import datetime

class BEVAPIClient:
    def __init__(self, base_url, api_key, api_secret=None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': 'BEV-API-Client/1.0',
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        })
    
    def _generate_signature(self, method, path, body=None, timestamp=None):
        """Generate HMAC signature for request authentication"""
        if not self.api_secret:
            return None
        
        if timestamp is None:
            timestamp = str(int(datetime.now().timestamp()))
        
        # Create signature string
        sig_string = f"{method.upper()}\n{path}\n{timestamp}"
        if body:
            sig_string += f"\n{body}"
        
        # Generate HMAC signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            sig_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make authenticated API request"""
        url = f"{self.base_url}{endpoint}"
        
        # Add signature headers if secret is provided
        if self.api_secret:
            body = kwargs.get('json', '')
            if body:
                body = json.dumps(body, sort_keys=True)
            
            signature, timestamp = self._generate_signature(
                method, endpoint, body
            )
            
            kwargs.setdefault('headers', {}).update({
                'X-Timestamp': timestamp,
                'X-Signature': signature
            })
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get(self, endpoint, **kwargs):
        return self._make_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint, **kwargs):
        return self._make_request('POST', endpoint, **kwargs)
    
    def put(self, endpoint, **kwargs):
        return self._make_request('PUT', endpoint, **kwargs)
    
    def delete(self, endpoint, **kwargs):
        return self._make_request('DELETE', endpoint, **kwargs)

# Usage example
client = BEVAPIClient(
    base_url='https://your-bev-instance',
    api_key='your-api-key',
    api_secret='your-api-secret'  # Optional for HMAC signing
)
```

### Token-Based Authentication (Optional)
```python
# JWT token authentication
class BEVTokenClient(BEVAPIClient):
    def __init__(self, base_url, username=None, password=None, token=None):
        super().__init__(base_url, api_key=None)
        self.token = token
        
        if not token and username and password:
            self.token = self._authenticate(username, password)
        
        if self.token:
            self.session.headers['Authorization'] = f'Bearer {self.token}'
    
    def _authenticate(self, username, password):
        """Authenticate and get JWT token"""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/login",
            json={'username': username, 'password': password}
        )
        response.raise_for_status()
        return response.json()['access_token']
    
    def refresh_token(self):
        """Refresh JWT token"""
        response = self.session.post(
            f"{self.base_url}/api/v1/auth/refresh",
            headers={'Authorization': f'Bearer {self.token}'}
        )
        response.raise_for_status()
        self.token = response.json()['access_token']
        self.session.headers['Authorization'] = f'Bearer {self.token}'
```

---

## Core API Workflows

### Investigation Lifecycle Management
```python
# Complete investigation workflow
class InvestigationWorkflow:
    def __init__(self, client):
        self.client = client
    
    def create_investigation(self, name, description, targets, classification='INTERNAL'):
        """Create a new investigation"""
        investigation_data = {
            'name': name,
            'description': description,
            'targets': targets,
            'classification': classification,
            'created_at': datetime.now().isoformat(),
            'status': 'ACTIVE'
        }
        
        response = self.client.post(
            '/api/v1/investigations',
            json=investigation_data
        )
        
        return response['investigation_id']
    
    def add_target_to_investigation(self, investigation_id, target_type, target_value):
        """Add a target to an existing investigation"""
        target_data = {
            'type': target_type,  # email, domain, ip, username, etc.
            'value': target_value,
            'added_at': datetime.now().isoformat()
        }
        
        return self.client.post(
            f'/api/v1/investigations/{investigation_id}/targets',
            json=target_data
        )
    
    def start_analysis(self, investigation_id, analysis_types=None):
        """Start analysis for all targets in investigation"""
        if analysis_types is None:
            analysis_types = ['breach', 'social', 'crypto', 'darknet']
        
        analysis_request = {
            'investigation_id': investigation_id,
            'analysis_types': analysis_types,
            'priority': 'normal',
            'options': {
                'tor_enabled': True,
                'deep_analysis': True,
                'correlation_enabled': True
            }
        }
        
        return self.client.post(
            '/analyzer/v1/batch',
            json=analysis_request
        )
    
    def get_investigation_status(self, investigation_id):
        """Get current status of investigation"""
        return self.client.get(f'/api/v1/investigations/{investigation_id}')
    
    def get_analysis_results(self, investigation_id, result_type=None):
        """Get analysis results for investigation"""
        params = {}
        if result_type:
            params['type'] = result_type
        
        return self.client.get(
            f'/api/v1/investigations/{investigation_id}/results',
            params=params
        )
    
    def generate_report(self, investigation_id, report_format='json'):
        """Generate investigation report"""
        report_request = {
            'investigation_id': investigation_id,
            'format': report_format,
            'include_evidence': True,
            'include_timeline': True,
            'include_graph': True
        }
        
        return self.client.post(
            '/api/v1/investigations/{investigation_id}/report',
            json=report_request
        )

# Usage example
workflow = InvestigationWorkflow(client)

# Create investigation
investigation_id = workflow.create_investigation(
    name="Phishing Campaign Analysis",
    description="Analysis of suspected phishing email campaign",
    targets=[
        {"type": "email", "value": "suspect@example.com"},
        {"type": "domain", "value": "phishing-site.com"}
    ]
)

# Start analysis
analysis_job = workflow.start_analysis(investigation_id)

# Check status
status = workflow.get_investigation_status(investigation_id)
print(f"Investigation status: {status['status']}")

# Get results when complete
if status['status'] == 'COMPLETED':
    results = workflow.get_analysis_results(investigation_id)
    report = workflow.generate_report(investigation_id, 'pdf')
```

### Batch Analysis Operations
```python
# Batch analysis workflow
class BatchAnalysisWorkflow:
    def __init__(self, client):
        self.client = client
    
    def submit_batch_analysis(self, targets, analysis_types, options=None):
        """Submit batch analysis job"""
        if options is None:
            options = {
                'tor_enabled': True,
                'parallel_limit': 5,
                'timeout': 3600,
                'retry_attempts': 3
            }
        
        batch_request = {
            'targets': targets,
            'analysis_types': analysis_types,
            'options': options,
            'callback_url': None,  # Optional webhook for completion
            'priority': 'normal'
        }
        
        response = self.client.post('/analyzer/v1/batch', json=batch_request)
        return response['job_id']
    
    def get_batch_status(self, job_id):
        """Get status of batch analysis job"""
        return self.client.get(f'/analyzer/v1/batch/{job_id}/status')
    
    def get_batch_results(self, job_id, format='json'):
        """Get results from completed batch job"""
        params = {'format': format}
        return self.client.get(f'/analyzer/v1/batch/{job_id}/results', params=params)
    
    def cancel_batch_job(self, job_id):
        """Cancel running batch job"""
        return self.client.delete(f'/analyzer/v1/batch/{job_id}')

# Example: Batch email analysis
batch_workflow = BatchAnalysisWorkflow(client)

# Submit batch job
targets = [
    {"type": "email", "value": "user1@example.com"},
    {"type": "email", "value": "user2@example.com"},
    {"type": "email", "value": "user3@example.com"}
]

job_id = batch_workflow.submit_batch_analysis(
    targets=targets,
    analysis_types=['breach', 'social'],
    options={'parallel_limit': 3}
)

# Monitor progress
import time
while True:
    status = batch_workflow.get_batch_status(job_id)
    print(f"Job {job_id}: {status['status']} - {status['progress']}%")
    
    if status['status'] in ['COMPLETED', 'FAILED']:
        break
    
    time.sleep(10)

# Get results
if status['status'] == 'COMPLETED':
    results = batch_workflow.get_batch_results(job_id)
```

---

## Analysis Pipeline Integration

### Custom Analyzer Development
```python
# Custom analyzer integration
class CustomAnalyzer:
    def __init__(self, name, version, description):
        self.name = name
        self.version = version
        self.description = description
        self.supported_types = []
        
    def register_analyzer(self, client):
        """Register custom analyzer with BEV"""
        analyzer_config = {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'supported_types': self.supported_types,
            'endpoint': f'/analyzer/custom/{self.name}',
            'timeout': 300,
            'requires_tor': False
        }
        
        return client.post('/analyzer/v1/register', json=analyzer_config)
    
    def analyze(self, target_type, target_value, options=None):
        """Perform analysis (implement in subclass)"""
        raise NotImplementedError("Subclass must implement analyze method")

# Example custom analyzer
class ShodanAnalyzer(CustomAnalyzer):
    def __init__(self, api_key):
        super().__init__(
            name="shodan_analyzer",
            version="1.0.0",
            description="Shodan IP/domain intelligence"
        )
        self.api_key = api_key
        self.supported_types = ['ip', 'domain']
    
    def analyze(self, target_type, target_value, options=None):
        """Analyze IP or domain using Shodan"""
        import shodan
        
        api = shodan.Shodan(self.api_key)
        
        try:
            if target_type == 'ip':
                host_info = api.host(target_value)
                return {
                    'analyzer': self.name,
                    'target': target_value,
                    'target_type': target_type,
                    'results': {
                        'open_ports': host_info.get('ports', []),
                        'services': host_info.get('data', []),
                        'vulns': host_info.get('vulns', []),
                        'location': {
                            'country': host_info.get('country_name'),
                            'city': host_info.get('city'),
                            'org': host_info.get('org')
                        }
                    },
                    'confidence': 0.9,
                    'timestamp': datetime.now().isoformat()
                }
            
            elif target_type == 'domain':
                dns_info = api.dns.domain_info(target_value)
                return {
                    'analyzer': self.name,
                    'target': target_value,
                    'target_type': target_type,
                    'results': {
                        'subdomains': dns_info.get('subdomains', []),
                        'dns_records': dns_info.get('data', []),
                        'more_info': dns_info.get('more_info', False)
                    },
                    'confidence': 0.8,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'analyzer': self.name,
                'target': target_value,
                'target_type': target_type,
                'error': str(e),
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }

# Register and use custom analyzer
shodan_analyzer = ShodanAnalyzer(api_key='your-shodan-api-key')
shodan_analyzer.register_analyzer(client)

# Use in analysis pipeline
result = shodan_analyzer.analyze('ip', '8.8.8.8')
```

### Workflow Automation Integration
```python
# Airflow DAG integration
class AirflowIntegration:
    def __init__(self, client):
        self.client = client
    
    def create_investigation_dag(self, investigation_id, dag_config):
        """Create Airflow DAG for investigation"""
        dag_definition = {
            'dag_id': f'investigation_{investigation_id}',
            'description': f'Automated analysis for investigation {investigation_id}',
            'schedule_interval': dag_config.get('schedule', None),
            'start_date': datetime.now().isoformat(),
            'catchup': False,
            'tasks': [
                {
                    'task_id': 'start_analysis',
                    'type': 'BEVAnalysisOperator',
                    'params': {
                        'investigation_id': investigation_id,
                        'analysis_types': dag_config.get('analysis_types', ['breach', 'social'])
                    }
                },
                {
                    'task_id': 'wait_for_completion',
                    'type': 'BEVWaitOperator',
                    'params': {
                        'investigation_id': investigation_id,
                        'timeout': 3600
                    },
                    'upstream_tasks': ['start_analysis']
                },
                {
                    'task_id': 'generate_report',
                    'type': 'BEVReportOperator',
                    'params': {
                        'investigation_id': investigation_id,
                        'format': 'pdf'
                    },
                    'upstream_tasks': ['wait_for_completion']
                }
            ]
        }
        
        return self.client.post('/workflow/v1/airflow/dags', json=dag_definition)
    
    def trigger_dag(self, dag_id, config=None):
        """Trigger DAG execution"""
        trigger_config = {
            'dag_id': dag_id,
            'conf': config or {},
            'execution_date': datetime.now().isoformat()
        }
        
        return self.client.post('/workflow/v1/airflow/trigger', json=trigger_config)
    
    def get_dag_status(self, dag_id, execution_date=None):
        """Get DAG execution status"""
        params = {'dag_id': dag_id}
        if execution_date:
            params['execution_date'] = execution_date
        
        return self.client.get('/workflow/v1/airflow/status', params=params)

# n8n workflow integration
class N8nIntegration:
    def __init__(self, client):
        self.client = client
    
    def create_alert_workflow(self, workflow_config):
        """Create n8n workflow for alerts"""
        workflow_definition = {
            'name': workflow_config['name'],
            'active': True,
            'nodes': [
                {
                    'name': 'BEV Webhook',
                    'type': 'webhook',
                    'parameters': {
                        'path': f'/webhook/{workflow_config["name"]}',
                        'method': 'POST'
                    },
                    'position': [250, 300]
                },
                {
                    'name': 'Process Alert',
                    'type': 'function',
                    'parameters': {
                        'code': '''
                        const alert = items[0].json;
                        
                        // Process alert data
                        const processedAlert = {
                            severity: alert.severity,
                            message: alert.message,
                            investigation_id: alert.investigation_id,
                            timestamp: new Date().toISOString()
                        };
                        
                        return [processedAlert];
                        '''
                    },
                    'position': [450, 300]
                },
                {
                    'name': 'Send Email',
                    'type': 'email',
                    'parameters': {
                        'to': workflow_config['email_recipients'],
                        'subject': 'BEV Alert: {{$node["Process Alert"].json["severity"]}}',
                        'body': 'Alert: {{$node["Process Alert"].json["message"]}}'
                    },
                    'position': [650, 300]
                }
            ],
            'connections': {
                'BEV Webhook': {
                    'main': [
                        [{'node': 'Process Alert', 'type': 'main', 'index': 0}]
                    ]
                },
                'Process Alert': {
                    'main': [
                        [{'node': 'Send Email', 'type': 'main', 'index': 0}]
                    ]
                }
            }
        }
        
        return self.client.post('/workflow/v1/n8n/workflows', json=workflow_definition)
```

---

## Data Export & Import Workflows

### Export Operations
```python
# Data export workflows
class DataExportWorkflow:
    def __init__(self, client):
        self.client = client
    
    def export_investigation_data(self, investigation_id, format='json', include_raw=False):
        """Export complete investigation data"""
        export_config = {
            'investigation_id': investigation_id,
            'format': format,  # json, csv, xml, pdf
            'include_raw_data': include_raw,
            'include_metadata': True,
            'include_timeline': True,
            'include_relationships': True
        }
        
        # Start export job
        response = self.client.post('/api/v1/export/investigation', json=export_config)
        export_job_id = response['job_id']
        
        # Poll for completion
        while True:
            status = self.client.get(f'/api/v1/export/jobs/{export_job_id}')
            
            if status['status'] == 'COMPLETED':
                return status['download_url']
            elif status['status'] == 'FAILED':
                raise Exception(f"Export failed: {status['error']}")
            
            time.sleep(5)
    
    def export_graph_data(self, investigation_id, format='graphml'):
        """Export graph visualization data"""
        export_config = {
            'investigation_id': investigation_id,
            'format': format,  # graphml, gexf, json, cytoscape
            'include_attributes': True,
            'include_weights': True
        }
        
        return self.client.post('/graph/v1/export', json=export_config)
    
    def export_timeline_data(self, investigation_id, format='json'):
        """Export timeline data"""
        export_config = {
            'investigation_id': investigation_id,
            'format': format,  # json, csv, timeline.js
            'time_range': None,  # Optional: limit time range
            'event_types': None  # Optional: filter event types
        }
        
        return self.client.post('/api/v1/export/timeline', json=export_config)
    
    def bulk_export(self, investigation_ids, format='zip'):
        """Export multiple investigations as archive"""
        export_config = {
            'investigation_ids': investigation_ids,
            'format': format,
            'include_reports': True,
            'include_evidence': True
        }
        
        response = self.client.post('/api/v1/export/bulk', json=export_config)
        return response['download_url']

# Import operations
class DataImportWorkflow:
    def __init__(self, client):
        self.client = client
    
    def import_investigation_data(self, file_path, merge_strategy='create_new'):
        """Import investigation data from file"""
        
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'merge_strategy': merge_strategy,  # create_new, merge_existing, overwrite
                'validate_schema': True,
                'import_relationships': True
            }
            
            response = self.client.post(
                '/api/v1/import/investigation',
                files=files,
                data=data
            )
        
        return response['investigation_id']
    
    def import_indicators(self, indicators, source='manual'):
        """Import threat indicators"""
        import_data = {
            'indicators': indicators,
            'source': source,
            'confidence': 0.8,
            'tags': ['imported'],
            'auto_analyze': True
        }
        
        return self.client.post('/api/v1/import/indicators', json=import_data)
    
    def import_from_csv(self, csv_file, mapping_config):
        """Import data from CSV with field mapping"""
        with open(csv_file, 'rb') as f:
            files = {'file': f}
            data = {
                'mapping': json.dumps(mapping_config),
                'has_header': True,
                'delimiter': ',',
                'auto_detect_types': True
            }
            
            response = self.client.post(
                '/api/v1/import/csv',
                files=files,
                data=data
            )
        
        return response['import_job_id']

# Usage examples
export_workflow = DataExportWorkflow(client)

# Export investigation as JSON
json_data = export_workflow.export_investigation_data(
    investigation_id='inv_123',
    format='json',
    include_raw=True
)

# Export graph for Cytoscape
graph_data = export_workflow.export_graph_data(
    investigation_id='inv_123',
    format='cytoscape'
)

# Import workflow
import_workflow = DataImportWorkflow(client)

# Import threat indicators
indicators = [
    {'type': 'ip', 'value': '192.168.1.100', 'description': 'Suspicious IP'},
    {'type': 'domain', 'value': 'malicious.com', 'description': 'Known bad domain'}
]

import_result = import_workflow.import_indicators(indicators)
```

---

## Real-time Integration Patterns

### WebSocket Integration
```python
# Real-time WebSocket client
import websocket
import json
import threading

class BEVWebSocketClient:
    def __init__(self, ws_url, api_key):
        self.ws_url = ws_url
        self.api_key = api_key
        self.ws = None
        self.callbacks = {}
        
    def connect(self):
        """Connect to WebSocket server"""
        headers = [f"Authorization: Bearer {self.api_key}"]
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            header=headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Run in background thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def on_open(self, ws):
        print("WebSocket connection opened")
        
        # Subscribe to analysis updates
        self.subscribe('analysis_updates')
        self.subscribe('system_alerts')
    
    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            event_type = data.get('type')
            
            if event_type in self.callbacks:
                for callback in self.callbacks[event_type]:
                    callback(data)
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")
    
    def subscribe(self, event_type):
        """Subscribe to event type"""
        message = {
            'action': 'subscribe',
            'event_type': event_type
        }
        self.ws.send(json.dumps(message))
    
    def add_callback(self, event_type, callback):
        """Add callback for event type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def send_message(self, message):
        """Send message to server"""
        self.ws.send(json.dumps(message))

# Usage example
def on_analysis_update(data):
    print(f"Analysis update: {data['investigation_id']} - {data['status']}")

def on_system_alert(data):
    print(f"System alert: {data['severity']} - {data['message']}")

# Connect and set up callbacks
ws_client = BEVWebSocketClient('wss://your-bev-instance/ws/v1/analysis', api_key)
ws_client.add_callback('analysis_updates', on_analysis_update)
ws_client.add_callback('system_alerts', on_system_alert)
ws_client.connect()
```

### Server-Sent Events Integration
```python
# Server-Sent Events client
import sseclient
import requests

class BEVEventStream:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        
    def stream_analysis_events(self, investigation_id=None):
        """Stream analysis events"""
        url = f"{self.base_url}/events/v1/analysis"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        params = {}
        
        if investigation_id:
            params['investigation_id'] = investigation_id
        
        response = requests.get(url, headers=headers, params=params, stream=True)
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            yield json.loads(event.data)
    
    def stream_system_events(self):
        """Stream system monitoring events"""
        url = f"{self.base_url}/events/v1/system"
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        response = requests.get(url, headers=headers, stream=True)
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            yield json.loads(event.data)

# Usage example
event_stream = BEVEventStream('https://your-bev-instance', api_key)

# Stream analysis events for specific investigation
for event in event_stream.stream_analysis_events('inv_123'):
    print(f"Event: {event['type']} - {event['data']}")
    
    if event['type'] == 'analysis_complete':
        print("Analysis completed!")
        break
```

---

## SDK and Client Libraries

### Python SDK
```python
# Complete Python SDK
class BEVSDK:
    def __init__(self, base_url, api_key, api_secret=None):
        self.client = BEVAPIClient(base_url, api_key, api_secret)
        
        # Initialize workflow managers
        self.investigations = InvestigationWorkflow(self.client)
        self.batch_analysis = BatchAnalysisWorkflow(self.client)
        self.export = DataExportWorkflow(self.client)
        self.import_data = DataImportWorkflow(self.client)
        
        # Initialize integrations
        self.airflow = AirflowIntegration(self.client)
        self.n8n = N8nIntegration(self.client)
    
    def analyze_email(self, email, analysis_types=None):
        """Quick email analysis"""
        if analysis_types is None:
            analysis_types = ['breach', 'social']
        
        targets = [{'type': 'email', 'value': email}]
        job_id = self.batch_analysis.submit_batch_analysis(targets, analysis_types)
        
        # Wait for completion
        while True:
            status = self.batch_analysis.get_batch_status(job_id)
            if status['status'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        
        if status['status'] == 'COMPLETED':
            return self.batch_analysis.get_batch_results(job_id)
        else:
            raise Exception(f"Analysis failed: {status['error']}")
    
    def analyze_domain(self, domain, include_subdomains=True):
        """Quick domain analysis"""
        targets = [{'type': 'domain', 'value': domain}]
        
        if include_subdomains:
            # Get subdomains first
            subdomain_result = self.client.get(
                f'/analyzer/v1/domain/{domain}/subdomains'
            )
            
            for subdomain in subdomain_result['subdomains']:
                targets.append({'type': 'domain', 'value': subdomain})
        
        job_id = self.batch_analysis.submit_batch_analysis(
            targets, 
            ['domain_info', 'reputation', 'ssl_cert']
        )
        
        # Wait for completion and return results
        while True:
            status = self.batch_analysis.get_batch_status(job_id)
            if status['status'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(5)
        
        return self.batch_analysis.get_batch_results(job_id)
    
    def search(self, query, search_type='all'):
        """Universal search across all data"""
        search_params = {
            'query': query,
            'type': search_type,
            'limit': 100,
            'include_metadata': True
        }
        
        return self.client.get('/api/v1/search', params=search_params)
    
    def get_system_health(self):
        """Get system health status"""
        return self.client.get('/api/v1/health')

# Usage examples
sdk = BEVSDK('https://your-bev-instance', 'your-api-key')

# Quick email analysis
email_results = sdk.analyze_email('suspect@example.com')
print(f"Found in {len(email_results['breach_data'])} breaches")

# Domain analysis
domain_results = sdk.analyze_domain('example.com')
print(f"Domain reputation: {domain_results['reputation_score']}")

# Universal search
search_results = sdk.search('phishing')
print(f"Found {len(search_results['items'])} results")
```

### JavaScript/Node.js SDK
```javascript
// Node.js SDK
class BEVAPI {
    constructor(baseUrl, apiKey, apiSecret = null) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.axios = require('axios');
        
        // Set default headers
        this.axios.defaults.headers.common['X-API-Key'] = apiKey;
        this.axios.defaults.headers.common['Content-Type'] = 'application/json';
    }
    
    async makeRequest(method, endpoint, data = null) {
        const url = `${this.baseUrl}${endpoint}`;
        
        try {
            const response = await this.axios({
                method,
                url,
                data,
                headers: this.generateHeaders(method, endpoint, data)
            });
            
            return response.data;
        } catch (error) {
            throw new Error(`API request failed: ${error.response?.data?.message || error.message}`);
        }
    }
    
    generateHeaders(method, endpoint, data) {
        const headers = {};
        
        if (this.apiSecret) {
            const timestamp = Math.floor(Date.now() / 1000).toString();
            const bodyString = data ? JSON.stringify(data) : '';
            const sigString = `${method.toUpperCase()}\n${endpoint}\n${timestamp}\n${bodyString}`;
            
            const crypto = require('crypto');
            const signature = crypto
                .createHmac('sha256', this.apiSecret)
                .update(sigString)
                .digest('hex');
            
            headers['X-Timestamp'] = timestamp;
            headers['X-Signature'] = signature;
        }
        
        return headers;
    }
    
    // Investigation methods
    async createInvestigation(name, description, targets) {
        const data = {
            name,
            description,
            targets,
            created_at: new Date().toISOString(),
            status: 'ACTIVE'
        };
        
        return await this.makeRequest('POST', '/api/v1/investigations', data);
    }
    
    async getInvestigation(investigationId) {
        return await this.makeRequest('GET', `/api/v1/investigations/${investigationId}`);
    }
    
    async startAnalysis(investigationId, analysisTypes = ['breach', 'social']) {
        const data = {
            investigation_id: investigationId,
            analysis_types: analysisTypes,
            priority: 'normal'
        };
        
        return await this.makeRequest('POST', '/analyzer/v1/batch', data);
    }
    
    // Quick analysis methods
    async analyzeEmail(email, analysisTypes = ['breach', 'social']) {
        const targets = [{ type: 'email', value: email }];
        const response = await this.makeRequest('POST', '/analyzer/v1/batch', {
            targets,
            analysis_types: analysisTypes
        });
        
        const jobId = response.job_id;
        
        // Poll for completion
        while (true) {
            const status = await this.makeRequest('GET', `/analyzer/v1/batch/${jobId}/status`);
            
            if (status.status === 'COMPLETED') {
                return await this.makeRequest('GET', `/analyzer/v1/batch/${jobId}/results`);
            } else if (status.status === 'FAILED') {
                throw new Error(`Analysis failed: ${status.error}`);
            }
            
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
    }
    
    async search(query, searchType = 'all') {
        const params = new URLSearchParams({
            query,
            type: searchType,
            limit: '100'
        });
        
        return await this.makeRequest('GET', `/api/v1/search?${params}`);
    }
}

// Usage example
const bev = new BEVAPI('https://your-bev-instance', 'your-api-key');

// Async function example
async function runAnalysis() {
    try {
        // Create investigation
        const investigation = await bev.createInvestigation(
            'Email Investigation',
            'Analyzing suspicious email',
            [{ type: 'email', value: 'suspect@example.com' }]
        );
        
        console.log(`Created investigation: ${investigation.investigation_id}`);
        
        // Start analysis
        const analysis = await bev.startAnalysis(investigation.investigation_id);
        console.log(`Started analysis job: ${analysis.job_id}`);
        
        // Quick email analysis
        const emailResults = await bev.analyzeEmail('suspect@example.com');
        console.log('Email analysis results:', emailResults);
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

runAnalysis();
```

---

## Error Handling & Retry Logic

### Robust Error Handling
```python
# Advanced error handling and retry logic
import time
import random
from functools import wraps

class BEVAPIError(Exception):
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class RateLimitError(BEVAPIError):
    def __init__(self, retry_after=None):
        super().__init__("Rate limit exceeded")
        self.retry_after = retry_after

def retry_with_backoff(max_retries=3, base_delay=1, max_delay=60):
    """Decorator for exponential backoff retry"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except RateLimitError as e:
                    if attempt == max_retries:
                        raise
                    
                    # Wait for retry_after if provided, otherwise use exponential backoff
                    delay = e.retry_after or min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1) * delay  # Add jitter
                    time.sleep(delay + jitter)
                
                except BEVAPIError as e:
                    if attempt == max_retries or e.status_code in [400, 401, 403, 404]:
                        raise  # Don't retry client errors
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, 0.1) * delay
                    time.sleep(delay + jitter)
                
                except Exception as e:
                    if attempt == max_retries:
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator

class RobustBEVClient(BEVAPIClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests
    
    def _make_request(self, method, endpoint, **kwargs):
        """Make request with enhanced error handling"""
        
        # Respect rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        try:
            response = super()._make_request(method, endpoint, **kwargs)
            return response
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            
            if status_code == 429:  # Rate limited
                retry_after = int(e.response.headers.get('Retry-After', 60))
                raise RateLimitError(retry_after)
            
            elif status_code >= 500:  # Server error
                raise BEVAPIError(
                    f"Server error: {e.response.text}",
                    status_code,
                    e.response
                )
            
            elif status_code >= 400:  # Client error
                raise BEVAPIError(
                    f"Client error: {e.response.text}",
                    status_code,
                    e.response
                )
            
            else:
                raise BEVAPIError(f"HTTP error: {e}", status_code, e.response)
        
        except requests.exceptions.ConnectionError as e:
            raise BEVAPIError(f"Connection error: {e}")
        
        except requests.exceptions.Timeout as e:
            raise BEVAPIError(f"Request timeout: {e}")
        
        except requests.exceptions.RequestException as e:
            raise BEVAPIError(f"Request error: {e}")
    
    @retry_with_backoff(max_retries=3)
    def get(self, endpoint, **kwargs):
        return super().get(endpoint, **kwargs)
    
    @retry_with_backoff(max_retries=3)
    def post(self, endpoint, **kwargs):
        return super().post(endpoint, **kwargs)
    
    @retry_with_backoff(max_retries=2)  # Fewer retries for write operations
    def put(self, endpoint, **kwargs):
        return super().put(endpoint, **kwargs)
    
    @retry_with_backoff(max_retries=2)
    def delete(self, endpoint, **kwargs):
        return super().delete(endpoint, **kwargs)

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
            else:
                raise BEVAPIError("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

# Usage with circuit breaker
class BEVClientWithCircuitBreaker(RobustBEVClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.circuit_breaker = CircuitBreaker()
    
    def _make_request(self, method, endpoint, **kwargs):
        return self.circuit_breaker.call(
            super()._make_request,
            method,
            endpoint,
            **kwargs
        )
```

---

## Rate Limiting & Performance

### Rate Limiting Strategies
```python
# Client-side rate limiting
import time
from collections import deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_requests, time_window):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self.lock = Lock()
    
    def acquire(self):
        with self.lock:
            now = time.time()
            
            # Remove old requests outside the time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            # Calculate wait time
            oldest_request = self.requests[0]
            wait_time = oldest_request + self.time_window - now
            return wait_time

class RateLimitedBEVClient(RobustBEVClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Different rate limits for different endpoints
        self.rate_limiters = {
            'analysis': RateLimiter(max_requests=10, time_window=60),  # 10 per minute
            'search': RateLimiter(max_requests=100, time_window=60),   # 100 per minute
            'default': RateLimiter(max_requests=50, time_window=60)    # 50 per minute
        }
    
    def _get_rate_limiter(self, endpoint):
        """Get appropriate rate limiter for endpoint"""
        if '/analyzer/' in endpoint:
            return self.rate_limiters['analysis']
        elif '/search' in endpoint:
            return self.rate_limiters['search']
        else:
            return self.rate_limiters['default']
    
    def _make_request(self, method, endpoint, **kwargs):
        rate_limiter = self._get_rate_limiter(endpoint)
        
        # Check rate limit
        result = rate_limiter.acquire()
        if result is not True:
            wait_time = result
            print(f"Rate limited, waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            rate_limiter.acquire()  # Should succeed now
        
        return super()._make_request(method, endpoint, **kwargs)

# Performance optimization techniques
class OptimizedBEVClient(RateLimitedBEVClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _get_cache_key(self, method, endpoint, params=None):
        """Generate cache key for request"""
        key_parts = [method, endpoint]
        if params:
            key_parts.append(str(sorted(params.items())))
        return '|'.join(key_parts)
    
    def _is_cache_valid(self, cache_entry):
        """Check if cache entry is still valid"""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def get(self, endpoint, **kwargs):
        """GET with caching"""
        params = kwargs.get('params')
        cache_key = self._get_cache_key('GET', endpoint, params)
        
        # Check cache
        if cache_key in self.response_cache:
            cache_entry = self.response_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['data']
        
        # Make request and cache result
        response = super().get(endpoint, **kwargs)
        self.response_cache[cache_key] = {
            'data': response,
            'timestamp': time.time()
        }
        
        return response
    
    def batch_requests(self, requests):
        """Execute multiple requests in optimal order"""
        # Group requests by rate limiter
        grouped_requests = {}
        for req in requests:
            limiter_key = 'analysis' if '/analyzer/' in req['endpoint'] else 'default'
            if limiter_key not in grouped_requests:
                grouped_requests[limiter_key] = []
            grouped_requests[limiter_key].append(req)
        
        results = []
        
        # Execute each group with appropriate rate limiting
        for limiter_key, group in grouped_requests.items():
            for req in group:
                method = req['method']
                endpoint = req['endpoint']
                kwargs = req.get('kwargs', {})
                
                try:
                    result = self._make_request(method, endpoint, **kwargs)
                    results.append({'success': True, 'data': result})
                except Exception as e:
                    results.append({'success': False, 'error': str(e)})
        
        return results
```

---

## Integration Examples

### Complete Integration Example
```python
# Complete integration example: Automated threat hunting
class ThreatHuntingIntegration:
    def __init__(self, bev_client):
        self.client = bev_client
        self.investigation_workflow = InvestigationWorkflow(bev_client)
        self.export_workflow = DataExportWorkflow(bev_client)
        
    def hunt_phishing_campaign(self, initial_indicators):
        """Automated phishing campaign investigation"""
        
        print("🎯 Starting phishing campaign hunt...")
        
        # 1. Create investigation
        investigation_id = self.investigation_workflow.create_investigation(
            name=f"Phishing Hunt - {datetime.now().strftime('%Y%m%d_%H%M')}",
            description="Automated phishing campaign investigation",
            targets=initial_indicators,
            classification='RESTRICTED'
        )
        
        print(f"📋 Created investigation: {investigation_id}")
        
        # 2. Start initial analysis
        analysis_job = self.investigation_workflow.start_analysis(
            investigation_id,
            analysis_types=['breach', 'social', 'domain_info', 'reputation']
        )
        
        print(f"🔍 Started analysis job: {analysis_job['job_id']}")
        
        # 3. Monitor and expand investigation
        expanded_targets = set()
        iteration = 1
        max_iterations = 3
        
        while iteration <= max_iterations:
            print(f"🔄 Investigation iteration {iteration}/{max_iterations}")
            
            # Wait for current analysis to complete
            self._wait_for_analysis_completion(investigation_id)
            
            # Get results and expand targets
            results = self.investigation_workflow.get_analysis_results(investigation_id)
            new_targets = self._extract_expansion_targets(results)
            
            # Add new targets if found
            for target in new_targets:
                if target not in expanded_targets:
                    self.investigation_workflow.add_target_to_investigation(
                        investigation_id, 
                        target['type'], 
                        target['value']
                    )
                    expanded_targets.add(target['value'])
                    print(f"➕ Added target: {target['type']} - {target['value']}")
            
            # Start analysis for new targets
            if new_targets:
                self.investigation_workflow.start_analysis(
                    investigation_id,
                    analysis_types=['breach', 'social', 'crypto']
                )
            
            iteration += 1
        
        # 4. Generate comprehensive report
        print("📊 Generating final report...")
        
        final_results = self.investigation_workflow.get_analysis_results(investigation_id)
        report_url = self.investigation_workflow.generate_report(
            investigation_id, 
            'pdf'
        )
        
        # 5. Export data for further analysis
        export_data = self.export_workflow.export_investigation_data(
            investigation_id,
            format='json',
            include_raw=True
        )
        
        # 6. Generate threat intelligence
        threat_intel = self._generate_threat_intelligence(final_results)
        
        return {
            'investigation_id': investigation_id,
            'total_targets': len(expanded_targets) + len(initial_indicators),
            'threat_score': threat_intel['threat_score'],
            'attribution_confidence': threat_intel['attribution_confidence'],
            'report_url': report_url,
            'export_data': export_data,
            'recommendations': threat_intel['recommendations']
        }
    
    def _wait_for_analysis_completion(self, investigation_id, timeout=3600):
        """Wait for investigation analysis to complete"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.investigation_workflow.get_investigation_status(investigation_id)
            
            if status['analysis_status'] == 'COMPLETED':
                return True
            elif status['analysis_status'] == 'FAILED':
                raise Exception("Analysis failed")
            
            time.sleep(30)
        
        raise Exception("Analysis timeout")
    
    def _extract_expansion_targets(self, results):
        """Extract potential targets for investigation expansion"""
        new_targets = []
        
        # Extract emails from social media results
        for social_result in results.get('social_profiles', []):
            for email in social_result.get('associated_emails', []):
                new_targets.append({'type': 'email', 'value': email})
        
        # Extract domains from breach data
        for breach_result in results.get('breach_data', []):
            if 'domain' in breach_result:
                new_targets.append({'type': 'domain', 'value': breach_result['domain']})
        
        # Extract related crypto addresses
        for crypto_result in results.get('crypto_analysis', []):
            for related_addr in crypto_result.get('related_addresses', []):
                new_targets.append({'type': 'crypto_address', 'value': related_addr})
        
        return new_targets
    
    def _generate_threat_intelligence(self, results):
        """Generate threat intelligence from investigation results"""
        
        # Calculate threat score based on findings
        threat_factors = {
            'breach_count': len(results.get('breach_data', [])),
            'darknet_mentions': len(results.get('darknet_findings', [])),
            'crypto_risk_score': max([r.get('risk_score', 0) for r in results.get('crypto_analysis', [])], default=0),
            'domain_reputation': min([r.get('reputation_score', 1.0) for r in results.get('domain_analysis', [])], default=1.0)
        }
        
        # Weighted threat score calculation
        threat_score = (
            min(threat_factors['breach_count'] * 0.1, 0.3) +
            min(threat_factors['darknet_mentions'] * 0.2, 0.4) +
            threat_factors['crypto_risk_score'] * 0.2 +
            (1.0 - threat_factors['domain_reputation']) * 0.1
        )
        
        # Attribution confidence
        attribution_indicators = [
            'shared_infrastructure',
            'common_tactics',
            'overlapping_timeframes',
            'related_personas'
        ]
        
        attribution_score = sum([
            1 for indicator in attribution_indicators 
            if self._check_attribution_indicator(results, indicator)
        ]) / len(attribution_indicators)
        
        # Generate recommendations
        recommendations = []
        
        if threat_score > 0.7:
            recommendations.append("HIGH PRIORITY: Immediate threat response recommended")
        if threat_factors['breach_count'] > 5:
            recommendations.append("Multiple breach exposures detected - credential monitoring advised")
        if threat_factors['darknet_mentions'] > 0:
            recommendations.append("Darknet activity detected - enhanced monitoring recommended")
        
        return {
            'threat_score': threat_score,
            'attribution_confidence': attribution_score,
            'threat_factors': threat_factors,
            'recommendations': recommendations
        }
    
    def _check_attribution_indicator(self, results, indicator):
        """Check for specific attribution indicators"""
        # Implementation would analyze results for attribution patterns
        # This is a simplified version
        return len(results.get('social_profiles', [])) > 2

# Usage example
client = OptimizedBEVClient('https://your-bev-instance', 'your-api-key')
threat_hunter = ThreatHuntingIntegration(client)

# Start automated threat hunt
initial_indicators = [
    {'type': 'email', 'value': 'phishing@suspicious-domain.com'},
    {'type': 'domain', 'value': 'suspicious-domain.com'},
    {'type': 'ip', 'value': '192.168.1.100'}
]

hunt_results = threat_hunter.hunt_phishing_campaign(initial_indicators)

print(f"🎯 Hunt completed:")
print(f"   Investigation ID: {hunt_results['investigation_id']}")
print(f"   Total targets analyzed: {hunt_results['total_targets']}")
print(f"   Threat score: {hunt_results['threat_score']:.2f}")
print(f"   Attribution confidence: {hunt_results['attribution_confidence']:.2f}")
print(f"   Report available at: {hunt_results['report_url']}")

for recommendation in hunt_results['recommendations']:
    print(f"   📋 {recommendation}")
```

---

*Last Updated: 2025-09-19*
*Framework Version: BEV OSINT v2.0*
*Classification: INTERNAL*
*Document Version: 1.0*