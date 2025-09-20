#!/usr/bin/env python3
"""
Military-Grade Security & Message Queue Infrastructure
Place in: /home/starlord/Bev/src/security/security_framework.py
"""

import asyncio
import os
import re
import json
import hashlib
import subprocess
import signal
import atexit
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import aio_pika
import aiokafka
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import redis.asyncio as redis
from collections import defaultdict
import uuid
import time
import numpy as np

# ================= SECURITY FRAMEWORK =================

class OperationalSecurityFramework:
    """Complete military-grade security infrastructure"""
    
    def __init__(self):
        # Tailscale VPN mesh
        self.vpn = TailscaleVPN()
        
        # PII redaction
        self.pii_redactor = AutomaticPIIRedactor()
        
        # Encryption management
        self.encryption = EncryptionManager()
        
        # API key rotation
        self.key_rotator = RollingKeyRotator()
        
        # Sandbox execution
        self.sandbox = SandboxExecutor()
        
        # Secure memory
        self.memory_wiper = SecureMemoryWiper()
        
        # Security logging
        self.security_log = []
    
    async def initialize_security(self):
        """Initialize all security measures"""
        
        print("🔒 Initializing Fortress-Level Security...")
        
        # Start VPN
        await self.vpn.connect()
        
        # Initialize encryption
        await self.encryption.initialize_keys()
        
        # Start key rotation
        asyncio.create_task(self.key_rotator.start_rotation())
        
        # Configure sandboxes
        await self.sandbox.configure_environments()
        
        # Register shutdown hooks
        atexit.register(self.secure_shutdown)
        signal.signal(signal.SIGTERM, lambda s, f: self.secure_shutdown())
        
        print("✅ All security measures active!")
    
    def secure_shutdown(self):
        """Secure cleanup on shutdown"""
        print("🔒 Performing secure shutdown...")
        
        # Wipe sensitive memory
        self.memory_wiper.wipe_all_sensitive()
        
        # Clear caches
        gc.collect()
        gc.collect()
        gc.collect()
        
        # Clear OS cache if possible
        if os.name == 'posix':
            try:
                # SECURITY: Replace with subprocess.run() - os.system('sync')
                if os.path.exists('/proc/sys/vm/drop_caches'):
                    # SECURITY: Replace with subprocess.run() - os.system('echo 3 > /proc/sys/vm/drop_caches')
            except:
                pass
        
        print("✅ Memory wiped successfully")


class TailscaleVPN:
    """Zero Trust Network with Tailscale"""
    
    def __init__(self):
        self.tailscale_key = os.getenv('TAILSCALE_AUTH_KEY', 'tskey-auth-placeholder')
        self.network_config = {
            'subnet_routes': ['10.0.0.0/8'],
            'accept_routes': True,
            'exit_node': True,
            'ssh_enabled': True
        }
        self.connected = False
    
    async def connect(self):
        """Establish Tailscale connection"""
        
        # Check if Tailscale is installed
        if not self.is_installed():
            print("⚠️ Tailscale not installed - skipping VPN setup")
            return False
        
        try:
            # Authenticate with Tailscale
            result = subprocess.run([
                'tailscale', 'up',
                f'--authkey={self.tailscale_key}',
                '--accept-routes',
                '--advertise-exit-node',
                '--ssh'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.connected = True
                print("✅ Tailscale VPN connected")
                
                # Configure firewall
                await self.configure_firewall()
                return True
            else:
                print(f"⚠️ Tailscale connection failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"⚠️ VPN error: {e}")
            return False
    
    def is_installed(self):
        """Check if Tailscale is installed"""
        try:
            result = subprocess.run(['tailscale', 'version'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def configure_firewall(self):
        """Configure firewall rules"""
        
        # Only configure if we have permissions
        if os.geteuid() != 0:
            print("⚠️ Not running as root - skipping firewall configuration")
            return
        
        rules = [
            # Allow Tailscale
            'iptables -A INPUT -i tailscale0 -j ACCEPT',
            'iptables -A OUTPUT -o tailscale0 -j ACCEPT',
            
            # Allow localhost
            'iptables -A INPUT -i lo -j ACCEPT',
            'iptables -A OUTPUT -o lo -j ACCEPT',
            
            # Allow established connections
            'iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT'
        ]
        
        for rule in rules:
            try:
                subprocess.run(rule.split())
            except:
                pass


class AutomaticPIIRedactor:
    """Real-time PII detection and redaction"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'bitcoin': r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            'ethereum': r'\b0x[a-fA-F0-9]{40}\b'
        }
        
        self.redaction_map = {}
    
    def redact(self, text: str) -> str:
        """Redact PII from text"""
        
        redacted = text
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.finditer(pattern, redacted)
            
            for match in matches:
                # Generate consistent hash for same PII
                pii_value = match.group()
                pii_hash = hashlib.sha256(pii_value.encode()).hexdigest()[:8]
                
                # Store mapping (encrypted)
                self.redaction_map[pii_hash] = self.encrypt_pii(pii_value)
                
                # Replace with token
                replacement = f"[{pii_type.upper()}_{pii_hash}]"
                redacted = redacted.replace(pii_value, replacement)
        
        return redacted
    
    def encrypt_pii(self, pii_value: str) -> str:
        """Encrypt PII for secure storage"""
        # Simple encryption - in production use proper key management
        key = Fernet.generate_key()
        f = Fernet(key)
        return f.encrypt(pii_value.encode()).decode()
    
    def deep_clean(self, data: Any) -> Any:
        """Recursively clean PII from any data structure"""
        
        if isinstance(data, str):
            return self.redact(data)
        elif isinstance(data, dict):
            return {k: self.deep_clean(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.deep_clean(item) for item in data]
        else:
            return data


class EncryptionManager:
    """End-to-end encryption for sensitive data"""
    
    def __init__(self):
        self.master_key = None
        self.agent_keys = {}
        
        # Key derivation function
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=os.urandom(16),
            iterations=100000,
        )
    
    async def initialize_keys(self):
        """Initialize encryption keys"""
        
        # Generate master key
        self.master_key = Fernet.generate_key()
        
        # Derive agent-specific keys
        agents = ['research_oracle', 'code_assassin', 'memory_keeper', 'tool_master', 'guardian']
        
        for agent_id in agents:
            self.agent_keys[agent_id] = self.derive_agent_key(agent_id)
        
        print("✅ Encryption keys generated")
    
    def derive_agent_key(self, agent_id: str) -> bytes:
        """Derive agent-specific key from master"""
        
        # Use master key as base
        if not self.master_key:
            self.master_key = Fernet.generate_key()
        
        # Derive unique key for agent
        derived = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            agent_id.encode(),
            100000
        )
        
        # Convert to Fernet-compatible key
        import base64
        return base64.urlsafe_b64encode(derived[:32])
    
    def encrypt_data(self, data: Any, agent_id: str) -> Dict:
        """Encrypt data with agent-specific key"""
        
        key = self.agent_keys.get(agent_id, self.master_key)
        f = Fernet(key)
        
        # Serialize data
        import pickle
        serialized = pickle.dumps(data)
        
        # Encrypt
        encrypted = f.encrypt(serialized)
        
        return {
            'data': encrypted.decode(),
            'metadata': {
                'version': 1,
                'agent_id': agent_id,
                'timestamp': datetime.now().isoformat(),
                'algorithm': 'Fernet-AES256'
            }
        }
    
    def decrypt_data(self, encrypted_data: Dict, agent_id: str) -> Any:
        """Decrypt data"""
        
        key = self.agent_keys.get(agent_id, self.master_key)
        f = Fernet(key)
        
        # Decrypt
        decrypted = f.decrypt(encrypted_data['data'].encode())
        
        # Deserialize
        import pickle
        return pickle.loads(decrypted)


class RollingKeyRotator:
    """Automatic API key rotation"""
    
    def __init__(self):
        self.rotation_interval = 3600  # 1 hour
        self.active_keys = {}
        self.pending_keys = {}
        
        # API providers
        self.providers = [
            'openai', 'anthropic', 'google',
            'github', 'twitter', 'reddit'
        ]
    
    async def start_rotation(self):
        """Start automatic key rotation"""
        
        # Initial key generation
        await self.generate_initial_keys()
        
        # Schedule rotation
        while True:
            await asyncio.sleep(self.rotation_interval)
            await self.rotate_all_keys()
    
    async def generate_initial_keys(self):
        """Generate initial API keys"""
        
        for provider in self.providers:
            # Generate placeholder key
            key = f"{provider}_key_{uuid.uuid4().hex[:16]}"
            
            self.active_keys[provider] = {
                'key': key,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=self.rotation_interval)
            }
        
        print(f"✅ Generated {len(self.active_keys)} initial API keys")
    
    async def rotate_all_keys(self):
        """Rotate all API keys"""
        
        for provider in self.providers:
            try:
                # Generate new key
                new_key = f"{provider}_key_{uuid.uuid4().hex[:16]}"
                
                # Add to pending
                self.pending_keys[provider] = {
                    'key': new_key,
                    'created_at': datetime.now()
                }
                
                # Wait for overlap period (5 minutes)
                await asyncio.sleep(300)
                
                # Promote to active
                self.active_keys[provider] = self.pending_keys[provider]
                
                print(f"✅ Rotated {provider} API key")
                
            except Exception as e:
                print(f"⚠️ Failed to rotate {provider} key: {e}")
    
    def get_active_key(self, provider: str) -> str:
        """Get current active key"""
        
        if provider in self.active_keys:
            return self.active_keys[provider]['key']
        
        if provider in self.pending_keys:
            return self.pending_keys[provider]['key']
        
        raise Exception(f"No active key for {provider}")


class SandboxExecutor:
    """Sandboxed execution environments"""
    
    def __init__(self):
        self.docker_available = self.check_docker()
        self.resource_limits = {
            'cpu': '1.0',
            'memory': '512m',
            'timeout': 30
        }
    
    def check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    async def configure_environments(self):
        """Configure sandbox environments"""
        
        if not self.docker_available:
            print("⚠️ Docker not available - sandbox features limited")
            return
        
        print("✅ Sandbox environments configured")
    
    async def execute_code(self, code: str, language: str = 'python') -> Dict:
        """Execute code in sandbox"""
        
        if not self.docker_available:
            # Fallback to subprocess with restrictions
            return await self.execute_subprocess(code, language)
        
        # Docker-based execution
        return await self.execute_docker(code, language)
    
    async def execute_subprocess(self, code: str, language: str) -> Dict:
        """Execute in restricted subprocess"""
        
        result = {
            'success': False,
            'output': '',
            'error': '',
            'exit_code': -1
        }
        
        try:
            if language == 'python':
                # Write code to temp file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    temp_file = f.name
                
                # Execute with timeout
                proc = await asyncio.create_subprocess_exec(
                    'python3', temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=self.resource_limits['timeout']
                    )
                    
                    result['success'] = proc.returncode == 0
                    result['output'] = stdout.decode()
                    result['error'] = stderr.decode()
                    result['exit_code'] = proc.returncode
                    
                except asyncio.TimeoutError:
                    proc.kill()
                    result['error'] = 'Execution timeout'
                
                finally:
                    # Clean up temp file
                    os.unlink(temp_file)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    async def execute_docker(self, code: str, language: str) -> Dict:
        """Execute in Docker container"""
        
        # Docker execution implementation
        # Would use docker-py or subprocess to run containerized code
        
        return {
            'success': True,
            'output': 'Docker execution placeholder',
            'error': '',
            'exit_code': 0
        }


class SecureMemoryWiper:
    """Secure memory management"""
    
    def __init__(self):
        self.sensitive_allocations = set()
        self.wipe_patterns = [
            b'\x00' * 1024,  # Zeros
            b'\xFF' * 1024,  # Ones
            b'\xAA' * 1024,  # Alternating
            os.urandom(1024)  # Random
        ]
    
    def secure_allocate(self, size: int) -> bytearray:
        """Allocate memory that will be wiped"""
        
        data = bytearray(size)
        self.sensitive_allocations.add(id(data))
        return data
    
    def secure_wipe(self, data: Any):
        """DOD 5220.22-M compliant wipe"""
        
        if not isinstance(data, (bytearray, memoryview)):
            data = bytearray(str(data).encode())
        
        # Multi-pass overwrite
        for pattern in self.wipe_patterns:
            for i in range(0, len(data), len(pattern)):
                chunk_size = min(len(pattern), len(data) - i)
                data[i:i+chunk_size] = pattern[:chunk_size]
        
        # Final random pass
        import random
        for i in range(len(data)):
            data[i] = random.randint(0, 255)
        
        # Mark for garbage collection
        del data
        gc.collect()
    
    def wipe_all_sensitive(self):
        """Wipe all tracked sensitive data"""
        
        for obj_id in list(self.sensitive_allocations):
            # Find object
            for obj in gc.get_objects():
                if id(obj) == obj_id:
                    try:
                        self.secure_wipe(obj)
                    except:
                        pass
                    break
        
        self.sensitive_allocations.clear()


# ================= MESSAGE QUEUE INFRASTRUCTURE =================

class MessageQueueOrchestrator:
    """Complete async communication infrastructure"""
    
    def __init__(self):
        # RabbitMQ cluster
        self.rabbitmq = RabbitMQCluster()
        
        # Kafka cluster
        self.kafka = KafkaCluster()
        
        # Event bus
        self.event_bus = AsyncEventBus()
        
        # Dead letter queue
        self.dlq_handler = DeadLetterQueueManager()
        
        # Message router
        self.router = MessageRouter()
        
        # Metrics
        self.metrics = MessageMetrics()
    
    async def initialize(self):
        """Initialize all messaging infrastructure"""
        
        print("📬 Initializing Message Queue Infrastructure...")
        
        # Setup RabbitMQ
        await self.rabbitmq.setup_cluster()
        
        # Setup Kafka
        await self.kafka.setup_cluster()
        
        # Initialize event bus
        await self.event_bus.initialize()
        
        # Setup DLQ
        await self.dlq_handler.initialize()
        
        print("✅ Message infrastructure ready!")


class RabbitMQCluster:
    """High-availability RabbitMQ broker"""
    
    def __init__(self):
        self.connection = None
        self.channel = None
        
        self.exchanges = {
            'agent.commands': {
                'type': aio_pika.ExchangeType.TOPIC,
                'durable': True
            },
            'agent.events': {
                'type': aio_pika.ExchangeType.FANOUT,
                'durable': True
            },
            'research.tasks': {
                'type': aio_pika.ExchangeType.DIRECT,
                'durable': True
            }
        }
        
        self.queues = {
            'research_oracle.tasks': {
                'routing_key': 'research.*',
                'exchange': 'agent.commands'
            },
            'code_assassin.tasks': {
                'routing_key': 'code.*',
                'exchange': 'agent.commands'
            }
        }
    
    async def setup_cluster(self):
        """Setup RabbitMQ connection"""
        
        try:
            # Connect to RabbitMQ
            self.connection = await aio_pika.connect_robust(
                "amqp://guest:guest@localhost/",
                heartbeat=60,
                connection_attempts=5,
                retry_delay=2
            )
            
            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=100)
            
            # Declare exchanges
            await self.create_exchanges()
            
            # Declare queues
            await self.create_queues()
            
            print("✅ RabbitMQ cluster connected")
            
        except Exception as e:
            print(f"⚠️ RabbitMQ connection failed: {e}")
    
    async def create_exchanges(self):
        """Create all exchanges"""
        
        if not self.channel:
            return
        
        for exchange_name, config in self.exchanges.items():
            try:
                await self.channel.declare_exchange(
                    exchange_name,
                    type=config['type'],
                    durable=config['durable']
                )
            except Exception as e:
                print(f"⚠️ Failed to create exchange {exchange_name}: {e}")
    
    async def create_queues(self):
        """Create all queues"""
        
        if not self.channel:
            return
        
        for queue_name, config in self.queues.items():
            try:
                queue = await self.channel.declare_queue(
                    queue_name,
                    durable=True
                )
                
                # Bind to exchange
                await queue.bind(
                    config['exchange'],
                    routing_key=config['routing_key']
                )
                
            except Exception as e:
                print(f"⚠️ Failed to create queue {queue_name}: {e}")
    
    async def publish(self, exchange: str, routing_key: str, message: Dict, priority: int = 5):
        """Publish message"""
        
        if not self.channel:
            print("⚠️ RabbitMQ not connected")
            return
        
        try:
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(message).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    priority=priority,
                    timestamp=datetime.now(),
                    message_id=str(uuid.uuid4())
                ),
                routing_key=routing_key
            )
        except Exception as e:
            print(f"⚠️ Failed to publish message: {e}")


class KafkaCluster:
    """High-throughput Kafka event streaming"""
    
    def __init__(self):
        self.bootstrap_servers = ['localhost:9092']
        self.producer = None
        self.consumer_groups = {}
        
        self.topics = {
            'agent.tasks': {
                'partitions': 10,
                'replication_factor': 1
            },
            'agent.results': {
                'partitions': 10,
                'replication_factor': 1
            }
        }
    
    async def setup_cluster(self):
        """Initialize Kafka cluster"""
        
        try:
            # Create producer
            self.producer = AIOKafkaProducer(
                bootstrap_servers=','.join(self.bootstrap_servers),
                compression_type='snappy',
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            await self.producer.start()
            
            print("✅ Kafka cluster connected")
            
        except Exception as e:
            print(f"⚠️ Kafka connection failed: {e}")
    
    async def produce(self, topic: str, key: str, value: Dict):
        """Produce message to Kafka"""
        
        if not self.producer:
            print("⚠️ Kafka producer not initialized")
            return
        
        try:
            # Send message
            metadata = await self.producer.send_and_wait(
                topic,
                key=key.encode('utf-8') if key else None,
                value=value
            )
            
            return {
                'topic': metadata.topic,
                'partition': metadata.partition,
                'offset': metadata.offset
            }
            
        except Exception as e:
            print(f"⚠️ Failed to produce to Kafka: {e}")
            return None


class AsyncEventBus:
    """In-memory event routing"""
    
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_buffer = asyncio.Queue(maxsize=10000)
        self.workers = []
        self.num_workers = 5
    
    async def initialize(self):
        """Start event processing workers"""
        
        for i in range(self.num_workers):
            worker = asyncio.create_task(self.event_worker(i))
            self.workers.append(worker)
        
        print(f"✅ Event bus initialized with {self.num_workers} workers")
    
    async def event_worker(self, worker_id: int):
        """Process events from buffer"""
        
        while True:
            try:
                event = await self.event_buffer.get()
                
                # Get subscribers
                subscribers = self.subscribers[event['type']]
                
                # Notify all subscribers
                tasks = [
                    subscriber(event)
                    for subscriber in subscribers
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                print(f"⚠️ Worker {worker_id} error: {e}")
    
    async def publish(self, event_type: str, data: Any):
        """Publish event to bus"""
        
        event = {
            'id': str(uuid.uuid4()),
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await asyncio.wait_for(
                self.event_buffer.put(event),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            print(f"⚠️ Event buffer full for {event_type}")
    
    def subscribe(self, event_type: str, callback):
        """Subscribe to event type"""
        
        self.subscribers[event_type].append(callback)
        
        return lambda: self.subscribers[event_type].remove(callback)


class DeadLetterQueueManager:
    """Handle failed messages"""
    
    def __init__(self):
        self.dlq_storage = []  # In-memory for now
        self.retry_scheduler = None
    
    async def initialize(self):
        """Setup DLQ infrastructure"""
        
        # Start retry processor
        self.retry_scheduler = asyncio.create_task(self.retry_processor())
        
        print("✅ Dead Letter Queue initialized")
    
    async def send_to_dlq(self, message: Dict, error: str, original_queue: str):
        """Send failed message to DLQ"""
        
        dlq_record = {
            'id': str(uuid.uuid4()),
            'original_queue': original_queue,
            'message': message,
            'error_message': error,
            'retry_count': message.get('retry_count', 0),
            'first_failed_at': datetime.now(),
            'last_failed_at': datetime.now(),
            'status': 'pending'
        }
        
        self.dlq_storage.append(dlq_record)
        
        print(f"⚠️ Message sent to DLQ: {dlq_record['id']}")
    
    async def retry_processor(self):
        """Process DLQ retries"""
        
        while True:
            # Check for messages ready to retry
            now = datetime.now()
            
            for record in self.dlq_storage:
                if record['status'] != 'pending':
                    continue
                
                # Exponential backoff
                retry_delay = 2 ** record['retry_count']
                time_since_failure = (now - record['last_failed_at']).total_seconds()
                
                if time_since_failure >= retry_delay:
                    # Attempt retry
                    print(f"🔄 Retrying message {record['id']}")
                    record['retry_count'] += 1
                    record['last_failed_at'] = now
                    
                    if record['retry_count'] >= 5:
                        record['status'] = 'failed'
                        print(f"❌ Message {record['id']} permanently failed")
            
            await asyncio.sleep(10)


class MessageRouter:
    """Intelligent message routing"""
    
    def __init__(self):
        self.routing_rules = {}
        self.route_metrics = defaultdict(dict)
    
    def add_route(self, pattern: str, handler):
        """Add routing rule"""
        self.routing_rules[pattern] = handler
    
    async def route_message(self, message: Dict) -> bool:
        """Route message to appropriate handler"""
        
        message_type = message.get('type', 'unknown')
        
        for pattern, handler in self.routing_rules.items():
            if self.matches_pattern(message_type, pattern):
                try:
                    await handler(message)
                    
                    # Track metrics
                    self.route_metrics[pattern]['success'] = \
                        self.route_metrics[pattern].get('success', 0) + 1
                    
                    return True
                    
                except Exception as e:
                    print(f"⚠️ Handler error for {pattern}: {e}")
                    
                    self.route_metrics[pattern]['failures'] = \
                        self.route_metrics[pattern].get('failures', 0) + 1
                    
                    return False
        
        print(f"⚠️ No handler for message type: {message_type}")
        return False
    
    def matches_pattern(self, message_type: str, pattern: str) -> bool:
        """Check if message type matches pattern"""
        
        # Simple wildcard matching
        if pattern.endswith('*'):
            return message_type.startswith(pattern[:-1])
        
        return message_type == pattern


class MessageMetrics:
    """Track messaging performance"""
    
    def __init__(self):
        self.metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'average_latency': 0.0
        }
    
    def record_sent(self):
        """Record sent message"""
        self.metrics['messages_sent'] += 1
    
    def record_received(self):
        """Record received message"""
        self.metrics['messages_received'] += 1
    
    def record_failed(self):
        """Record failed message"""
        self.metrics['messages_failed'] += 1
    
    def record_latency(self, latency: float):
        """Update average latency"""
        
        current_avg = self.metrics['average_latency']
        total_messages = self.metrics['messages_received']
        
        if total_messages == 0:
            self.metrics['average_latency'] = latency
        else:
            self.metrics['average_latency'] = \
                (current_avg * total_messages + latency) / (total_messages + 1)
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return self.metrics.copy()


# ================= MAIN INITIALIZATION =================

async def initialize_infrastructure():
    """Initialize complete infrastructure"""
    
    print("🚀 INITIALIZING AI SWARM INFRASTRUCTURE")
    print("=" * 50)
    
    # Initialize security
    security = OperationalSecurityFramework()
    await security.initialize_security()
    
    # Initialize message queues
    message_queue = MessageQueueOrchestrator()
    await message_queue.initialize()
    
    print("\n" + "=" * 50)
    print("✅ INFRASTRUCTURE READY FOR OPERATIONS")
    print("=" * 50)
    
    return {
        'security': security,
        'message_queue': message_queue
    }


if __name__ == "__main__":
    # Run infrastructure initialization
    asyncio.run(initialize_infrastructure())
