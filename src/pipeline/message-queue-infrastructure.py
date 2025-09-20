import os
#!/usr/bin/env python3
"""
Enterprise Message Queue Infrastructure
RabbitMQ cluster, Kafka streaming, dead letter queues, event orchestration
"""

import asyncio
import json
import pickle
import msgpack
import lz4.frame
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import aio_pika
from aio_pika import ExchangeType, DeliveryMode
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import redis.asyncio as redis
from collections import defaultdict, deque
import hashlib
import uuid
import logging
from enum import Enum

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 10
    HIGH = 7
    NORMAL = 5
    LOW = 3
    BULK = 1

@dataclass
class Message:
    """Universal message format"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    payload: Any = None
    headers: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None  # Time to live in seconds
    retry_count: int = 0
    max_retries: int = 3
    source: Optional[str] = None
    destination: Optional[str] = None
    routing_key: Optional[str] = None

@dataclass
class EventSubscription:
    """Event subscription definition"""
    event_pattern: str
    handler: Callable
    filter_func: Optional[Callable] = None
    priority: int = 5
    async_handler: bool = True

class RabbitMQCluster:
    """High-availability RabbitMQ cluster management"""
    
    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes
        self.connections = {}
        self.channels = {}
        self.exchanges = {}
        self.queues = {}
        self.current_node = 0
        
        # Exchange definitions
        self.exchange_configs = {
            'agent.direct': {
                'type': ExchangeType.DIRECT,
                'durable': True,
                'auto_delete': False
            },
            'agent.topic': {
                'type': ExchangeType.TOPIC,
                'durable': True,
                'auto_delete': False
            },
            'agent.fanout': {
                'type': ExchangeType.FANOUT,
                'durable': True,
                'auto_delete': False
            },
            'agent.headers': {
                'type': ExchangeType.HEADERS,
                'durable': True,
                'auto_delete': False
            },
            'dlq.exchange': {
                'type': ExchangeType.DIRECT,
                'durable': True,
                'auto_delete': False
            }
        }
        
        # Queue definitions
        self.queue_configs = {
            'research.tasks': {
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 86400000,  # 24 hours
                    'x-max-length': 10000,
                    'x-dead-letter-exchange': 'dlq.exchange',
                    'x-dead-letter-routing-key': 'research.dlq'
                }
            },
            'osint.requests': {
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 3600000,  # 1 hour
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'dlq.exchange',
                    'x-dead-letter-routing-key': 'osint.dlq'
                }
            },
            'memory.operations': {
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-message-ttl': 7200000,  # 2 hours
                    'x-dead-letter-exchange': 'dlq.exchange',
                    'x-dead-letter-routing-key': 'memory.dlq'
                }
            },
            'agent.commands': {
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-max-priority': 10,
                    'x-dead-letter-exchange': 'dlq.exchange',
                    'x-dead-letter-routing-key': 'agent.dlq'
                }
            },
            'security.alerts': {
                'durable': True,
                'exclusive': False,
                'auto_delete': False,
                'arguments': {
                    'x-max-priority': 10,
                    'x-message-ttl': 86400000
                }
            }
        }
        
        self.dlq_configs = {
            'research.dlq': {'durable': True, 'arguments': {'x-message-ttl': 604800000}},  # 7 days
            'osint.dlq': {'durable': True, 'arguments': {'x-message-ttl': 259200000}},  # 3 days
            'memory.dlq': {'durable': True, 'arguments': {'x-message-ttl': 172800000}},  # 2 days
            'agent.dlq': {'durable': True, 'arguments': {'x-message-ttl': 86400000}},  # 1 day
        }
    
    async def setup_cluster(self):
        """Initialize RabbitMQ cluster connections"""
        for node in self.nodes:
            try:
                connection = await aio_pika.connect_robust(
                    host=node['host'],
                    port=node.get('port', 5672),
                    login=node.get('user', 'guest'),
                    password=node.get('password', 'guest'),
                    virtualhost=node.get('vhost', '/'),
                    connection_attempts=3,
                    retry_delay=5.0
                )
                
                self.connections[node['host']] = connection
                self.channels[node['host']] = await connection.channel()
                
                # Set channel QoS
                await self.channels[node['host']].set_qos(prefetch_count=100)
                
                print(f"✅ Connected to RabbitMQ node: {node['host']}")
                
            except Exception as e:
                print(f"❌ Failed to connect to {node['host']}: {e}")
    
    async def create_exchanges(self):
        """Create all exchanges on all nodes"""
        for host, channel in self.channels.items():
            for exchange_name, config in self.exchange_configs.items():
                exchange = await channel.declare_exchange(
                    name=exchange_name,
                    type=config['type'],
                    durable=config['durable'],
                    auto_delete=config['auto_delete']
                )
                
                if host not in self.exchanges:
                    self.exchanges[host] = {}
                self.exchanges[host][exchange_name] = exchange
                
                print(f"📨 Created exchange '{exchange_name}' on {host}")
    
    async def create_queues(self):
        """Create all queues with bindings"""
        for host, channel in self.channels.items():
            # Create main queues
            for queue_name, config in self.queue_configs.items():
                queue = await channel.declare_queue(
                    name=queue_name,
                    durable=config['durable'],
                    exclusive=config.get('exclusive', False),
                    auto_delete=config.get('auto_delete', False),
                    arguments=config.get('arguments', {})
                )
                
                if host not in self.queues:
                    self.queues[host] = {}
                self.queues[host][queue_name] = queue
                
                # Bind to appropriate exchanges
                if 'research' in queue_name:
                    await queue.bind(self.exchanges[host]['agent.topic'], 'research.*')
                elif 'osint' in queue_name:
                    await queue.bind(self.exchanges[host]['agent.topic'], 'osint.*')
                elif 'memory' in queue_name:
                    await queue.bind(self.exchanges[host]['agent.topic'], 'memory.*')
                elif 'agent' in queue_name:
                    await queue.bind(self.exchanges[host]['agent.direct'], 'agent.command')
                elif 'security' in queue_name:
                    await queue.bind(self.exchanges[host]['agent.fanout'])
                
                print(f"📥 Created queue '{queue_name}' on {host}")
            
            # Create dead letter queues
            for dlq_name, config in self.dlq_configs.items():
                dlq = await channel.declare_queue(
                    name=dlq_name,
                    durable=config['durable'],
                    arguments=config.get('arguments', {})
                )
                
                # Bind to DLQ exchange
                await dlq.bind(self.exchanges[host]['dlq.exchange'], dlq_name)
                
                self.queues[host][dlq_name] = dlq
                print(f"☠️ Created DLQ '{dlq_name}' on {host}")
    
    async def publish(self, message: Message, exchange: str = 'agent.topic'):
        """Publish message with automatic failover"""
        # Round-robin node selection
        host = self.nodes[self.current_node]['host']
        self.current_node = (self.current_node + 1) % len(self.nodes)
        
        try:
            channel = self.channels[host]
            exchange_obj = self.exchanges[host][exchange]
            
            # Prepare message
            aio_message = aio_pika.Message(
                body=msgpack.packb(message.payload),
                headers=message.headers,
                content_type='application/msgpack',
                content_encoding='lz4' if len(str(message.payload)) > 1000 else None,
                priority=message.priority.value,
                correlation_id=message.correlation_id,
                reply_to=message.source,
                expiration=str(message.ttl * 1000) if message.ttl else None,
                message_id=message.id,
                timestamp=message.timestamp,
                delivery_mode=DeliveryMode.PERSISTENT
            )
            
            # Compress if needed
            if len(aio_message.body) > 1000:
                aio_message.body = lz4.frame.compress(aio_message.body)
            
            # Publish
            await exchange_obj.publish(
                aio_message,
                routing_key=message.routing_key or ''
            )
            
            return True
            
        except Exception as e:
            print(f"❌ RabbitMQ publish failed on {host}: {e}")
            
            # Try failover to next node
            if len(self.connections) > 1:
                return await self.publish(message, exchange)
            
            return False
    
    async def consume(self, queue_name: str, handler: Callable, 
                      auto_ack: bool = False):
        """Consume messages from queue"""
        # Select least loaded node
        host = self._select_best_node()
        queue = self.queues[host][queue_name]
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process(ignore_processed=True):
                try:
                    # Decompress if needed
                    body = message.body
                    if message.content_encoding == 'lz4':
                        body = lz4.frame.decompress(body)
                    
                    # Deserialize
                    payload = msgpack.unpackb(body, raw=False)
                    
                    # Create Message object
                    msg = Message(
                        id=message.message_id,
                        correlation_id=message.correlation_id,
                        payload=payload,
                        headers=dict(message.headers) if message.headers else {},
                        priority=MessagePriority(message.priority or 5),
                        timestamp=message.timestamp,
                        source=message.reply_to,
                        routing_key=message.routing_key
                    )
                    
                    # Process
                    result = await handler(msg)
                    
                    if not auto_ack and result:
                        await message.ack()
                    elif not auto_ack and not result:
                        # Retry logic
                        msg.retry_count += 1
                        if msg.retry_count < msg.max_retries:
                            # Requeue with delay
                            await asyncio.sleep(msg.retry_count * 5)
                            await message.reject(requeue=True)
                        else:
                            # Send to DLQ
                            await message.reject(requeue=False)
                            
                except Exception as e:
                    print(f"❌ Message processing error: {e}")
                    await message.reject(requeue=False)
        
        # Start consuming
        await queue.consume(process_message, no_ack=auto_ack)
        print(f"👂 Consuming from '{queue_name}' on {host}")
    
    def _select_best_node(self) -> str:
        """Select node with best performance"""
        # Simple round-robin for now
        hosts = list(self.channels.keys())
        return hosts[self.current_node % len(hosts)]

class KafkaCluster:
    """Apache Kafka cluster for event streaming"""
    
    def __init__(self, brokers: List[str]):
        self.brokers = brokers
        self.producer = None
        self.consumers = {}
        
        # Topic configurations
        self.topics = {
            'agent.events': {
                'num_partitions': 10,
                'replication_factor': 3,
                'retention_ms': 604800000,  # 7 days
                'compression_type': 'lz4'
            },
            'research.stream': {
                'num_partitions': 5,
                'replication_factor': 2,
                'retention_ms': 259200000,  # 3 days
                'compression_type': 'snappy'
            },
            'osint.firehose': {
                'num_partitions': 20,
                'replication_factor': 3,
                'retention_ms': 86400000,  # 1 day
                'compression_type': 'lz4'
            },
            'metrics.timeseries': {
                'num_partitions': 3,
                'replication_factor': 2,
                'retention_ms': 2592000000,  # 30 days
                'compression_type': 'gzip'
            },
            'audit.log': {
                'num_partitions': 1,
                'replication_factor': 3,
                'retention_ms': 31536000000,  # 1 year
                'compression_type': 'gzip'
            }
        }
    
    async def setup_cluster(self):
        """Initialize Kafka connections"""
        # Create producer
        self.producer = AIOKafkaProducer(
            bootstrap_servers=','.join(self.brokers),
            compression_type='lz4',
            max_batch_size=65536,
            linger_ms=100,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=5,
            value_serializer=lambda v: msgpack.packb(v),
            key_serializer=lambda k: k.encode('utf-8') if k else None
        )
        
        await self.producer.start()
        print(f"✅ Kafka producer connected to {self.brokers}")
    
    async def create_topics(self):
        """Create Kafka topics (requires admin client)"""
        # In production, use kafka-python AdminClient
        # This is placeholder for topic creation
        for topic_name, config in self.topics.items():
            print(f"📊 Topic '{topic_name}' configured: {config['num_partitions']} partitions")
    
    async def produce(self, topic: str, key: str, value: Any, 
                     partition: Optional[int] = None):
        """Produce message to Kafka topic"""
        try:
            # Send to Kafka
            record_metadata = await self.producer.send_and_wait(
                topic=topic,
                key=key,
                value=value,
                partition=partition
            )
            
            return {
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset
            }
            
        except KafkaError as e:
            print(f"❌ Kafka produce error: {e}")
            return None
    
    async def consume(self, topics: Union[str, List[str]], 
                     group_id: str, handler: Callable):
        """Consume from Kafka topics"""
        if isinstance(topics, str):
            topics = [topics]
        
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=','.join(self.brokers),
            group_id=group_id,
            value_deserializer=lambda v: msgpack.unpackb(v, raw=False),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            enable_auto_commit=False,  # Manual commit for reliability
            auto_offset_reset='earliest',
            max_poll_records=100
        )
        
        self.consumers[group_id] = consumer
        await consumer.start()
        
        print(f"👂 Kafka consumer '{group_id}' started for topics: {topics}")
        
        try:
            async for msg in consumer:
                # Process message
                result = await handler({
                    'topic': msg.topic,
                    'partition': msg.partition,
                    'offset': msg.offset,
                    'key': msg.key,
                    'value': msg.value,
                    'timestamp': msg.timestamp
                })
                
                # Commit if successful
                if result:
                    await consumer.commit()
                    
        except Exception as e:
            print(f"❌ Kafka consumer error: {e}")
        finally:
            await consumer.stop()
    
    async def stream_processor(self, source_topic: str, 
                              sink_topic: str, 
                              transform_func: Callable):
        """Kafka Streams-like processing"""
        group_id = f"processor_{source_topic}_to_{sink_topic}"
        
        async def process_and_forward(msg):
            try:
                # Transform message
                transformed = await transform_func(msg['value'])
                
                # Forward to sink topic
                await self.produce(
                    topic=sink_topic,
                    key=msg['key'],
                    value=transformed
                )
                
                return True
                
            except Exception as e:
                print(f"❌ Stream processing error: {e}")
                return False
        
        # Start consuming and processing
        await self.consume(source_topic, group_id, process_and_forward)

class DeadLetterQueueManager:
    """Manage dead letter queues and message recovery"""
    
    def __init__(self, rabbitmq: RabbitMQCluster):
        self.rabbitmq = rabbitmq
        self.retry_policies = {}
        self.recovery_handlers = {}
        
    async def initialize(self):
        """Setup DLQ monitoring"""
        # Monitor all DLQs
        for dlq_name in self.rabbitmq.dlq_configs.keys():
            asyncio.create_task(self._monitor_dlq(dlq_name))
    
    async def _monitor_dlq(self, dlq_name: str):
        """Monitor DLQ for recovery attempts"""
        while True:
            try:
                # Check DLQ size periodically
                await asyncio.sleep(60)  # Check every minute
                
                # Process messages in DLQ
                await self.process_dlq(dlq_name)
                
            except Exception as e:
                print(f"❌ DLQ monitor error for {dlq_name}: {e}")
                await asyncio.sleep(5)
    
    async def process_dlq(self, dlq_name: str):
        """Process messages in dead letter queue"""
        host = list(self.rabbitmq.queues.keys())[0]
        
        if dlq_name not in self.rabbitmq.queues[host]:
            return
        
        queue = self.rabbitmq.queues[host][dlq_name]
        
        # Get message count
        message_count = queue.declaration_result.message_count if hasattr(queue, 'declaration_result') else 0
        
        if message_count > 0:
            print(f"⚠️ Processing {message_count} messages in DLQ: {dlq_name}")
            
            # Process messages
            async def handle_dlq_message(message: aio_pika.IncomingMessage):
                async with message.process():
                    try:
                        # Check if we have a recovery handler
                        queue_prefix = dlq_name.replace('.dlq', '')
                        
                        if queue_prefix in self.recovery_handlers:
                            # Attempt recovery
                            body = message.body
                            if message.content_encoding == 'lz4':
                                body = lz4.frame.decompress(body)
                            
                            payload = msgpack.unpackb(body, raw=False)
                            
                            recovered = await self.recovery_handlers[queue_prefix](payload)
                            
                            if recovered:
                                await message.ack()
                                print(f"✅ Recovered message from {dlq_name}")
                            else:
                                # Keep in DLQ
                                await message.reject(requeue=True)
                        else:
                            # No handler, log and acknowledge
                            print(f"⚠️ No recovery handler for {queue_prefix}")
                            await message.ack()
                            
                    except Exception as e:
                        print(f"❌ DLQ processing error: {e}")
                        await message.reject(requeue=True)
            
            # Consume a batch of messages
            await queue.consume(handle_dlq_message, no_ack=False)
    
    def register_recovery_handler(self, queue_prefix: str, handler: Callable):
        """Register handler for DLQ recovery"""
        self.recovery_handlers[queue_prefix] = handler
        print(f"🔧 Registered recovery handler for {queue_prefix}")

class AsyncEventBus:
    """Async event bus for loose coupling"""
    
    def __init__(self):
        self.subscriptions = defaultdict(list)
        self.event_history = deque(maxlen=10000)
        self.metrics = defaultdict(int)
        
    async def initialize(self):
        """Initialize event bus"""
        print("🚌 Event bus initialized")
    
    def subscribe(self, event_pattern: str, handler: Callable, 
                 priority: int = 5):
        """Subscribe to events"""
        subscription = EventSubscription(
            event_pattern=event_pattern,
            handler=handler,
            priority=priority
        )
        
        self.subscriptions[event_pattern].append(subscription)
        
        # Sort by priority
        self.subscriptions[event_pattern].sort(
            key=lambda x: x.priority, 
            reverse=True
        )
        
        print(f"📌 Subscribed to '{event_pattern}'")
    
    async def publish(self, event_type: str, data: Any):
        """Publish event to all subscribers"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now(),
            'id': str(uuid.uuid4())
        }
        
        # Add to history
        self.event_history.append(event)
        
        # Update metrics
        self.metrics[event_type] += 1
        
        # Find matching subscriptions
        handlers = []
        for pattern, subs in self.subscriptions.items():
            if self._matches_pattern(event_type, pattern):
                handlers.extend(subs)
        
        # Sort by priority
        handlers.sort(key=lambda x: x.priority, reverse=True)
        
        # Execute handlers
        for subscription in handlers:
            try:
                if subscription.async_handler:
                    asyncio.create_task(subscription.handler(event))
                else:
                    subscription.handler(event)
                    
            except Exception as e:
                print(f"❌ Event handler error: {e}")
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event matches pattern (supports wildcards)"""
        if pattern == '*':
            return True
        
        if '*' in pattern:
            # Convert pattern to regex
            import re
            regex_pattern = pattern.replace('.', r'\.').replace('*', '.*')
            return bool(re.match(f"^{regex_pattern}$", event_type))
        
        return event_type == pattern

class MessageRouter:
    """Intelligent message routing"""
    
    def __init__(self, rabbitmq: RabbitMQCluster, kafka: KafkaCluster):
        self.rabbitmq = rabbitmq
        self.kafka = kafka
        
        # Routing rules
        self.routes = {
            'research.*': {'broker': 'rabbitmq', 'exchange': 'agent.topic'},
            'osint.*': {'broker': 'rabbitmq', 'exchange': 'agent.topic'},
            'memory.*': {'broker': 'rabbitmq', 'exchange': 'agent.topic'},
            'agent.command': {'broker': 'rabbitmq', 'exchange': 'agent.direct'},
            'metrics.*': {'broker': 'kafka', 'topic': 'metrics.timeseries'},
            'audit.*': {'broker': 'kafka', 'topic': 'audit.log'},
            'stream.*': {'broker': 'kafka', 'topic': 'research.stream'}
        }
    
    async def route(self, message: Message):
        """Route message based on rules"""
        routing_key = message.routing_key or message.destination or ''
        
        # Find matching route
        for pattern, config in self.routes.items():
            if self._matches_pattern(routing_key, pattern):
                if config['broker'] == 'rabbitmq':
                    return await self.rabbitmq.publish(
                        message, 
                        config['exchange']
                    )
                elif config['broker'] == 'kafka':
                    return await self.kafka.produce(
                        config['topic'],
                        message.id,
                        message.payload
                    )
        
        # Default route
        return await self.rabbitmq.publish(message, 'agent.topic')
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Pattern matching for routing keys"""
        import re
        regex = pattern.replace('.', r'\.').replace('*', '[^.]*')
        return bool(re.match(f"^{regex}$", key))

class MessageQueueOrchestrator:
    """Main orchestrator for all messaging"""
    
    def __init__(self, config: Dict[str, Any]):
        # RabbitMQ cluster
        self.rabbitmq = RabbitMQCluster(config['rabbitmq_nodes'])
        
        # Kafka cluster  
        self.kafka = KafkaCluster(config['kafka_brokers'])
        
        # Event bus
        self.event_bus = AsyncEventBus()
        
        # Dead letter queue manager
        self.dlq_manager = DeadLetterQueueManager(self.rabbitmq)
        
        # Message router
        self.router = MessageRouter(self.rabbitmq, self.kafka)
        
        # Metrics
        self.metrics = defaultdict(int)
        
    async def initialize(self):
        """Initialize all messaging infrastructure"""
        print("🚀 Initializing message queue infrastructure...")
        
        # Setup RabbitMQ
        await self.rabbitmq.setup_cluster()
        await self.rabbitmq.create_exchanges()
        await self.rabbitmq.create_queues()
        
        # Setup Kafka
        await self.kafka.setup_cluster()
        await self.kafka.create_topics()
        
        # Initialize event bus
        await self.event_bus.initialize()
        
        # Setup DLQ monitoring
        await self.dlq_manager.initialize()
        
        print("✅ Message queue infrastructure ready!")
    
    async def send_message(self, message: Message):
        """Send message through appropriate channel"""
        self.metrics['messages_sent'] += 1
        
        # Route message
        success = await self.router.route(message)
        
        if not success:
            self.metrics['send_failures'] += 1
            
            # Publish failure event
            await self.event_bus.publish('message.failed', {
                'message_id': message.id,
                'reason': 'routing_failed'
            })
        
        return success
    
    async def process_research_tasks(self, handler: Callable):
        """Process research task queue"""
        await self.rabbitmq.consume('research.tasks', handler)
    
    async def process_osint_requests(self, handler: Callable):
        """Process OSINT request queue"""
        await self.rabbitmq.consume('osint.requests', handler)
    
    async def stream_events(self, handler: Callable):
        """Stream events from Kafka"""
        await self.kafka.consume(
            ['agent.events', 'research.stream'],
            'event_processor',
            handler
        )
    
    async def publish_metrics(self, metrics: Dict[str, Any]):
        """Publish metrics to Kafka"""
        await self.kafka.produce(
            'metrics.timeseries',
            str(datetime.now()),
            metrics
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get messaging statistics"""
        return {
            'messages_sent': self.metrics['messages_sent'],
            'send_failures': self.metrics['send_failures'],
            'event_bus_events': len(self.event_bus.event_history),
            'event_types': dict(self.event_bus.metrics)
        }

# Example usage
async def main():
    config = {
        'rabbitmq_nodes': [
            {'host': 'localhost', 'port': 5672, 'user': 'guest', 'password': 'guest'},
            {'host': 'localhost', 'port': 5673, 'user': 'guest', 'password': 'guest'},
            {'host': 'localhost', 'port': 5674, 'user': 'guest', 'password': 'guest'}
        ],
        'kafka_brokers': [
            'localhost:9092',
            'localhost:9093',
            'localhost:9094'
        ]
    }
    
    orchestrator = MessageQueueOrchestrator(config)
    await orchestrator.initialize()
    
    # Example message
    message = Message(
        payload={'task': 'investigate', 'target': 'example.com'},
        routing_key='research.osint',
        priority=MessagePriority.HIGH
    )
    
    await orchestrator.send_message(message)
    
    # Setup handlers
    async def research_handler(msg: Message):
        print(f"Processing research task: {msg.payload}")
        return True
    
    # Start consuming
    await orchestrator.process_research_tasks(research_handler)

if __name__ == "__main__":
    asyncio.run(main())
