import os
#!/usr/bin/env python3
"""
Enterprise Message Queue Manager
Complete RabbitMQ and Kafka integration with DLQ and Saga support
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aio_pika
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import asyncpg
import redis.asyncio as redis
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

@dataclass
class QueueConfig:
    """Message queue configuration"""
    name: str
    durable: bool = True
    exclusive: bool = False
    auto_delete: bool = False
    max_length: int = 100000
    message_ttl: int = 86400000  # 24 hours
    max_priority: int = 10
    dlx_exchange: str = "dlx"
    dlx_routing_key: str = ""

class MessageQueueOrchestrator:
    """Enterprise message queue orchestrator with RabbitMQ and Kafka"""
    
    def __init__(self):
        # RabbitMQ settings
        self.rabbitmq_urls = [
            "amqp://admin:BevSwarm2024!@localhost:5672/",
            "amqp://admin:BevSwarm2024!@localhost:5673/",
            "amqp://admin:BevSwarm2024!@localhost:5674/"
        ]
        self.rabbitmq_connections = []
        self.rabbitmq_channels = []
        
        # Kafka settings
        self.kafka_bootstrap = "localhost:9092,localhost:9093,localhost:9094"
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumers: Dict[str, AIOKafkaConsumer] = {}
        
        # Redis for state
        self.redis_client: Optional[redis.Redis] = None
        
        # PostgreSQL for persistence
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # DLQ handler
        self.dlq_handlers: Dict[str, Callable] = {}
        
        # Saga orchestration
        self.active_sagas: Dict[str, 'Saga'] = {}
    
    async def initialize(self):
        """Initialize all message queue infrastructure"""
        
        # Initialize RabbitMQ cluster
        await self._init_rabbitmq()
        
        # Initialize Kafka cluster
        await self._init_kafka()
        
        # Initialize Redis
        self.redis_client = await redis.from_url("redis://localhost:6379")
        
        # Initialize PostgreSQL
        self.db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='swarm_admin',
            password=os.getenv('DB_PASSWORD', 'dev_password'),
            database='ai_swarm',
            min_size=5,
            max_size=20
        )
        
        # Create DLQ table
        await self._init_dlq_storage()
        
        # Start background workers
        asyncio.create_task(self._dlq_processor())
        asyncio.create_task(self._saga_timeout_handler())
        
        logger.info("Message Queue Orchestrator initialized")
    
    async def _init_rabbitmq(self):
        """Initialize RabbitMQ cluster connections"""
        
        for url in self.rabbitmq_urls:
            try:
                connection = await aio_pika.connect_robust(
                    url,
                    heartbeat=60,
                    connection_attempts=5,
                    retry_delay=2
                )
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=100)
                
                self.rabbitmq_connections.append(connection)
                self.rabbitmq_channels.append(channel)
                
                logger.info(f"Connected to RabbitMQ: {url}")
                
            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ {url}: {e}")
        
        # Setup standard exchanges
        await self._setup_rabbitmq_topology()
    
    async def _setup_rabbitmq_topology(self):
        """Setup RabbitMQ exchanges and queues"""
        
        if not self.rabbitmq_channels:
            return
        
        channel = self.rabbitmq_channels[0]
        
        # Standard exchanges
        exchanges = {
            'agent.direct': aio_pika.ExchangeType.DIRECT,
            'agent.topic': aio_pika.ExchangeType.TOPIC,
            'agent.fanout': aio_pika.ExchangeType.FANOUT,
            'priority.tasks': aio_pika.ExchangeType.DIRECT,
            'dlx': aio_pika.ExchangeType.TOPIC
        }
        
        for exchange_name, exchange_type in exchanges.items():
            exchange = await channel.declare_exchange(
                exchange_name,
                exchange_type,
                durable=True
            )
            logger.info(f"Declared exchange: {exchange_name}")
        
        # Standard queues
        queues = [
            QueueConfig(
                name='research.tasks',
                max_priority=10,
                dlx_exchange='dlx',
                dlx_routing_key='research.dlq'
            ),
            QueueConfig(
                name='code.generation',
                max_priority=5,
                dlx_exchange='dlx',
                dlx_routing_key='code.dlq'
            ),
            QueueConfig(
                name='memory.operations',
                message_ttl=3600000,  # 1 hour
                dlx_exchange='dlx',
                dlx_routing_key='memory.dlq'
            ),
            QueueConfig(
                name='tool.execution',
                max_length=50000,
                dlx_exchange='dlx',
                dlx_routing_key='tool.dlq'
            ),
            QueueConfig(
                name='security.alerts',
                max_priority=10,
                dlx_exchange='dlx',
                dlx_routing_key='security.dlq'
            )
        ]
        
        for queue_config in queues:
            await self._declare_queue(channel, queue_config)
    
    async def _declare_queue(self, channel, config: QueueConfig):
        """Declare a queue with configuration"""
        
        arguments = {
            'x-message-ttl': config.message_ttl,
            'x-max-length': config.max_length,
            'x-overflow': 'reject-publish',
            'x-dead-letter-exchange': config.dlx_exchange,
            'x-dead-letter-routing-key': config.dlx_routing_key
        }
        
        if config.max_priority:
            arguments['x-max-priority'] = config.max_priority
        
        queue = await channel.declare_queue(
            config.name,
            durable=config.durable,
            exclusive=config.exclusive,
            auto_delete=config.auto_delete,
            arguments=arguments
        )
        
        logger.info(f"Declared queue: {config.name}")
        return queue
    
    async def _init_kafka(self):
        """Initialize Kafka producer and setup topics"""
        
        # Create producer
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode() if k else None,
            acks='all',
            compression_type='snappy',
            max_batch_size=32768,
            linger_ms=10
        )
        await self.kafka_producer.start()
        
        logger.info("Kafka producer initialized")
        
        # Create standard topics (handled by auto-creation in docker-compose)
        topics = [
            'agent.events',
            'agent.metrics',
            'research.results',
            'security.incidents',
            'dlq.messages'
        ]
        
        # Topics are auto-created with proper replication
    
    async def _init_dlq_storage(self):
        """Initialize dead letter queue storage"""
        
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS dead_letter_queue (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_queue VARCHAR(255),
                    message_id VARCHAR(255),
                    correlation_id VARCHAR(255),
                    payload JSONB,
                    error_message TEXT,
                    error_count INTEGER DEFAULT 1,
                    first_error_at TIMESTAMPTZ DEFAULT NOW(),
                    last_error_at TIMESTAMPTZ DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'pending',
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_dlq_status ON dead_letter_queue(status);
                CREATE INDEX IF NOT EXISTS idx_dlq_source ON dead_letter_queue(source_queue);
                CREATE INDEX IF NOT EXISTS idx_dlq_error_at ON dead_letter_queue(last_error_at);
            ''')
    
    async def publish_to_rabbitmq(self, exchange: str, routing_key: str, 
                                  message: Dict[str, Any], priority: int = 5,
                                  correlation_id: str = None):
        """Publish message to RabbitMQ with failover"""
        
        message_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Round-robin channel selection
        channel_index = hash(routing_key) % len(self.rabbitmq_channels)
        channel = self.rabbitmq_channels[channel_index]
        
        try:
            exchange_obj = await channel.get_exchange(exchange)
            
            message_body = aio_pika.Message(
                body=json.dumps(message).encode(),
                message_id=message_id,
                correlation_id=correlation_id,
                priority=priority,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                timestamp=datetime.utcnow(),
                headers={
                    'x-retry-count': 0,
                    'x-original-timestamp': datetime.utcnow().isoformat()
                }
            )
            
            await exchange_obj.publish(
                message_body,
                routing_key=routing_key
            )
            
            logger.info(f"Published to RabbitMQ: {exchange}/{routing_key}", 
                       message_id=message_id)
            
            return message_id
            
        except Exception as e:
            logger.error(f"RabbitMQ publish failed: {e}")
            
            # Send to DLQ
            await self._send_to_dlq(
                source_queue=f"{exchange}/{routing_key}",
                message_id=message_id,
                correlation_id=correlation_id,
                payload=message,
                error_message=str(e)
            )
            
            raise
    
    async def publish_to_kafka(self, topic: str, key: str, value: Dict[str, Any],
                              headers: Dict[str, str] = None):
        """Publish message to Kafka"""
        
        try:
            record_metadata = await self.kafka_producer.send_and_wait(
                topic,
                value=value,
                key=key,
                headers=[(k, v.encode()) for k, v in (headers or {}).items()]
            )
            
            logger.info(f"Published to Kafka: {topic}", 
                       partition=record_metadata.partition,
                       offset=record_metadata.offset)
            
            return {
                'topic': record_metadata.topic,
                'partition': record_metadata.partition,
                'offset': record_metadata.offset
            }
            
        except KafkaError as e:
            logger.error(f"Kafka publish failed: {e}")
            
            # Send to DLQ
            await self._send_to_dlq(
                source_queue=f"kafka/{topic}",
                message_id=str(uuid.uuid4()),
                correlation_id=None,
                payload=value,
                error_message=str(e)
            )
            
            raise
    
    async def consume_from_rabbitmq(self, queue_name: str, 
                                   handler: Callable[[Dict], Any],
                                   auto_ack: bool = False):
        """Consume messages from RabbitMQ queue"""
        
        channel = self.rabbitmq_channels[0]
        queue = await channel.get_queue(queue_name)
        
        async def process_message(message: aio_pika.IncomingMessage):
            async with message.process(requeue=not auto_ack):
                try:
                    body = json.loads(message.body.decode())
                    
                    # Process message
                    result = await handler(body)
                    
                    # Store result if correlation ID present
                    if message.correlation_id:
                        await self.redis_client.setex(
                            f"result:{message.correlation_id}",
                            300,  # 5 minutes TTL
                            json.dumps(result)
                        )
                    
                    logger.info(f"Processed message from {queue_name}", 
                               message_id=message.message_id)
                    
                except Exception as e:
                    logger.error(f"Message processing failed: {e}")
                    
                    # Check retry count
                    retry_count = message.headers.get('x-retry-count', 0)
                    
                    if retry_count < 3:
                        # Retry with exponential backoff
                        await asyncio.sleep(2 ** retry_count)
                        
                        # Republish with incremented retry count
                        new_message = aio_pika.Message(
                            body=message.body,
                            headers={**message.headers, 'x-retry-count': retry_count + 1}
                        )
                        
                        await channel.default_exchange.publish(
                            new_message,
                            routing_key=queue_name
                        )
                    else:
                        # Max retries reached - send to DLQ
                        await self._send_to_dlq(
                            source_queue=queue_name,
                            message_id=str(message.message_id),
                            correlation_id=str(message.correlation_id),
                            payload=json.loads(message.body.decode()),
                            error_message=str(e)
                        )
        
        await queue.consume(process_message)
        logger.info(f"Started consuming from {queue_name}")
    
    async def consume_from_kafka(self, topics: List[str], group_id: str,
                                handler: Callable[[Dict], Any]):
        """Consume messages from Kafka topics"""
        
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=self.kafka_bootstrap,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset='earliest',
            enable_auto_commit=False,
            max_poll_records=100
        )
        
        await consumer.start()
        self.kafka_consumers[group_id] = consumer
        
        try:
            async for message in consumer:
                try:
                    # Process message
                    result = await handler(message.value)
                    
                    # Commit offset
                    await consumer.commit()
                    
                    logger.info(f"Processed Kafka message", 
                               topic=message.topic,
                               partition=message.partition,
                               offset=message.offset)
                    
                except Exception as e:
                    logger.error(f"Kafka message processing failed: {e}")
                    
                    # Send to DLQ
                    await self._send_to_dlq(
                        source_queue=f"kafka/{message.topic}",
                        message_id=f"{message.partition}-{message.offset}",
                        correlation_id=None,
                        payload=message.value,
                        error_message=str(e)
                    )
        
        finally:
            await consumer.stop()
    
    async def _send_to_dlq(self, source_queue: str, message_id: str,
                          correlation_id: str, payload: Dict[str, Any],
                          error_message: str):
        """Send failed message to dead letter queue"""
        
        # Store in database
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO dead_letter_queue 
                (source_queue, message_id, correlation_id, payload, error_message)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (message_id) DO UPDATE
                SET error_count = dead_letter_queue.error_count + 1,
                    last_error_at = NOW(),
                    error_message = $5
            ''', source_queue, message_id, correlation_id, 
                json.dumps(payload), error_message)
        
        # Publish to Kafka DLQ topic
        await self.kafka_producer.send_and_wait(
            'dlq.messages',
            value={
                'source_queue': source_queue,
                'message_id': message_id,
                'correlation_id': correlation_id,
                'payload': payload,
                'error': error_message,
                'timestamp': datetime.utcnow().isoformat()
            }
        )
        
        logger.warning(f"Message sent to DLQ", 
                      source=source_queue, 
                      message_id=message_id)
    
    async def _dlq_processor(self):
        """Process dead letter queue messages"""
        
        while True:
            try:
                async with self.db_pool.acquire() as conn:
                    # Get pending DLQ messages
                    messages = await conn.fetch('''
                        SELECT * FROM dead_letter_queue
                        WHERE status = 'pending'
                        AND error_count < 5
                        AND last_error_at < NOW() - INTERVAL '1 minute' * POW(2, error_count)
                        LIMIT 10
                    ''')
                    
                    for msg in messages:
                        source = msg['source_queue']
                        
                        # Check if we have a handler
                        if source in self.dlq_handlers:
                            try:
                                # Process with custom handler
                                await self.dlq_handlers[source](msg['payload'])
                                
                                # Mark as processed
                                await conn.execute('''
                                    UPDATE dead_letter_queue
                                    SET status = 'processed'
                                    WHERE id = $1
                                ''', msg['id'])
                                
                                logger.info(f"DLQ message processed", 
                                          message_id=msg['message_id'])
                                
                            except Exception as e:
                                logger.error(f"DLQ processing failed: {e}")
                                
                                # Update error count
                                await conn.execute('''
                                    UPDATE dead_letter_queue
                                    SET error_count = error_count + 1,
                                        last_error_at = NOW()
                                    WHERE id = $1
                                ''', msg['id'])
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"DLQ processor error: {e}")
                await asyncio.sleep(30)

@dataclass
class SagaStep:
    """Individual step in a saga"""
    name: str
    action: Callable
    compensator: Callable
    timeout: int = 30

class Saga:
    """Distributed transaction saga implementation"""
    
    def __init__(self, saga_id: str, steps: List[SagaStep]):
        self.saga_id = saga_id
        self.steps = steps
        self.completed_steps: List[SagaStep] = []
        self.status = "pending"
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute saga with automatic compensation on failure"""
        
        self.status = "running"
        results = []
        
        try:
            for step in self.steps:
                logger.info(f"Executing saga step: {step.name}", 
                           saga_id=self.saga_id)
                
                # Execute step with timeout
                try:
                    result = await asyncio.wait_for(
                        step.action(context),
                        timeout=step.timeout
                    )
                    
                    results.append({
                        'step': step.name,
                        'result': result,
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    self.completed_steps.append(step)
                    
                    # Update context for next step
                    context[f"{step.name}_result"] = result
                    
                except asyncio.TimeoutError:
                    raise Exception(f"Step {step.name} timed out")
            
            self.status = "completed"
            self.end_time = datetime.utcnow()
            
            return {
                'saga_id': self.saga_id,
                'status': 'success',
                'results': results,
                'duration': (self.end_time - self.start_time).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Saga failed: {e}", saga_id=self.saga_id)
            
            # Compensate in reverse order
            self.status = "compensating"
            
            for step in reversed(self.completed_steps):
                try:
                    logger.info(f"Compensating step: {step.name}", 
                               saga_id=self.saga_id)
                    
                    await asyncio.wait_for(
                        step.compensator(context),
                        timeout=step.timeout
                    )
                    
                except Exception as comp_error:
                    logger.error(f"Compensation failed for {step.name}: {comp_error}")
            
            self.status = "failed"
            self.end_time = datetime.utcnow()
            
            return {
                'saga_id': self.saga_id,
                'status': 'failed',
                'error': str(e),
                'compensated_steps': [s.name for s in self.completed_steps],
                'duration': (self.end_time - self.start_time).total_seconds()
            }

class SagaOrchestrator:
    """Orchestrate distributed sagas across the system"""
    
    def __init__(self, queue_manager: MessageQueueOrchestrator):
        self.queue_manager = queue_manager
        self.active_sagas: Dict[str, Saga] = {}
    
    async def start_saga(self, saga_type: str, context: Dict[str, Any]) -> str:
        """Start a new saga based on type"""
        
        saga_id = str(uuid.uuid4())
        
        # Define saga steps based on type
        if saga_type == "research_pipeline":
            steps = [
                SagaStep(
                    name="collect_data",
                    action=self._collect_research_data,
                    compensator=self._cleanup_research_data
                ),
                SagaStep(
                    name="analyze_data",
                    action=self._analyze_research_data,
                    compensator=self._cleanup_analysis
                ),
                SagaStep(
                    name="generate_report",
                    action=self._generate_research_report,
                    compensator=self._cleanup_report
                )
            ]
        elif saga_type == "code_generation":
            steps = [
                SagaStep(
                    name="design_architecture",
                    action=self._design_code_architecture,
                    compensator=self._cleanup_architecture
                ),
                SagaStep(
                    name="generate_code",
                    action=self._generate_code,
                    compensator=self._cleanup_code
                ),
                SagaStep(
                    name="test_code",
                    action=self._test_code,
                    compensator=self._cleanup_tests
                )
            ]
        else:
            raise ValueError(f"Unknown saga type: {saga_type}")
        
        saga = Saga(saga_id, steps)
        self.active_sagas[saga_id] = saga
        
        # Execute saga asynchronously
        asyncio.create_task(self._execute_saga(saga, context))
        
        return saga_id
    
    async def _execute_saga(self, saga: Saga, context: Dict[str, Any]):
        """Execute saga and publish results"""
        
        result = await saga.execute(context)
        
        # Publish saga completion event
        await self.queue_manager.publish_to_kafka(
            'agent.events',
            key=f"saga.{saga.saga_id}",
            value={
                'event': 'saga_completed',
                'saga_id': saga.saga_id,
                'result': result
            }
        )
        
        # Remove from active sagas
        del self.active_sagas[saga.saga_id]
    
    # Saga step implementations
    async def _collect_research_data(self, context: Dict[str, Any]) -> Dict:
        """Collect research data step"""
        # Implementation here
        return {'data_collected': True}
    
    async def _cleanup_research_data(self, context: Dict[str, Any]):
        """Cleanup research data on failure"""
        # Implementation here
        pass
    
    async def _analyze_research_data(self, context: Dict[str, Any]) -> Dict:
        """Analyze research data step"""
        # Implementation here
        return {'analysis_complete': True}
    
    async def _cleanup_analysis(self, context: Dict[str, Any]):
        """Cleanup analysis on failure"""
        # Implementation here
        pass
    
    async def _generate_research_report(self, context: Dict[str, Any]) -> Dict:
        """Generate research report step"""
        # Implementation here
        return {'report_generated': True}
    
    async def _cleanup_report(self, context: Dict[str, Any]):
        """Cleanup report on failure"""
        # Implementation here
        pass
    
    async def _design_code_architecture(self, context: Dict[str, Any]) -> Dict:
        """Design code architecture step"""
        # Implementation here
        return {'architecture_designed': True}
    
    async def _cleanup_architecture(self, context: Dict[str, Any]):
        """Cleanup architecture on failure"""
        # Implementation here
        pass
    
    async def _generate_code(self, context: Dict[str, Any]) -> Dict:
        """Generate code step"""
        # Implementation here
        return {'code_generated': True}
    
    async def _cleanup_code(self, context: Dict[str, Any]):
        """Cleanup code on failure"""
        # Implementation here
        pass
    
    async def _test_code(self, context: Dict[str, Any]) -> Dict:
        """Test code step"""
        # Implementation here
        return {'tests_passed': True}
    
    async def _cleanup_tests(self, context: Dict[str, Any]):
        """Cleanup tests on failure"""
        # Implementation here
        pass

# Deployment script
async def deploy_message_queue_infrastructure():
    """Deploy complete message queue infrastructure"""
    
    # Initialize orchestrator
    orchestrator = MessageQueueOrchestrator()
    await orchestrator.initialize()
    
    # Create saga orchestrator
    saga_orchestrator = SagaOrchestrator(orchestrator)
    
    # Register DLQ handlers
    async def handle_research_dlq(message):
        logger.info(f"Handling research DLQ message: {message}")
        # Custom handling logic
    
    orchestrator.dlq_handlers['research.tasks'] = handle_research_dlq
    
    # Start consumers
    async def research_handler(message):
        logger.info(f"Processing research task: {message}")
        return {'processed': True}
    
    await orchestrator.consume_from_rabbitmq(
        'research.tasks',
        research_handler
    )
    
    # Start Kafka consumer
    await orchestrator.consume_from_kafka(
        ['agent.events'],
        'event_processor_group',
        lambda msg: logger.info(f"Event received: {msg}")
    )
    
    logger.info("Message queue infrastructure deployed")
    
    return orchestrator, saga_orchestrator

if __name__ == "__main__":
    asyncio.run(deploy_message_queue_infrastructure())
