#!/usr/bin/env python3
"""
BEV OSINT Framework - Queue Manager
Advanced queue management with RabbitMQ/Kafka integration, priority queuing,
backpressure handling, and intelligent message routing.
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import pickle
import uuid

# Message queue libraries
import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaError
import redis.asyncio as redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueueType(Enum):
    """Queue implementation types"""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    MEMORY = "memory"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class QueueMessage:
    """Universal message container"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    payload: Any = None
    priority: MessagePriority = MessagePriority.MEDIUM
    routing_key: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    delay_until: Optional[datetime] = None
    status: MessageStatus = MessageStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'message_id': self.message_id,
            'payload': self.payload,
            'priority': self.priority.value,
            'routing_key': self.routing_key,
            'headers': self.headers,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'delay_until': self.delay_until.isoformat() if self.delay_until else None,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create message from dictionary"""
        message = cls()
        message.message_id = data.get('message_id', str(uuid.uuid4()))
        message.payload = data.get('payload')
        message.priority = MessagePriority(data.get('priority', MessagePriority.MEDIUM.value))
        message.routing_key = data.get('routing_key', '')
        message.headers = data.get('headers', {})
        message.created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        message.expires_at = datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None
        message.retry_count = data.get('retry_count', 0)
        message.max_retries = data.get('max_retries', 3)
        message.delay_until = datetime.fromisoformat(data['delay_until']) if data.get('delay_until') else None
        message.status = MessageStatus(data.get('status', MessageStatus.PENDING.value))
        return message

    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at and datetime.now() > self.expires_at

    @property
    def is_delayed(self) -> bool:
        """Check if message is still delayed"""
        return self.delay_until and datetime.now() < self.delay_until

    @property
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries


@dataclass
class QueueStats:
    """Queue statistics tracking"""
    messages_enqueued: int = 0
    messages_dequeued: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    messages_dead_lettered: int = 0
    total_processing_time: float = 0.0
    queue_size: int = 0
    active_consumers: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate processing success rate"""
        total = self.messages_processed + self.messages_failed
        return self.messages_processed / total if total > 0 else 0.0

    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time"""
        return self.total_processing_time / self.messages_processed if self.messages_processed > 0 else 0.0


class BaseQueueBackend:
    """Abstract base class for queue backends"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = QueueStats()

    async def initialize(self):
        """Initialize the queue backend"""
        pass

    async def close(self):
        """Close the queue backend"""
        pass

    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue a message"""
        raise NotImplementedError

    async def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue a message"""
        raise NotImplementedError

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        pass

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message"""
        pass

    async def get_queue_size(self, queue_name: str) -> int:
        """Get current queue size"""
        raise NotImplementedError

    async def purge_queue(self, queue_name: str):
        """Purge all messages from queue"""
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return asdict(self.stats)


class RabbitMQBackend(BaseQueueBackend):
    """RabbitMQ queue backend implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.connection_url = config.get('rabbitmq_url', 'amqp://guest:guest@rabbitmq-1:5672/')
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self.queues = {}

    async def initialize(self):
        """Initialize RabbitMQ connection"""
        try:
            self.connection = await aio_pika.connect_robust(self.connection_url)
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)

            # Declare default exchange
            self.exchanges['default'] = await self.channel.declare_exchange(
                'bev_requests',
                ExchangeType.TOPIC,
                durable=True
            )

            logger.info("RabbitMQ backend initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RabbitMQ: {e}")
            raise

    async def close(self):
        """Close RabbitMQ connection"""
        if self.connection:
            await self.connection.close()

    async def _ensure_queue(self, queue_name: str):
        """Ensure queue exists"""
        if queue_name not in self.queues:
            # Create priority queue arguments
            queue_args = {
                'x-max-priority': 5,  # Support 5 priority levels
                'x-message-ttl': 3600000,  # 1 hour TTL
                'x-dead-letter-exchange': 'bev_requests_dlx'
            }

            self.queues[queue_name] = await self.channel.declare_queue(
                queue_name,
                durable=True,
                arguments=queue_args
            )

            # Bind queue to exchange
            await self.queues[queue_name].bind(
                self.exchanges['default'],
                routing_key=f"requests.{queue_name}"
            )

    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue message to RabbitMQ"""
        await self._ensure_queue(queue_name)

        # Convert message to RabbitMQ format
        body = json.dumps(message.to_dict()).encode()

        # Set message properties
        properties = {
            'message_id': message.message_id,
            'priority': message.priority.value,
            'delivery_mode': DeliveryMode.PERSISTENT,
            'headers': message.headers
        }

        if message.expires_at:
            properties['expiration'] = str(int((message.expires_at - datetime.now()).total_seconds() * 1000))

        rabbitmq_message = Message(body, **properties)

        # Publish message
        await self.exchanges['default'].publish(
            rabbitmq_message,
            routing_key=f"requests.{queue_name}"
        )

        self.stats.messages_enqueued += 1
        logger.debug(f"Enqueued message {message.message_id} to {queue_name}")

    async def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue message from RabbitMQ"""
        await self._ensure_queue(queue_name)

        try:
            # Get message with timeout
            async with self.queues[queue_name].iterator(timeout=timeout) as queue_iter:
                async for rabbitmq_message in queue_iter:
                    async with rabbitmq_message.process():
                        # Convert back to QueueMessage
                        data = json.loads(rabbitmq_message.body.decode())
                        message = QueueMessage.from_dict(data)

                        # Check if message is delayed
                        if message.is_delayed:
                            # Requeue with delay
                            await asyncio.sleep(1)
                            continue

                        # Check if message is expired
                        if message.is_expired:
                            logger.warning(f"Message {message.message_id} expired, moving to DLQ")
                            # Message will be auto-acked and moved to DLQ
                            continue

                        self.stats.messages_dequeued += 1
                        message.status = MessageStatus.PROCESSING
                        return message

        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error(f"Error dequeuing from {queue_name}: {e}")

        return None

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        self.stats.messages_processed += 1
        message.status = MessageStatus.COMPLETED

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message"""
        self.stats.messages_failed += 1

        if requeue and message.can_retry:
            message.retry_count += 1
            message.status = MessageStatus.RETRYING
            # Add exponential backoff
            delay = 2 ** message.retry_count
            message.delay_until = datetime.now() + timedelta(seconds=delay)
            await self.enqueue(queue_name, message)
            self.stats.messages_retried += 1
        else:
            message.status = MessageStatus.DEAD_LETTER
            self.stats.messages_dead_lettered += 1

    async def get_queue_size(self, queue_name: str) -> int:
        """Get current queue size"""
        await self._ensure_queue(queue_name)
        queue_info = await self.channel.queue_declare(queue_name, passive=True)
        return queue_info.method.message_count

    async def purge_queue(self, queue_name: str):
        """Purge all messages from queue"""
        await self._ensure_queue(queue_name)
        await self.queues[queue_name].purge()


class KafkaBackend(BaseQueueBackend):
    """Kafka queue backend implementation"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bootstrap_servers = config.get('kafka_brokers', 'kafka-1:9092,kafka-2:9092,kafka-3:9092')
        self.producer = None
        self.consumers = {}
        self.topic_prefix = config.get('topic_prefix', 'bev_requests')

    async def initialize(self):
        """Initialize Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                compression_type='gzip',
                acks='all',
                retries=3
            )
            await self.producer.start()
            logger.info("Kafka backend initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka: {e}")
            raise

    async def close(self):
        """Close Kafka connections"""
        if self.producer:
            await self.producer.stop()

        for consumer in self.consumers.values():
            await consumer.stop()

    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue message to Kafka"""
        topic = f"{self.topic_prefix}_{queue_name}"

        # Add priority to headers for ordering
        headers = [
            ('priority', str(message.priority.value).encode()),
            ('message_id', message.message_id.encode()),
            ('created_at', message.created_at.isoformat().encode())
        ]

        # Add custom headers
        for key, value in message.headers.items():
            headers.append((key, str(value).encode()))

        try:
            await self.producer.send(
                topic,
                value=message.to_dict(),
                headers=headers,
                partition=None  # Let Kafka decide partition
            )
            self.stats.messages_enqueued += 1
            logger.debug(f"Enqueued message {message.message_id} to {topic}")
        except KafkaError as e:
            logger.error(f"Failed to enqueue message to Kafka: {e}")
            raise

    async def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue message from Kafka"""
        topic = f"{self.topic_prefix}_{queue_name}"

        # Create consumer if not exists
        if queue_name not in self.consumers:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"bev_consumer_{queue_name}",
                auto_offset_reset='earliest',
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                enable_auto_commit=False
            )
            await consumer.start()
            self.consumers[queue_name] = consumer

        consumer = self.consumers[queue_name]

        try:
            # Get message with timeout
            data = await asyncio.wait_for(consumer.getone(), timeout=timeout)

            # Convert to QueueMessage
            message = QueueMessage.from_dict(data.value)

            # Check if message is delayed
            if message.is_delayed:
                # In Kafka, we can't easily requeue, so we'll wait
                await asyncio.sleep((message.delay_until - datetime.now()).total_seconds())

            # Check if message is expired
            if message.is_expired:
                logger.warning(f"Message {message.message_id} expired")
                await consumer.commit()
                return None

            self.stats.messages_dequeued += 1
            message.status = MessageStatus.PROCESSING
            return message

        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error dequeuing from Kafka: {e}")
            return None

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        if queue_name in self.consumers:
            await self.consumers[queue_name].commit()
        self.stats.messages_processed += 1
        message.status = MessageStatus.COMPLETED

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message - Kafka doesn't have traditional NACK"""
        self.stats.messages_failed += 1

        if requeue and message.can_retry:
            message.retry_count += 1
            message.status = MessageStatus.RETRYING
            delay = 2 ** message.retry_count
            message.delay_until = datetime.now() + timedelta(seconds=delay)
            await self.enqueue(f"{queue_name}_retry", message)
            self.stats.messages_retried += 1
        else:
            message.status = MessageStatus.DEAD_LETTER
            await self.enqueue(f"{queue_name}_dlq", message)
            self.stats.messages_dead_lettered += 1

        # Commit the original message
        if queue_name in self.consumers:
            await self.consumers[queue_name].commit()

    async def get_queue_size(self, queue_name: str) -> int:
        """Get approximate queue size (Kafka doesn't provide exact count easily)"""
        # This is a simplified implementation
        # In practice, you'd need to check partition offsets
        return 0

    async def purge_queue(self, queue_name: str):
        """Purge queue - not easily supported in Kafka"""
        logger.warning("Kafka doesn't support queue purging - consider topic deletion/recreation")


class RedisBackend(BaseQueueBackend):
    """Redis queue backend implementation using lists and sorted sets"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.redis_url = config.get('redis_url', 'redis://redis:6379/1')
        self.redis_client = None

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis backend initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue message to Redis"""
        # Use priority as score for sorted set
        score = message.priority.value * 1000000 + int(time.time())

        # Serialize message
        serialized = json.dumps(message.to_dict())

        # Add to priority queue (sorted set)
        await self.redis_client.zadd(
            f"queue:{queue_name}:priority",
            {serialized: score}
        )

        # Also add to regular list for FIFO within same priority
        await self.redis_client.lpush(
            f"queue:{queue_name}:fifo",
            serialized
        )

        self.stats.messages_enqueued += 1
        logger.debug(f"Enqueued message {message.message_id} to {queue_name}")

    async def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue message from Redis"""
        try:
            # Try priority queue first
            result = await self.redis_client.zpopmin(f"queue:{queue_name}:priority")

            if not result:
                # Fallback to FIFO queue with blocking pop
                result = await self.redis_client.brpop(
                    f"queue:{queue_name}:fifo",
                    timeout=int(timeout)
                )
                if result:
                    _, serialized = result
                else:
                    return None
            else:
                serialized, _ = result[0]

            # Deserialize message
            data = json.loads(serialized)
            message = QueueMessage.from_dict(data)

            # Check if message is delayed
            if message.is_delayed:
                # Requeue with delay
                await asyncio.sleep(1)
                await self.enqueue(queue_name, message)
                return None

            # Check if message is expired
            if message.is_expired:
                logger.warning(f"Message {message.message_id} expired")
                return None

            # Move to processing set
            await self.redis_client.zadd(
                f"queue:{queue_name}:processing",
                {serialized: time.time()}
            )

            self.stats.messages_dequeued += 1
            message.status = MessageStatus.PROCESSING
            return message

        except Exception as e:
            logger.error(f"Error dequeuing from Redis: {e}")
            return None

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        serialized = json.dumps(message.to_dict())
        await self.redis_client.zrem(f"queue:{queue_name}:processing", serialized)
        self.stats.messages_processed += 1
        message.status = MessageStatus.COMPLETED

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message"""
        serialized = json.dumps(message.to_dict())
        await self.redis_client.zrem(f"queue:{queue_name}:processing", serialized)
        self.stats.messages_failed += 1

        if requeue and message.can_retry:
            message.retry_count += 1
            message.status = MessageStatus.RETRYING
            delay = 2 ** message.retry_count
            message.delay_until = datetime.now() + timedelta(seconds=delay)
            await self.enqueue(queue_name, message)
            self.stats.messages_retried += 1
        else:
            message.status = MessageStatus.DEAD_LETTER
            await self.redis_client.lpush(
                f"queue:{queue_name}:dlq",
                json.dumps(message.to_dict())
            )
            self.stats.messages_dead_lettered += 1

    async def get_queue_size(self, queue_name: str) -> int:
        """Get current queue size"""
        priority_size = await self.redis_client.zcard(f"queue:{queue_name}:priority")
        fifo_size = await self.redis_client.llen(f"queue:{queue_name}:fifo")
        return priority_size + fifo_size

    async def purge_queue(self, queue_name: str):
        """Purge all messages from queue"""
        keys = [
            f"queue:{queue_name}:priority",
            f"queue:{queue_name}:fifo",
            f"queue:{queue_name}:processing",
            f"queue:{queue_name}:dlq"
        ]
        await self.redis_client.delete(*keys)


class MemoryBackend(BaseQueueBackend):
    """In-memory queue backend for testing and fallback"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.queues = defaultdict(lambda: {
            'priority': [],
            'fifo': deque(),
            'processing': {},
            'dlq': deque()
        })
        self.max_queue_size = config.get('max_queue_size', 10000)

    async def initialize(self):
        """Initialize memory backend"""
        logger.info("Memory backend initialized")

    async def close(self):
        """Close memory backend"""
        self.queues.clear()

    async def enqueue(self, queue_name: str, message: QueueMessage):
        """Enqueue message to memory"""
        queue = self.queues[queue_name]

        # Check queue size limit
        total_size = len(queue['priority']) + len(queue['fifo'])
        if total_size >= self.max_queue_size:
            raise Exception(f"Queue {queue_name} is full")

        # Add to priority queue or FIFO based on priority
        if message.priority.value <= 2:  # High priority
            queue['priority'].append(message)
            queue['priority'].sort(key=lambda m: m.priority.value)
        else:
            queue['fifo'].appendleft(message)

        self.stats.messages_enqueued += 1

    async def dequeue(self, queue_name: str, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue message from memory"""
        queue = self.queues[queue_name]

        # Check priority queue first
        if queue['priority']:
            message = queue['priority'].pop(0)
        elif queue['fifo']:
            message = queue['fifo'].pop()
        else:
            # Wait for message with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
                if queue['priority'] or queue['fifo']:
                    break
            else:
                return None

            # Try again after wait
            if queue['priority']:
                message = queue['priority'].pop(0)
            elif queue['fifo']:
                message = queue['fifo'].pop()
            else:
                return None

        # Check if message is delayed
        if message.is_delayed:
            await asyncio.sleep((message.delay_until - datetime.now()).total_seconds())

        # Check if message is expired
        if message.is_expired:
            logger.warning(f"Message {message.message_id} expired")
            return None

        # Move to processing
        queue['processing'][message.message_id] = message
        self.stats.messages_dequeued += 1
        message.status = MessageStatus.PROCESSING
        return message

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        queue = self.queues[queue_name]
        queue['processing'].pop(message.message_id, None)
        self.stats.messages_processed += 1
        message.status = MessageStatus.COMPLETED

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message"""
        queue = self.queues[queue_name]
        queue['processing'].pop(message.message_id, None)
        self.stats.messages_failed += 1

        if requeue and message.can_retry:
            message.retry_count += 1
            message.status = MessageStatus.RETRYING
            delay = 2 ** message.retry_count
            message.delay_until = datetime.now() + timedelta(seconds=delay)
            await self.enqueue(queue_name, message)
            self.stats.messages_retried += 1
        else:
            message.status = MessageStatus.DEAD_LETTER
            queue['dlq'].appendleft(message)
            self.stats.messages_dead_lettered += 1

    async def get_queue_size(self, queue_name: str) -> int:
        """Get current queue size"""
        queue = self.queues[queue_name]
        return len(queue['priority']) + len(queue['fifo'])

    async def purge_queue(self, queue_name: str):
        """Purge all messages from queue"""
        queue = self.queues[queue_name]
        queue['priority'].clear()
        queue['fifo'].clear()
        queue['processing'].clear()


class QueueManager:
    """
    Advanced queue manager with multiple backend support,
    priority queuing, and intelligent message routing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.queue_type = QueueType(config.get('queue_type', 'rabbitmq'))
        self.backend = None

        # Load balancing and failover
        self.primary_backend = None
        self.fallback_backends = []

        # Message routing
        self.routing_rules = config.get('routing_rules', {})
        self.default_queue = config.get('default_queue', 'default')

        # Backpressure handling
        self.enable_backpressure = config.get('enable_backpressure', True)
        self.backpressure_threshold = config.get('backpressure_threshold', 1000)
        self.backpressure_strategy = config.get('backpressure_strategy', 'drop_low_priority')

        # Performance tracking
        self.global_stats = QueueStats()
        self.queue_stats = defaultdict(QueueStats)

        # Message handlers
        self.message_handlers = {}
        self.consumer_tasks = {}

        logger.info(f"QueueManager configured with {self.queue_type.value} backend")

    async def initialize(self):
        """Initialize the queue manager"""
        # Initialize primary backend
        self.backend = self._create_backend(self.queue_type, self.config)
        await self.backend.initialize()
        self.primary_backend = self.backend

        # Initialize fallback backends if configured
        fallback_configs = self.config.get('fallback_backends', [])
        for fallback_config in fallback_configs:
            fallback_type = QueueType(fallback_config['type'])
            fallback_backend = self._create_backend(fallback_type, fallback_config)
            await fallback_backend.initialize()
            self.fallback_backends.append(fallback_backend)

        logger.info(f"QueueManager initialized with {len(self.fallback_backends)} fallback backends")

    async def close(self):
        """Close the queue manager"""
        # Stop all consumer tasks
        for task in self.consumer_tasks.values():
            task.cancel()

        await asyncio.gather(*self.consumer_tasks.values(), return_exceptions=True)

        # Close backends
        if self.primary_backend:
            await self.primary_backend.close()

        for backend in self.fallback_backends:
            await backend.close()

    def _create_backend(self, queue_type: QueueType, config: Dict[str, Any]) -> BaseQueueBackend:
        """Create appropriate backend instance"""
        if queue_type == QueueType.RABBITMQ:
            return RabbitMQBackend(config)
        elif queue_type == QueueType.KAFKA:
            return KafkaBackend(config)
        elif queue_type == QueueType.REDIS:
            return RedisBackend(config)
        elif queue_type == QueueType.MEMORY:
            return MemoryBackend(config)
        else:
            raise ValueError(f"Unsupported queue type: {queue_type}")

    def _route_message(self, message: QueueMessage) -> str:
        """Determine target queue for message based on routing rules"""
        # Check routing rules
        for pattern, queue_name in self.routing_rules.items():
            if pattern in message.routing_key:
                return queue_name

        # Check priority routing
        if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
            return f"{self.default_queue}_priority"

        return self.default_queue

    async def _check_backpressure(self, queue_name: str) -> bool:
        """Check if backpressure should be applied"""
        if not self.enable_backpressure:
            return False

        queue_size = await self.backend.get_queue_size(queue_name)
        return queue_size >= self.backpressure_threshold

    async def _handle_backpressure(self, message: QueueMessage) -> bool:
        """Handle backpressure based on strategy"""
        if self.backpressure_strategy == 'drop_low_priority':
            # Drop low priority messages
            if message.priority in [MessagePriority.LOW, MessagePriority.BACKGROUND]:
                logger.warning(f"Dropping low priority message {message.message_id} due to backpressure")
                return False

        elif self.backpressure_strategy == 'delay':
            # Add delay to message
            message.delay_until = datetime.now() + timedelta(seconds=10)

        elif self.backpressure_strategy == 'reject':
            # Reject message
            logger.warning(f"Rejecting message {message.message_id} due to backpressure")
            return False

        return True

    async def enqueue(self, message: QueueMessage) -> bool:
        """Enqueue message with intelligent routing and backpressure handling"""
        try:
            # Route message to appropriate queue
            queue_name = self._route_message(message)

            # Check backpressure
            if await self._check_backpressure(queue_name):
                if not await self._handle_backpressure(message):
                    return False

            # Try primary backend first
            try:
                await self.backend.enqueue(queue_name, message)
                self.global_stats.messages_enqueued += 1
                self.queue_stats[queue_name].messages_enqueued += 1
                return True

            except Exception as e:
                logger.warning(f"Primary backend failed: {e}, trying fallbacks")

                # Try fallback backends
                for fallback in self.fallback_backends:
                    try:
                        await fallback.enqueue(queue_name, message)
                        self.global_stats.messages_enqueued += 1
                        self.queue_stats[queue_name].messages_enqueued += 1
                        return True
                    except Exception as fe:
                        logger.warning(f"Fallback backend failed: {fe}")

                # All backends failed
                raise Exception("All queue backends failed")

        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            return False

    async def dequeue(self, queue_name: Optional[str] = None, timeout: float = 1.0) -> Optional[QueueMessage]:
        """Dequeue message from specified or default queue"""
        target_queue = queue_name or self.default_queue

        try:
            # Try primary backend first
            message = await self.backend.dequeue(target_queue, timeout)

            if message:
                self.global_stats.messages_dequeued += 1
                self.queue_stats[target_queue].messages_dequeued += 1
                return message

            # Try fallback backends if no message from primary
            for fallback in self.fallback_backends:
                message = await fallback.dequeue(target_queue, timeout=0.1)
                if message:
                    self.global_stats.messages_dequeued += 1
                    self.queue_stats[target_queue].messages_dequeued += 1
                    return message

        except Exception as e:
            logger.error(f"Failed to dequeue message: {e}")

        return None

    async def ack(self, queue_name: str, message: QueueMessage):
        """Acknowledge message processing"""
        try:
            await self.backend.ack(queue_name, message)
            self.global_stats.messages_processed += 1
            self.queue_stats[queue_name].messages_processed += 1
        except Exception as e:
            logger.error(f"Failed to ack message: {e}")

    async def nack(self, queue_name: str, message: QueueMessage, requeue: bool = True):
        """Negative acknowledge message"""
        try:
            await self.backend.nack(queue_name, message, requeue)
            self.global_stats.messages_failed += 1
            self.queue_stats[queue_name].messages_failed += 1
        except Exception as e:
            logger.error(f"Failed to nack message: {e}")

    def register_handler(self, queue_name: str, handler: Callable):
        """Register message handler for queue"""
        self.message_handlers[queue_name] = handler
        logger.info(f"Registered handler for queue {queue_name}")

    async def start_consumer(self, queue_name: str, concurrency: int = 1):
        """Start consumer for specified queue"""
        if queue_name in self.consumer_tasks:
            logger.warning(f"Consumer already running for queue {queue_name}")
            return

        # Start consumer tasks
        tasks = []
        for i in range(concurrency):
            task = asyncio.create_task(self._consumer_worker(queue_name, i))
            tasks.append(task)

        self.consumer_tasks[queue_name] = asyncio.gather(*tasks)
        logger.info(f"Started {concurrency} consumers for queue {queue_name}")

    async def stop_consumer(self, queue_name: str):
        """Stop consumer for specified queue"""
        if queue_name in self.consumer_tasks:
            self.consumer_tasks[queue_name].cancel()
            del self.consumer_tasks[queue_name]
            logger.info(f"Stopped consumer for queue {queue_name}")

    async def _consumer_worker(self, queue_name: str, worker_id: int):
        """Consumer worker coroutine"""
        logger.debug(f"Consumer worker {worker_id} started for queue {queue_name}")

        while True:
            try:
                # Get message
                message = await self.dequeue(queue_name, timeout=1.0)
                if not message:
                    continue

                # Process message
                if queue_name in self.message_handlers:
                    handler = self.message_handlers[queue_name]
                    start_time = time.time()

                    try:
                        # Execute handler
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)

                        # Acknowledge successful processing
                        await self.ack(queue_name, message)

                        # Update stats
                        processing_time = time.time() - start_time
                        self.global_stats.total_processing_time += processing_time
                        self.queue_stats[queue_name].total_processing_time += processing_time

                    except Exception as e:
                        logger.error(f"Handler error for message {message.message_id}: {e}")
                        await self.nack(queue_name, message, requeue=True)

                else:
                    logger.warning(f"No handler registered for queue {queue_name}")
                    await self.nack(queue_name, message, requeue=False)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer worker error: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Consumer worker {worker_id} stopped for queue {queue_name}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive queue statistics"""
        return {
            'global_stats': asdict(self.global_stats),
            'queue_stats': {name: asdict(stats) for name, stats in self.queue_stats.items()},
            'backend_stats': self.backend.get_stats() if self.backend else {},
            'active_consumers': list(self.consumer_tasks.keys()),
            'queue_type': self.queue_type.value
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on queue manager"""
        health_status = {
            'status': 'healthy',
            'backend_type': self.queue_type.value,
            'primary_backend_healthy': True,
            'fallback_backends_available': len(self.fallback_backends),
            'active_consumers': len(self.consumer_tasks),
            'issues': []
        }

        # Check primary backend
        try:
            if hasattr(self.backend, 'health_check'):
                backend_health = await self.backend.health_check()
                if backend_health.get('status') != 'healthy':
                    health_status['primary_backend_healthy'] = False
                    health_status['issues'].append("Primary backend unhealthy")
        except Exception as e:
            health_status['primary_backend_healthy'] = False
            health_status['issues'].append(f"Primary backend error: {e}")

        # Check if failure rate is too high
        if self.global_stats.success_rate < 0.9:
            health_status['issues'].append(f"Low success rate: {self.global_stats.success_rate:.2%}")
            health_status['status'] = 'warning'

        if health_status['issues']:
            health_status['status'] = 'degraded' if health_status['primary_backend_healthy'] else 'critical'

        return health_status


# Factory function
def create_queue_manager(config: Dict[str, Any]) -> QueueManager:
    """Create and configure a QueueManager instance"""
    return QueueManager(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'queue_type': 'memory',  # Use memory for testing
            'max_queue_size': 1000,
            'enable_backpressure': True,
            'backpressure_threshold': 100,
            'backpressure_strategy': 'drop_low_priority',
            'routing_rules': {
                'high_priority': 'priority_queue',
                'crawl': 'crawler_queue'
            }
        }

        queue_manager = create_queue_manager(config)
        await queue_manager.initialize()

        try:
            # Register a simple handler
            async def test_handler(message: QueueMessage):
                print(f"Processing message: {message.message_id}")
                await asyncio.sleep(0.1)  # Simulate processing

            queue_manager.register_handler('default', test_handler)

            # Start consumer
            await queue_manager.start_consumer('default', concurrency=2)

            # Send test messages
            for i in range(10):
                message = QueueMessage(
                    payload=f"Test message {i}",
                    priority=MessagePriority.MEDIUM,
                    routing_key="test"
                )
                await queue_manager.enqueue(message)

            # Wait for processing
            await asyncio.sleep(2)

            # Get statistics
            stats = queue_manager.get_statistics()
            print(f"Statistics: {json.dumps(stats, indent=2, default=str)}")

            # Health check
            health = await queue_manager.health_check()
            print(f"Health: {health}")

        finally:
            await queue_manager.close()

    asyncio.run(main())