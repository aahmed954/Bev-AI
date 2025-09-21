#!/usr/bin/env python3
"""
OSINT Integration Layer for Advanced Avatar System
Connects BEV OSINT infrastructure with real-time avatar responses
Supports breach, darknet, crypto, and social media investigations
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum
from collections import deque, defaultdict
import hashlib
import uuid

# Database connections
import asyncpg
from neo4j import AsyncGraphDatabase
import redis.asyncio as redis
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SearchRequest, Filter, FieldCondition

# Message queuing and event streaming
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import aio_pika
from nats.aio.client import Client as NATS

# WebSocket and REST
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import httpx

# OSINT analysis tools
from typing_extensions import Protocol
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OSINTEventType(Enum):
    """OSINT investigation event types"""
    # Discovery events
    BREACH_DISCOVERED = "breach_discovered"
    DARKNET_ACTIVITY = "darknet_activity"
    CRYPTO_TRANSACTION = "crypto_transaction"
    SOCIAL_PROFILE_FOUND = "social_profile_found"

    # Analysis events
    PATTERN_DETECTED = "pattern_detected"
    CORRELATION_FOUND = "correlation_found"
    THREAT_IDENTIFIED = "threat_identified"
    VULNERABILITY_FOUND = "vulnerability_found"

    # Investigation milestones
    INVESTIGATION_STARTED = "investigation_started"
    INVESTIGATION_PROGRESS = "investigation_progress"
    INVESTIGATION_COMPLETE = "investigation_complete"
    BREAKTHROUGH_MOMENT = "breakthrough_moment"

    # Alert events
    CRITICAL_ALERT = "critical_alert"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ANOMALY_DETECTED = "anomaly_detected"

class ThreatLevel(IntEnum):
    """Threat severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class InvestigationType(Enum):
    """OSINT investigation types"""
    BREACH_DATABASE = "breach_database"
    DARKNET_MARKET = "darknet_market"
    CRYPTOCURRENCY = "cryptocurrency"
    SOCIAL_MEDIA = "social_media"
    THREAT_INTELLIGENCE = "threat_intelligence"
    GRAPH_ANALYSIS = "graph_analysis"
    VULNERABILITY_SCAN = "vulnerability_scan"
    IDENTITY_VERIFICATION = "identity_verification"

@dataclass
class OSINTEvent:
    """OSINT event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: OSINTEventType = OSINTEventType.INVESTIGATION_PROGRESS
    investigation_type: InvestigationType = InvestigationType.THREAT_INTELLIGENCE
    timestamp: datetime = field(default_factory=datetime.now)

    # Event details
    title: str = ""
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    # Impact assessment
    threat_level: ThreatLevel = ThreatLevel.NONE
    confidence: float = 0.5
    priority: int = 5

    # Avatar response hints
    suggested_emotion: str = "neutral"
    response_template: str = ""
    animation_cue: str = ""

    # Tracking
    source_analyzer: str = ""
    investigation_id: str = ""
    correlation_ids: List[str] = field(default_factory=list)

@dataclass
class InvestigationState:
    """Current state of OSINT investigation"""
    investigation_id: str
    investigation_type: InvestigationType
    started_at: datetime

    # Progress tracking
    phase: str = "initialization"  # initialization, collection, analysis, correlation, reporting
    progress: float = 0.0
    milestones_completed: List[str] = field(default_factory=list)

    # Findings
    findings_count: int = 0
    breaches_found: int = 0
    threats_identified: int = 0
    patterns_detected: int = 0

    # Current activity
    current_activity: str = ""
    last_update: datetime = field(default_factory=datetime.now)

    # Performance metrics
    events_processed: int = 0
    processing_time: float = 0.0
    memory_usage: float = 0.0

class OSINTEventProcessor:
    """Processes OSINT events and generates avatar responses"""

    def __init__(self):
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.processed_events: deque = deque(maxlen=1000)
        self.active_investigations: Dict[str, InvestigationState] = {}

        # Response templates based on event types
        self.response_templates = {
            OSINTEventType.BREACH_DISCOVERED: [
                "I've found credential exposures in the {source} breach database.",
                "New breach data discovered! {count} records potentially compromised.",
                "Alert: Sensitive data exposure detected in recent breach."
            ],
            OSINTEventType.PATTERN_DETECTED: [
                "Interesting pattern emerging from the data analysis...",
                "I'm seeing correlations between {entity1} and {entity2}.",
                "Pattern recognition complete - this looks significant!"
            ],
            OSINTEventType.THREAT_IDENTIFIED: [
                "⚠️ Critical threat identified: {threat_description}",
                "Threat assessment complete - immediate action recommended.",
                "I've detected suspicious activity requiring attention."
            ],
            OSINTEventType.BREAKTHROUGH_MOMENT: [
                "🎯 Breakthrough! I've connected the pieces!",
                "This is it! The correlation reveals {discovery}",
                "Excellent finding! This changes our understanding completely."
            ],
            OSINTEventType.INVESTIGATION_COMPLETE: [
                "Investigation complete. {findings_count} key findings documented.",
                "Analysis finished. Threat level assessed as {threat_level}.",
                "All data processed. Ready for next investigation."
            ]
        }

        # Emotion mapping for events
        self.emotion_map = {
            OSINTEventType.BREACH_DISCOVERED: "alert",
            OSINTEventType.DARKNET_ACTIVITY: "focused",
            OSINTEventType.CRYPTO_TRANSACTION: "analyzing",
            OSINTEventType.PATTERN_DETECTED: "excited",
            OSINTEventType.CORRELATION_FOUND: "breakthrough",
            OSINTEventType.THREAT_IDENTIFIED: "concerned",
            OSINTEventType.BREAKTHROUGH_MOMENT: "excited",
            OSINTEventType.INVESTIGATION_COMPLETE: "satisfied",
            OSINTEventType.CRITICAL_ALERT: "alert",
            OSINTEventType.INVESTIGATION_STARTED: "focused"
        }

    async def process_event(self, event: OSINTEvent) -> Dict[str, Any]:
        """Process single OSINT event and generate avatar response"""

        # Update investigation state
        if event.investigation_id in self.active_investigations:
            investigation = self.active_investigations[event.investigation_id]
            investigation.events_processed += 1
            investigation.last_update = datetime.now()

            # Update findings based on event type
            if "breach" in event.event_type.value:
                investigation.breaches_found += 1
            elif "threat" in event.event_type.value:
                investigation.threats_identified += 1
            elif "pattern" in event.event_type.value:
                investigation.patterns_detected += 1

            investigation.findings_count = (
                investigation.breaches_found +
                investigation.threats_identified +
                investigation.patterns_detected
            )

        # Generate avatar response
        response = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'emotion': self.emotion_map.get(event.event_type, "neutral"),
            'response_text': self._generate_response_text(event),
            'animation_cue': event.animation_cue or self._get_animation_cue(event),
            'threat_level': event.threat_level.value,
            'priority': event.priority,
            'investigation_progress': self._calculate_progress(event.investigation_id)
        }

        # Add to processed queue
        self.processed_events.append(event)

        return response

    def _generate_response_text(self, event: OSINTEvent) -> str:
        """Generate contextual response text for event"""

        templates = self.response_templates.get(event.event_type, [])
        if not templates:
            return f"Processing {event.event_type.value.replace('_', ' ')}..."

        # Select template and format with event data
        import random
        template = random.choice(templates)

        # Format template with event data
        try:
            response = template.format(**event.data)
        except KeyError:
            response = template.format(
                source=event.source_analyzer,
                count=event.data.get('count', 'multiple'),
                threat_description=event.description,
                entity1=event.data.get('entity1', 'target'),
                entity2=event.data.get('entity2', 'source'),
                discovery=event.title,
                findings_count=event.data.get('findings_count', 'several'),
                threat_level=event.threat_level.name.lower()
            )

        return response

    def _get_animation_cue(self, event: OSINTEvent) -> str:
        """Get animation cue for avatar based on event"""

        animation_cues = {
            OSINTEventType.BREACH_DISCOVERED: "alert_pose",
            OSINTEventType.PATTERN_DETECTED: "thinking_gesture",
            OSINTEventType.BREAKTHROUGH_MOMENT: "eureka_animation",
            OSINTEventType.THREAT_IDENTIFIED: "concern_expression",
            OSINTEventType.INVESTIGATION_COMPLETE: "satisfied_nod"
        }

        return animation_cues.get(event.event_type, "idle_animation")

    def _calculate_progress(self, investigation_id: str) -> float:
        """Calculate investigation progress"""

        if investigation_id not in self.active_investigations:
            return 0.0

        investigation = self.active_investigations[investigation_id]
        return min(1.0, investigation.progress)

class DatabaseConnector:
    """Manages connections to all BEV databases"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.neo4j_driver: Optional[AsyncGraphDatabase.driver] = None
        self.redis_client: Optional[redis.Redis] = None
        self.qdrant_client: Optional[QdrantClient] = None

    async def initialize(self):
        """Initialize all database connections"""

        # PostgreSQL connection pool
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host=self.config['postgres']['host'],
                port=self.config['postgres']['port'],
                user=self.config['postgres']['user'],
                password=self.config['postgres']['password'],
                database=self.config['postgres']['database'],
                min_size=10,
                max_size=20
            )
            logger.info("PostgreSQL connection pool established")
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")

        # Neo4j driver
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config['neo4j']['uri'],
                auth=(
                    self.config['neo4j']['user'],
                    self.config['neo4j']['password']
                )
            )
            logger.info("Neo4j driver initialized")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")

        # Redis client
        try:
            self.redis_client = await redis.from_url(
                self.config['redis']['url'],
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")

        # Qdrant vector database
        try:
            self.qdrant_client = QdrantClient(
                host=self.config['qdrant']['host'],
                port=self.config['qdrant']['port']
            )
            logger.info("Qdrant vector database connected")
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")

    async def query_investigation_data(self, investigation_id: str) -> Dict[str, Any]:
        """Query investigation data from PostgreSQL"""

        if not self.postgres_pool:
            return {}

        async with self.postgres_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT investigation_id, investigation_type, status,
                       started_at, findings, threat_indicators, progress
                FROM osint_investigations
                WHERE investigation_id = $1
                """,
                investigation_id
            )

            if row:
                return dict(row)
            return {}

    async def get_graph_relationships(self, entity_id: str) -> List[Dict]:
        """Get entity relationships from Neo4j"""

        if not self.neo4j_driver:
            return []

        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $entity_id})-[r]->(related)
                RETURN type(r) as relationship, related.id as related_id,
                       related.name as related_name, r.confidence as confidence
                LIMIT 50
                """,
                entity_id=entity_id
            )

            relationships = []
            async for record in result:
                relationships.append({
                    'relationship': record['relationship'],
                    'related_id': record['related_id'],
                    'related_name': record['related_name'],
                    'confidence': record['confidence']
                })

            return relationships

    async def cache_investigation_state(self, state: InvestigationState):
        """Cache investigation state in Redis"""

        if not self.redis_client:
            return

        key = f"osint:investigation:{state.investigation_id}"
        value = json.dumps(asdict(state), default=str)

        await self.redis_client.setex(key, 3600, value)

    async def semantic_search(self, query_vector: List[float],
                            collection: str = "osint_findings") -> List[Dict]:
        """Perform semantic search in Qdrant"""

        if not self.qdrant_client:
            return []

        search_result = self.qdrant_client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=10
        )

        return [
            {
                'id': point.id,
                'score': point.score,
                'payload': point.payload
            }
            for point in search_result
        ]

    async def close(self):
        """Close all database connections"""

        if self.postgres_pool:
            await self.postgres_pool.close()

        if self.neo4j_driver:
            await self.neo4j_driver.close()

        if self.redis_client:
            await self.redis_client.close()

class OSINTAnalyzerConnector:
    """Connects to OSINT analyzer services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analyzer_endpoints = {
            'breach': config.get('breach_analyzer_url', 'http://localhost:8081'),
            'darknet': config.get('darknet_analyzer_url', 'http://localhost:8082'),
            'crypto': config.get('crypto_analyzer_url', 'http://localhost:8083'),
            'social': config.get('social_analyzer_url', 'http://localhost:8084')
        }
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def trigger_breach_analysis(self, target: str) -> Dict[str, Any]:
        """Trigger breach database analysis"""

        try:
            response = await self.http_client.post(
                f"{self.analyzer_endpoints['breach']}/analyze",
                json={'target': target, 'deep_scan': True}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Breach analysis failed: {e}")
            return {'error': str(e)}

    async def monitor_darknet_activity(self, keywords: List[str]) -> AsyncGenerator:
        """Stream darknet monitoring results"""

        try:
            async with self.http_client.stream(
                'POST',
                f"{self.analyzer_endpoints['darknet']}/monitor",
                json={'keywords': keywords}
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        yield json.loads(line)
        except Exception as e:
            logger.error(f"Darknet monitoring error: {e}")

    async def track_crypto_transactions(self, wallet_address: str) -> Dict[str, Any]:
        """Track cryptocurrency transactions"""

        try:
            response = await self.http_client.post(
                f"{self.analyzer_endpoints['crypto']}/track",
                json={'wallet': wallet_address, 'depth': 3}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Crypto tracking failed: {e}")
            return {'error': str(e)}

    async def analyze_social_profile(self, username: str,
                                    platforms: List[str]) -> Dict[str, Any]:
        """Analyze social media profiles"""

        try:
            response = await self.http_client.post(
                f"{self.analyzer_endpoints['social']}/profile",
                json={'username': username, 'platforms': platforms}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Social analysis failed: {e}")
            return {'error': str(e)}

class MessageQueueHandler:
    """Handles message queue communication for OSINT events"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.kafka_producer: Optional[AIOKafkaProducer] = None
        self.kafka_consumer: Optional[AIOKafkaConsumer] = None
        self.rabbitmq_connection: Optional[aio_pika.Connection] = None
        self.rabbitmq_channel: Optional[aio_pika.Channel] = None
        self.nats_client: Optional[NATS] = None

    async def initialize(self):
        """Initialize message queue connections"""

        # Kafka setup
        try:
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers=self.config.get('kafka_brokers', 'localhost:9092'),
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()

            self.kafka_consumer = AIOKafkaConsumer(
                'osint-events',
                bootstrap_servers=self.config.get('kafka_brokers', 'localhost:9092'),
                group_id='avatar-integration',
                value_deserializer=lambda v: json.loads(v.decode())
            )
            await self.kafka_consumer.start()

            logger.info("Kafka connections established")
        except Exception as e:
            logger.error(f"Kafka initialization failed: {e}")

        # RabbitMQ setup
        try:
            self.rabbitmq_connection = await aio_pika.connect_robust(
                self.config.get('rabbitmq_url', 'amqp://guest:guest@localhost/')
            )
            self.rabbitmq_channel = await self.rabbitmq_connection.channel()

            # Declare OSINT events exchange
            await self.rabbitmq_channel.declare_exchange(
                'osint_events',
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )

            logger.info("RabbitMQ connection established")
        except Exception as e:
            logger.error(f"RabbitMQ initialization failed: {e}")

        # NATS setup for real-time streaming
        try:
            self.nats_client = NATS()
            await self.nats_client.connect(
                servers=[self.config.get('nats_url', 'nats://localhost:4222')]
            )
            logger.info("NATS streaming connected")
        except Exception as e:
            logger.error(f"NATS initialization failed: {e}")

    async def publish_event(self, event: OSINTEvent):
        """Publish OSINT event to all message queues"""

        event_data = asdict(event)
        event_data['timestamp'] = event.timestamp.isoformat()

        # Kafka
        if self.kafka_producer:
            try:
                await self.kafka_producer.send(
                    'osint-events',
                    value=event_data
                )
            except Exception as e:
                logger.error(f"Kafka publish failed: {e}")

        # RabbitMQ
        if self.rabbitmq_channel:
            try:
                routing_key = f"osint.{event.investigation_type.value}.{event.event_type.value}"
                await self.rabbitmq_channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(event_data).encode(),
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                    ),
                    routing_key=routing_key
                )
            except Exception as e:
                logger.error(f"RabbitMQ publish failed: {e}")

        # NATS
        if self.nats_client:
            try:
                subject = f"osint.{event.event_type.value}"
                await self.nats_client.publish(
                    subject,
                    json.dumps(event_data).encode()
                )
            except Exception as e:
                logger.error(f"NATS publish failed: {e}")

    async def consume_events(self) -> AsyncGenerator[OSINTEvent, None]:
        """Consume OSINT events from message queues"""

        if self.kafka_consumer:
            async for msg in self.kafka_consumer:
                try:
                    event_data = msg.value
                    event = self._deserialize_event(event_data)
                    yield event
                except Exception as e:
                    logger.error(f"Event deserialization failed: {e}")

    def _deserialize_event(self, data: Dict) -> OSINTEvent:
        """Deserialize event data to OSINTEvent object"""

        # Convert string enums back to enum types
        data['event_type'] = OSINTEventType(data['event_type'])
        data['investigation_type'] = InvestigationType(data['investigation_type'])
        data['threat_level'] = ThreatLevel(data['threat_level'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        return OSINTEvent(**data)

class OSINTIntegrationLayer:
    """Main integration layer connecting OSINT to Avatar system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()

        # Core components
        self.event_processor = OSINTEventProcessor()
        self.db_connector = DatabaseConnector(self.config['databases'])
        self.analyzer_connector = OSINTAnalyzerConnector(self.config)
        self.mq_handler = MessageQueueHandler(self.config['messaging'])

        # WebSocket connections for avatar communication
        self.avatar_websocket: Optional[WebSocket] = None
        self.websocket_clients: List[WebSocket] = []

        # Performance tracking
        self.metrics = {
            'events_processed': 0,
            'average_response_time': 0.0,
            'active_investigations': 0,
            'memory_usage': 0.0
        }

        # Background tasks
        self.background_tasks = []
        self.running = False

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'databases': {
                'postgres': {
                    'host': 'localhost',
                    'port': 5432,
                    'user': 'researcher',
                    'password': 'osint_research_2024',
                    'database': 'osint'
                },
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'user': 'neo4j',
                    'password': 'BevGraphMaster2024'
                },
                'redis': {
                    'url': 'redis://localhost:6379'
                },
                'qdrant': {
                    'host': 'localhost',
                    'port': 6333
                }
            },
            'messaging': {
                'kafka_brokers': 'localhost:9092',
                'rabbitmq_url': 'amqp://guest:guest@localhost/',
                'nats_url': 'nats://localhost:4222'
            },
            'avatar_websocket_url': 'ws://localhost:8091/ws',
            'response_timeout': 100,  # milliseconds
            'batch_size': 10
        }

    async def initialize(self):
        """Initialize all integration components"""

        logger.info("Initializing OSINT Integration Layer...")

        # Initialize database connections
        await self.db_connector.initialize()

        # Initialize message queue handlers
        await self.mq_handler.initialize()

        # Connect to avatar system
        await self._connect_to_avatar()

        self.running = True

        # Start background tasks
        self.background_tasks.append(
            asyncio.create_task(self._event_processing_loop())
        )
        self.background_tasks.append(
            asyncio.create_task(self._message_consumer_loop())
        )

        logger.info("OSINT Integration Layer initialized successfully")

    async def _connect_to_avatar(self):
        """Establish WebSocket connection to avatar system"""

        try:
            import websockets
            self.avatar_websocket = await websockets.connect(
                self.config['avatar_websocket_url']
            )
            logger.info("Connected to avatar system")
        except Exception as e:
            logger.error(f"Avatar connection failed: {e}")

    async def _event_processing_loop(self):
        """Main event processing loop"""

        while self.running:
            try:
                # Process events from queue
                if not self.event_processor.event_queue.empty():
                    event = await self.event_processor.event_queue.get()

                    start_time = time.time()

                    # Process event and generate response
                    response = await self.event_processor.process_event(event)

                    # Send to avatar system
                    await self._send_to_avatar(response)

                    # Update metrics
                    processing_time = (time.time() - start_time) * 1000
                    self.metrics['events_processed'] += 1
                    self.metrics['average_response_time'] = (
                        (self.metrics['average_response_time'] *
                         (self.metrics['events_processed'] - 1) +
                         processing_time) / self.metrics['events_processed']
                    )

                    # Cache investigation state
                    if event.investigation_id in self.event_processor.active_investigations:
                        state = self.event_processor.active_investigations[event.investigation_id]
                        await self.db_connector.cache_investigation_state(state)

                await asyncio.sleep(0.01)  # Small delay to prevent CPU spinning

            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(0.1)

    async def _message_consumer_loop(self):
        """Consume events from message queues"""

        while self.running:
            try:
                async for event in self.mq_handler.consume_events():
                    await self.event_processor.event_queue.put(event)
            except Exception as e:
                logger.error(f"Message consumption error: {e}")
                await asyncio.sleep(1.0)

    async def _send_to_avatar(self, response: Dict[str, Any]):
        """Send processed response to avatar system"""

        if self.avatar_websocket:
            try:
                message = {
                    'type': 'osint_update',
                    'data': response
                }
                await self.avatar_websocket.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Avatar communication error: {e}")
                # Try to reconnect
                await self._connect_to_avatar()

        # Also broadcast to any connected WebSocket clients
        await self._broadcast_to_clients(response)

    async def _broadcast_to_clients(self, data: Dict[str, Any]):
        """Broadcast to all connected WebSocket clients"""

        disconnected = []
        for client in self.websocket_clients:
            try:
                await client.send_text(json.dumps(data))
            except:
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.remove(client)

    async def start_investigation(self, investigation_type: InvestigationType,
                                target: str, params: Dict = None) -> str:
        """Start new OSINT investigation"""

        investigation_id = str(uuid.uuid4())

        # Create investigation state
        state = InvestigationState(
            investigation_id=investigation_id,
            investigation_type=investigation_type,
            started_at=datetime.now()
        )

        self.event_processor.active_investigations[investigation_id] = state

        # Create start event
        event = OSINTEvent(
            event_type=OSINTEventType.INVESTIGATION_STARTED,
            investigation_type=investigation_type,
            investigation_id=investigation_id,
            title=f"Starting {investigation_type.value} investigation",
            description=f"Investigating target: {target}",
            data={'target': target, 'params': params or {}},
            suggested_emotion="focused"
        )

        # Queue event for processing
        await self.event_processor.event_queue.put(event)

        # Trigger appropriate analyzer
        if investigation_type == InvestigationType.BREACH_DATABASE:
            asyncio.create_task(self._run_breach_investigation(investigation_id, target))
        elif investigation_type == InvestigationType.DARKNET_MARKET:
            asyncio.create_task(self._run_darknet_investigation(investigation_id, target))
        elif investigation_type == InvestigationType.CRYPTOCURRENCY:
            asyncio.create_task(self._run_crypto_investigation(investigation_id, target))
        elif investigation_type == InvestigationType.SOCIAL_MEDIA:
            asyncio.create_task(self._run_social_investigation(investigation_id, target))

        return investigation_id

    async def _run_breach_investigation(self, investigation_id: str, target: str):
        """Run breach database investigation"""

        try:
            # Trigger breach analysis
            result = await self.analyzer_connector.trigger_breach_analysis(target)

            if 'breaches' in result:
                for breach in result['breaches']:
                    event = OSINTEvent(
                        event_type=OSINTEventType.BREACH_DISCOVERED,
                        investigation_type=InvestigationType.BREACH_DATABASE,
                        investigation_id=investigation_id,
                        title=f"Breach found: {breach['name']}",
                        description=breach.get('description', ''),
                        data=breach,
                        threat_level=ThreatLevel.HIGH if breach.get('sensitive') else ThreatLevel.MEDIUM,
                        suggested_emotion="alert"
                    )
                    await self.event_processor.event_queue.put(event)

            # Mark investigation complete
            complete_event = OSINTEvent(
                event_type=OSINTEventType.INVESTIGATION_COMPLETE,
                investigation_type=InvestigationType.BREACH_DATABASE,
                investigation_id=investigation_id,
                title="Breach investigation complete",
                data={'total_breaches': len(result.get('breaches', []))},
                suggested_emotion="satisfied"
            )
            await self.event_processor.event_queue.put(complete_event)

        except Exception as e:
            logger.error(f"Breach investigation failed: {e}")

    async def _run_darknet_investigation(self, investigation_id: str, target: str):
        """Run darknet market investigation"""

        try:
            keywords = [target] if isinstance(target, str) else target

            async for activity in self.analyzer_connector.monitor_darknet_activity(keywords):
                event = OSINTEvent(
                    event_type=OSINTEventType.DARKNET_ACTIVITY,
                    investigation_type=InvestigationType.DARKNET_MARKET,
                    investigation_id=investigation_id,
                    title=f"Darknet activity: {activity.get('market', 'Unknown')}",
                    description=activity.get('description', ''),
                    data=activity,
                    threat_level=self._assess_darknet_threat(activity),
                    suggested_emotion="focused"
                )
                await self.event_processor.event_queue.put(event)

                # Check for patterns
                if activity.get('pattern_detected'):
                    pattern_event = OSINTEvent(
                        event_type=OSINTEventType.PATTERN_DETECTED,
                        investigation_type=InvestigationType.DARKNET_MARKET,
                        investigation_id=investigation_id,
                        title="Pattern detected in darknet activity",
                        data={'pattern': activity['pattern']},
                        suggested_emotion="excited"
                    )
                    await self.event_processor.event_queue.put(pattern_event)

        except Exception as e:
            logger.error(f"Darknet investigation failed: {e}")

    async def _run_crypto_investigation(self, investigation_id: str, wallet: str):
        """Run cryptocurrency tracking investigation"""

        try:
            result = await self.analyzer_connector.track_crypto_transactions(wallet)

            for transaction in result.get('transactions', []):
                event = OSINTEvent(
                    event_type=OSINTEventType.CRYPTO_TRANSACTION,
                    investigation_type=InvestigationType.CRYPTOCURRENCY,
                    investigation_id=investigation_id,
                    title=f"Transaction: {transaction['hash'][:16]}...",
                    data=transaction,
                    threat_level=self._assess_crypto_risk(transaction),
                    suggested_emotion="analyzing"
                )
                await self.event_processor.event_queue.put(event)

            # Check for suspicious patterns
            if result.get('suspicious_activity'):
                alert_event = OSINTEvent(
                    event_type=OSINTEventType.SUSPICIOUS_ACTIVITY,
                    investigation_type=InvestigationType.CRYPTOCURRENCY,
                    investigation_id=investigation_id,
                    title="Suspicious cryptocurrency activity detected",
                    data=result['suspicious_activity'],
                    threat_level=ThreatLevel.HIGH,
                    suggested_emotion="alert"
                )
                await self.event_processor.event_queue.put(alert_event)

        except Exception as e:
            logger.error(f"Crypto investigation failed: {e}")

    async def _run_social_investigation(self, investigation_id: str, username: str):
        """Run social media investigation"""

        try:
            platforms = ['twitter', 'instagram', 'linkedin', 'facebook']
            result = await self.analyzer_connector.analyze_social_profile(username, platforms)

            for platform, profile in result.get('profiles', {}).items():
                if profile:
                    event = OSINTEvent(
                        event_type=OSINTEventType.SOCIAL_PROFILE_FOUND,
                        investigation_type=InvestigationType.SOCIAL_MEDIA,
                        investigation_id=investigation_id,
                        title=f"Profile found on {platform}",
                        data=profile,
                        suggested_emotion="discovered"
                    )
                    await self.event_processor.event_queue.put(event)

            # Check for cross-platform correlations
            if result.get('correlations'):
                correlation_event = OSINTEvent(
                    event_type=OSINTEventType.CORRELATION_FOUND,
                    investigation_type=InvestigationType.SOCIAL_MEDIA,
                    investigation_id=investigation_id,
                    title="Cross-platform correlation identified",
                    data=result['correlations'],
                    suggested_emotion="breakthrough"
                )
                await self.event_processor.event_queue.put(correlation_event)

        except Exception as e:
            logger.error(f"Social investigation failed: {e}")

    def _assess_darknet_threat(self, activity: Dict) -> ThreatLevel:
        """Assess threat level from darknet activity"""

        indicators = activity.get('threat_indicators', [])
        if 'exploit' in indicators or 'zero_day' in indicators:
            return ThreatLevel.CRITICAL
        elif 'malware' in indicators or 'ransomware' in indicators:
            return ThreatLevel.HIGH
        elif 'credentials' in indicators:
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    def _assess_crypto_risk(self, transaction: Dict) -> ThreatLevel:
        """Assess risk level from crypto transaction"""

        amount = transaction.get('amount', 0)
        if transaction.get('mixer_detected'):
            return ThreatLevel.HIGH
        elif amount > 100000:  # Large transaction
            return ThreatLevel.MEDIUM
        return ThreatLevel.LOW

    async def get_investigation_status(self, investigation_id: str) -> Dict[str, Any]:
        """Get current status of investigation"""

        if investigation_id in self.event_processor.active_investigations:
            state = self.event_processor.active_investigations[investigation_id]
            return asdict(state)

        # Try to load from database
        data = await self.db_connector.query_investigation_data(investigation_id)
        return data or {'error': 'Investigation not found'}

    async def correlate_findings(self, investigation_ids: List[str]) -> Dict[str, Any]:
        """Correlate findings across multiple investigations"""

        correlations = []

        # Get all events from specified investigations
        all_events = []
        for event in self.event_processor.processed_events:
            if event.investigation_id in investigation_ids:
                all_events.append(event)

        # Simple correlation logic (can be enhanced)
        entities = defaultdict(list)
        for event in all_events:
            for key, value in event.data.items():
                if key in ['email', 'username', 'wallet', 'ip_address']:
                    entities[value].append(event.investigation_id)

        # Find entities appearing in multiple investigations
        for entity, investigations in entities.items():
            if len(set(investigations)) > 1:
                correlations.append({
                    'entity': entity,
                    'investigations': list(set(investigations)),
                    'confidence': len(investigations) / len(investigation_ids)
                })

        if correlations:
            # Create breakthrough event
            breakthrough_event = OSINTEvent(
                event_type=OSINTEventType.BREAKTHROUGH_MOMENT,
                investigation_type=InvestigationType.GRAPH_ANALYSIS,
                title="Major correlation discovered!",
                data={'correlations': correlations},
                threat_level=ThreatLevel.HIGH,
                suggested_emotion="breakthrough"
            )
            await self.event_processor.event_queue.put(breakthrough_event)

        return {'correlations': correlations}

    async def generate_threat_report(self, investigation_id: str) -> Dict[str, Any]:
        """Generate comprehensive threat report for investigation"""

        if investigation_id not in self.event_processor.active_investigations:
            return {'error': 'Investigation not found'}

        state = self.event_processor.active_investigations[investigation_id]

        # Collect all events for this investigation
        events = [e for e in self.event_processor.processed_events
                 if e.investigation_id == investigation_id]

        # Calculate threat metrics
        max_threat = max([e.threat_level for e in events]) if events else ThreatLevel.NONE

        report = {
            'investigation_id': investigation_id,
            'investigation_type': state.investigation_type.value,
            'duration': (datetime.now() - state.started_at).total_seconds(),
            'findings': {
                'total_events': len(events),
                'breaches_found': state.breaches_found,
                'threats_identified': state.threats_identified,
                'patterns_detected': state.patterns_detected
            },
            'threat_assessment': {
                'level': max_threat.name,
                'score': max_threat.value / 4.0,
                'critical_findings': [e.title for e in events if e.threat_level == ThreatLevel.CRITICAL]
            },
            'timeline': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'event': e.title,
                    'threat_level': e.threat_level.name
                }
                for e in sorted(events, key=lambda x: x.timestamp)
            ]
        }

        return report

    async def handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections from clients"""

        await websocket.accept()
        self.websocket_clients.append(websocket)

        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message['type'] == 'start_investigation':
                    investigation_id = await self.start_investigation(
                        InvestigationType(message['investigation_type']),
                        message['target'],
                        message.get('params')
                    )
                    await websocket.send_text(json.dumps({
                        'type': 'investigation_started',
                        'investigation_id': investigation_id
                    }))

                elif message['type'] == 'get_status':
                    status = await self.get_investigation_status(message['investigation_id'])
                    await websocket.send_text(json.dumps({
                        'type': 'status_update',
                        'data': status
                    }))

                elif message['type'] == 'correlate':
                    correlations = await self.correlate_findings(message['investigation_ids'])
                    await websocket.send_text(json.dumps({
                        'type': 'correlation_results',
                        'data': correlations
                    }))

        except WebSocketDisconnect:
            self.websocket_clients.remove(websocket)
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            if websocket in self.websocket_clients:
                self.websocket_clients.remove(websocket)

    async def shutdown(self):
        """Graceful shutdown"""

        logger.info("Shutting down OSINT Integration Layer...")

        self.running = False

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Close database connections
        await self.db_connector.close()

        # Close WebSocket connections
        if self.avatar_websocket:
            await self.avatar_websocket.close()

        for client in self.websocket_clients:
            await client.close()

        logger.info("OSINT Integration Layer shutdown complete")

# FastAPI application
app = FastAPI(title="OSINT Integration Layer", version="1.0.0")

# Global integration instance
integration: Optional[OSINTIntegrationLayer] = None

@app.on_event("startup")
async def startup_event():
    """Initialize integration layer on startup"""
    global integration

    integration = OSINTIntegrationLayer()
    await integration.initialize()
    logger.info("OSINT Integration Layer ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global integration

    if integration:
        await integration.shutdown()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    if integration:
        await integration.handle_websocket(websocket)

@app.post("/investigation/start")
async def start_investigation(
    investigation_type: str,
    target: str,
    params: Optional[Dict] = None
):
    """Start new OSINT investigation"""

    if not integration:
        raise HTTPException(status_code=503, detail="Service not ready")

    investigation_id = await integration.start_investigation(
        InvestigationType(investigation_type),
        target,
        params
    )

    return {"investigation_id": investigation_id}

@app.get("/investigation/{investigation_id}/status")
async def get_investigation_status(investigation_id: str):
    """Get investigation status"""

    if not integration:
        raise HTTPException(status_code=503, detail="Service not ready")

    status = await integration.get_investigation_status(investigation_id)
    return status

@app.post("/investigation/correlate")
async def correlate_investigations(investigation_ids: List[str]):
    """Correlate findings across investigations"""

    if not integration:
        raise HTTPException(status_code=503, detail="Service not ready")

    correlations = await integration.correlate_findings(investigation_ids)
    return correlations

@app.get("/investigation/{investigation_id}/report")
async def generate_report(investigation_id: str):
    """Generate threat report for investigation"""

    if not integration:
        raise HTTPException(status_code=503, detail="Service not ready")

    report = await integration.generate_threat_report(investigation_id)
    return report

@app.get("/metrics")
async def get_metrics():
    """Get integration layer metrics"""

    if not integration:
        raise HTTPException(status_code=503, detail="Service not ready")

    return integration.metrics

@app.get("/health")
async def health_check():
    """Health check endpoint"""

    return {
        "status": "healthy" if integration and integration.running else "initializing",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8092,
        loop="uvloop",
        log_level="info"
    )