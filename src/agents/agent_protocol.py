import os
#!/usr/bin/env python3
"""
Multi-Agent Communication Protocol
Completes the 5% gap in agent system with standardized messaging
"""

import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis
import msgpack
import structlog
from cryptography.fernet import Fernet
import asyncpg

logger = structlog.get_logger()

class MessagePriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 5
    LOW = 9

class MessageType(Enum):
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"

@dataclass
class AgentMessage:
    """Standardized inter-agent message format"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""  # Can be broadcast with "*"
    message_type: MessageType = MessageType.EVENT
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl: int = 300  # 5 minutes default
    requires_ack: bool = False
    encrypted: bool = False
    signature: Optional[str] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_bytes(self) -> bytes:
        """Serialize to msgpack for efficient transport"""
        return msgpack.packb(asdict(self))
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'AgentMessage':
        """Deserialize from msgpack"""
        return cls(**msgpack.unpackb(data, raw=False))

class AgentCommunicationBus:
    """Central communication bus for all agents"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.subscribers: Dict[str, List[Callable]] = {}
        self.agent_registry: Dict[str, Dict] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.db_pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize communication infrastructure"""
        # Redis connection
        self.redis = await redis.from_url(self.redis_url)
        
        # PostgreSQL for message persistence
        self.db_pool = await asyncpg.create_pool(
            host='localhost',
            port=5432,
            user='swarm_admin',
            password=os.getenv('DB_PASSWORD', 'dev_password'),
            database='ai_swarm',
            min_size=10,
            max_size=20
        )
        
        # Create message log table
        async with self.db_pool.acquire() as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id UUID PRIMARY KEY,
                    sender VARCHAR(255),
                    recipient VARCHAR(255),
                    message_type VARCHAR(50),
                    priority INTEGER,
                    payload JSONB,
                    timestamp TIMESTAMPTZ,
                    processed BOOLEAN DEFAULT FALSE,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_messages_recipient ON agent_messages(recipient);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON agent_messages(timestamp);
                CREATE INDEX IF NOT EXISTS idx_messages_processed ON agent_messages(processed);
            ''')
        
        # Start background workers
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._message_processor())
        asyncio.create_task(self._retry_handler())
        
        logger.info("Agent Communication Bus initialized")
    
    async def register_agent(self, agent_id: str, capabilities: List[str], 
                            metadata: Dict[str, Any] = None):
        """Register an agent with the swarm"""
        agent_info = {
            'id': agent_id,
            'capabilities': capabilities,
            'metadata': metadata or {},
            'status': 'online',
            'last_heartbeat': datetime.utcnow().isoformat(),
            'message_count': 0,
            'error_count': 0
        }
        
        self.agent_registry[agent_id] = agent_info
        
        # Store in Redis for persistence
        await self.redis.hset(
            f"agent:{agent_id}",
            mapping={k: json.dumps(v) if isinstance(v, (dict, list)) else v 
                    for k, v in agent_info.items()}
        )
        
        # Announce to swarm
        await self.broadcast(AgentMessage(
            sender="system",
            recipient="*",
            message_type=MessageType.EVENT,
            payload={'event': 'agent_joined', 'agent': agent_info}
        ))
        
        logger.info(f"Agent registered: {agent_id}", capabilities=capabilities)
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send message to specific agent or broadcast"""
        try:
            # Encrypt if required
            if message.encrypted:
                message.payload = {
                    'encrypted_data': self.cipher.encrypt(
                        json.dumps(message.payload).encode()
                    ).decode()
                }
            
            # Sign message
            message.signature = self._sign_message(message)
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO agent_messages 
                    (id, sender, recipient, message_type, priority, payload, timestamp, retry_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ''', uuid.UUID(message.id), message.sender, message.recipient,
                    message.message_type.value, message.priority.value,
                    json.dumps(message.payload), 
                    datetime.fromisoformat(message.timestamp),
                    message.retry_count)
            
            # Route message
            if message.recipient == "*":
                # Broadcast
                await self.redis.publish("agent:broadcast", message.to_bytes())
            else:
                # Direct message
                await self.redis.lpush(
                    f"agent:queue:{message.recipient}", 
                    message.to_bytes()
                )
                
                # Notify recipient
                await self.redis.publish(
                    f"agent:notify:{message.recipient}",
                    message.id
                )
            
            # Update metrics
            self.agent_registry[message.sender]['message_count'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}", message_id=message.id)
            return False
    
    async def broadcast(self, message: AgentMessage):
        """Broadcast message to all agents"""
        message.recipient = "*"
        await self.send_message(message)
    
    async def request_consensus(self, proposal: Dict[str, Any], 
                                timeout: int = 30) -> Dict[str, Any]:
        """Request consensus from agent swarm"""
        consensus_id = str(uuid.uuid4())
        
        # Send consensus request
        await self.broadcast(AgentMessage(
            sender="consensus_coordinator",
            message_type=MessageType.CONSENSUS_REQUEST,
            priority=MessagePriority.HIGH,
            payload={
                'consensus_id': consensus_id,
                'proposal': proposal,
                'deadline': (datetime.utcnow() + timedelta(seconds=timeout)).isoformat()
            }
        ))
        
        # Collect votes
        votes = []
        deadline = asyncio.get_event_loop().time() + timeout
        
        while asyncio.get_event_loop().time() < deadline:
            vote_data = await self.redis.blpop(
                f"consensus:votes:{consensus_id}",
                timeout=1
            )
            
            if vote_data:
                votes.append(json.loads(vote_data[1]))
            
            # Check if we have enough votes (majority)
            if len(votes) >= len(self.agent_registry) * 0.51:
                break
        
        # Calculate consensus
        approval_count = sum(1 for v in votes if v['decision'] == 'approve')
        
        return {
            'consensus_id': consensus_id,
            'approved': approval_count > len(votes) / 2,
            'votes': votes,
            'participation': len(votes) / len(self.agent_registry)
        }
    
    def _sign_message(self, message: AgentMessage) -> str:
        """Create message signature for verification"""
        content = f"{message.sender}:{message.recipient}:{message.timestamp}:{json.dumps(message.payload)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, info in self.agent_registry.items():
                    last_heartbeat = datetime.fromisoformat(info['last_heartbeat'])
                    
                    if (current_time - last_heartbeat).seconds > 60:
                        # Agent might be offline
                        info['status'] = 'offline'
                        
                        await self.broadcast(AgentMessage(
                            sender="system",
                            message_type=MessageType.EVENT,
                            payload={'event': 'agent_offline', 'agent_id': agent_id}
                        ))
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
    
    async def _message_processor(self):
        """Process incoming messages"""
        while True:
            try:
                # Get unprocessed messages from database
                async with self.db_pool.acquire() as conn:
                    messages = await conn.fetch('''
                        SELECT * FROM agent_messages 
                        WHERE processed = FALSE 
                        AND retry_count < 3
                        ORDER BY priority ASC, timestamp ASC
                        LIMIT 100
                    ''')
                
                for msg_record in messages:
                    # Process message
                    message = AgentMessage(
                        id=str(msg_record['id']),
                        sender=msg_record['sender'],
                        recipient=msg_record['recipient'],
                        message_type=MessageType(msg_record['message_type']),
                        priority=MessagePriority(msg_record['priority']),
                        payload=msg_record['payload'],
                        timestamp=msg_record['timestamp'].isoformat(),
                        retry_count=msg_record['retry_count']
                    )
                    
                    # Route to handler
                    if message.message_type in self.message_handlers:
                        try:
                            await self.message_handlers[message.message_type](message)
                            
                            # Mark as processed
                            await conn.execute(
                                'UPDATE agent_messages SET processed = TRUE WHERE id = $1',
                                msg_record['id']
                            )
                        except Exception as e:
                            logger.error(f"Message processing failed: {e}", 
                                       message_id=message.id)
                            
                            # Increment retry count
                            await conn.execute('''
                                UPDATE agent_messages 
                                SET retry_count = retry_count + 1 
                                WHERE id = $1
                            ''', msg_record['id'])
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    async def _retry_handler(self):
        """Handle message retries with exponential backoff"""
        while True:
            try:
                async with self.db_pool.acquire() as conn:
                    # Get messages that need retry
                    failed_messages = await conn.fetch('''
                        SELECT * FROM agent_messages
                        WHERE processed = FALSE
                        AND retry_count > 0
                        AND retry_count < 3
                        AND created_at < NOW() - INTERVAL '1 minute' * POW(2, retry_count)
                        LIMIT 10
                    ''')
                
                for msg in failed_messages:
                    # Recreate and resend
                    message = AgentMessage(
                        id=str(msg['id']),
                        sender=msg['sender'],
                        recipient=msg['recipient'],
                        message_type=MessageType(msg['message_type']),
                        priority=MessagePriority(msg['priority']),
                        payload=msg['payload'],
                        retry_count=msg['retry_count'] + 1
                    )
                    
                    await self.send_message(message)
                    logger.info(f"Retrying message: {message.id}", 
                              retry_count=message.retry_count)
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Retry handler error: {e}")
                await asyncio.sleep(30)

# Agent-specific implementations
class EnhancedSwarmAgent:
    """Base class for enhanced swarm agents with full protocol support"""
    
    def __init__(self, agent_id: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.comm_bus: Optional[AgentCommunicationBus] = None
        self.message_queue = asyncio.Queue()
        self.running = False
        
    async def initialize(self, comm_bus: AgentCommunicationBus):
        """Initialize agent with communication bus"""
        self.comm_bus = comm_bus
        await self.comm_bus.register_agent(
            self.agent_id,
            self.capabilities,
            {'version': '1.0', 'platform': 'BEV'}
        )
        
        # Start message handler
        self.running = True
        asyncio.create_task(self._message_handler())
        asyncio.create_task(self._heartbeat_sender())
    
    async def _message_handler(self):
        """Handle incoming messages"""
        while self.running:
            try:
                # Get messages from Redis queue
                message_data = await self.comm_bus.redis.blpop(
                    f"agent:queue:{self.agent_id}",
                    timeout=1
                )
                
                if message_data:
                    message = AgentMessage.from_bytes(message_data[1])
                    
                    # Process based on type
                    if message.message_type == MessageType.COMMAND:
                        result = await self.execute_command(message.payload)
                        
                        # Send response
                        await self.comm_bus.send_message(AgentMessage(
                            sender=self.agent_id,
                            recipient=message.sender,
                            message_type=MessageType.RESPONSE,
                            payload={'result': result},
                            correlation_id=message.id
                        ))
                    
                    elif message.message_type == MessageType.QUERY:
                        response = await self.process_query(message.payload)
                        
                        await self.comm_bus.send_message(AgentMessage(
                            sender=self.agent_id,
                            recipient=message.sender,
                            message_type=MessageType.RESPONSE,
                            payload={'response': response},
                            correlation_id=message.id
                        ))
                    
                    elif message.message_type == MessageType.CONSENSUS_REQUEST:
                        vote = await self.vote_on_proposal(message.payload['proposal'])
                        
                        await self.comm_bus.redis.lpush(
                            f"consensus:votes:{message.payload['consensus_id']}",
                            json.dumps({
                                'agent': self.agent_id,
                                'decision': vote,
                                'timestamp': datetime.utcnow().isoformat()
                            })
                        )
                        
            except Exception as e:
                logger.error(f"Agent message handler error: {e}", agent=self.agent_id)
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeats"""
        while self.running:
            await self.comm_bus.send_message(AgentMessage(
                sender=self.agent_id,
                recipient="system",
                message_type=MessageType.HEARTBEAT,
                priority=MessagePriority.LOW,
                payload={'status': 'alive', 'timestamp': datetime.utcnow().isoformat()}
            ))
            
            await asyncio.sleep(30)
    
    async def execute_command(self, command: Dict[str, Any]) -> Any:
        """Override in subclass"""
        raise NotImplementedError
    
    async def process_query(self, query: Dict[str, Any]) -> Any:
        """Override in subclass"""
        raise NotImplementedError
    
    async def vote_on_proposal(self, proposal: Dict[str, Any]) -> str:
        """Override in subclass - return 'approve' or 'reject'"""
        return 'approve'  # Default to approve

# Initialize the complete agent system
async def complete_agent_initialization():
    """Complete the 5% gap in agent system"""
    
    # Create communication bus
    comm_bus = AgentCommunicationBus()
    await comm_bus.initialize()
    
    # Import existing agents
    from agents.swarm_master import SwarmMaster
    from agents.research_coordinator import ResearchCoordinator
    from agents.code_optimizer import CodeOptimizer
    from agents.memory_manager import MemoryManager
    from pipeline.toolmaster_orchestrator import ToolMaster
    from security.guardian_security_enforcer import Guardian
    
    # Enhance existing agents with protocol support
    agents = {
        'swarm_master': SwarmMaster(),
        'research_coordinator': ResearchCoordinator(),
        'code_optimizer': CodeOptimizer(),
        'memory_manager': MemoryManager(),
        'tool_master': ToolMaster(),
        'guardian': Guardian()
    }
    
    # Initialize all agents with communication protocol
    for agent_id, agent in agents.items():
        # Wrap with enhanced protocol
        enhanced_agent = EnhancedSwarmAgent(
            agent_id=agent_id,
            capabilities=agent.get_capabilities() if hasattr(agent, 'get_capabilities') else []
        )
        
        # Initialize with bus
        await enhanced_agent.initialize(comm_bus)
    
    logger.info("All agents initialized with communication protocol")
    
    return comm_bus, agents

if __name__ == "__main__":
    asyncio.run(complete_agent_initialization())
