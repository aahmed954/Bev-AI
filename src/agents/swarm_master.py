"""
ORACLE1 Swarm Master - Democratic & Hierarchical Swarm Coordination
Supports both democratic consensus and hierarchical command structures
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import redis.asyncio as redis
import structlog
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from scipy.stats import entropy
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure structured logging
logger = structlog.get_logger()

# Metrics
swarm_agents = Gauge('swarm_agents_total', 'Total agents in swarm')
consensus_rounds = Counter('consensus_rounds_total', 'Total consensus rounds')
coordination_latency = Histogram('coordination_latency_seconds', 'Coordination latency')
task_assignments = Counter('task_assignments_total', 'Total task assignments')
swarm_health = Gauge('swarm_health_score', 'Overall swarm health score')

app = FastAPI(title="ORACLE1 Swarm Master", version="3.0.0")


class CoordinationMode(str, Enum):
    """Swarm coordination modes"""
    DEMOCRATIC = "democratic"
    HIERARCHICAL = "hierarchical"
    HYBRID = "hybrid"
    AUTONOMOUS = "autonomous"
    CONSENSUS = "consensus"


class AgentRole(str, Enum):
    """Agent roles in the swarm"""
    LEADER = "leader"
    WORKER = "worker"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    MONITOR = "monitor"
    VALIDATOR = "validator"


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class Agent:
    """Represents an agent in the swarm"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: AgentRole = AgentRole.WORKER
    capabilities: Set[str] = field(default_factory=set)
    performance_score: float = 1.0
    availability: float = 1.0
    current_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    reputation: float = 1.0
    specializations: List[str] = field(default_factory=list)


@dataclass
class SwarmTask:
    """Represents a task to be executed by the swarm"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    requirements: Set[str] = field(default_factory=set)
    payload: Dict[str, Any] = field(default_factory=dict)
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None


class ConsensusAlgorithm:
    """Implements various consensus algorithms for democratic coordination"""

    def __init__(self):
        self.voting_history = defaultdict(list)

    async def raft_consensus(self, agents: List[Agent], proposal: Dict) -> bool:
        """Raft consensus algorithm implementation"""
        # Elect leader
        leader = max(agents, key=lambda a: (a.reputation, a.performance_score))

        # Leader proposes
        votes = []
        for agent in agents:
            if agent.id != leader.id:
                # Simulate voting based on agent characteristics
                vote_weight = agent.reputation * agent.availability
                votes.append(vote_weight > 0.5)

        # Calculate consensus
        approval_rate = sum(votes) / len(votes) if votes else 0
        return approval_rate > 0.5

    async def byzantine_fault_tolerance(self, agents: List[Agent], proposal: Dict) -> bool:
        """Byzantine Fault Tolerant consensus"""
        if len(agents) < 4:
            return await self.simple_majority(agents, proposal)

        # Implement PBFT-like consensus
        faulty_tolerance = (len(agents) - 1) // 3
        required_votes = len(agents) - faulty_tolerance

        votes = []
        for agent in agents:
            # Each agent votes based on their evaluation
            trust_score = agent.reputation * agent.performance_score
            votes.append(trust_score > 0.6)

        return sum(votes) >= required_votes

    async def simple_majority(self, agents: List[Agent], proposal: Dict) -> bool:
        """Simple majority voting"""
        votes = [agent.reputation > 0.5 for agent in agents]
        return sum(votes) > len(votes) / 2

    async def weighted_voting(self, agents: List[Agent], proposal: Dict) -> float:
        """Weighted voting based on agent reputation and performance"""
        if not agents:
            return 0.0

        total_weight = sum(a.reputation * a.performance_score for a in agents)
        if total_weight == 0:
            return 0.0

        weighted_votes = sum(
            (a.reputation * a.performance_score) / total_weight
            for a in agents
            if self._agent_approves(a, proposal)
        )
        return weighted_votes

    def _agent_approves(self, agent: Agent, proposal: Dict) -> bool:
        """Simulate agent approval based on various factors"""
        # Simplified approval logic
        if proposal.get('priority') == TaskPriority.CRITICAL.value:
            return True
        if agent.availability < 0.3:
            return False
        return agent.reputation > 0.5


class HierarchicalCoordinator:
    """Implements hierarchical coordination strategies"""

    def __init__(self):
        self.hierarchy = nx.DiGraph()
        self.command_chain = []

    def build_hierarchy(self, agents: List[Agent]):
        """Build command hierarchy based on agent roles and performance"""
        self.hierarchy.clear()

        # Sort agents by role priority and performance
        sorted_agents = sorted(
            agents,
            key=lambda a: (
                self._role_priority(a.role),
                a.performance_score,
                a.reputation
            ),
            reverse=True
        )

        # Build hierarchy tree
        if sorted_agents:
            root = sorted_agents[0]
            self.hierarchy.add_node(root.id, agent=root)

            # Add remaining agents in layers
            layer_size = 3  # Each node can manage 3 subordinates
            current_layer = [root.id]
            remaining = sorted_agents[1:]

            while remaining and current_layer:
                next_layer = []
                for parent_id in current_layer:
                    for _ in range(min(layer_size, len(remaining))):
                        if not remaining:
                            break
                        child = remaining.pop(0)
                        self.hierarchy.add_node(child.id, agent=child)
                        self.hierarchy.add_edge(parent_id, child.id)
                        next_layer.append(child.id)
                current_layer = next_layer

    def _role_priority(self, role: AgentRole) -> int:
        """Get priority value for agent role"""
        priorities = {
            AgentRole.LEADER: 5,
            AgentRole.COORDINATOR: 4,
            AgentRole.SPECIALIST: 3,
            AgentRole.VALIDATOR: 2,
            AgentRole.WORKER: 1,
            AgentRole.MONITOR: 0
        }
        return priorities.get(role, 0)

    def get_command_chain(self, target_agent_id: str) -> List[str]:
        """Get command chain to reach target agent"""
        if not self.hierarchy.has_node(target_agent_id):
            return []

        # Find path from root to target
        roots = [n for n in self.hierarchy.nodes() if self.hierarchy.in_degree(n) == 0]
        if not roots:
            return []

        try:
            path = nx.shortest_path(self.hierarchy, roots[0], target_agent_id)
            return path
        except nx.NetworkXNoPath:
            return []

    async def delegate_task(self, task: SwarmTask, agents_dict: Dict[str, Agent]) -> List[str]:
        """Delegate task through hierarchy"""
        assigned = []

        # Find suitable agents based on requirements
        suitable_agents = [
            agent_id for agent_id in self.hierarchy.nodes()
            if self._agent_suitable_for_task(agents_dict.get(agent_id), task)
        ]

        # Assign through command chain
        for agent_id in suitable_agents[:3]:  # Assign to up to 3 agents
            command_chain = self.get_command_chain(agent_id)
            if command_chain:
                assigned.append(agent_id)

        return assigned

    def _agent_suitable_for_task(self, agent: Optional[Agent], task: SwarmTask) -> bool:
        """Check if agent is suitable for task"""
        if not agent:
            return False

        # Check capability match
        if task.requirements and not task.requirements.intersection(agent.capabilities):
            return False

        # Check availability
        if agent.availability < 0.3:
            return False

        # Check current load
        if len(agent.current_tasks) >= 5:
            return False

        return True


class SwarmMaster:
    """Main swarm coordination engine"""

    def __init__(self, mode: CoordinationMode = CoordinationMode.HYBRID):
        self.mode = mode
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, SwarmTask] = {}
        self.consensus = ConsensusAlgorithm()
        self.hierarchy = HierarchicalCoordinator()
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.task_queue = asyncio.Queue()
        self.coordination_graph = nx.Graph()
        self._running = False

    async def initialize(self):
        """Initialize swarm master connections"""
        try:
            # Connect to Redis
            self.redis_client = await redis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=True
            )

            # Initialize Kafka
            self.kafka_producer = AIOKafkaProducer(
                bootstrap_servers='localhost:9092',
                value_serializer=lambda v: json.dumps(v).encode()
            )
            await self.kafka_producer.start()

            self.kafka_consumer = AIOKafkaConsumer(
                'swarm-events',
                bootstrap_servers='localhost:9092',
                value_deserializer=lambda m: json.loads(m.decode())
            )
            await self.kafka_consumer.start()

            self._running = True
            logger.info("Swarm master initialized", mode=self.mode)

        except Exception as e:
            logger.error("Failed to initialize swarm master", error=str(e))
            raise

    async def register_agent(self, agent: Agent) -> bool:
        """Register a new agent in the swarm"""
        try:
            self.agents[agent.id] = agent
            swarm_agents.set(len(self.agents))

            # Update coordination graph
            self.coordination_graph.add_node(agent.id, agent=agent)

            # Rebuild hierarchy if in hierarchical mode
            if self.mode in [CoordinationMode.HIERARCHICAL, CoordinationMode.HYBRID]:
                self.hierarchy.build_hierarchy(list(self.agents.values()))

            # Store in Redis
            await self.redis_client.hset(
                f"agent:{agent.id}",
                mapping={
                    "name": agent.name,
                    "role": agent.role.value,
                    "capabilities": json.dumps(list(agent.capabilities)),
                    "performance": agent.performance_score,
                    "reputation": agent.reputation
                }
            )

            # Publish registration event
            await self.kafka_producer.send(
                'swarm-events',
                {
                    'type': 'agent_registered',
                    'agent_id': agent.id,
                    'timestamp': datetime.utcnow().isoformat()
                }
            )

            logger.info("Agent registered", agent_id=agent.id, role=agent.role.value)
            return True

        except Exception as e:
            logger.error("Failed to register agent", error=str(e))
            return False

    async def coordinate_task(self, task: SwarmTask) -> Dict[str, Any]:
        """Coordinate task execution based on current mode"""
        start_time = time.time()

        try:
            self.tasks[task.id] = task
            result = {}

            if self.mode == CoordinationMode.DEMOCRATIC:
                result = await self._democratic_coordination(task)
            elif self.mode == CoordinationMode.HIERARCHICAL:
                result = await self._hierarchical_coordination(task)
            elif self.mode == CoordinationMode.HYBRID:
                result = await self._hybrid_coordination(task)
            elif self.mode == CoordinationMode.CONSENSUS:
                result = await self._consensus_coordination(task)
            else:  # AUTONOMOUS
                result = await self._autonomous_coordination(task)

            # Record metrics
            coordination_latency.observe(time.time() - start_time)
            task_assignments.inc()

            return result

        except Exception as e:
            logger.error("Task coordination failed", task_id=task.id, error=str(e))
            return {"status": "failed", "error": str(e)}

    async def _democratic_coordination(self, task: SwarmTask) -> Dict[str, Any]:
        """Democratic task coordination through voting"""
        eligible_agents = self._find_eligible_agents(task)

        if not eligible_agents:
            return {"status": "no_agents_available"}

        # Create proposal
        proposal = {
            "task_id": task.id,
            "priority": task.priority.value,
            "requirements": list(task.requirements),
            "deadline": task.deadline.isoformat() if task.deadline else None
        }

        # Get consensus
        if await self.consensus.byzantine_fault_tolerance(eligible_agents, proposal):
            # Weighted voting for agent selection
            agent_scores = {}
            for agent in eligible_agents:
                score = await self.consensus.weighted_voting([agent], proposal)
                agent_scores[agent.id] = score

            # Select top agents
            selected_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            task.assigned_agents = [agent_id for agent_id, _ in selected_agents]

            # Update task status
            task.status = "assigned"

            return {
                "status": "coordinated",
                "mode": "democratic",
                "assigned_agents": task.assigned_agents,
                "consensus_score": sum(s for _, s in selected_agents) / len(selected_agents)
            }

        return {"status": "no_consensus"}

    async def _hierarchical_coordination(self, task: SwarmTask) -> Dict[str, Any]:
        """Hierarchical task coordination through command chain"""
        assigned = await self.hierarchy.delegate_task(task, self.agents)

        if assigned:
            task.assigned_agents = assigned
            task.status = "assigned"

            return {
                "status": "coordinated",
                "mode": "hierarchical",
                "assigned_agents": assigned,
                "command_chains": [self.hierarchy.get_command_chain(a) for a in assigned]
            }

        return {"status": "delegation_failed"}

    async def _hybrid_coordination(self, task: SwarmTask) -> Dict[str, Any]:
        """Hybrid coordination combining democratic and hierarchical approaches"""
        # Use hierarchical for critical tasks
        if task.priority == TaskPriority.CRITICAL:
            return await self._hierarchical_coordination(task)

        # Use democratic for normal tasks
        result = await self._democratic_coordination(task)

        # Fall back to hierarchical if democratic fails
        if result.get("status") != "coordinated":
            result = await self._hierarchical_coordination(task)
            result["fallback"] = True

        return result

    async def _consensus_coordination(self, task: SwarmTask) -> Dict[str, Any]:
        """Full consensus coordination using Raft algorithm"""
        eligible_agents = self._find_eligible_agents(task)

        if len(eligible_agents) < 3:
            return {"status": "insufficient_agents"}

        proposal = {
            "task_id": task.id,
            "priority": task.priority.value,
            "agents": [a.id for a in eligible_agents[:5]]
        }

        if await self.consensus.raft_consensus(eligible_agents, proposal):
            task.assigned_agents = proposal["agents"]
            task.status = "assigned"
            consensus_rounds.inc()

            return {
                "status": "coordinated",
                "mode": "consensus",
                "assigned_agents": task.assigned_agents,
                "consensus_type": "raft"
            }

        return {"status": "consensus_failed"}

    async def _autonomous_coordination(self, task: SwarmTask) -> Dict[str, Any]:
        """Autonomous coordination where agents self-organize"""
        # Broadcast task to all eligible agents
        eligible_agents = self._find_eligible_agents(task)

        if not eligible_agents:
            return {"status": "no_agents_available"}

        # Agents bid based on their suitability
        bids = []
        for agent in eligible_agents:
            suitability = self._calculate_suitability(agent, task)
            bids.append((agent.id, suitability))

        # Select highest bidders
        bids.sort(key=lambda x: x[1], reverse=True)
        task.assigned_agents = [bid[0] for bid in bids[:3]]
        task.status = "assigned"

        return {
            "status": "coordinated",
            "mode": "autonomous",
            "assigned_agents": task.assigned_agents,
            "bids": dict(bids[:3])
        }

    def _find_eligible_agents(self, task: SwarmTask) -> List[Agent]:
        """Find agents eligible for a task"""
        eligible = []

        for agent in self.agents.values():
            # Check availability
            if agent.availability < 0.2:
                continue

            # Check capabilities match
            if task.requirements and not task.requirements.intersection(agent.capabilities):
                continue

            # Check workload
            if len(agent.current_tasks) >= 5:
                continue

            eligible.append(agent)

        return eligible

    def _calculate_suitability(self, agent: Agent, task: SwarmTask) -> float:
        """Calculate agent suitability for task"""
        score = 0.0

        # Base score from performance and reputation
        score += agent.performance_score * 0.3
        score += agent.reputation * 0.3

        # Availability bonus
        score += agent.availability * 0.2

        # Capability match bonus
        if task.requirements:
            match_ratio = len(task.requirements.intersection(agent.capabilities)) / len(task.requirements)
            score += match_ratio * 0.2

        # Workload penalty
        workload_penalty = len(agent.current_tasks) / 10
        score -= workload_penalty * 0.1

        return max(0.0, min(1.0, score))

    async def get_swarm_health(self) -> Dict[str, Any]:
        """Get overall swarm health metrics"""
        if not self.agents:
            return {"health": 0.0, "status": "no_agents"}

        # Calculate health metrics
        total_agents = len(self.agents)
        active_agents = sum(1 for a in self.agents.values() if a.availability > 0.5)
        avg_performance = np.mean([a.performance_score for a in self.agents.values()])
        avg_reputation = np.mean([a.reputation for a in self.agents.values()])

        # Check connectivity
        if self.coordination_graph.number_of_nodes() > 1:
            connectivity = nx.edge_connectivity(self.coordination_graph)
        else:
            connectivity = 0

        # Calculate overall health score
        health_score = (
            (active_agents / total_agents) * 0.3 +
            avg_performance * 0.3 +
            avg_reputation * 0.2 +
            min(connectivity / 3, 1.0) * 0.2
        )

        swarm_health.set(health_score)

        return {
            "health": health_score,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "avg_performance": avg_performance,
            "avg_reputation": avg_reputation,
            "connectivity": connectivity,
            "mode": self.mode.value,
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"])
        }

    async def cleanup(self):
        """Cleanup resources"""
        self._running = False

        if self.kafka_producer:
            await self.kafka_producer.stop()
        if self.kafka_consumer:
            await self.kafka_consumer.stop()
        if self.redis_client:
            await self.redis_client.close()


# Global swarm instance
swarm = SwarmMaster()


@app.on_event("startup")
async def startup():
    """Initialize swarm on startup"""
    await swarm.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await swarm.cleanup()


@app.get("/health")
async def health():
    """Health check endpoint"""
    health_status = await swarm.get_swarm_health()
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


class AgentRegistration(BaseModel):
    """Agent registration request"""
    name: str
    role: AgentRole
    capabilities: List[str]
    specializations: List[str] = []


@app.post("/agents/register")
async def register_agent(registration: AgentRegistration):
    """Register a new agent"""
    agent = Agent(
        name=registration.name,
        role=registration.role,
        capabilities=set(registration.capabilities),
        specializations=registration.specializations
    )

    if await swarm.register_agent(agent):
        return {"status": "registered", "agent_id": agent.id}

    raise HTTPException(status_code=500, detail="Registration failed")


class TaskSubmission(BaseModel):
    """Task submission request"""
    type: str
    priority: TaskPriority = TaskPriority.MEDIUM
    requirements: List[str] = []
    payload: Dict[str, Any] = {}
    deadline: Optional[str] = None
    dependencies: List[str] = []


@app.post("/tasks/submit")
async def submit_task(submission: TaskSubmission):
    """Submit a task for coordination"""
    task = SwarmTask(
        type=submission.type,
        priority=submission.priority,
        requirements=set(submission.requirements),
        payload=submission.payload,
        deadline=datetime.fromisoformat(submission.deadline) if submission.deadline else None,
        dependencies=submission.dependencies
    )

    result = await swarm.coordinate_task(task)
    return {"task_id": task.id, **result}


@app.get("/swarm/status")
async def swarm_status():
    """Get swarm status"""
    return {
        "mode": swarm.mode.value,
        "agents": len(swarm.agents),
        "tasks": len(swarm.tasks),
        "health": await swarm.get_swarm_health()
    }


@app.post("/swarm/mode")
async def set_coordination_mode(mode: CoordinationMode):
    """Change coordination mode"""
    swarm.mode = mode

    # Rebuild hierarchy if needed
    if mode in [CoordinationMode.HIERARCHICAL, CoordinationMode.HYBRID]:
        swarm.hierarchy.build_hierarchy(list(swarm.agents.values()))

    return {"mode": mode.value, "status": "updated"}


@app.websocket("/ws/swarm")
async def swarm_websocket(websocket: WebSocket):
    """WebSocket for real-time swarm updates"""
    await websocket.accept()

    try:
        while True:
            # Send periodic updates
            status = await swarm.get_swarm_health()
            await websocket.send_json(status)
            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)