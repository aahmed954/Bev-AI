"""
MCP Server Data Models
======================

Type-safe data models for MCP protocol and OSINT operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum
import uuid


class MCPMessageType(str, Enum):
    """MCP message types according to specification v1.0"""
    INITIALIZE = "initialize"
    INITIALIZED = "initialized"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"
    COMPLETION = "completion"
    LOGGING = "logging"
    PING = "ping"


class SecurityLevel(str, Enum):
    """Security levels for OSINT operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolCategory(str, Enum):
    """OSINT tool categories"""
    COLLECTION = "collection"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    INTELLIGENCE = "intelligence"
    SECURITY = "security"
    CRYPTO = "crypto"


@dataclass
class MCPClientInfo:
    """Client information for MCP connections"""
    name: str
    version: str
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    capabilities: Dict[str, Any] = field(default_factory=dict)
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class MCPServerInfo:
    """Server information for MCP responses"""
    name: str = "BEV-OSINT-MCP-Server"
    version: str = "1.0.0"
    protocol_version: str = "1.0"
    capabilities: Dict[str, Any] = field(default_factory=lambda: {
        "tools": True,
        "resources": True,
        "prompts": True,
        "logging": True,
        "experimental": {
            "websockets": True,
            "real_time_updates": True,
            "background_tasks": True
        }
    })


@dataclass
class MCPMessage:
    """Generic MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPError:
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Dict[str, Any]] = None


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    type: str
    description: str
    required: bool = False
    default: Optional[Any] = None
    enum_values: Optional[List[str]] = None
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


@dataclass
class ToolDefinition:
    """MCP tool definition"""
    name: str
    description: str
    category: ToolCategory
    security_level: SecurityLevel
    parameters: List[ToolParameter] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    rate_limit: int = 60  # requests per minute
    timeout: int = 300  # seconds
    requires_auth: bool = True


@dataclass
class ToolResult:
    """Tool execution result"""
    tool_name: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OSINTTarget:
    """OSINT investigation target"""
    target_id: str
    target_type: Literal["email", "domain", "ip", "phone", "username", "hash", "wallet"]
    value: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class OSINTResult:
    """OSINT investigation result"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target: OSINTTarget
    tool_name: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    risk_score: float = 0.0
    sources: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    ioc_type: Literal["ip", "domain", "hash", "url", "email"]
    value: str
    threat_types: List[str] = field(default_factory=list)
    confidence: float = 0.0
    severity: SecurityLevel = SecurityLevel.MEDIUM
    sources: List[str] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """Neo4j graph node representation"""
    node_id: str
    labels: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """Neo4j graph relationship representation"""
    relationship_id: str
    start_node: str
    end_node: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CryptoTransaction:
    """Cryptocurrency transaction data"""
    transaction_id: str
    blockchain: str
    from_address: str
    to_address: str
    value: float
    currency: str
    timestamp: datetime
    block_height: Optional[int] = None
    confirmations: int = 0
    fee: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityScanResult:
    """Security scan result"""
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target: str
    scan_type: str
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    security_score: float = 0.0
    risk_level: SecurityLevel = SecurityLevel.MEDIUM
    scan_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditLogEntry:
    """Security audit log entry"""
    log_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    client_id: str
    user_id: Optional[str] = None
    action: str
    resource: str
    success: bool
    ip_address: str
    user_agent: str
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_data: Dict[str, Any] = field(default_factory=dict)
    security_level: SecurityLevel = SecurityLevel.MEDIUM


@dataclass
class RateLimitInfo:
    """Rate limiting information"""
    client_id: str
    requests_made: int
    requests_remaining: int
    reset_time: datetime
    window_start: datetime
    window_duration: int  # seconds