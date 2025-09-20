"""
MCP Server Implementation
========================

Main MCP server with WebSocket support, performance optimizations, and background processing.
"""

import asyncio
import json
import logging
import time
import uvloop
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import asdict
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import signal
import sys
from prometheus_client import Counter, Histogram, Gauge, generate_latest

from .models import (
    MCPMessage, MCPError, MCPClientInfo, MCPServerInfo, OSINTTarget,
    ToolResult, SecurityLevel, MCPMessageType
)
from .security import SecurityManager, SecurityConfig, AuthenticationError, AuthorizationError, RateLimitError
from .database import DatabaseManager, DatabaseConfig
from .tools import OSINTToolRegistry


logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'status'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active WebSocket connections')
TOOL_EXECUTIONS = Counter('mcp_tool_executions_total', 'Tool executions', ['tool_name', 'status'])
ERROR_COUNT = Counter('mcp_errors_total', 'Total errors', ['error_type'])


class MCPProtocolHandler:
    """Handles MCP protocol messages and routing"""
    
    def __init__(self, server_info: MCPServerInfo, tool_registry: OSINTToolRegistry,
                 security_manager: SecurityManager, db_manager: DatabaseManager):
        self.server_info = server_info
        self.tool_registry = tool_registry
        self.security_manager = security_manager
        self.db_manager = db_manager
    
    async def handle_message(self, message: MCPMessage, client_info: MCPClientInfo,
                           websocket: Optional[WebSocket] = None) -> MCPMessage:
        """Handle incoming MCP message"""
        try:
            if message.method == MCPMessageType.INITIALIZE:
                return await self._handle_initialize(message, client_info)
            
            elif message.method == MCPMessageType.TOOLS_LIST:
                return await self._handle_tools_list(message, client_info)
            
            elif message.method == MCPMessageType.TOOLS_CALL:
                return await self._handle_tools_call(message, client_info)
            
            elif message.method == MCPMessageType.RESOURCES_LIST:
                return await self._handle_resources_list(message, client_info)
            
            elif message.method == MCPMessageType.RESOURCES_READ:
                return await self._handle_resources_read(message, client_info)
            
            elif message.method == MCPMessageType.PING:
                return await self._handle_ping(message, client_info)
            
            else:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": -32601,
                        "message": f"Method '{message.method}' not found"
                    }
                )
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            ERROR_COUNT.labels(error_type="protocol_error").inc()
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": "Internal error",
                    "data": {"error": str(e)}
                }
            )
    
    async def _handle_initialize(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle initialize request"""
        client_capabilities = message.params.get("capabilities", {}) if message.params else {}
        client_info.capabilities = client_capabilities
        
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": self.server_info.protocol_version,
                "capabilities": self.server_info.capabilities,
                "serverInfo": {
                    "name": self.server_info.name,
                    "version": self.server_info.version
                }
            }
        )
    
    async def _handle_tools_list(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle tools list request"""
        tool_definitions = self.tool_registry.get_tool_definitions()
        
        tools = []
        for tool_def in tool_definitions:
            tools.append({
                "name": tool_def.name,
                "description": tool_def.description,
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description,
                            **({"enum": param.enum_values} if param.enum_values else {}),
                            **({"pattern": param.pattern} if param.pattern else {}),
                            **({"minLength": param.min_length} if param.min_length else {}),
                            **({"maxLength": param.max_length} if param.max_length else {})
                        }
                        for param in tool_def.parameters
                    },
                    "required": [p.name for p in tool_def.parameters if p.required]
                }
            })
        
        return MCPMessage(
            id=message.id,
            result={"tools": tools}
        )
    
    async def _handle_tools_call(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle tool call request"""
        if not message.params:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": "Invalid params"}
            )
        
        tool_name = message.params.get("name")
        arguments = message.params.get("arguments", {})
        
        if not tool_name:
            return MCPMessage(
                id=message.id,
                error={"code": -32602, "message": "Tool name required"}
            )
        
        try:
            # Create target from arguments
            target = OSINTTarget(
                target_id=f"{tool_name}_{int(time.time())}",
                target_type=arguments.get("target_type", "unknown"),
                value=arguments.get("target_value", ""),
                metadata={"client_id": client_info.client_id}
            )
            
            # Execute tool
            with REQUEST_DURATION.time():
                result = await self.tool_registry.execute_tool(
                    tool_name, target, arguments, client_info.client_id
                )
            
            TOOL_EXECUTIONS.labels(
                tool_name=tool_name,
                status="success" if result.success else "error"
            ).inc()
            
            if result.success:
                return MCPMessage(
                    id=message.id,
                    result={
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result.data, indent=2, default=str)
                            }
                        ],
                        "isError": False
                    }
                )
            else:
                return MCPMessage(
                    id=message.id,
                    result={
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Tool execution failed: {result.error}"
                            }
                        ],
                        "isError": True
                    }
                )
        
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            ERROR_COUNT.labels(error_type="tool_execution").inc()
            return MCPMessage(
                id=message.id,
                error={
                    "code": -32603,
                    "message": "Tool execution failed",
                    "data": {"error": str(e)}
                }
            )
    
    async def _handle_resources_list(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle resources list request"""
        # Placeholder for resource management
        return MCPMessage(
            id=message.id,
            result={"resources": []}
        )
    
    async def _handle_resources_read(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle resource read request"""
        # Placeholder for resource reading
        return MCPMessage(
            id=message.id,
            result={"contents": []}
        )
    
    async def _handle_ping(self, message: MCPMessage, client_info: MCPClientInfo) -> MCPMessage:
        """Handle ping request"""
        return MCPMessage(
            id=message.id,
            result={"pong": True, "timestamp": datetime.now().isoformat()}
        )


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_info: Dict[str, MCPClientInfo] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, client_info: MCPClientInfo):
        """Add new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_info[client_id] = client_info
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_info:
            del self.client_info[client_id]
        ACTIVE_CONNECTIONS.set(len(self.active_connections))
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: MCPMessage):
        """Send message to specific client"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_text(json.dumps(asdict(message), default=str))
            except (ConnectionClosedError, ConnectionClosedOK):
                self.disconnect(client_id)
    
    async def broadcast(self, message: MCPMessage, exclude: Optional[Set[str]] = None):
        """Broadcast message to all clients"""
        if exclude is None:
            exclude = set()
        
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await websocket.send_text(json.dumps(asdict(message), default=str))
                except (ConnectionClosedError, ConnectionClosedOK):
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)


class BackgroundTaskManager:
    """Manages background tasks for performance optimization"""
    
    def __init__(self, security_manager: SecurityManager, db_manager: DatabaseManager):
        self.security_manager = security_manager
        self.db_manager = db_manager
        self.running_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start background tasks"""
        tasks = [
            self._session_cleanup_task(),
            self._cache_optimization_task(),
            self._health_check_task(),
            self._metrics_collection_task()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.running_tasks.add(task)
            # Remove completed tasks
            task.add_done_callback(self.running_tasks.discard)
        
        logger.info("Background tasks started")
    
    async def stop(self):
        """Stop all background tasks"""
        self.shutdown_event.set()
        
        # Cancel all running tasks
        for task in self.running_tasks:
            task.cancel()
        
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks, return_exceptions=True)
        
        logger.info("Background tasks stopped")
    
    async def _session_cleanup_task(self):
        """Clean up expired sessions periodically"""
        while not self.shutdown_event.is_set():
            try:
                await self.security_manager.cleanup_expired_sessions()
                await asyncio.sleep(300)  # 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _cache_optimization_task(self):
        """Optimize cache usage periodically"""
        while not self.shutdown_event.is_set():
            try:
                # Implement cache optimization logic here
                await asyncio.sleep(600)  # 10 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _health_check_task(self):
        """Perform health checks periodically"""
        while not self.shutdown_event.is_set():
            try:
                health = await self.db_manager.health_check()
                logger.info(f"Health check: {health}")
                await asyncio.sleep(60)  # 1 minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_task(self):
        """Collect custom metrics periodically"""
        while not self.shutdown_event.is_set():
            try:
                # Collect custom metrics
                await asyncio.sleep(30)  # 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)


class MCPServer:
    """Main MCP Server implementation"""
    
    def __init__(self, security_config: SecurityConfig, db_config: DatabaseConfig,
                 host: str = "0.0.0.0", port: int = 3010):
        self.host = host
        self.port = port
        self.security_config = security_config
        self.db_config = db_config
        
        # Core components
        self.db_manager: Optional[DatabaseManager] = None
        self.security_manager: Optional[SecurityManager] = None
        self.tool_registry: Optional[OSINTToolRegistry] = None
        self.protocol_handler: Optional[MCPProtocolHandler] = None
        self.connection_manager = ConnectionManager()
        self.background_tasks: Optional[BackgroundTaskManager] = None
        
        # FastAPI app
        self.app = FastAPI(
            title="BEV OSINT MCP Server",
            description="Secure MCP server for OSINT operations",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            if not self.db_manager:
                raise HTTPException(status_code=503, detail="Server not initialized")
            
            health = await self.db_manager.health_check()
            all_healthy = all(health.values())
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "services": health
            }
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            return generate_latest()
        
        @self.app.get("/info")
        async def server_info():
            """Server information endpoint"""
            if not self.protocol_handler:
                raise HTTPException(status_code=503, detail="Server not initialized")
            
            return asdict(self.protocol_handler.server_info)
        
        @self.app.websocket("/mcp")
        async def websocket_endpoint(websocket: WebSocket):
            """Main MCP WebSocket endpoint"""
            await self._handle_websocket(websocket)
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connection"""
        client_id = None
        
        try:
            # Wait for initial message with authentication
            await websocket.accept()
            
            # Expect authentication message first
            auth_data = await websocket.receive_text()
            auth_message = json.loads(auth_data)
            
            # Extract authentication token
            token = auth_message.get("token")
            if not token:
                await websocket.close(code=4001, reason="Authentication required")
                return
            
            # Authenticate client
            try:
                client_info = websocket.client
                payload = await self.security_manager.authenticate_request(
                    token=token,
                    client_ip=client_info.host if client_info else "unknown",
                    user_agent="websocket"
                )
                
                client_id = payload["client_id"]
                client_info_obj = MCPClientInfo(
                    name=payload.get("client_name", "Unknown"),
                    version="1.0.0",
                    client_id=client_id
                )
                
                await self.connection_manager.connect(websocket, client_id, client_info_obj)
                
                # Send connection success
                success_msg = MCPMessage(
                    id="auth_success",
                    result={"authenticated": True, "client_id": client_id}
                )
                await websocket.send_text(json.dumps(asdict(success_msg), default=str))
                
            except (AuthenticationError, AuthorizationError) as e:
                await websocket.close(code=4003, reason=str(e))
                return
            
            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message_data = json.loads(data)
                    
                    # Convert to MCPMessage
                    message = MCPMessage(**message_data)
                    
                    REQUEST_COUNT.labels(
                        method=message.method or "unknown",
                        status="received"
                    ).inc()
                    
                    # Handle message
                    response = await self.protocol_handler.handle_message(
                        message, client_info_obj, websocket
                    )
                    
                    # Send response
                    await websocket.send_text(json.dumps(asdict(response), default=str))
                    
                    REQUEST_COUNT.labels(
                        method=message.method or "unknown",
                        status="responded"
                    ).inc()
                
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    error_msg = MCPMessage(
                        error={"code": -32700, "message": "Parse error"}
                    )
                    await websocket.send_text(json.dumps(asdict(error_msg), default=str))
                except Exception as e:
                    logger.error(f"WebSocket message handling error: {e}")
                    error_msg = MCPMessage(
                        error={"code": -32603, "message": "Internal error"}
                    )
                    await websocket.send_text(json.dumps(asdict(error_msg), default=str))
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        
        finally:
            if client_id:
                self.connection_manager.disconnect(client_id)
    
    async def initialize(self):
        """Initialize server components"""
        logger.info("Initializing MCP server...")
        
        # Initialize database connections
        self.db_manager = DatabaseManager(self.db_config, None)  # Security manager not ready yet
        await self.db_manager.initialize_all()
        
        # Initialize security manager
        self.security_manager = SecurityManager(self.security_config, self.db_manager.redis.client)
        
        # Update database manager with security manager
        self.db_manager.security = self.security_manager
        self.db_manager.postgres.security = self.security_manager
        self.db_manager.neo4j.security = self.security_manager
        
        # Initialize tool registry
        self.tool_registry = OSINTToolRegistry(self.db_manager, self.security_manager)
        
        # Initialize protocol handler
        server_info = MCPServerInfo()
        self.protocol_handler = MCPProtocolHandler(
            server_info, self.tool_registry, self.security_manager, self.db_manager
        )
        
        # Start background tasks
        self.background_tasks = BackgroundTaskManager(self.security_manager, self.db_manager)
        await self.background_tasks.start()
        
        logger.info("MCP server initialized successfully")
    
    async def shutdown(self):
        """Shutdown server components"""
        logger.info("Shutting down MCP server...")
        
        if self.background_tasks:
            await self.background_tasks.stop()
        
        if self.db_manager:
            await self.db_manager.close_all()
        
        logger.info("MCP server shutdown complete")
    
    async def run(self):
        """Run the server"""
        # Set up uvloop for better performance
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        # Initialize server
        await self.initialize()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            asyncio.create_task(self.shutdown())
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run server
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True,
            loop="uvloop"
        )
        
        server = uvicorn.Server(config)
        
        try:
            await server.serve()
        finally:
            await self.shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager"""
    # Startup
    logger.info("Starting up...")
    yield
    # Shutdown
    logger.info("Shutting down...")


# Factory function for creating server instance
def create_server(security_config: SecurityConfig, db_config: DatabaseConfig,
                 host: str = "0.0.0.0", port: int = 3010) -> MCPServer:
    """Create MCP server instance"""
    return MCPServer(security_config, db_config, host, port)