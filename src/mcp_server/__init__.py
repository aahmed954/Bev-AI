"""
BEV OSINT Framework - Model Context Protocol (MCP) Server
=========================================================

Secure MCP server implementation with enhanced security features and OSINT tool integration.

Security Features:
- JWT-based authentication
- Input validation and sanitization
- SQL injection prevention
- Rate limiting (100 requests/minute)
- Command injection protection
- Comprehensive audit logging
- Encrypted credential management

Performance Features:
- Async/await operations
- Database connection pooling
- Redis caching
- Background task processing
- WebSocket support

Integration:
- PostgreSQL, Neo4j, Redis, Elasticsearch
- 8 specialized OSINT tools
- BEV infrastructure components
"""

__version__ = "1.0.0"
__author__ = "BEV OSINT Framework"
__license__ = "MIT"

from .server import MCPServer
from .auth import AuthManager
from .tools import OSINTToolRegistry
from .security import SecurityManager

__all__ = [
    "MCPServer",
    "AuthManager", 
    "OSINTToolRegistry",
    "SecurityManager"
]