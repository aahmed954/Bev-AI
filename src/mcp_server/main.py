"""
MCP Server Entry Point
=====================

Main entry point for the BEV OSINT MCP Server with configuration loading.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional
import uvloop
from cryptography.fernet import Fernet

from .server import create_server
from .security import SecurityConfig
from .database import DatabaseConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/mcp_server.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


def load_config_from_env() -> tuple[SecurityConfig, DatabaseConfig]:
    """Load configuration from environment variables"""
    
    # Security configuration
    jwt_secret = os.getenv('JWT_SECRET', 'BevJWTSecret2024ForTokenSigning')
    if len(jwt_secret) < 32:
        logger.warning("JWT secret is too short, generating new one")
        jwt_secret = Fernet.generate_key().decode()
    
    encryption_key = os.getenv('DATA_ENCRYPTION_KEY', 'BevDataEncryptionKey2024VerySecure32Chars')
    if len(encryption_key) < 32:
        logger.warning("Encryption key is too short, generating new one")
        encryption_key = Fernet.generate_key().decode()
    
    allowed_networks = os.getenv('ALLOWED_NETWORKS', '172.30.0.0/16,127.0.0.1/32').split(',')
    
    security_config = SecurityConfig(
        jwt_secret=jwt_secret,
        jwt_algorithm=os.getenv('JWT_ALGORITHM', 'HS256'),
        jwt_expiry_hours=int(os.getenv('JWT_EXPIRY_HOURS', '24')),
        rate_limit_requests=int(os.getenv('API_RATE_LIMIT', '100')),
        rate_limit_window=int(os.getenv('API_RATE_WINDOW', '60')),
        max_request_size=int(os.getenv('MAX_REQUEST_SIZE', '10000000')),
        allowed_networks=allowed_networks,
        encryption_key=encryption_key,
        password_min_length=int(os.getenv('PASSWORD_MIN_LENGTH', '12')),
        session_timeout=int(os.getenv('SESSION_TIMEOUT', '3600')),
        audit_log_retention_days=int(os.getenv('AUDIT_LOG_RETENTION_DAYS', '90'))
    )
    
    # Database configuration
    postgres_uri = os.getenv('POSTGRES_URI', 'postgresql://bev_admin:BevSecureDB2024!@172.21.0.2:5432/osint')
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://172.21.0.3:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD', 'BevGraph2024!')
    
    db_config = DatabaseConfig(
        postgres_uri=postgres_uri,
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        redis_host=os.getenv('REDIS_HOST', '172.21.0.4'),
        redis_port=int(os.getenv('REDIS_PORT', '6379')),
        redis_password=os.getenv('REDIS_PASSWORD', 'BevCache2024!'),
        elasticsearch_host=os.getenv('ELASTICSEARCH_HOST', '172.21.0.5'),
        elasticsearch_port=int(os.getenv('ELASTICSEARCH_PORT', '9200')),
        postgres_min_connections=int(os.getenv('POSTGRES_MIN_CONNECTIONS', '5')),
        postgres_max_connections=int(os.getenv('POSTGRES_MAX_CONNECTIONS', '20')),
        redis_max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', '20')),
        connection_timeout=int(os.getenv('CONNECTION_TIMEOUT', '30')),
        query_timeout=int(os.getenv('QUERY_TIMEOUT', '300'))
    )
    
    return security_config, db_config


def validate_environment():
    """Validate environment and dependencies"""
    logger.info("Validating environment...")
    
    # Check required environment variables
    required_vars = [
        'POSTGRES_URI', 'NEO4J_URI', 'NEO4J_USER', 'NEO4J_PASSWORD',
        'REDIS_HOST', 'REDIS_PASSWORD', 'JWT_SECRET'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    # Check file permissions
    log_dir = Path('/var/log')
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.warning("Cannot create log directory, will log to stdout only")
    
    # Check network connectivity (basic)
    try:
        import socket
        
        # Test database connections
        postgres_host = os.getenv('POSTGRES_HOST', '172.21.0.2')
        postgres_port = int(os.getenv('POSTGRES_PORT', '5432'))
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((postgres_host, postgres_port))
        sock.close()
        
        if result != 0:
            logger.warning(f"Cannot connect to PostgreSQL at {postgres_host}:{postgres_port}")
    
    except Exception as e:
        logger.warning(f"Network connectivity check failed: {e}")
    
    logger.info("Environment validation complete")
    return True


async def main():
    """Main entry point"""
    logger.info("Starting BEV OSINT MCP Server...")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        sys.exit(1)
    
    # Load configuration
    try:
        security_config, db_config = load_config_from_env()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Get server settings
    host = os.getenv('MCP_SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('MCP_SERVER_PORT', '3010'))
    
    # Create and run server
    try:
        logger.info(f"Creating MCP server on {host}:{port}")
        server = create_server(security_config, db_config, host, port)
        
        logger.info("Starting MCP server...")
        await server.run()
    
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    
    logger.info("MCP server stopped")


def run_server():
    """Synchronous entry point for running the server"""
    # Use uvloop for better performance
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_server()