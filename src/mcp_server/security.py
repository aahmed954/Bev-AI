"""
Security Manager for MCP Server
===============================

Comprehensive security implementation including authentication, authorization,
input validation, rate limiting, and audit logging.
"""

import asyncio
import hashlib
import hmac
import jwt
import re
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from functools import wraps
import logging
from cryptography.fernet import Fernet
import redis.asyncio as aioredis
import ipaddress

from .models import (
    SecurityLevel, AuditLogEntry, RateLimitInfo, MCPClientInfo,
    ToolDefinition
)


logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related errors"""
    pass


class AuthenticationError(SecurityError):
    """Authentication failures"""
    pass


class AuthorizationError(SecurityError):
    """Authorization failures"""
    pass


class RateLimitError(SecurityError):
    """Rate limiting violations"""
    pass


class InputValidationError(SecurityError):
    """Input validation failures"""
    pass


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    max_request_size: int = 10_000_000  # 10MB
    allowed_networks: List[str] = None
    encryption_key: Optional[str] = None
    password_min_length: int = 12
    session_timeout: int = 3600  # seconds
    audit_log_retention_days: int = 90


class InputValidator:
    """Input validation and sanitization"""
    
    # Regex patterns for validation
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    DOMAIN_PATTERN = re.compile(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$')
    IP_PATTERN = re.compile(r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    HASH_PATTERN = re.compile(r'^[a-fA-F0-9]+$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]{3,50}$')
    
    # Dangerous patterns that could indicate injection attacks
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(--|\#|/\*|\*/)",
        r"(\bxp_cmdshell\b|\bsp_executesql\b)"
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"(\||&|;|`|\$\(|\$\{)",
        r"(\bnc\b|\bnetcat\b|\bwget\b|\bcurl\b|\bbash\b|\bsh\b)",
        r"(\.\./|\.\.\\\\)",
        r"(/etc/passwd|/etc/shadow|cmd\.exe|powershell\.exe)"
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>"
    ]

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        if not email or len(email) > 254:
            return False
        return bool(cls.EMAIL_PATTERN.match(email))

    @classmethod
    def validate_domain(cls, domain: str) -> bool:
        """Validate domain format"""
        if not domain or len(domain) > 253:
            return False
        return bool(cls.DOMAIN_PATTERN.match(domain))

    @classmethod
    def validate_ip(cls, ip: str) -> bool:
        """Validate IP address format"""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_hash(cls, hash_value: str, hash_type: str = None) -> bool:
        """Validate hash format"""
        if not hash_value:
            return False
        
        # Check basic hex format
        if not cls.HASH_PATTERN.match(hash_value):
            return False
        
        # Check length based on hash type
        if hash_type:
            expected_lengths = {
                'md5': 32,
                'sha1': 40,
                'sha256': 64,
                'sha512': 128
            }
            expected_length = expected_lengths.get(hash_type.lower())
            if expected_length and len(hash_value) != expected_length:
                return False
        
        return True

    @classmethod
    def sanitize_input(cls, value: str, max_length: int = 1000) -> str:
        """Sanitize input string"""
        if not isinstance(value, str):
            raise InputValidationError("Input must be a string")
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        # Remove null bytes and control characters
        value = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        return value

    @classmethod
    def check_injection_patterns(cls, value: str) -> List[str]:
        """Check for injection attack patterns"""
        threats = []
        value_lower = value.lower()
        
        # Check SQL injection
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append("sql_injection")
                break
        
        # Check command injection
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append("command_injection")
                break
        
        # Check XSS
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value_lower, re.IGNORECASE):
                threats.append("xss")
                break
        
        return threats

    @classmethod
    def validate_tool_parameters(cls, tool_def: ToolDefinition, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters against definition"""
        validated_params = {}
        
        for param_def in tool_def.parameters:
            param_name = param_def.name
            param_value = params.get(param_name)
            
            # Check required parameters
            if param_def.required and param_value is None:
                raise InputValidationError(f"Required parameter '{param_name}' is missing")
            
            if param_value is None:
                param_value = param_def.default
                if param_value is None:
                    continue
            
            # Type validation
            if param_def.type == "string":
                if not isinstance(param_value, str):
                    raise InputValidationError(f"Parameter '{param_name}' must be a string")
                
                # Length validation
                if param_def.min_length and len(param_value) < param_def.min_length:
                    raise InputValidationError(f"Parameter '{param_name}' too short")
                if param_def.max_length and len(param_value) > param_def.max_length:
                    raise InputValidationError(f"Parameter '{param_name}' too long")
                
                # Pattern validation
                if param_def.pattern and not re.match(param_def.pattern, param_value):
                    raise InputValidationError(f"Parameter '{param_name}' format invalid")
                
                # Enum validation
                if param_def.enum_values and param_value not in param_def.enum_values:
                    raise InputValidationError(f"Parameter '{param_name}' not in allowed values")
                
                # Sanitize and check for injection
                param_value = cls.sanitize_input(param_value, param_def.max_length or 1000)
                threats = cls.check_injection_patterns(param_value)
                if threats:
                    raise InputValidationError(f"Parameter '{param_name}' contains potential threats: {threats}")
            
            elif param_def.type == "integer":
                if not isinstance(param_value, int):
                    try:
                        param_value = int(param_value)
                    except (ValueError, TypeError):
                        raise InputValidationError(f"Parameter '{param_name}' must be an integer")
            
            elif param_def.type == "boolean":
                if not isinstance(param_value, bool):
                    if isinstance(param_value, str):
                        param_value = param_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        param_value = bool(param_value)
            
            validated_params[param_name] = param_value
        
        return validated_params


class AuthManager:
    """Authentication and authorization management"""
    
    def __init__(self, config: SecurityConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.cipher = Fernet(config.encryption_key.encode() if config.encryption_key else Fernet.generate_key())
    
    def generate_jwt(self, client_info: MCPClientInfo, user_id: Optional[str] = None) -> str:
        """Generate JWT token for client"""
        payload = {
            'client_id': client_info.client_id,
            'client_name': client_info.name,
            'user_id': user_id,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiry_hours)
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    async def validate_jwt(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm])
            
            # Check if token is blacklisted
            blacklist_key = f"jwt_blacklist:{hashlib.sha256(token.encode()).hexdigest()}"
            if await self.redis.exists(blacklist_key):
                raise AuthenticationError("Token has been revoked")
            
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")
    
    async def revoke_token(self, token: str):
        """Revoke JWT token by adding to blacklist"""
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        blacklist_key = f"jwt_blacklist:{token_hash}"
        
        # Add to blacklist with expiration matching token expiry
        try:
            payload = jwt.decode(token, self.config.jwt_secret, algorithms=[self.config.jwt_algorithm], options={"verify_exp": False})
            exp_time = payload.get('exp')
            if exp_time:
                expiry_seconds = max(0, exp_time - int(time.time()))
                await self.redis.setex(blacklist_key, expiry_seconds, "revoked")
        except jwt.InvalidTokenError:
            # If we can't decode, just set a reasonable expiry
            await self.redis.setex(blacklist_key, self.config.jwt_expiry_hours * 3600, "revoked")
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = hashed.split(':')
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return hmac.compare_digest(stored_hash, password_hash.hex())
        except ValueError:
            return False
    
    def check_network_access(self, client_ip: str) -> bool:
        """Check if client IP is allowed"""
        if not self.config.allowed_networks:
            return True
        
        try:
            client_addr = ipaddress.ip_address(client_ip)
            for network in self.config.allowed_networks:
                if client_addr in ipaddress.ip_network(network, strict=False):
                    return True
            return False
        except ValueError:
            return False


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, redis_client: aioredis.Redis, config: SecurityConfig):
        self.redis = redis_client
        self.config = config
    
    async def check_rate_limit(self, client_id: str, tool_name: Optional[str] = None) -> RateLimitInfo:
        """Check and update rate limit for client"""
        # Use tool-specific or global rate limit
        window_duration = self.config.rate_limit_window
        max_requests = self.config.rate_limit_requests
        
        # Create rate limit key
        key_parts = ["rate_limit", client_id]
        if tool_name:
            key_parts.append(tool_name)
        rate_key = ":".join(key_parts)
        
        window_start = datetime.now().replace(second=0, microsecond=0)
        window_key = f"{rate_key}:{int(window_start.timestamp())}"
        
        # Get current count
        current_count = await self.redis.get(window_key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= max_requests:
            reset_time = window_start + timedelta(seconds=window_duration)
            raise RateLimitError(f"Rate limit exceeded. Reset at {reset_time}")
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, window_duration)
        await pipe.execute()
        
        return RateLimitInfo(
            client_id=client_id,
            requests_made=current_count + 1,
            requests_remaining=max_requests - current_count - 1,
            reset_time=window_start + timedelta(seconds=window_duration),
            window_start=window_start,
            window_duration=window_duration
        )


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self, redis_client: aioredis.Redis, config: SecurityConfig):
        self.redis = redis_client
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.audit")
    
    async def log_action(self, log_entry: AuditLogEntry):
        """Log security action"""
        # Store in Redis for immediate access
        log_key = f"audit_log:{log_entry.timestamp.strftime('%Y%m%d')}:{log_entry.log_id}"
        log_data = {
            'timestamp': log_entry.timestamp.isoformat(),
            'client_id': log_entry.client_id,
            'user_id': log_entry.user_id or '',
            'action': log_entry.action,
            'resource': log_entry.resource,
            'success': str(log_entry.success),
            'ip_address': log_entry.ip_address,
            'user_agent': log_entry.user_agent,
            'security_level': log_entry.security_level.value,
            'request_data': str(log_entry.request_data),
            'response_data': str(log_entry.response_data)
        }
        
        await self.redis.hset(log_key, mapping=log_data)
        await self.redis.expire(log_key, self.config.audit_log_retention_days * 24 * 3600)
        
        # Also log to standard logging
        self.logger.info(
            f"AUDIT: {log_entry.action} on {log_entry.resource} by {log_entry.client_id} "
            f"from {log_entry.ip_address} - {'SUCCESS' if log_entry.success else 'FAILED'}"
        )
    
    async def get_audit_logs(self, client_id: Optional[str] = None, 
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[AuditLogEntry]:
        """Retrieve audit logs with filters"""
        # This is a simplified implementation
        # In production, you'd want to use a proper time-series database
        logs = []
        
        if not start_date:
            start_date = datetime.now() - timedelta(days=7)
        if not end_date:
            end_date = datetime.now()
        
        # Search through date range
        current_date = start_date
        while current_date <= end_date and len(logs) < limit:
            date_pattern = f"audit_log:{current_date.strftime('%Y%m%d')}:*"
            keys = await self.redis.keys(date_pattern)
            
            for key in keys[:limit - len(logs)]:
                log_data = await self.redis.hgetall(key)
                if log_data and (not client_id or log_data.get('client_id') == client_id):
                    # Convert back to AuditLogEntry (simplified)
                    logs.append(log_data)
            
            current_date += timedelta(days=1)
        
        return logs


class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, config: SecurityConfig, redis_client: aioredis.Redis):
        self.config = config
        self.redis = redis_client
        self.auth_manager = AuthManager(config, redis_client)
        self.rate_limiter = RateLimiter(redis_client, config)
        self.audit_logger = AuditLogger(redis_client, config)
        self.validator = InputValidator()
        
        # Active sessions tracking
        self.active_sessions: Dict[str, datetime] = {}
    
    async def authenticate_request(self, token: str, client_ip: str, user_agent: str) -> Dict[str, Any]:
        """Authenticate incoming request"""
        # Check network access
        if not self.auth_manager.check_network_access(client_ip):
            raise AuthorizationError(f"Access denied from IP: {client_ip}")
        
        # Validate token
        payload = await self.auth_manager.validate_jwt(token)
        
        # Update session activity
        client_id = payload['client_id']
        self.active_sessions[client_id] = datetime.now()
        
        return payload
    
    async def authorize_tool_access(self, client_id: str, tool_name: str, security_level: SecurityLevel) -> bool:
        """Authorize tool access based on client permissions"""
        # Check rate limits
        await self.rate_limiter.check_rate_limit(client_id, tool_name)
        
        # Here you would implement more sophisticated authorization logic
        # For now, we'll allow access if authenticated
        return True
    
    async def validate_and_sanitize_params(self, tool_def: ToolDefinition, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize tool parameters"""
        return self.validator.validate_tool_parameters(tool_def, params)
    
    async def log_request(self, client_id: str, action: str, resource: str, 
                         ip_address: str, user_agent: str, success: bool,
                         request_data: Dict[str, Any], response_data: Dict[str, Any],
                         security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """Log security event"""
        log_entry = AuditLogEntry(
            client_id=client_id,
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            request_data=request_data,
            response_data=response_data,
            security_level=security_level
        )
        
        await self.audit_logger.log_action(log_entry)
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        expired_sessions = [
            client_id for client_id, last_activity in self.active_sessions.items()
            if (now - last_activity).total_seconds() > self.config.session_timeout
        ]
        
        for client_id in expired_sessions:
            del self.active_sessions[client_id]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


def require_auth(security_level: SecurityLevel = SecurityLevel.MEDIUM):
    """Decorator for requiring authentication on endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # This would be implemented in the actual server context
            # where request context is available
            return await func(self, *args, **kwargs)
        return wrapper
    return decorator