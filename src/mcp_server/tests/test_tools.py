"""
Comprehensive tests for OSINT tools
===================================

Tests for tool execution, parameter validation, and integration with security features.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from mcp_server.tools import (
    OSINTToolRegistry, OSINTCollector, ThreatAnalyzer, GraphAnalyzer,
    OSINTToolBase
)
from mcp_server.models import (
    OSINTTarget, ToolResult, SecurityLevel, ToolCategory,
    ThreatIntelligence
)
from mcp_server.database import DatabaseManager
from mcp_server.security import SecurityManager


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing"""
    db_manager = Mock(spec=DatabaseManager)
    
    # Mock PostgreSQL manager
    db_manager.postgres = Mock()
    db_manager.postgres.store_osint_result = AsyncMock(return_value="test_result_id")
    db_manager.postgres.store_threat_intelligence = AsyncMock(return_value=True)
    db_manager.postgres.get_threat_intelligence = AsyncMock(return_value=None)
    
    # Mock Neo4j manager
    db_manager.neo4j = Mock()
    db_manager.neo4j.find_related_entities = AsyncMock(return_value=[])
    db_manager.neo4j.execute_cypher = AsyncMock(return_value=[])
    
    # Mock Redis manager
    db_manager.redis = Mock()
    db_manager.redis.cache_result = AsyncMock()
    db_manager.redis.get_cached_result = AsyncMock(return_value=None)
    
    # Mock Elasticsearch manager
    db_manager.elasticsearch = Mock()
    db_manager.elasticsearch.search_documents = AsyncMock(return_value=[])
    
    return db_manager


@pytest.fixture
def mock_security_manager():
    """Mock security manager for testing"""
    security_manager = Mock(spec=SecurityManager)
    security_manager.authorize_tool_access = AsyncMock(return_value=True)
    security_manager.validate_and_sanitize_params = AsyncMock(side_effect=lambda tool_def, params: params)
    security_manager.log_request = AsyncMock()
    return security_manager


@pytest.fixture
def sample_osint_target():
    """Sample OSINT target for testing"""
    return OSINTTarget(
        target_id="test_target_123",
        target_type="domain",
        value="example.com",
        metadata={"source": "test"}
    )


class TestOSINTCollector:
    """Test OSINT data collection tool"""
    
    @pytest.fixture
    def osint_collector(self, mock_db_manager, mock_security_manager):
        return OSINTCollector(mock_db_manager, mock_security_manager)
    
    def test_tool_definition(self, osint_collector):
        """Test tool definition structure"""
        tool_def = osint_collector.get_tool_definition()
        
        assert tool_def.name == "collect_osint"
        assert tool_def.category == ToolCategory.COLLECTION
        assert tool_def.security_level == SecurityLevel.MEDIUM
        assert len(tool_def.parameters) == 5
        
        # Check required parameters
        param_names = [p.name for p in tool_def.parameters]
        assert "target_type" in param_names
        assert "target_value" in param_names
        
        # Check parameter validation
        target_type_param = next(p for p in tool_def.parameters if p.name == "target_type")
        assert target_type_param.required is True
        assert "email" in target_type_param.enum_values
    
    @pytest.mark.asyncio
    async def test_execute_domain_collection(self, osint_collector, sample_osint_target):
        """Test domain OSINT collection"""
        params = {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois,virustotal",
            "use_tor": False,
            "max_results": 100
        }
        
        with patch.object(osint_collector, '_collect_whois', return_value={"whois_data": "test"}), \
             patch.object(osint_collector, '_collect_virustotal', return_value={"reputation": "clean"}):
            
            result = await osint_collector.execute(sample_osint_target, params)
            
            assert result.success is True
            assert result.tool_name == "collect_osint"
            assert "results" in result.data
            assert "whois" in result.data["results"]
            assert "virustotal" in result.data["results"]
    
    @pytest.mark.asyncio
    async def test_execute_email_collection(self, osint_collector):
        """Test email OSINT collection"""
        email_target = OSINTTarget(
            target_id="test_email",
            target_type="email",
            value="test@example.com"
        )
        
        params = {
            "target_type": "email",
            "target_value": "test@example.com",
            "sources": "dehashed",
            "use_tor": False,
            "max_results": 50
        }
        
        with patch.object(osint_collector, '_collect_dehashed', 
                         return_value={"breaches": [], "password_hashes": []}):
            
            result = await osint_collector.execute(email_target, params)
            
            assert result.success is True
            assert "dehashed" in result.data["results"]
    
    @pytest.mark.asyncio
    async def test_execute_ip_collection(self, osint_collector):
        """Test IP address OSINT collection"""
        ip_target = OSINTTarget(
            target_id="test_ip",
            target_type="ip",
            value="192.168.1.1"
        )
        
        params = {
            "target_type": "ip",
            "target_value": "192.168.1.1",
            "sources": "shodan",
            "use_tor": False,
            "max_results": 100
        }
        
        with patch.object(osint_collector, '_collect_shodan', 
                         return_value={"ports": [22, 80, 443], "services": ["ssh", "http", "https"]}):
            
            result = await osint_collector.execute(ip_target, params)
            
            assert result.success is True
            assert "shodan" in result.data["results"]
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_target(self, osint_collector):
        """Test execution with invalid target"""
        invalid_target = OSINTTarget(
            target_id="test_invalid",
            target_type="email",
            value="invalid.email.format"  # Invalid email
        )
        
        params = {
            "target_type": "email",
            "target_value": "invalid.email.format",
            "sources": "dehashed"
        }
        
        result = await osint_collector.execute(invalid_target, params)
        
        assert result.success is False
        assert "Invalid target" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_with_source_failure(self, osint_collector, sample_osint_target):
        """Test execution when a source fails"""
        params = {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois,virustotal"
        }
        
        # Mock one source to succeed and one to fail
        with patch.object(osint_collector, '_collect_whois', return_value={"whois_data": "test"}), \
             patch.object(osint_collector, '_collect_virustotal', side_effect=Exception("API error")):
            
            result = await osint_collector.execute(sample_osint_target, params)
            
            assert result.success is True  # Should still succeed if at least one source works
            assert "whois" in result.data["results"]
            assert "virustotal" in result.data["results"]
            assert "error" in result.data["results"]["virustotal"]
    
    def test_validate_target_types(self, osint_collector):
        """Test target validation for different types"""
        # Valid targets
        assert osint_collector._validate_target(OSINTTarget("1", "email", "test@example.com"))
        assert osint_collector._validate_target(OSINTTarget("2", "domain", "example.com"))
        assert osint_collector._validate_target(OSINTTarget("3", "ip", "192.168.1.1"))
        assert osint_collector._validate_target(OSINTTarget("4", "username", "testuser"))
        
        # Invalid targets
        assert not osint_collector._validate_target(OSINTTarget("5", "email", "invalid.email"))
        assert not osint_collector._validate_target(OSINTTarget("6", "domain", "invalid..domain"))
        assert not osint_collector._validate_target(OSINTTarget("7", "ip", "256.1.1.1"))
    
    @pytest.mark.asyncio
    async def test_safe_command_execution(self, osint_collector):
        """Test safe command execution with whitelisting"""
        # Allowed command
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = Mock()
            mock_process.communicate.return_value = (b"test output", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process
            
            result = await osint_collector._execute_safe_command(['whois', 'example.com'])
            assert result == "test output"
        
        # Disallowed command
        with pytest.raises(ValueError):
            await osint_collector._execute_safe_command(['rm', '-rf', '/'])


class TestThreatAnalyzer:
    """Test threat analysis tool"""
    
    @pytest.fixture
    def threat_analyzer(self, mock_db_manager, mock_security_manager):
        return ThreatAnalyzer(mock_db_manager, mock_security_manager)
    
    def test_tool_definition(self, threat_analyzer):
        """Test tool definition structure"""
        tool_def = threat_analyzer.get_tool_definition()
        
        assert tool_def.name == "analyze_threat"
        assert tool_def.category == ToolCategory.ANALYSIS
        assert tool_def.security_level == SecurityLevel.HIGH
        
        # Check parameters
        param_names = [p.name for p in tool_def.parameters]
        assert "ioc_type" in param_names
        assert "ioc_value" in param_names
        assert "include_ml_analysis" in param_names
    
    @pytest.mark.asyncio
    async def test_execute_ip_analysis(self, threat_analyzer):
        """Test IP address threat analysis"""
        ip_target = OSINTTarget(
            target_id="test_ip_threat",
            target_type="ip",
            value="192.168.1.1"
        )
        
        params = {
            "ioc_type": "ip",
            "ioc_value": "192.168.1.1",
            "include_ml_analysis": True
        }
        
        result = await threat_analyzer.execute(ip_target, params)
        
        assert result.success is True
        assert result.data["ioc_type"] == "ip"
        assert result.data["ioc_value"] == "192.168.1.1"
        assert "ml_analysis" in result.data
        assert "confidence" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_domain_analysis(self, threat_analyzer):
        """Test domain threat analysis"""
        domain_target = OSINTTarget(
            target_id="test_domain_threat",
            target_type="domain",
            value="malicious-example.com"
        )
        
        params = {
            "ioc_type": "domain",
            "ioc_value": "malicious-example.com",
            "context": "Suspicious domain found in logs",
            "include_ml_analysis": True
        }
        
        result = await threat_analyzer.execute(domain_target, params)
        
        assert result.success is True
        assert result.data["ioc_type"] == "domain"
        assert "realtime_analysis" in result.data
        assert "ml_analysis" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_hash_analysis(self, threat_analyzer):
        """Test hash threat analysis"""
        hash_target = OSINTTarget(
            target_id="test_hash_threat",
            target_type="hash",
            value="d41d8cd98f00b204e9800998ecf8427e"
        )
        
        params = {
            "ioc_type": "hash",
            "ioc_value": "d41d8cd98f00b204e9800998ecf8427e",
            "include_ml_analysis": False
        }
        
        result = await threat_analyzer.execute(hash_target, params)
        
        assert result.success is True
        assert result.data["ioc_type"] == "hash"
        assert "ml_analysis" not in result.data or result.data["ml_analysis"] is None
    
    @pytest.mark.asyncio
    async def test_execute_with_existing_intelligence(self, threat_analyzer, mock_db_manager):
        """Test analysis with existing threat intelligence"""
        # Mock existing threat intelligence
        mock_db_manager.postgres.get_threat_intelligence.return_value = {
            "severity": "high",
            "confidence": 0.9,
            "threat_types": ["malware", "phishing"],
            "sources": ["intelligence_feed_1"]
        }
        
        target = OSINTTarget(
            target_id="test_known_threat",
            target_type="domain",
            value="known-bad.com"
        )
        
        params = {
            "ioc_type": "domain",
            "ioc_value": "known-bad.com"
        }
        
        result = await threat_analyzer.execute(target, params)
        
        assert result.success is True
        assert result.data["threat_level"] == "high"
        assert result.data["confidence"] == 0.9
        assert "malware" in result.data["threat_types"]
    
    def test_validate_ioc_formats(self, threat_analyzer):
        """Test IOC format validation"""
        # Valid IOCs
        assert threat_analyzer._validate_ioc("ip", "192.168.1.1")
        assert threat_analyzer._validate_ioc("domain", "example.com")
        assert threat_analyzer._validate_ioc("email", "test@example.com")
        assert threat_analyzer._validate_ioc("hash", "d41d8cd98f00b204e9800998ecf8427e")
        assert threat_analyzer._validate_ioc("url", "https://example.com/path")
        
        # Invalid IOCs
        assert not threat_analyzer._validate_ioc("ip", "256.1.1.1")
        assert not threat_analyzer._validate_ioc("domain", "invalid..domain")
        assert not threat_analyzer._validate_ioc("email", "invalid.email")
        assert not threat_analyzer._validate_ioc("hash", "not_hex_chars")
        assert not threat_analyzer._validate_ioc("url", "not_a_url")
    
    def test_calculate_entropy(self, threat_analyzer):
        """Test entropy calculation for ML features"""
        # High entropy string (random)
        high_entropy = "a8f5f167f44f4964e6c998dee827110c"
        entropy = threat_analyzer._calculate_entropy(high_entropy)
        assert entropy > 3.0
        
        # Low entropy string (predictable)
        low_entropy = "aaaaaaaaaaaaaaaa"
        entropy = threat_analyzer._calculate_entropy(low_entropy)
        assert entropy < 1.0
        
        # Empty string
        entropy = threat_analyzer._calculate_entropy("")
        assert entropy == 0


class TestGraphAnalyzer:
    """Test graph analysis tool"""
    
    @pytest.fixture
    def graph_analyzer(self, mock_db_manager, mock_security_manager):
        return GraphAnalyzer(mock_db_manager, mock_security_manager)
    
    def test_tool_definition(self, graph_analyzer):
        """Test tool definition structure"""
        tool_def = graph_analyzer.get_tool_definition()
        
        assert tool_def.name == "graph_analysis"
        assert tool_def.category == ToolCategory.VISUALIZATION
        assert tool_def.security_level == SecurityLevel.MEDIUM
        
        # Check parameters
        param_names = [p.name for p in tool_def.parameters]
        assert "entity_value" in param_names
        assert "analysis_type" in param_names
        assert "max_depth" in param_names
        assert "min_confidence" in param_names
    
    @pytest.mark.asyncio
    async def test_execute_relationships_analysis(self, graph_analyzer, mock_db_manager):
        """Test relationship analysis"""
        mock_db_manager.neo4j.find_related_entities.return_value = [
            {"value": "related1.com", "labels": ["Domain"], "distance": 1},
            {"value": "related2.com", "labels": ["Domain"], "distance": 2}
        ]
        
        target = OSINTTarget(
            target_id="test_graph",
            target_type="domain",
            value="example.com"
        )
        
        params = {
            "entity_value": "example.com",
            "analysis_type": "relationships",
            "max_depth": 2
        }
        
        result = await graph_analyzer.execute(target, params)
        
        assert result.success is True
        assert result.data["analysis_type"] == "relationships"
        assert "relationships" in result.data
        assert len(result.data["relationships"]) == 2
    
    @pytest.mark.asyncio
    async def test_execute_clusters_analysis(self, graph_analyzer, mock_db_manager):
        """Test cluster analysis"""
        mock_db_manager.neo4j.execute_cypher.return_value = [
            {"cluster": ["entity1.com", "entity2.com", "entity3.com"]}
        ]
        
        target = OSINTTarget(
            target_id="test_cluster",
            target_type="domain",
            value="example.com"
        )
        
        params = {
            "entity_value": "example.com",
            "analysis_type": "clusters",
            "min_confidence": 0.7
        }
        
        result = await graph_analyzer.execute(target, params)
        
        assert result.success is True
        assert result.data["analysis_type"] == "clusters"
        assert "clusters" in result.data
    
    @pytest.mark.asyncio
    async def test_execute_centrality_analysis(self, graph_analyzer, mock_db_manager):
        """Test centrality analysis"""
        mock_db_manager.neo4j.execute_cypher.return_value = [
            {"degree_centrality": 15, "unique_connections": 10}
        ]
        
        target = OSINTTarget(
            target_id="test_centrality",
            target_type="domain",
            value="hub.example.com"
        )
        
        params = {
            "entity_value": "hub.example.com",
            "analysis_type": "centrality"
        }
        
        result = await graph_analyzer.execute(target, params)
        
        assert result.success is True
        assert result.data["analysis_type"] == "centrality"
        assert "centrality" in result.data
        assert result.data["centrality"]["degree_centrality"] == 15


class TestOSINTToolRegistry:
    """Test tool registry functionality"""
    
    @pytest.fixture
    def tool_registry(self, mock_db_manager, mock_security_manager):
        return OSINTToolRegistry(mock_db_manager, mock_security_manager)
    
    def test_registry_initialization(self, tool_registry):
        """Test tool registry initialization"""
        tools = tool_registry.get_all_tools()
        
        # Check that core tools are registered
        assert "collect_osint" in tools
        assert "analyze_threat" in tools
        assert "graph_analysis" in tools
        
        # Check tool instances
        assert isinstance(tools["collect_osint"], OSINTCollector)
        assert isinstance(tools["analyze_threat"], ThreatAnalyzer)
        assert isinstance(tools["graph_analysis"], GraphAnalyzer)
    
    def test_get_tool_definitions(self, tool_registry):
        """Test getting all tool definitions"""
        definitions = tool_registry.get_tool_definitions()
        
        assert len(definitions) >= 3  # At least the core tools
        
        # Check definition structure
        for definition in definitions:
            assert hasattr(definition, 'name')
            assert hasattr(definition, 'description')
            assert hasattr(definition, 'category')
            assert hasattr(definition, 'security_level')
            assert hasattr(definition, 'parameters')
    
    def test_get_tool_by_name(self, tool_registry):
        """Test getting specific tool by name"""
        collector = tool_registry.get_tool("collect_osint")
        assert isinstance(collector, OSINTCollector)
        
        analyzer = tool_registry.get_tool("analyze_threat")
        assert isinstance(analyzer, ThreatAnalyzer)
        
        # Non-existent tool
        assert tool_registry.get_tool("non_existent_tool") is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_success(self, tool_registry, mock_security_manager):
        """Test successful tool execution through registry"""
        target = OSINTTarget(
            target_id="test_registry",
            target_type="domain",
            value="example.com"
        )
        
        params = {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois"
        }
        
        # Mock the actual tool execution
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = Mock(security_level=SecurityLevel.MEDIUM)
        mock_tool.execute = AsyncMock(return_value=ToolResult(
            tool_name="collect_osint",
            success=True,
            data={"test": "data"}
        ))
        
        tool_registry.tools["collect_osint"] = mock_tool
        
        result = await tool_registry.execute_tool(
            "collect_osint", target, params, "test_client"
        )
        
        assert result.success is True
        assert result.tool_name == "collect_osint"
        
        # Verify security checks were called
        mock_security_manager.authorize_tool_access.assert_called_once()
        mock_security_manager.validate_and_sanitize_params.assert_called_once()
        mock_security_manager.log_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, tool_registry):
        """Test execution of non-existent tool"""
        target = OSINTTarget(
            target_id="test_registry",
            target_type="domain",
            value="example.com"
        )
        
        result = await tool_registry.execute_tool(
            "non_existent_tool", target, {}, "test_client"
        )
        
        assert result.success is False
        assert "not found" in result.error
    
    @pytest.mark.asyncio
    async def test_execute_tool_authorization_denied(self, tool_registry, mock_security_manager):
        """Test tool execution with authorization denied"""
        mock_security_manager.authorize_tool_access.return_value = False
        
        target = OSINTTarget(
            target_id="test_registry",
            target_type="domain",
            value="example.com"
        )
        
        result = await tool_registry.execute_tool(
            "collect_osint", target, {}, "test_client"
        )
        
        assert result.success is False
        assert "Access denied" in result.error


class TestToolIntegration:
    """Integration tests for tools with database and security"""
    
    @pytest.mark.asyncio
    async def test_full_osint_collection_flow(self, mock_db_manager, mock_security_manager):
        """Test complete OSINT collection flow"""
        collector = OSINTCollector(mock_db_manager, mock_security_manager)
        
        target = OSINTTarget(
            target_id="integration_test",
            target_type="domain",
            value="example.com"
        )
        
        params = {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois",
            "use_tor": False,
            "max_results": 10
        }
        
        with patch.object(collector, '_collect_whois', return_value={"domain_info": "test"}):
            result = await collector.execute(target, params)
            
            assert result.success is True
            
            # Verify database storage was called
            mock_db_manager.postgres.store_osint_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_threat_analysis_flow(self, mock_db_manager, mock_security_manager):
        """Test complete threat analysis flow"""
        analyzer = ThreatAnalyzer(mock_db_manager, mock_security_manager)
        
        target = OSINTTarget(
            target_id="threat_test",
            target_type="ip",
            value="192.168.1.1"
        )
        
        params = {
            "ioc_type": "ip",
            "ioc_value": "192.168.1.1",
            "include_ml_analysis": True
        }
        
        result = await analyzer.execute(target, params)
        
        assert result.success is True
        
        # Verify threat intelligence storage was called
        mock_db_manager.postgres.store_threat_intelligence.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_logging(self, mock_db_manager, mock_security_manager):
        """Test error handling and security logging"""
        collector = OSINTCollector(mock_db_manager, mock_security_manager)
        
        # Simulate database error
        mock_db_manager.postgres.store_osint_result.side_effect = Exception("Database error")
        
        target = OSINTTarget(
            target_id="error_test",
            target_type="domain",
            value="example.com"
        )
        
        params = {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois"
        }
        
        with patch.object(collector, '_collect_whois', return_value={"domain_info": "test"}):
            result = await collector.execute(target, params)
            
            # Should handle error gracefully
            assert result.success is False
            assert "Database error" in result.error


if __name__ == "__main__":
    pytest.main([__file__])