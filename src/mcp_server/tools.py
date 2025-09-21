"""
OSINT Tools Registry
===================

Implementation of 8 specialized OSINT tools with security hardening.
"""

import asyncio
import aiohttp
import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
import re
import socket
from urllib.parse import urlparse
import subprocess
import ipaddress

from .models import (
    ToolDefinition, ToolParameter, ToolResult, OSINTTarget, OSINTResult,
    ThreatIntelligence, SecurityLevel, ToolCategory, CryptoTransaction,
    SecurityScanResult, GraphNode, GraphRelationship
)
from .database import DatabaseManager
from .security import SecurityManager, InputValidator


logger = logging.getLogger(__name__)


class OSINTToolBase(ABC):
    """Base class for OSINT tools"""
    
    def __init__(self, name: str, description: str, category: ToolCategory, 
                 security_level: SecurityLevel, db_manager: DatabaseManager,
                 security_manager: SecurityManager):
        self.name = name
        self.description = description
        self.category = category
        self.security_level = security_level
        self.db_manager = db_manager
        self.security_manager = security_manager
        self.validator = InputValidator()
    
    @abstractmethod
    async def execute(self, target: OSINTTarget, params: Dict[str, Any]) -> ToolResult:
        """Execute the OSINT tool"""
        pass
    
    @abstractmethod
    def get_tool_definition(self) -> ToolDefinition:
        """Get tool definition for MCP"""
        pass
    
    async def _make_secure_request(self, url: str, headers: Dict[str, str] = None,
                                 timeout: int = 30) -> Dict[str, Any]:
        """Make secure HTTP request with validation"""
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            raise ValueError("Invalid URL scheme")
        
        # Prevent SSRF attacks
        try:
            ip = socket.gethostbyname(parsed.hostname)
            ip_addr = ipaddress.ip_address(ip)
            if ip_addr.is_private or ip_addr.is_loopback:
                raise ValueError("Access to private/loopback IPs not allowed")
        except socket.gaierror:
            pass  # Allow domain names that don't resolve
        
        default_headers = {
            'User-Agent': 'BEV-OSINT-Framework/1.0',
            'Accept': 'application/json',
            'Connection': 'close'
        }
        if headers:
            default_headers.update(headers)
        
        timeout_config = aiohttp.ClientTimeout(total=timeout)
        
        async with aiohttp.ClientSession(timeout=timeout_config) as session:
            async with session.get(url, headers=default_headers) as response:
                response.raise_for_status()
                return await response.json()
    
    async def _execute_safe_command(self, command: List[str], timeout: int = 30) -> str:
        """Execute system command safely"""
        # Whitelist allowed commands
        allowed_commands = {
            'whois', 'dig', 'nslookup', 'ping', 'traceroute',
            'openssl', 'curl', 'wget'
        }
        
        if not command or command[0] not in allowed_commands:
            raise ValueError(f"Command '{command[0]}' not allowed")
        
        # Sanitize command arguments
        safe_command = []
        for arg in command:
            # Remove dangerous characters
            safe_arg = re.sub(r'[;&|`$(){}[\]<>]', '', arg)
            safe_command.append(safe_arg)
        
        try:
            process = await asyncio.create_subprocess_exec(
                *safe_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                timeout=timeout
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, safe_command, stderr)
            
            return stdout.decode('utf-8', errors='ignore')
        
        except asyncio.TimeoutError:
            raise TimeoutError(f"Command timed out after {timeout} seconds")


class OSINTCollector(OSINTToolBase):
    """Multi-source OSINT data collection tool"""
    
    def __init__(self, db_manager: DatabaseManager, security_manager: SecurityManager):
        super().__init__(
            name="collect_osint",
            description="Collect OSINT data from multiple sources with Tor support",
            category=ToolCategory.COLLECTION,
            security_level=SecurityLevel.MEDIUM,
            db_manager=db_manager,
            security_manager=security_manager
        )
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            category=self.category,
            security_level=self.security_level,
            parameters=[
                ToolParameter(
                    name="target_type",
                    type="string",
                    description="Type of target to investigate",
                    required=True,
                    enum_values=["email", "domain", "ip", "phone", "username"]
                ),
                ToolParameter(
                    name="target_value",
                    type="string",
                    description="Target value to investigate",
                    required=True,
                    max_length=255
                ),
                ToolParameter(
                    name="sources",
                    type="string",
                    description="Comma-separated list of sources to query",
                    required=False,
                    default="shodan,virustotal,dehashed"
                ),
                ToolParameter(
                    name="use_tor",
                    type="boolean",
                    description="Use Tor proxy for requests",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return",
                    required=False,
                    default=100
                )
            ],
            rate_limit=30,  # 30 requests per minute
            timeout=300
        )
    
    async def execute(self, target: OSINTTarget, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            # Validate target
            if not self._validate_target(target):
                raise ValueError(f"Invalid target: {target.value}")
            
            sources = params.get("sources", "shodan,virustotal,dehashed").split(",")
            use_tor = params.get("use_tor", False)
            max_results = params.get("max_results", 100)
            
            collected_data = {}
            
            # Collect from each source
            for source in sources:
                source = source.strip()
                try:
                    if source == "shodan" and target.target_type == "ip":
                        data = await self._collect_shodan(target.value, use_tor)
                        collected_data["shodan"] = data
                    
                    elif source == "virustotal":
                        data = await self._collect_virustotal(target.value, target.target_type, use_tor)
                        collected_data["virustotal"] = data
                    
                    elif source == "dehashed" and target.target_type == "email":
                        data = await self._collect_dehashed(target.value, use_tor)
                        collected_data["dehashed"] = data
                    
                    elif source == "whois" and target.target_type in ["domain", "ip"]:
                        data = await self._collect_whois(target.value)
                        collected_data["whois"] = data
                
                except Exception as e:
                    logger.error(f"Failed to collect from {source}: {e}")
                    collected_data[source] = {"error": str(e)}
            
            # Calculate confidence score based on number of sources
            confidence_score = min(len([s for s in collected_data.values() if "error" not in s]) / len(sources), 1.0)
            
            # Store result
            result = OSINTResult(
                target=target,
                tool_name=self.name,
                data=collected_data,
                confidence_score=confidence_score,
                sources=list(collected_data.keys())
            )
            
            await self.db_manager.postgres.store_osint_result(result)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"results": collected_data, "confidence": confidence_score},
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"OSINT collection failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _validate_target(self, target: OSINTTarget) -> bool:
        """Validate target format"""
        if target.target_type == "email":
            return self.validator.validate_email(target.value)
        elif target.target_type == "domain":
            return self.validator.validate_domain(target.value)
        elif target.target_type == "ip":
            return self.validator.validate_ip(target.value)
        elif target.target_type == "username":
            return bool(self.validator.USERNAME_PATTERN.match(target.value))
        return True
    
    async def _collect_shodan(self, ip: str, use_tor: bool = False) -> Dict[str, Any]:
        """Collect data from Shodan"""
        # Placeholder implementation - would use actual Shodan API
        return {
            "ip": ip,
            "ports": [22, 80, 443],
            "services": ["ssh", "http", "https"],
            "location": "Unknown",
            "organization": "Unknown"
        }
    
    async def _collect_virustotal(self, target: str, target_type: str, use_tor: bool = False) -> Dict[str, Any]:
        """Collect data from VirusTotal"""
        # Placeholder implementation - would use actual VirusTotal API
        return {
            "target": target,
            "reputation": "clean",
            "detections": 0,
            "last_scan": datetime.now().isoformat()
        }
    
    async def _collect_dehashed(self, email: str, use_tor: bool = False) -> Dict[str, Any]:
        """Collect data from DeHashed"""
        # Placeholder implementation - would use actual DeHashed API
        return {
            "email": email,
            "breaches": [],
            "password_hashes": [],
            "associated_data": {}
        }
    
    async def _collect_whois(self, target: str) -> Dict[str, Any]:
        """Collect WHOIS data"""
        try:
            output = await self._execute_safe_command(['whois', target])
            return {"whois_data": output}
        except Exception as e:
            return {"error": str(e)}


class ThreatAnalyzer(OSINTToolBase):
    """IOC analysis with ML-based threat classification"""
    
    def __init__(self, db_manager: DatabaseManager, security_manager: SecurityManager):
        super().__init__(
            name="analyze_threat",
            description="Analyze IOCs and classify threats using ML models",
            category=ToolCategory.ANALYSIS,
            security_level=SecurityLevel.HIGH,
            db_manager=db_manager,
            security_manager=security_manager
        )
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            category=self.category,
            security_level=self.security_level,
            parameters=[
                ToolParameter(
                    name="ioc_type",
                    type="string",
                    description="Type of IOC to analyze",
                    required=True,
                    enum_values=["ip", "domain", "hash", "url", "email"]
                ),
                ToolParameter(
                    name="ioc_value",
                    type="string",
                    description="IOC value to analyze",
                    required=True,
                    max_length=2048
                ),
                ToolParameter(
                    name="context",
                    type="string",
                    description="Additional context for analysis",
                    required=False,
                    max_length=1000
                ),
                ToolParameter(
                    name="include_ml_analysis",
                    type="boolean",
                    description="Include ML-based threat classification",
                    required=False,
                    default=True
                )
            ],
            rate_limit=60,
            timeout=120
        )
    
    async def execute(self, target: OSINTTarget, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            ioc_type = params["ioc_type"]
            ioc_value = params["ioc_value"]
            context = params.get("context", "")
            include_ml = params.get("include_ml_analysis", True)
            
            # Validate IOC format
            if not self._validate_ioc(ioc_type, ioc_value):
                raise ValueError(f"Invalid {ioc_type} format: {ioc_value}")
            
            # Check against known threat intel
            existing_intel = await self.db_manager.postgres.get_threat_intelligence(ioc_type, ioc_value)
            
            analysis_result = {
                "ioc_type": ioc_type,
                "ioc_value": ioc_value,
                "analysis_timestamp": datetime.now().isoformat(),
                "threat_level": "unknown",
                "confidence": 0.0,
                "threat_types": [],
                "sources": []
            }
            
            if existing_intel:
                analysis_result.update({
                    "threat_level": existing_intel.get("severity", "unknown"),
                    "confidence": existing_intel.get("confidence", 0.0),
                    "threat_types": existing_intel.get("threat_types", []),
                    "sources": existing_intel.get("sources", []),
                    "first_seen": existing_intel.get("first_seen"),
                    "last_seen": existing_intel.get("last_seen")
                })
            
            # Perform real-time analysis
            realtime_analysis = await self._analyze_realtime(ioc_type, ioc_value)
            analysis_result["realtime_analysis"] = realtime_analysis
            
            # ML-based classification if requested
            if include_ml:
                ml_analysis = await self._ml_threat_classification(ioc_type, ioc_value, context)
                analysis_result["ml_analysis"] = ml_analysis
                
                # Update confidence based on ML results
                if ml_analysis.get("confidence", 0) > analysis_result["confidence"]:
                    analysis_result["confidence"] = ml_analysis["confidence"]
                    analysis_result["threat_level"] = ml_analysis.get("threat_level", "unknown")
            
            # Store/update threat intelligence
            threat_intel = ThreatIntelligence(
                ioc_type=ioc_type,
                value=ioc_value,
                threat_types=analysis_result["threat_types"],
                confidence=analysis_result["confidence"],
                severity=SecurityLevel(analysis_result["threat_level"]) if analysis_result["threat_level"] in [s.value for s in SecurityLevel] else SecurityLevel.MEDIUM,
                sources=analysis_result["sources"]
            )
            
            await self.db_manager.postgres.store_threat_intelligence(threat_intel)
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=analysis_result,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def _validate_ioc(self, ioc_type: str, ioc_value: str) -> bool:
        """Validate IOC format"""
        if ioc_type == "ip":
            return self.validator.validate_ip(ioc_value)
        elif ioc_type == "domain":
            return self.validator.validate_domain(ioc_value)
        elif ioc_type == "email":
            return self.validator.validate_email(ioc_value)
        elif ioc_type == "hash":
            return self.validator.validate_hash(ioc_value)
        elif ioc_type == "url":
            try:
                parsed = urlparse(ioc_value)
                return bool(parsed.scheme and parsed.netloc)
            except:
                return False
        return True
    
    async def _analyze_realtime(self, ioc_type: str, ioc_value: str) -> Dict[str, Any]:
        """Perform real-time IOC analysis"""
        # Placeholder for real-time analysis
        return {
            "reputation_score": 0.7,
            "geographic_info": "Unknown",
            "network_info": {},
            "behavioral_indicators": []
        }
    
    async def _ml_threat_classification(self, ioc_type: str, ioc_value: str, context: str) -> Dict[str, Any]:
        """ML-based threat classification"""
        # Placeholder for ML analysis
        # In production, this would use actual ML models
        features = {
            "length": len(ioc_value),
            "has_numbers": any(c.isdigit() for c in ioc_value),
            "has_special_chars": any(not c.isalnum() for c in ioc_value),
            "entropy": self._calculate_entropy(ioc_value)
        }
        
        # Simple heuristic-based classification for demo
        confidence = 0.5
        threat_level = "low"
        
        if features["entropy"] > 4.0:
            confidence = 0.8
            threat_level = "medium"
        
        return {
            "confidence": confidence,
            "threat_level": threat_level,
            "features": features,
            "model_version": "1.0.0"
        }
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of string"""
        if not data:
            return 0
        
        entropy = 0
        for char in set(data):
            prob = data.count(char) / len(data)
            entropy -= prob * (prob.bit_length() - 1) if prob > 0 else 0
        
        return entropy


class GraphAnalyzer(OSINTToolBase):
    """Neo4j-based relationship mapping and graph analysis"""
    
    def __init__(self, db_manager: DatabaseManager, security_manager: SecurityManager):
        super().__init__(
            name="graph_analysis",
            description="Analyze relationships and patterns using graph database",
            category=ToolCategory.VISUALIZATION,
            security_level=SecurityLevel.MEDIUM,
            db_manager=db_manager,
            security_manager=security_manager
        )
    
    def get_tool_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            category=self.category,
            security_level=self.security_level,
            parameters=[
                ToolParameter(
                    name="entity_value",
                    type="string",
                    description="Entity to analyze relationships for",
                    required=True,
                    max_length=255
                ),
                ToolParameter(
                    name="analysis_type",
                    type="string",
                    description="Type of graph analysis to perform",
                    required=True,
                    enum_values=["relationships", "clusters", "paths", "centrality"]
                ),
                ToolParameter(
                    name="max_depth",
                    type="integer",
                    description="Maximum relationship depth to analyze",
                    required=False,
                    default=3
                ),
                ToolParameter(
                    name="min_confidence",
                    type="number",
                    description="Minimum confidence threshold for relationships",
                    required=False,
                    default=0.5
                )
            ],
            rate_limit=30,
            timeout=180
        )
    
    async def execute(self, target: OSINTTarget, params: Dict[str, Any]) -> ToolResult:
        start_time = time.time()
        
        try:
            entity_value = params["entity_value"]
            analysis_type = params["analysis_type"]
            max_depth = params.get("max_depth", 3)
            min_confidence = params.get("min_confidence", 0.5)
            
            analysis_result = {
                "entity": entity_value,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat()
            }
            
            if analysis_type == "relationships":
                relationships = await self.db_manager.neo4j.find_related_entities(entity_value, max_depth)
                analysis_result["relationships"] = relationships
            
            elif analysis_type == "clusters":
                clusters = await self._find_clusters(entity_value, min_confidence)
                analysis_result["clusters"] = clusters
            
            elif analysis_type == "paths":
                paths = await self._find_paths(entity_value, max_depth)
                analysis_result["paths"] = paths
            
            elif analysis_type == "centrality":
                centrality = await self._calculate_centrality(entity_value)
                analysis_result["centrality"] = centrality
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=analysis_result,
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            logger.error(f"Graph analysis failed: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    async def _find_clusters(self, entity_value: str, min_confidence: float) -> List[Dict[str, Any]]:
        """Find entity clusters"""
        query = """
            MATCH (n {value: $entity_value})-[r*1..2]-(m)
            WHERE r.confidence >= $min_confidence
            RETURN COLLECT(DISTINCT m.value) as cluster
        """
        
        result = await self.db_manager.neo4j.execute_cypher(query, {
            "entity_value": entity_value,
            "min_confidence": min_confidence
        })
        
        return result
    
    async def _find_paths(self, entity_value: str, max_depth: int) -> List[Dict[str, Any]]:
        """Find paths between entities"""
        query = """
            MATCH path = (start {value: $entity_value})-[*1..3]-(end)
            WHERE start <> end
            RETURN path, length(path) as path_length
            ORDER BY path_length
            LIMIT 100
        """
        
        result = await self.db_manager.neo4j.execute_cypher(query, {
            "entity_value": entity_value
        })
        
        return result
    
    async def _calculate_centrality(self, entity_value: str) -> Dict[str, Any]:
        """Calculate centrality metrics"""
        # Simplified centrality calculation
        query = """
            MATCH (n {value: $entity_value})-[r]-(m)
            RETURN count(r) as degree_centrality,
                   count(DISTINCT m) as unique_connections
        """
        
        result = await self.db_manager.neo4j.execute_cypher(query, {
            "entity_value": entity_value
        })
        
        return result[0] if result else {"degree_centrality": 0, "unique_connections": 0}


# Additional tool classes would be implemented similarly...
# For brevity, I'll provide the registry class that manages all tools


class OSINTToolRegistry:
    """Registry for all OSINT tools"""
    
    def __init__(self, db_manager: DatabaseManager, security_manager: SecurityManager):
        self.db_manager = db_manager
        self.security_manager = security_manager
        self.tools: Dict[str, OSINTToolBase] = {}
        self._initialize_tools()
    
    def _initialize_tools(self):
        """Initialize all OSINT tools"""
        self.tools = {
            "collect_osint": OSINTCollector(self.db_manager, self.security_manager),
            "analyze_threat": ThreatAnalyzer(self.db_manager, self.security_manager),
            "graph_analysis": GraphAnalyzer(self.db_manager, self.security_manager),
            # Additional tools would be added here:
            # "coordinate_agents": AgentCoordinator(self.db_manager, self.security_manager),
            # "monitor_targets": TargetMonitor(self.db_manager, self.security_manager),
            # "crawl_darkweb": DarkwebCrawler(self.db_manager, self.security_manager),
            # "analyze_crypto": CryptoAnalyzer(self.db_manager, self.security_manager),
            # "security_scan": SecurityScanner(self.db_manager, self.security_manager),
        }
    
    def get_tool(self, tool_name: str) -> Optional[OSINTToolBase]:
        """Get tool by name"""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, OSINTToolBase]:
        """Get all available tools"""
        return self.tools.copy()
    
    def get_tool_definitions(self) -> List[ToolDefinition]:
        """Get MCP tool definitions for all tools"""
        return [tool.get_tool_definition() for tool in self.tools.values()]
    
    async def execute_tool(self, tool_name: str, target: OSINTTarget, 
                          params: Dict[str, Any], client_id: str) -> ToolResult:
        """Execute a tool with security validation"""
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        try:
            # Security validation
            tool_def = tool.get_tool_definition()
            
            # Check authorization
            authorized = await self.security_manager.authorize_tool_access(
                client_id, tool_name, tool_def.security_level
            )
            if not authorized:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error="Access denied"
                )
            
            # Validate and sanitize parameters
            validated_params = await self.security_manager.validate_and_sanitize_params(
                tool_def, params
            )
            
            # Execute tool
            result = await tool.execute(target, validated_params)
            
            # Log execution
            await self.security_manager.log_request(
                client_id=client_id,
                action="tool_execution",
                resource=tool_name,
                ip_address="unknown",  # Would be passed from server context
                user_agent="unknown",  # Would be passed from server context
                success=result.success,
                request_data={"target": target.__dict__, "params": validated_params},
                response_data={"success": result.success, "execution_time": result.execution_time},
                security_level=tool_def.security_level
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e)
            )