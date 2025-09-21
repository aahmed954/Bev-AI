#!/usr/bin/env python3
"""
Tool Master - Intelligent Tool Selection and Orchestration
Coordinates between all tools and agents for optimal task execution
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import inspect
import networkx as nx
from datetime import datetime
import aiohttp
import subprocess
import docker
import kubernetes
from concurrent.futures import ThreadPoolExecutor
import yaml
import re

class ToolCategory(Enum):
    OSINT = "osint"
    EXPLOITATION = "exploitation"
    ANALYSIS = "analysis"
    ENHANCEMENT = "enhancement"
    INFRASTRUCTURE = "infrastructure"
    COMMUNICATION = "communication"
    PERSISTENCE = "persistence"
    EVASION = "evasion"

@dataclass
class Tool:
    """Individual tool definition"""
    name: str
    category: ToolCategory
    capabilities: List[str]
    requirements: List[str]
    docker_image: Optional[str] = None
    api_endpoint: Optional[str] = None
    command: Optional[str] = None
    cost: float = 0.0  # Computational cost
    success_rate: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ToolChain:
    """Sequence of tools for complex operations"""
    name: str
    tools: List[Tool]
    data_flow: Dict[str, List[str]]  # Tool name -> dependent tools
    parallel_groups: List[List[str]]  # Tools that can run in parallel
    
class ToolRegistry:
    """Central registry of all available tools"""
    
    def __init__(self):
        self.tools = {}
        self.categories = {}
        self.capability_index = {}
        self._initialize_tools()
        
    def _initialize_tools(self):
        """Register all available tools"""
        
        # OSINT Tools
        self.register(Tool(
            name="shodan_scanner",
            category=ToolCategory.OSINT,
            capabilities=["port_scanning", "service_detection", "vulnerability_discovery"],
            requirements=["shodan_api_key"],
            api_endpoint="https://api.shodan.io",
            cost=0.1
        ))
        
        self.register(Tool(
            name="breach_hunter",
            category=ToolCategory.OSINT,
            capabilities=["credential_search", "breach_database", "password_dumps"],
            requirements=["tor_connection"],
            docker_image="breach_hunter:latest",
            cost=0.3
        ))
        
        self.register(Tool(
            name="social_mapper",
            category=ToolCategory.OSINT,
            capabilities=["social_media_correlation", "profile_matching", "relationship_mapping"],
            requirements=["api_keys"],
            docker_image="social_mapper:latest",
            cost=0.2
        ))
        
        # Exploitation Tools
        self.register(Tool(
            name="metasploit",
            category=ToolCategory.EXPLOITATION,
            capabilities=["exploit_execution", "payload_generation", "post_exploitation"],
            requirements=["target_info"],
            docker_image="metasploit/metasploit-framework",
            cost=0.5
        ))
        
        self.register(Tool(
            name="sqlmap",
            category=ToolCategory.EXPLOITATION,
            capabilities=["sql_injection", "database_extraction", "privilege_escalation"],
            requirements=["target_url"],
            docker_image="sqlmap:latest",
            cost=0.2
        ))
        
        # Analysis Tools
        self.register(Tool(
            name="ghidra",
            category=ToolCategory.ANALYSIS,
            capabilities=["reverse_engineering", "binary_analysis", "decompilation"],
            requirements=["binary_file"],
            docker_image="ghidra:latest",
            cost=0.7
        ))
        
        self.register(Tool(
            name="wireshark",
            category=ToolCategory.ANALYSIS,
            capabilities=["packet_analysis", "protocol_decode", "traffic_monitoring"],
            requirements=["pcap_file"],
            command="tshark",
            cost=0.1
        ))
        
        # Enhancement Tools
        self.register(Tool(
            name="watermark_remover",
            category=ToolCategory.ENHANCEMENT,
            capabilities=["watermark_detection", "watermark_removal", "image_restoration"],
            requirements=["image_file"],
            docker_image="watermark_assassin:latest",
            cost=0.4
        ))
        
        self.register(Tool(
            name="drm_stripper",
            category=ToolCategory.ENHANCEMENT,
            capabilities=["drm_analysis", "protection_removal", "content_liberation"],
            requirements=["protected_file"],
            docker_image="drm_research:latest",
            cost=0.6
        ))
        
        # Infrastructure Tools
        self.register(Tool(
            name="terraform",
            category=ToolCategory.INFRASTRUCTURE,
            capabilities=["infrastructure_deployment", "cloud_provisioning", "configuration_management"],
            requirements=["cloud_credentials"],
            command="terraform",
            cost=0.1
        ))
        
        self.register(Tool(
            name="docker_compose",
            category=ToolCategory.INFRASTRUCTURE,
            capabilities=["container_orchestration", "service_deployment", "network_creation"],
            requirements=["docker_daemon"],
            command="docker-compose",
            cost=0.1
        ))
        
        # Evasion Tools
        self.register(Tool(
            name="obfuscator",
            category=ToolCategory.EVASION,
            capabilities=["code_obfuscation", "signature_evasion", "polymorphic_generation"],
            requirements=["source_code"],
            docker_image="obfuscator:latest",
            cost=0.3
        ))
        
        self.register(Tool(
            name="proxy_chain",
            category=ToolCategory.EVASION,
            capabilities=["traffic_routing", "anonymization", "geo_spoofing"],
            requirements=["proxy_list"],
            docker_image="proxychains:latest",
            cost=0.2
        ))
    
    def register(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name] = tool
        
        # Index by category
        if tool.category not in self.categories:
            self.categories[tool.category] = []
        self.categories[tool.category].append(tool)
        
        # Index by capability
        for capability in tool.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(tool)
    
    def find_tools_by_capability(self, capability: str) -> List[Tool]:
        """Find tools that provide a specific capability"""
        return self.capability_index.get(capability, [])
    
    def find_best_tool(self, capability: str, constraints: Dict = None) -> Optional[Tool]:
        """Find optimal tool for a capability given constraints"""
        tools = self.find_tools_by_capability(capability)
        
        if not tools:
            return None
        
        # Score tools based on cost, success rate, and constraints
        scored_tools = []
        for tool in tools:
            score = tool.success_rate / (tool.cost + 0.1)  # Prevent division by zero
            
            # Check constraints
            if constraints:
                if 'max_cost' in constraints and tool.cost > constraints['max_cost']:
                    continue
                if 'required_category' in constraints and tool.category != constraints['required_category']:
                    continue
            
            scored_tools.append((tool, score))
        
        if not scored_tools:
            return None
        
        # Return tool with highest score
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        return scored_tools[0][0]

class ToolExecutor:
    """Executes tools and manages their lifecycle"""
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.active_tools = {}
        
    async def execute_tool(self, tool: Tool, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given inputs"""
        execution_id = f"{tool.name}_{datetime.now().timestamp()}"
        self.active_tools[execution_id] = {
            'tool': tool,
            'status': 'running',
            'started_at': datetime.now()
        }
        
        try:
            if tool.docker_image:
                result = await self._execute_docker_tool(tool, inputs)
            elif tool.api_endpoint:
                result = await self._execute_api_tool(tool, inputs)
            elif tool.command:
                result = await self._execute_command_tool(tool, inputs)
            else:
                raise ValueError(f"No execution method for tool {tool.name}")
            
            self.active_tools[execution_id]['status'] = 'completed'
            return result
            
        except Exception as e:
            self.active_tools[execution_id]['status'] = 'failed'
            self.active_tools[execution_id]['error'] = str(e)
            raise
        
    async def _execute_docker_tool(self, tool: Tool, inputs: Dict) -> Dict:
        """Execute tool via Docker container"""
        container = self.docker_client.containers.run(
            tool.docker_image,
            command=json.dumps(inputs),
            detach=True,
            remove=False,
            volumes={
                '/tmp/tool_input': {'bind': '/input', 'mode': 'rw'},
                '/tmp/tool_output': {'bind': '/output', 'mode': 'rw'}
            }
        )
        
        # Wait for completion
        result = await asyncio.to_thread(container.wait)
        logs = container.logs().decode('utf-8')
        container.remove()
        
        # Read output
        with open('/tmp/tool_output/result.json', 'r') as f:
            output = json.load(f)
        
        return {
            'status': 'success' if result['StatusCode'] == 0 else 'failed',
            'output': output,
            'logs': logs
        }
    
    async def _execute_api_tool(self, tool: Tool, inputs: Dict) -> Dict:
        """Execute tool via API call"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{tool.api_endpoint}/{tool.name}",
                json=inputs,
                headers={'Content-Type': 'application/json'}
            ) as response:
                result = await response.json()
                return {
                    'status': 'success' if response.status == 200 else 'failed',
                    'output': result
                }
    
    async def _execute_command_tool(self, tool: Tool, inputs: Dict) -> Dict:
        """Execute tool via system command"""
        # Build command with inputs
        cmd = [tool.command]
        for key, value in inputs.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Execute command
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            'status': 'success' if process.returncode == 0 else 'failed',
            'output': stdout.decode('utf-8'),
            'error': stderr.decode('utf-8') if stderr else None
        }

class ToolCoordinator:
    """Main coordinator for intelligent tool selection and execution"""
    
    def __init__(self):
        self.registry = ToolRegistry()
        self.executor = ToolExecutor()
        self.execution_history = []
        self.tool_graph = nx.DiGraph()
        self._build_tool_graph()
        
    def _build_tool_graph(self):
        """Build dependency graph of tools"""
        # Add all tools as nodes
        for tool_name, tool in self.registry.tools.items():
            self.tool_graph.add_node(tool_name, tool=tool)
        
        # Add edges based on data flow possibilities
        # OSINT tools feed into exploitation
        for osint_tool in self.registry.categories.get(ToolCategory.OSINT, []):
            for exploit_tool in self.registry.categories.get(ToolCategory.EXPLOITATION, []):
                self.tool_graph.add_edge(osint_tool.name, exploit_tool.name)
        
        # Analysis tools can process exploitation results
        for exploit_tool in self.registry.categories.get(ToolCategory.EXPLOITATION, []):
            for analysis_tool in self.registry.categories.get(ToolCategory.ANALYSIS, []):
                self.tool_graph.add_edge(exploit_tool.name, analysis_tool.name)
        
        # Enhancement tools process analysis results
        for analysis_tool in self.registry.categories.get(ToolCategory.ANALYSIS, []):
            for enhance_tool in self.registry.categories.get(ToolCategory.ENHANCEMENT, []):
                self.tool_graph.add_edge(analysis_tool.name, enhance_tool.name)
    
    async def plan_execution(self, goal: str, constraints: Dict = None) -> ToolChain:
        """Plan optimal tool chain for achieving goal"""
        # Parse goal to identify required capabilities
        required_capabilities = self._extract_capabilities(goal)
        
        # Find tools for each capability
        selected_tools = []
        for capability in required_capabilities:
            tool = self.registry.find_best_tool(capability, constraints)
            if tool:
                selected_tools.append(tool)
        
        # Optimize execution order using topological sort
        if len(selected_tools) > 1:
            tool_names = [t.name for t in selected_tools]
            subgraph = self.tool_graph.subgraph(tool_names)
            
            # Check for cycles
            if nx.is_directed_acyclic_graph(subgraph):
                execution_order = list(nx.topological_sort(subgraph))
            else:
                execution_order = tool_names
        else:
            execution_order = [selected_tools[0].name] if selected_tools else []
        
        # Identify parallel execution opportunities
        parallel_groups = self._identify_parallel_groups(execution_order)
        
        # Build data flow map
        data_flow = {}
        for i, tool_name in enumerate(execution_order[:-1]):
            data_flow[tool_name] = [execution_order[i+1]]
        
        return ToolChain(
            name=f"chain_for_{goal[:20]}",
            tools=[self.registry.tools[name] for name in execution_order],
            data_flow=data_flow,
            parallel_groups=parallel_groups
        )
    
    def _extract_capabilities(self, goal: str) -> List[str]:
        """Extract required capabilities from goal description"""
        capabilities = []
        
        # Pattern matching for common goals
        patterns = {
            r'scan|reconnaissance|discover': ['port_scanning', 'service_detection'],
            r'exploit|compromise|pwn': ['exploit_execution', 'payload_generation'],
            r'analyze|reverse|understand': ['binary_analysis', 'decompilation'],
            r'remove watermark|strip drm': ['watermark_removal', 'drm_analysis'],
            r'hide|evade|obfuscate': ['code_obfuscation', 'traffic_routing'],
            r'social|profile|person': ['social_media_correlation', 'profile_matching'],
            r'credential|password|breach': ['credential_search', 'breach_database'],
            r'sql|database|dump': ['sql_injection', 'database_extraction'],
        }
        
        goal_lower = goal.lower()
        for pattern, caps in patterns.items():
            if re.search(pattern, goal_lower):
                capabilities.extend(caps)
        
        return list(set(capabilities))  # Remove duplicates
    
    def _identify_parallel_groups(self, execution_order: List[str]) -> List[List[str]]:
        """Identify tools that can run in parallel"""
        parallel_groups = []
        
        if not execution_order:
            return parallel_groups
        
        # Tools with no dependencies between them can run in parallel
        for i in range(len(execution_order)):
            group = [execution_order[i]]
            
            # Check if next tools have no dependency on this one
            for j in range(i+1, len(execution_order)):
                if not self.tool_graph.has_edge(execution_order[i], execution_order[j]):
                    group.append(execution_order[j])
            
            if len(group) > 1:
                parallel_groups.append(group)
        
        return parallel_groups
    
    async def execute_chain(self, chain: ToolChain, initial_input: Dict) -> Dict[str, Any]:
        """Execute a tool chain"""
        results = {}
        current_input = initial_input
        
        print(f"🔧 Executing tool chain: {chain.name}")
        print(f"   Tools: {[t.name for t in chain.tools]}")
        
        # Check if we can parallelize
        if chain.parallel_groups:
            for group in chain.parallel_groups:
                if len(group) > 1:
                    # Execute tools in parallel
                    tasks = []
                    for tool_name in group:
                        tool = self.registry.tools[tool_name]
                        tasks.append(self.executor.execute_tool(tool, current_input))
                    
                    group_results = await asyncio.gather(*tasks)
                    
                    # Merge results
                    for tool_name, result in zip(group, group_results):
                        results[tool_name] = result
                        
                    # Use last result as input for next stage
                    current_input = group_results[-1].get('output', current_input)
        
        else:
            # Sequential execution
            for tool in chain.tools:
                print(f"   Executing: {tool.name}")
                
                result = await self.executor.execute_tool(tool, current_input)
                results[tool.name] = result
                
                # Use output as input for next tool
                if 'output' in result:
                    current_input = result['output']
                
                print(f"   ✓ {tool.name} completed")
        
        # Record execution
        self.execution_history.append({
            'chain': chain.name,
            'timestamp': datetime.now(),
            'tools_executed': [t.name for t in chain.tools],
            'results': results
        })
        
        return results
    
    async def autonomous_tool_discovery(self):
        """Discover and integrate new tools automatically"""
        print("🔍 Starting autonomous tool discovery...")
        
        # Scan for new Docker images
        available_images = self.docker_client.images.list()
        
        for image in available_images:
            tags = image.tags
            for tag in tags:
                if 'tool' in tag or 'scanner' in tag or 'exploit' in tag:
                    # Check if we already have this tool
                    tool_name = tag.split(':')[0].split('/')[-1]
                    
                    if tool_name not in self.registry.tools:
                        # Attempt to extract capabilities from image metadata
                        labels = image.attrs.get('Config', {}).get('Labels', {})
                        
                        if 'capabilities' in labels:
                            capabilities = json.loads(labels['capabilities'])
                            category = ToolCategory(labels.get('category', 'analysis'))
                            
                            # Register new tool
                            new_tool = Tool(
                                name=tool_name,
                                category=category,
                                capabilities=capabilities,
                                requirements=[],
                                docker_image=tag
                            )
                            
                            self.registry.register(new_tool)
                            print(f"   ✓ Discovered new tool: {tool_name}")
    
    async def optimize_tool_selection(self):
        """Learn from execution history to improve tool selection"""
        if len(self.execution_history) < 10:
            return
        
        # Analyze success rates
        tool_performance = {}
        
        for execution in self.execution_history[-100:]:  # Last 100 executions
            for tool_name, result in execution['results'].items():
                if tool_name not in tool_performance:
                    tool_performance[tool_name] = {'success': 0, 'total': 0}
                
                tool_performance[tool_name]['total'] += 1
                if result.get('status') == 'success':
                    tool_performance[tool_name]['success'] += 1
        
        # Update tool success rates
        for tool_name, perf in tool_performance.items():
            if tool_name in self.registry.tools:
                success_rate = perf['success'] / perf['total']
                self.registry.tools[tool_name].success_rate = success_rate
                
                print(f"   Updated {tool_name} success rate: {success_rate:.2%}")

# Example usage
async def main():
    coordinator = ToolCoordinator()
    
    # Plan execution for a goal
    chain = await coordinator.plan_execution(
        "scan target system and find SQL injection vulnerabilities",
        constraints={'max_cost': 1.0}
    )
    
    # Execute the chain
    results = await coordinator.execute_chain(chain, {
        'target': '192.168.1.100',
        'port_range': '1-65535'
    })
    
    print(f"\n📊 Execution Results:")
    for tool_name, result in results.items():
        print(f"   {tool_name}: {result.get('status')}")
    
    # Start autonomous optimization
    asyncio.create_task(coordinator.autonomous_tool_discovery())
    asyncio.create_task(coordinator.optimize_tool_selection())

if __name__ == "__main__":
    asyncio.run(main())
