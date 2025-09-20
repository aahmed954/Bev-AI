#!/usr/bin/env python3
"""
ToolMaster Worker for ORACLE1
Tool orchestration, workflow management, and system coordination
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import structlog
from celery import Task, group, chain, chord
from celery_app import app
from pydantic import BaseModel
import redis
import httpx
from dataclasses import dataclass
from enum import Enum

# Configure structured logging
logger = structlog.get_logger("toolmaster_worker")

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    CANCELLED = "cancelled"

class WorkflowStage(str, Enum):
    """Workflow execution stages"""
    PLANNING = "planning"
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"

class ToolType(str, Enum):
    """Tool categories"""
    OCR = "ocr"
    NLP = "nlp"
    GENETIC = "genetic"
    KNOWLEDGE = "knowledge"
    EDGE = "edge"
    DATABASE = "database"
    API = "api"
    CUSTOM = "custom"

class ToolDefinition(BaseModel):
    """Tool definition model"""
    tool_id: str
    tool_type: ToolType
    name: str
    description: str
    endpoint: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout: int = 300
    retry_count: int = 3
    dependencies: List[str] = []
    capabilities: List[str] = []

class WorkflowTask(BaseModel):
    """Workflow task model"""
    task_id: str
    tool_id: str
    input_data: Dict[str, Any]
    dependencies: List[str] = []
    timeout: int = 300
    retry_count: int = 3
    priority: int = 5
    stage: WorkflowStage = WorkflowStage.PENDING

class Workflow(BaseModel):
    """Workflow definition model"""
    workflow_id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    execution_strategy: str = "sequential"  # sequential, parallel, dag
    timeout: int = 3600
    retry_policy: Dict[str, Any] = {}

class ExecutionPlan(BaseModel):
    """Workflow execution plan"""
    workflow_id: str
    execution_id: str
    task_order: List[List[str]]  # List of task batches for execution
    estimated_duration: int
    resource_requirements: Dict[str, Any]
    dependencies_resolved: bool = True

class ToolMasterOrchestrator:
    """Main orchestration engine for ORACLE1"""

    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=4)
        self.tools_registry = {}
        self.active_workflows = {}
        self.execution_history = {}
        self.setup_default_tools()

    def setup_default_tools(self):
        """Setup default ORACLE1 tools"""
        try:
            # OCR Service
            self.register_tool(ToolDefinition(
                tool_id="ocr_service",
                tool_type=ToolType.OCR,
                name="OCR Text Extraction",
                description="Multi-language OCR processing with Tesseract",
                endpoint="http://ocr-service:8080/ocr/process",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file": {"type": "string"},
                        "language": {"type": "string", "default": "auto"},
                        "preprocessing": {"type": "boolean", "default": True}
                    },
                    "required": ["file"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "confidence": {"type": "number"},
                        "language": {"type": "string"}
                    }
                },
                capabilities=["multi_language", "pdf_processing", "image_processing"]
            ))

            # Document Analyzer
            self.register_tool(ToolDefinition(
                tool_id="document_analyzer",
                tool_type=ToolType.NLP,
                name="Document Analysis",
                description="NLP document analysis with entity extraction",
                endpoint="http://document-analyzer:8081/analyze",
                input_schema={
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "text": {"type": "string"},
                        "analysis_types": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["document_id", "text"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "entities": {"type": "array"},
                        "relationships": {"type": "array"},
                        "keywords": {"type": "array"}
                    }
                },
                capabilities=["entity_extraction", "relationship_mapping", "sentiment_analysis"]
            ))

            logger.info("Default tools registered", tool_count=len(self.tools_registry))

        except Exception as e:
            logger.error("Failed to setup default tools", error=str(e))

    def register_tool(self, tool: ToolDefinition) -> bool:
        """Register a new tool in the registry"""
        try:
            self.tools_registry[tool.tool_id] = tool

            # Cache in Redis
            self.redis_client.hset(
                "tools_registry",
                tool.tool_id,
                tool.json()
            )

            logger.info("Tool registered", tool_id=tool.tool_id, tool_type=tool.tool_type)
            return True

        except Exception as e:
            logger.error("Tool registration failed", tool_id=tool.tool_id, error=str(e))
            return False

    def get_tool(self, tool_id: str) -> Optional[ToolDefinition]:
        """Get tool definition by ID"""
        try:
            if tool_id in self.tools_registry:
                return self.tools_registry[tool_id]

            # Try to load from Redis
            tool_data = self.redis_client.hget("tools_registry", tool_id)
            if tool_data:
                tool = ToolDefinition.parse_raw(tool_data)
                self.tools_registry[tool_id] = tool
                return tool

            return None

        except Exception as e:
            logger.error("Failed to get tool", tool_id=tool_id, error=str(e))
            return None

    def create_workflow(self, workflow: Workflow) -> ExecutionPlan:
        """Create execution plan for workflow"""
        try:
            logger.info("Creating workflow execution plan", workflow_id=workflow.workflow_id)

            # Validate tools exist
            for task in workflow.tasks:
                if not self.get_tool(task.tool_id):
                    raise ValueError(f"Tool not found: {task.tool_id}")

            # Generate execution plan
            execution_plan = self._generate_execution_plan(workflow)

            # Store workflow
            self.active_workflows[workflow.workflow_id] = {
                "workflow": workflow,
                "execution_plan": execution_plan,
                "status": TaskStatus.PENDING,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }

            # Cache in Redis
            self.redis_client.setex(
                f"workflow:{workflow.workflow_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    "workflow": workflow.dict(),
                    "execution_plan": execution_plan.dict()
                }, default=str)
            )

            return execution_plan

        except Exception as e:
            logger.error("Workflow creation failed", workflow_id=workflow.workflow_id, error=str(e))
            raise

    def _generate_execution_plan(self, workflow: Workflow) -> ExecutionPlan:
        """Generate optimized execution plan"""
        try:
            execution_id = f"exec_{workflow.workflow_id}_{int(time.time())}"

            if workflow.execution_strategy == "sequential":
                task_order = [[task.task_id] for task in workflow.tasks]
            elif workflow.execution_strategy == "parallel":
                task_order = [[task.task_id for task in workflow.tasks]]
            else:  # DAG-based execution
                task_order = self._resolve_task_dependencies(workflow.tasks)

            # Estimate duration
            estimated_duration = self._estimate_execution_duration(workflow.tasks, task_order)

            # Calculate resource requirements
            resource_requirements = self._calculate_resource_requirements(workflow.tasks)

            return ExecutionPlan(
                workflow_id=workflow.workflow_id,
                execution_id=execution_id,
                task_order=task_order,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements,
                dependencies_resolved=True
            )

        except Exception as e:
            logger.error("Execution plan generation failed", error=str(e))
            raise

    def _resolve_task_dependencies(self, tasks: List[WorkflowTask]) -> List[List[str]]:
        """Resolve task dependencies using topological sort"""
        try:
            # Build dependency graph
            graph = {task.task_id: task.dependencies for task in tasks}
            task_order = []

            # Simple topological sort
            remaining_tasks = set(task.task_id for task in tasks)

            while remaining_tasks:
                # Find tasks with no remaining dependencies
                ready_tasks = []
                for task_id in remaining_tasks:
                    if all(dep not in remaining_tasks for dep in graph[task_id]):
                        ready_tasks.append(task_id)

                if not ready_tasks:
                    # Circular dependency or error
                    ready_tasks = list(remaining_tasks)  # Force execution

                task_order.append(ready_tasks)
                remaining_tasks -= set(ready_tasks)

            return task_order

        except Exception as e:
            logger.error("Dependency resolution failed", error=str(e))
            # Fallback to sequential execution
            return [[task.task_id] for task in tasks]

    def _estimate_execution_duration(self, tasks: List[WorkflowTask], task_order: List[List[str]]) -> int:
        """Estimate total execution duration"""
        try:
            total_duration = 0

            for batch in task_order:
                batch_duration = 0
                for task_id in batch:
                    task = next((t for t in tasks if t.task_id == task_id), None)
                    if task:
                        tool = self.get_tool(task.tool_id)
                        if tool:
                            batch_duration = max(batch_duration, tool.timeout)

                total_duration += batch_duration

            return total_duration

        except Exception as e:
            logger.error("Duration estimation failed", error=str(e))
            return 3600  # Default 1 hour

    def _calculate_resource_requirements(self, tasks: List[WorkflowTask]) -> Dict[str, Any]:
        """Calculate resource requirements for workflow"""
        try:
            requirements = {
                "cpu_cores": 0,
                "memory_mb": 0,
                "storage_mb": 0,
                "network_bandwidth": 0
            }

            for task in tasks:
                tool = self.get_tool(task.tool_id)
                if tool:
                    # Simple resource estimation based on tool type
                    if tool.tool_type == ToolType.OCR:
                        requirements["cpu_cores"] += 1
                        requirements["memory_mb"] += 512
                    elif tool.tool_type == ToolType.NLP:
                        requirements["cpu_cores"] += 2
                        requirements["memory_mb"] += 1024
                    elif tool.tool_type == ToolType.GENETIC:
                        requirements["cpu_cores"] += 4
                        requirements["memory_mb"] += 2048

            return requirements

        except Exception as e:
            logger.error("Resource calculation failed", error=str(e))
            return {"cpu_cores": 1, "memory_mb": 512, "storage_mb": 100, "network_bandwidth": 10}

    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute workflow using Celery"""
        try:
            logger.info("Starting workflow execution", workflow_id=workflow_id)

            workflow_data = self.active_workflows.get(workflow_id)
            if not workflow_data:
                raise ValueError(f"Workflow not found: {workflow_id}")

            workflow = workflow_data["workflow"]
            execution_plan = workflow_data["execution_plan"]

            # Update status
            workflow_data["status"] = TaskStatus.RUNNING
            workflow_data["updated_at"] = datetime.now()

            # Execute based on strategy
            if workflow.execution_strategy == "sequential":
                result = await self._execute_sequential(workflow, execution_plan)
            elif workflow.execution_strategy == "parallel":
                result = await self._execute_parallel(workflow, execution_plan)
            else:  # DAG execution
                result = await self._execute_dag(workflow, execution_plan)

            # Update final status
            workflow_data["status"] = TaskStatus.SUCCESS
            workflow_data["updated_at"] = datetime.now()
            workflow_data["result"] = result

            return result

        except Exception as e:
            logger.error("Workflow execution failed", workflow_id=workflow_id, error=str(e))
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TaskStatus.FAILURE
                self.active_workflows[workflow_id]["error"] = str(e)
            raise

    async def _execute_sequential(self, workflow: Workflow, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute workflow sequentially"""
        try:
            results = {}

            for task in workflow.tasks:
                logger.info("Executing task", task_id=task.task_id, tool_id=task.tool_id)

                # Execute task
                task_result = await self._execute_task(task, results)
                results[task.task_id] = task_result

            return {
                "workflow_id": workflow.workflow_id,
                "execution_strategy": "sequential",
                "task_results": results,
                "status": "completed",
                "execution_time": time.time()
            }

        except Exception as e:
            logger.error("Sequential execution failed", error=str(e))
            raise

    async def _execute_parallel(self, workflow: Workflow, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute workflow in parallel"""
        try:
            # Create Celery group for parallel execution
            task_signatures = []

            for task in workflow.tasks:
                signature = execute_tool_task.s(task.dict())
                task_signatures.append(signature)

            # Execute parallel group
            job = group(task_signatures)()

            # Wait for results
            results = {}
            for i, result in enumerate(job.get()):
                task_id = workflow.tasks[i].task_id
                results[task_id] = result

            return {
                "workflow_id": workflow.workflow_id,
                "execution_strategy": "parallel",
                "task_results": results,
                "status": "completed",
                "execution_time": time.time()
            }

        except Exception as e:
            logger.error("Parallel execution failed", error=str(e))
            raise

    async def _execute_dag(self, workflow: Workflow, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """Execute workflow as DAG"""
        try:
            results = {}

            for batch in execution_plan.task_order:
                # Execute batch in parallel
                batch_tasks = [task for task in workflow.tasks if task.task_id in batch]

                if len(batch_tasks) == 1:
                    # Single task
                    task_result = await self._execute_task(batch_tasks[0], results)
                    results[batch_tasks[0].task_id] = task_result
                else:
                    # Parallel batch
                    task_signatures = [execute_tool_task.s(task.dict()) for task in batch_tasks]
                    job = group(task_signatures)()

                    batch_results = job.get()
                    for i, result in enumerate(batch_results):
                        results[batch_tasks[i].task_id] = result

            return {
                "workflow_id": workflow.workflow_id,
                "execution_strategy": "dag",
                "task_results": results,
                "status": "completed",
                "execution_time": time.time()
            }

        except Exception as e:
            logger.error("DAG execution failed", error=str(e))
            raise

    async def _execute_task(self, task: WorkflowTask, previous_results: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual task"""
        try:
            tool = self.get_tool(task.tool_id)
            if not tool:
                raise ValueError(f"Tool not found: {task.tool_id}")

            # Prepare input data (may include results from previous tasks)
            input_data = task.input_data.copy()

            # Inject dependency results
            for dep_task_id in task.dependencies:
                if dep_task_id in previous_results:
                    input_data[f"dep_{dep_task_id}"] = previous_results[dep_task_id]

            # Execute via HTTP API
            async with httpx.AsyncClient(timeout=task.timeout) as client:
                response = await client.post(
                    tool.endpoint,
                    json=input_data,
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"Tool execution failed: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error("Task execution failed", task_id=task.task_id, error=str(e))
            raise

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        try:
            workflow_data = self.active_workflows.get(workflow_id)
            if not workflow_data:
                # Try to load from Redis
                cached_data = self.redis_client.get(f"workflow:{workflow_id}")
                if cached_data:
                    return json.loads(cached_data)
                return {"error": "Workflow not found"}

            return {
                "workflow_id": workflow_id,
                "status": workflow_data["status"],
                "created_at": workflow_data["created_at"].isoformat(),
                "updated_at": workflow_data["updated_at"].isoformat(),
                "task_count": len(workflow_data["workflow"].tasks),
                "execution_plan": workflow_data["execution_plan"].dict() if "execution_plan" in workflow_data else None
            }

        except Exception as e:
            logger.error("Failed to get workflow status", workflow_id=workflow_id, error=str(e))
            return {"error": str(e)}

    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel running workflow"""
        try:
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = TaskStatus.CANCELLED
                self.active_workflows[workflow_id]["updated_at"] = datetime.now()

                logger.info("Workflow cancelled", workflow_id=workflow_id)
                return True

            return False

        except Exception as e:
            logger.error("Failed to cancel workflow", workflow_id=workflow_id, error=str(e))
            return False

# Initialize ToolMaster orchestrator
toolmaster = ToolMasterOrchestrator()

class ToolMasterTask(Task):
    """Custom Celery task for tool orchestration"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error("ToolMaster task failed",
                    task_id=task_id,
                    exception=str(exc),
                    traceback=str(einfo))

@app.task(bind=True, base=ToolMasterTask, queue='tool_orchestration')
def execute_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute complete workflow"""
    try:
        logger.info("Processing workflow execution", task_id=self.request.id)

        workflow = Workflow(**workflow_data)

        # Create execution plan
        execution_plan = toolmaster.create_workflow(workflow)

        # Execute workflow (run async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(toolmaster.execute_workflow(workflow.workflow_id))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error("Workflow execution failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=60, max_retries=3)

@app.task(bind=True, base=ToolMasterTask, queue='tool_orchestration')
def execute_tool_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute individual tool task"""
    try:
        logger.info("Processing tool task", task_id=self.request.id)

        task = WorkflowTask(**task_data)

        # Execute task (run async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(toolmaster._execute_task(task, {}))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error("Tool task execution failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=30, max_retries=3)

@app.task(bind=True, base=ToolMasterTask, queue='tool_orchestration')
def register_tool(self, tool_data: Dict[str, Any]) -> bool:
    """Register new tool in registry"""
    try:
        logger.info("Registering tool", tool_id=tool_data.get('tool_id'))

        tool = ToolDefinition(**tool_data)
        result = toolmaster.register_tool(tool)

        return result

    except Exception as e:
        logger.error("Tool registration failed", error=str(e))
        raise

@app.task(bind=True, base=ToolMasterTask, queue='tool_orchestration')
def orchestrate_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """Orchestrate complex processing pipeline"""
    try:
        logger.info("Processing pipeline orchestration", task_id=self.request.id)

        # Create workflow from pipeline configuration
        workflow_tasks = []

        for i, stage in enumerate(pipeline_config.get('stages', [])):
            task = WorkflowTask(
                task_id=f"stage_{i}",
                tool_id=stage['tool_id'],
                input_data=stage['input_data'],
                dependencies=stage.get('dependencies', []),
                timeout=stage.get('timeout', 300)
            )
            workflow_tasks.append(task)

        workflow = Workflow(
            workflow_id=f"pipeline_{self.request.id}",
            name=pipeline_config.get('name', 'Pipeline'),
            description=pipeline_config.get('description', ''),
            tasks=workflow_tasks,
            execution_strategy=pipeline_config.get('execution_strategy', 'sequential')
        )

        # Execute workflow
        execution_plan = toolmaster.create_workflow(workflow)

        # Execute workflow (run async in sync context)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(toolmaster.execute_workflow(workflow.workflow_id))
            return result
        finally:
            loop.close()

    except Exception as e:
        logger.error("Pipeline orchestration failed", task_id=self.request.id, error=str(e))
        raise self.retry(exc=e, countdown=120, max_retries=2)

@app.task(bind=True, base=ToolMasterTask, queue='tool_orchestration')
def monitor_system_health(self) -> Dict[str, Any]:
    """Monitor ORACLE1 system health"""
    try:
        logger.info("Monitoring system health", task_id=self.request.id)

        health_status = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "healthy",
            "components": {},
            "active_workflows": len(toolmaster.active_workflows),
            "registered_tools": len(toolmaster.tools_registry)
        }

        # Check tool endpoints
        for tool_id, tool in toolmaster.tools_registry.items():
            try:
                # Simple health check via HTTP
                import requests
                response = requests.get(f"{tool.endpoint.replace('/analyze', '').replace('/process', '')}/health", timeout=5)
                health_status["components"][tool_id] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                health_status["components"][tool_id] = {
                    "status": "unreachable",
                    "error": str(e)
                }

        # Overall system health
        unhealthy_components = [comp for comp, status in health_status["components"].items()
                              if status["status"] != "healthy"]

        if unhealthy_components:
            health_status["system_status"] = "degraded"
            health_status["issues"] = unhealthy_components

        return health_status

    except Exception as e:
        logger.error("System health monitoring failed", task_id=self.request.id, error=str(e))
        raise

if __name__ == "__main__":
    # Test ToolMaster functionality
    test_workflow = Workflow(
        workflow_id="test_workflow_001",
        name="Test Document Processing",
        description="Test OCR -> Analysis workflow",
        tasks=[
            WorkflowTask(
                task_id="ocr_task",
                tool_id="ocr_service",
                input_data={"file": "test_document.pdf"}
            ),
            WorkflowTask(
                task_id="analysis_task",
                tool_id="document_analyzer",
                input_data={"document_id": "test_001"},
                dependencies=["ocr_task"]
            )
        ],
        execution_strategy="sequential"
    )

    print("ToolMaster orchestrator ready for workflow management")