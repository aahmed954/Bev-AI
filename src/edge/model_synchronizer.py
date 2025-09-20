"""
Model Synchronization System for BEV OSINT Framework

Manages model distribution, versioning, and synchronization across
edge computing nodes with automatic updates and deployment coordination.
"""

import asyncio
import time
import logging
import json
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import aiofiles
import asyncpg
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError
import requests
from huggingface_hub import hf_hub_download, HfApi
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

class ModelStatus(Enum):
    """Model synchronization status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"

class SyncPriority(Enum):
    """Model synchronization priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

@dataclass
class ModelVersion:
    """Model version information"""
    model_name: str
    version: str
    model_path: str
    model_size_bytes: int
    checksum: str
    release_date: datetime
    compatibility_version: str
    deployment_regions: List[str]
    priority: SyncPriority
    metadata: Dict[str, Any]

@dataclass
class ModelRepository:
    """Model repository configuration"""
    name: str
    type: str  # 'huggingface', 's3', 'local', 'custom'
    base_url: str
    credentials: Dict[str, str]
    sync_enabled: bool
    sync_interval_hours: int
    retry_attempts: int
    timeout_seconds: int

@dataclass
class SyncTask:
    """Model synchronization task"""
    task_id: str
    model_name: str
    version: str
    source_repository: str
    target_regions: List[str]
    priority: SyncPriority
    status: ModelStatus
    progress_percent: float
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    retry_count: int

class ModelSynchronizer:
    """
    Model Synchronization Manager

    Handles model distribution, versioning, and automatic updates across
    edge computing nodes with intelligent scheduling and error recovery.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.repositories: Dict[str, ModelRepository] = {}
        self.model_versions: Dict[str, List[ModelVersion]] = {}
        self.sync_tasks: Dict[str, SyncTask] = {}
        self.active_downloads: Dict[str, asyncio.Task] = {}
        self.edge_nodes: Dict[str, str] = {}  # node_id -> endpoint

        # Configuration
        self.base_model_path = Path("/opt/models")
        self.temp_download_path = Path("/tmp/model_downloads")
        self.max_concurrent_downloads = 3
        self.chunk_size = 1024 * 1024  # 1MB chunks
        self.health_check_interval = 300  # 5 minutes

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.sync_counter = Counter(
            'model_sync_total',
            'Total model synchronizations',
            ['model', 'region', 'status'],
            registry=self.registry
        )
        self.download_speed_gauge = Gauge(
            'model_download_speed_mbps',
            'Model download speed',
            ['model'],
            registry=self.registry
        )
        self.sync_time_histogram = Histogram(
            'model_sync_duration_seconds',
            'Model synchronization duration',
            ['model', 'region'],
            registry=self.registry
        )
        self.model_size_gauge = Gauge(
            'model_size_bytes',
            'Model size in bytes',
            ['model', 'version'],
            registry=self.registry
        )

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        self.s3_client: Optional[boto3.client] = None
        self.hf_api: Optional[HfApi] = None

    async def initialize(self):
        """Initialize model synchronizer"""
        try:
            self.logger.info("Initializing Model Synchronizer")

            # Create directories
            self.base_model_path.mkdir(parents=True, exist_ok=True)
            self.temp_download_path.mkdir(parents=True, exist_ok=True)

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for model downloads
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                user="postgres",
                password=os.getenv('DB_PASSWORD', 'dev_password'),
                database="bev_osint",
                min_size=3,
                max_size=10
            )

            # Create database tables
            await self._create_tables()

            # Initialize external clients
            await self._initialize_external_clients()

            # Load repositories configuration
            await self._load_repositories()

            # Load edge nodes configuration
            await self._load_edge_nodes()

            # Start background tasks
            asyncio.create_task(self._sync_scheduler())
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._cleanup_manager())

            self.logger.info("Model Synchronizer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize model synchronizer: {e}")
            raise

    async def _create_tables(self):
        """Create database tables for model synchronization"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_repositories (
                    name VARCHAR(64) PRIMARY KEY,
                    type VARCHAR(32) NOT NULL,
                    base_url TEXT NOT NULL,
                    credentials JSONB,
                    sync_enabled BOOLEAN DEFAULT TRUE,
                    sync_interval_hours INTEGER DEFAULT 24,
                    retry_attempts INTEGER DEFAULT 3,
                    timeout_seconds INTEGER DEFAULT 300,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_name VARCHAR(64) NOT NULL,
                    version VARCHAR(32) NOT NULL,
                    model_path TEXT NOT NULL,
                    model_size_bytes BIGINT NOT NULL,
                    checksum VARCHAR(64) NOT NULL,
                    release_date TIMESTAMP NOT NULL,
                    compatibility_version VARCHAR(32) NOT NULL,
                    deployment_regions JSONB NOT NULL,
                    priority INTEGER NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_name, version)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_tasks (
                    task_id VARCHAR(64) PRIMARY KEY,
                    model_name VARCHAR(64) NOT NULL,
                    version VARCHAR(32) NOT NULL,
                    source_repository VARCHAR(64) NOT NULL,
                    target_regions JSONB NOT NULL,
                    priority INTEGER NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    progress_percent DECIMAL(5,2) DEFAULT 0.0,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_nodes (
                    node_id VARCHAR(64) PRIMARY KEY,
                    region VARCHAR(32) NOT NULL,
                    endpoint TEXT NOT NULL,
                    model_endpoint TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_health_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    deployed_models JSONB DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def _initialize_external_clients(self):
        """Initialize external API clients"""
        try:
            # Initialize S3 client
            try:
                self.s3_client = boto3.client('s3')
                # Test credentials
                self.s3_client.list_buckets()
                self.logger.info("S3 client initialized successfully")
            except NoCredentialsError:
                self.logger.warning("S3 credentials not found, S3 repository disabled")
                self.s3_client = None
            except Exception as e:
                self.logger.warning(f"S3 client initialization failed: {e}")
                self.s3_client = None

            # Initialize Hugging Face API
            try:
                self.hf_api = HfApi()
                self.logger.info("Hugging Face API initialized successfully")
            except Exception as e:
                self.logger.warning(f"Hugging Face API initialization failed: {e}")
                self.hf_api = None

        except Exception as e:
            self.logger.error(f"External client initialization failed: {e}")

    async def _load_repositories(self):
        """Load model repositories configuration"""
        # Default repositories
        default_repositories = {
            "huggingface": ModelRepository(
                name="huggingface",
                type="huggingface",
                base_url="https://huggingface.co",
                credentials={},
                sync_enabled=True,
                sync_interval_hours=24,
                retry_attempts=3,
                timeout_seconds=300
            ),
            "local": ModelRepository(
                name="local",
                type="local",
                base_url="file:///opt/models",
                credentials={},
                sync_enabled=True,
                sync_interval_hours=1,
                retry_attempts=1,
                timeout_seconds=60
            )
        }

        for repo_name, repo in default_repositories.items():
            self.repositories[repo_name] = repo

            # Store in database
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO model_repositories
                        (name, type, base_url, credentials, sync_enabled, sync_interval_hours, retry_attempts, timeout_seconds)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        ON CONFLICT (name) DO UPDATE SET
                            type = EXCLUDED.type,
                            base_url = EXCLUDED.base_url,
                            sync_enabled = EXCLUDED.sync_enabled,
                            updated_at = CURRENT_TIMESTAMP
                    """, repo.name, repo.type, repo.base_url, json.dumps(repo.credentials),
                    repo.sync_enabled, repo.sync_interval_hours, repo.retry_attempts, repo.timeout_seconds)
            except Exception as e:
                self.logger.error(f"Failed to store repository {repo_name}: {e}")

    async def _load_edge_nodes(self):
        """Load edge nodes configuration"""
        # Default edge nodes based on regions
        edge_node_configs = {
            "edge-us-east-001": {
                "region": "us-east",
                "endpoint": "http://172.30.0.47:8000",
                "model_endpoint": "http://172.30.0.47:8001"
            },
            "edge-us-west-001": {
                "region": "us-west",
                "endpoint": "http://172.30.0.48:8000",
                "model_endpoint": "http://172.30.0.48:8001"
            },
            "edge-eu-central-001": {
                "region": "eu-central",
                "endpoint": "http://172.30.0.49:8000",
                "model_endpoint": "http://172.30.0.49:8001"
            },
            "edge-asia-pacific-001": {
                "region": "asia-pacific",
                "endpoint": "http://172.30.0.50:8000",
                "model_endpoint": "http://172.30.0.50:8001"
            }
        }

        for node_id, config in edge_node_configs.items():
            self.edge_nodes[node_id] = config["endpoint"]

            # Store in database
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO edge_nodes
                        (node_id, region, endpoint, model_endpoint, is_active)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (node_id) DO UPDATE SET
                            region = EXCLUDED.region,
                            endpoint = EXCLUDED.endpoint,
                            model_endpoint = EXCLUDED.model_endpoint,
                            updated_at = CURRENT_TIMESTAMP
                    """, node_id, config["region"], config["endpoint"], config["model_endpoint"], True)
            except Exception as e:
                self.logger.error(f"Failed to store edge node {node_id}: {e}")

    async def add_model_version(self, model_version: ModelVersion):
        """Add a new model version for synchronization"""
        try:
            # Store in memory
            if model_version.model_name not in self.model_versions:
                self.model_versions[model_version.model_name] = []

            self.model_versions[model_version.model_name].append(model_version)

            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO model_versions
                    (model_name, version, model_path, model_size_bytes, checksum,
                     release_date, compatibility_version, deployment_regions, priority, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (model_name, version) DO UPDATE SET
                        model_path = EXCLUDED.model_path,
                        model_size_bytes = EXCLUDED.model_size_bytes,
                        checksum = EXCLUDED.checksum,
                        deployment_regions = EXCLUDED.deployment_regions,
                        priority = EXCLUDED.priority,
                        metadata = EXCLUDED.metadata
                """, model_version.model_name, model_version.version, model_version.model_path,
                model_version.model_size_bytes, model_version.checksum, model_version.release_date,
                model_version.compatibility_version, json.dumps(model_version.deployment_regions),
                model_version.priority.value, json.dumps(model_version.metadata))

            # Update metrics
            self.model_size_gauge.labels(
                model=model_version.model_name,
                version=model_version.version
            ).set(model_version.model_size_bytes)

            self.logger.info(f"Added model version {model_version.model_name}:{model_version.version}")

        except Exception as e:
            self.logger.error(f"Failed to add model version: {e}")
            raise

    async def sync_model_to_regions(self, model_name: str, version: str, regions: List[str], priority: SyncPriority = SyncPriority.NORMAL) -> str:
        """Synchronize a model version to specific regions"""
        task_id = f"sync_{model_name}_{version}_{int(time.time())}"

        try:
            # Create sync task
            sync_task = SyncTask(
                task_id=task_id,
                model_name=model_name,
                version=version,
                source_repository="huggingface",  # Default to HuggingFace
                target_regions=regions,
                priority=priority,
                status=ModelStatus.PENDING,
                progress_percent=0.0,
                started_at=None,
                completed_at=None,
                error_message=None,
                retry_count=0
            )

            self.sync_tasks[task_id] = sync_task

            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO sync_tasks
                    (task_id, model_name, version, source_repository, target_regions, priority, status)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, task_id, model_name, version, sync_task.source_repository,
                json.dumps(regions), priority.value, ModelStatus.PENDING.value)

            # Schedule task for execution
            asyncio.create_task(self._execute_sync_task(task_id))

            self.logger.info(f"Created sync task {task_id} for {model_name}:{version} to regions {regions}")
            return task_id

        except Exception as e:
            self.logger.error(f"Failed to create sync task: {e}")
            raise

    async def _execute_sync_task(self, task_id: str):
        """Execute a synchronization task"""
        start_time = time.time()

        try:
            task = self.sync_tasks[task_id]
            task.status = ModelStatus.DOWNLOADING
            task.started_at = datetime.utcnow()

            await self._update_task_status(task_id)

            # Download model if not already available
            model_path = await self._download_model(task.model_name, task.version, task.source_repository)

            if not model_path:
                raise Exception("Model download failed")

            task.status = ModelStatus.VALIDATING
            task.progress_percent = 50.0
            await self._update_task_status(task_id)

            # Validate model
            if not await self._validate_model(model_path, task.model_name, task.version):
                raise Exception("Model validation failed")

            task.status = ModelStatus.DEPLOYING
            task.progress_percent = 75.0
            await self._update_task_status(task_id)

            # Deploy to target regions
            deployment_results = await self._deploy_to_regions(model_path, task.model_name, task.version, task.target_regions)

            # Check deployment results
            successful_regions = [region for region, success in deployment_results.items() if success]
            failed_regions = [region for region, success in deployment_results.items() if not success]

            if failed_regions:
                task.error_message = f"Deployment failed for regions: {failed_regions}"
                if successful_regions:
                    task.status = ModelStatus.DEPLOYED  # Partial success
                else:
                    task.status = ModelStatus.FAILED
            else:
                task.status = ModelStatus.DEPLOYED

            task.progress_percent = 100.0
            task.completed_at = datetime.utcnow()

            # Update metrics
            sync_duration = time.time() - start_time
            for region in successful_regions:
                self.sync_counter.labels(
                    model=task.model_name,
                    region=region,
                    status="success"
                ).inc()
                self.sync_time_histogram.labels(
                    model=task.model_name,
                    region=region
                ).observe(sync_duration)

            for region in failed_regions:
                self.sync_counter.labels(
                    model=task.model_name,
                    region=region,
                    status="failed"
                ).inc()

            await self._update_task_status(task_id)

            if task.status == ModelStatus.DEPLOYED:
                self.logger.info(f"Sync task {task_id} completed successfully")
            else:
                self.logger.warning(f"Sync task {task_id} completed with errors: {task.error_message}")

        except Exception as e:
            task = self.sync_tasks[task_id]
            task.status = ModelStatus.FAILED
            task.error_message = str(e)
            task.completed_at = datetime.utcnow()

            await self._update_task_status(task_id)

            self.logger.error(f"Sync task {task_id} failed: {e}")

    async def _download_model(self, model_name: str, version: str, repository: str) -> Optional[Path]:
        """Download model from repository"""
        try:
            model_dir = self.base_model_path / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)

            # Check if model already exists and is valid
            if (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists():
                self.logger.info(f"Model {model_name}:{version} already exists locally")
                return model_dir

            # Download based on repository type
            if repository == "huggingface":
                return await self._download_from_huggingface(model_name, version, model_dir)
            elif repository == "s3":
                return await self._download_from_s3(model_name, version, model_dir)
            elif repository == "local":
                return await self._copy_from_local(model_name, version, model_dir)
            else:
                raise Exception(f"Unknown repository type: {repository}")

        except Exception as e:
            self.logger.error(f"Model download failed: {e}")
            return None

    async def _download_from_huggingface(self, model_name: str, version: str, target_dir: Path) -> Path:
        """Download model from Hugging Face Hub"""
        try:
            self.logger.info(f"Downloading {model_name}:{version} from Hugging Face")

            # Common model files to download
            files_to_download = [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt"
            ]

            # Try to download model files (prefer safetensors over bin)
            model_files = []
            try:
                # List repository files
                api = HfApi()
                repo_files = api.list_repo_files(repo_id=model_name, revision=version if version != "latest" else None)

                # Find model weight files
                for file in repo_files:
                    if file.endswith(('.safetensors', '.bin', '.pt', '.pth')):
                        model_files.append(file)

                # Prioritize safetensors
                safetensor_files = [f for f in model_files if f.endswith('.safetensors')]
                if safetensor_files:
                    files_to_download.extend(safetensor_files)
                else:
                    bin_files = [f for f in model_files if f.endswith('.bin')]
                    files_to_download.extend(bin_files)

            except Exception as e:
                self.logger.warning(f"Could not list repository files: {e}")
                # Fallback to common files
                files_to_download.extend(["pytorch_model.bin", "model.safetensors"])

            # Download files
            downloaded_files = []
            for filename in files_to_download:
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        revision=version if version != "latest" else None,
                        cache_dir=str(target_dir.parent),
                        local_dir=str(target_dir),
                        local_dir_use_symlinks=False
                    )
                    downloaded_files.append(downloaded_path)
                    self.logger.debug(f"Downloaded {filename}")
                except Exception as e:
                    self.logger.debug(f"Could not download {filename}: {e}")

            if not downloaded_files:
                raise Exception("No files were successfully downloaded")

            self.logger.info(f"Successfully downloaded {len(downloaded_files)} files for {model_name}:{version}")
            return target_dir

        except Exception as e:
            self.logger.error(f"Hugging Face download failed: {e}")
            raise

    async def _download_from_s3(self, model_name: str, version: str, target_dir: Path) -> Path:
        """Download model from S3 bucket"""
        # Implementation for S3 download
        # This would require S3 bucket configuration and AWS credentials
        raise NotImplementedError("S3 download not yet implemented")

    async def _copy_from_local(self, model_name: str, version: str, target_dir: Path) -> Path:
        """Copy model from local repository"""
        try:
            local_repo = self.repositories["local"]
            source_path = Path(local_repo.base_url.replace("file://", "")) / model_name / version

            if not source_path.exists():
                raise Exception(f"Local model path does not exist: {source_path}")

            # Copy model files
            shutil.copytree(source_path, target_dir, dirs_exist_ok=True)

            self.logger.info(f"Copied model {model_name}:{version} from local repository")
            return target_dir

        except Exception as e:
            self.logger.error(f"Local copy failed: {e}")
            raise

    async def _validate_model(self, model_path: Path, model_name: str, version: str) -> bool:
        """Validate downloaded model"""
        try:
            # Basic file existence checks
            config_file = model_path / "config.json"
            if not config_file.exists():
                self.logger.error("Model config.json not found")
                return False

            # Check for model weights
            weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("*.bin"))
            if not weight_files:
                self.logger.error("No model weight files found")
                return False

            # Validate config JSON
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    if "model_type" not in config:
                        self.logger.warning("Model config missing model_type")
            except Exception as e:
                self.logger.error(f"Invalid config.json: {e}")
                return False

            # Calculate and store checksum
            checksum = await self._calculate_model_checksum(model_path)

            # TODO: Add more sophisticated validation
            # - Load model with transformers library
            # - Test inference
            # - Verify model architecture

            self.logger.info(f"Model validation successful for {model_name}:{version}")
            return True

        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False

    async def _calculate_model_checksum(self, model_path: Path) -> str:
        """Calculate checksum for model directory"""
        hasher = hashlib.sha256()

        for file_path in sorted(model_path.rglob("*")):
            if file_path.is_file():
                async with aiofiles.open(file_path, 'rb') as f:
                    while chunk := await f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    async def _deploy_to_regions(self, model_path: Path, model_name: str, version: str, regions: List[str]) -> Dict[str, bool]:
        """Deploy model to edge nodes in specified regions"""
        deployment_results = {}

        # Get edge nodes for target regions
        target_nodes = {}
        async with self.db_pool.acquire() as conn:
            for region in regions:
                nodes = await conn.fetch("""
                    SELECT node_id, endpoint, model_endpoint
                    FROM edge_nodes
                    WHERE region = $1 AND is_active = TRUE
                """, region)

                target_nodes[region] = [(row['node_id'], row['model_endpoint']) for row in nodes]

        # Deploy to each region
        for region, nodes in target_nodes.items():
            region_success = True

            for node_id, model_endpoint in nodes:
                try:
                    success = await self._deploy_to_node(model_path, model_name, version, node_id, model_endpoint)
                    if not success:
                        region_success = False
                        self.logger.error(f"Failed to deploy to node {node_id}")
                except Exception as e:
                    region_success = False
                    self.logger.error(f"Deployment to node {node_id} failed: {e}")

            deployment_results[region] = region_success

        return deployment_results

    async def _deploy_to_node(self, model_path: Path, model_name: str, version: str, node_id: str, model_endpoint: str) -> bool:
        """Deploy model to a specific edge node"""
        try:
            # Create deployment payload
            deployment_data = {
                "name": model_name,
                "version": version,
                "type": "custom",
                "path": str(model_path),
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9
            }

            # Send deployment request to edge node
            async with self.session.post(f"{model_endpoint}/load_model", json=deployment_data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success", False):
                        self.logger.info(f"Successfully deployed {model_name}:{version} to {node_id}")
                        return True
                    else:
                        self.logger.error(f"Deployment failed on {node_id}: {result.get('error', 'Unknown error')}")
                        return False
                else:
                    error_text = await response.text()
                    self.logger.error(f"Deployment request failed for {node_id}: {response.status} - {error_text}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to deploy to node {node_id}: {e}")
            return False

    async def _update_task_status(self, task_id: str):
        """Update sync task status in database"""
        try:
            task = self.sync_tasks[task_id]

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE sync_tasks
                    SET status = $1, progress_percent = $2, started_at = $3,
                        completed_at = $4, error_message = $5, retry_count = $6, updated_at = CURRENT_TIMESTAMP
                    WHERE task_id = $7
                """, task.status.value, task.progress_percent, task.started_at,
                task.completed_at, task.error_message, task.retry_count, task_id)

        except Exception as e:
            self.logger.error(f"Failed to update task status: {e}")

    async def _sync_scheduler(self):
        """Background scheduler for automatic model synchronization"""
        while True:
            try:
                await self._check_for_updates()
                await asyncio.sleep(3600)  # Check every hour
            except Exception as e:
                self.logger.error(f"Sync scheduler error: {e}")
                await asyncio.sleep(3600)

    async def _check_for_updates(self):
        """Check for model updates and schedule synchronization"""
        try:
            # Check each repository for updates
            for repo_name, repo in self.repositories.items():
                if not repo.sync_enabled:
                    continue

                # Repository-specific update checking logic
                if repo.type == "huggingface":
                    await self._check_huggingface_updates(repo)
                elif repo.type == "local":
                    await self._check_local_updates(repo)

        except Exception as e:
            self.logger.error(f"Update check failed: {e}")

    async def _check_huggingface_updates(self, repo: ModelRepository):
        """Check for updates in Hugging Face repository"""
        # Implementation for checking HuggingFace model updates
        # This would involve checking model cards, commits, etc.
        pass

    async def _check_local_updates(self, repo: ModelRepository):
        """Check for updates in local repository"""
        # Implementation for checking local repository updates
        pass

    async def _health_monitor(self):
        """Monitor edge node health and model deployment status"""
        while True:
            try:
                await self._check_edge_node_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _check_edge_node_health(self):
        """Check health of all edge nodes"""
        async with self.db_pool.acquire() as conn:
            nodes = await conn.fetch("SELECT node_id, endpoint FROM edge_nodes WHERE is_active = TRUE")

            for node in nodes:
                try:
                    async with self.session.get(f"{node['endpoint']}/health") as response:
                        if response.status == 200:
                            await conn.execute("""
                                UPDATE edge_nodes
                                SET last_health_check = CURRENT_TIMESTAMP
                                WHERE node_id = $1
                            """, node['node_id'])
                        else:
                            self.logger.warning(f"Node {node['node_id']} health check failed: {response.status}")
                except Exception as e:
                    self.logger.warning(f"Node {node['node_id']} health check error: {e}")

    async def _cleanup_manager(self):
        """Cleanup old models and temporary files"""
        while True:
            try:
                await self._cleanup_old_downloads()
                await self._cleanup_old_models()
                await asyncio.sleep(3600)  # Cleanup every hour
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)

    async def _cleanup_old_downloads(self):
        """Clean up old temporary download files"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            for file_path in self.temp_download_path.rglob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()

        except Exception as e:
            self.logger.error(f"Download cleanup failed: {e}")

    async def _cleanup_old_models(self):
        """Clean up old model versions"""
        try:
            # Keep only the latest 3 versions of each model
            for model_name, versions in self.model_versions.items():
                if len(versions) > 3:
                    # Sort by release date and keep newest
                    sorted_versions = sorted(versions, key=lambda v: v.release_date, reverse=True)
                    versions_to_remove = sorted_versions[3:]

                    for version_to_remove in versions_to_remove:
                        model_path = self.base_model_path / model_name / version_to_remove.version
                        if model_path.exists():
                            shutil.rmtree(model_path)
                            self.logger.info(f"Cleaned up old model version: {model_name}:{version_to_remove.version}")

        except Exception as e:
            self.logger.error(f"Model cleanup failed: {e}")

    async def get_sync_status(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Get synchronization status"""
        if task_id:
            # Get specific task status
            task = self.sync_tasks.get(task_id)
            if not task:
                return {"error": "Task not found"}

            return {
                "task_id": task.task_id,
                "model_name": task.model_name,
                "version": task.version,
                "status": task.status.value,
                "progress_percent": task.progress_percent,
                "target_regions": task.target_regions,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message,
                "retry_count": task.retry_count
            }
        else:
            # Get overall sync status
            return {
                "total_tasks": len(self.sync_tasks),
                "active_downloads": len(self.active_downloads),
                "repositories": len(self.repositories),
                "edge_nodes": len(self.edge_nodes),
                "model_versions": sum(len(versions) for versions in self.model_versions.values()),
                "tasks_by_status": {
                    status.value: len([task for task in self.sync_tasks.values() if task.status == status])
                    for status in ModelStatus
                }
            }

    async def cleanup(self):
        """Cleanup synchronizer resources"""
        try:
            # Cancel active downloads
            for task in self.active_downloads.values():
                task.cancel()

            # Close HTTP session
            if self.session:
                await self.session.close()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            self.logger.info("Model Synchronizer cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of ModelSynchronizer"""
    synchronizer = ModelSynchronizer()

    try:
        await synchronizer.initialize()

        # Add a model version
        model_version = ModelVersion(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            version="latest",
            model_path="microsoft/Phi-3-mini-4k-instruct",
            model_size_bytes=4000000000,  # 4GB estimate
            checksum="",
            release_date=datetime.utcnow(),
            compatibility_version="1.0",
            deployment_regions=["us-east", "us-west"],
            priority=SyncPriority.HIGH,
            metadata={"model_type": "phi-3", "quantization": "none"}
        )

        await synchronizer.add_model_version(model_version)

        # Sync to regions
        task_id = await synchronizer.sync_model_to_regions(
            model_name="microsoft/Phi-3-mini-4k-instruct",
            version="latest",
            regions=["us-east", "us-west"],
            priority=SyncPriority.HIGH
        )

        print(f"Started sync task: {task_id}")

        # Monitor progress
        while True:
            status = await synchronizer.get_sync_status(task_id)
            print(f"Sync progress: {status['progress_percent']:.1f}% - {status['status']}")

            if status['status'] in ['deployed', 'failed']:
                break

            await asyncio.sleep(5)

        print("Sync completed")

    finally:
        await synchronizer.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())