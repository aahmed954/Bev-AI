# BEV AI Assistant Platform - Developer Guide and Operations Manual

## Overview

This comprehensive guide provides developers and operations teams with the essential information needed to extend, maintain, and operate the BEV AI Assistant Platform. The guide covers development workflows, platform extension patterns, operational procedures, and troubleshooting methodologies for the revolutionary AI-powered cybersecurity research platform.

### **Target Audience**
- **Developers**: Platform extension, custom analyzer development, AI model integration
- **DevOps Engineers**: Deployment automation, infrastructure management, monitoring
- **Security Researchers**: Custom tool development, investigation workflow optimization
- **System Administrators**: Daily operations, maintenance, troubleshooting

## Development Environment Setup

### **Prerequisites and System Requirements**

#### **Hardware Requirements by Development Type**
```yaml
Frontend_Development:
  CPU: 8+ cores x86_64
  RAM: 16GB minimum, 32GB recommended
  GPU: Optional (RTX 3060+ for avatar development)
  Storage: 500GB+ SSD

Backend_Development:
  CPU: 16+ cores x86_64
  RAM: 32GB minimum, 64GB recommended
  GPU: RTX 3080+ for AI model development
  Storage: 1TB+ SSD

Full_Stack_Development:
  CPU: 16+ cores x86_64
  RAM: 64GB minimum
  GPU: RTX 4090 for complete development experience
  Storage: 2TB+ SSD
```

#### **Software Dependencies**
```bash
#!/bin/bash
# setup-development-environment.sh - Complete development environment setup

echo "🛠️ BEV Development Environment Setup"

# Core development tools
install_core_tools() {
    echo "📦 Installing core development tools..."

    # Update system
    sudo apt update && sudo apt upgrade -y

    # Essential development packages
    sudo apt install -y \
        build-essential \
        git \
        curl \
        wget \
        vim \
        tmux \
        htop \
        tree \
        jq \
        unzip \
        software-properties-common

    # Python development environment
    sudo apt install -y \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip

    # Node.js development environment
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs

    # Container development tools
    install_container_tools
}

# Container and orchestration tools
install_container_tools() {
    echo "🐳 Installing container development tools..."

    # Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER

    # Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose

    # NVIDIA Container Toolkit (for GPU development)
    install_nvidia_container_toolkit
}

# NVIDIA Container Toolkit for GPU development
install_nvidia_container_toolkit() {
    echo "🎮 Installing NVIDIA Container Toolkit..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker for NVIDIA
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
}

# Development IDE and tools
install_development_ides() {
    echo "💻 Installing development IDEs and tools..."

    # Visual Studio Code
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
    sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
    sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
    sudo apt update
    sudo apt install -y code

    # PyCharm Professional (optional)
    sudo snap install pycharm-professional --classic

    # Development utilities
    sudo apt install -y \
        git-flow \
        meld \
        httpie \
        postgresql-client \
        redis-tools
}

# Python development environment
setup_python_environment() {
    echo "🐍 Setting up Python development environment..."

    # Create virtual environment
    python3.11 -m venv venv-bev-dev
    source venv-bev-dev/bin/activate

    # Upgrade pip and install development tools
    pip install --upgrade pip setuptools wheel

    # Core development dependencies
    pip install \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
        transformers \
        langchain \
        fastapi \
        uvicorn \
        sqlalchemy \
        psycopg2-binary \
        redis \
        neo4j \
        pytest \
        pytest-cov \
        black \
        flake8 \
        mypy \
        pre-commit

    # Development environment configuration
    echo 'source venv-bev-dev/bin/activate' >> ~/.bashrc
}

# Git configuration for BEV development
configure_git_environment() {
    echo "📝 Configuring Git for BEV development..."

    # Git hooks and pre-commit
    pre-commit install

    # Git configuration for BEV
    git config --local core.autocrlf false
    git config --local pull.rebase true
    git config --local branch.autosetupmerge always
    git config --local branch.autosetuprebase always

    # Create development branch
    git checkout -b feature/development-setup
}

# Main installation
main() {
    install_core_tools
    install_development_ides
    setup_python_environment
    configure_git_environment

    echo "✅ BEV development environment setup complete!"
    echo "🚀 Next steps:"
    echo "  1. Restart terminal to apply group changes"
    echo "  2. cd ~/Projects/Bev && source venv-bev-dev/bin/activate"
    echo "  3. Run: pytest tests/ to validate environment"
}

main "$@"
```

### **IDE Configuration and Extensions**

#### **Visual Studio Code Configuration**
```json
// .vscode/settings.json - BEV project configuration
{
    "python.defaultInterpreterPath": "./venv-bev-dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/.git": false
    },
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "docker.enableDockerComposeLanguageService": true,
    "yaml.schemas": {
        "https://json.schemastore.org/docker-compose.json": "docker-compose*.yml"
    }
}
```

#### **Recommended Extensions**
```json
// .vscode/extensions.json - Recommended extensions for BEV development
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-toolsai.jupyter",
        "ms-vscode.docker",
        "redhat.vscode-yaml",
        "ms-vscode.vscode-json",
        "github.copilot",
        "ms-vscode-remote.remote-containers",
        "bradlc.vscode-tailwindcss",
        "esbenp.prettier-vscode"
    ]
}
```

## Platform Architecture for Developers

### **Core Architecture Patterns**

#### **Service-Oriented Architecture (SOA)**
```python
# Core architectural pattern for BEV services
class BEVServiceBase:
    """
    Base class for all BEV services
    Implements common patterns: logging, metrics, health checks, configuration
    """

    def __init__(self, service_name: str, config: Dict):
        self.service_name = service_name
        self.config = config
        self.logger = self._setup_logging()
        self.metrics = self._setup_metrics()
        self.health_checker = self._setup_health_checker()

    def _setup_logging(self) -> logging.Logger:
        """Standardized logging configuration for all services"""
        logger = logging.getLogger(self.service_name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def _setup_metrics(self) -> PrometheusMetrics:
        """Standardized metrics collection for monitoring"""
        return PrometheusMetrics(service_name=self.service_name)

    def _setup_health_checker(self) -> HealthChecker:
        """Standardized health checking for service monitoring"""
        return HealthChecker(service_name=self.service_name)

    async def startup(self):
        """Service startup lifecycle hook"""
        self.logger.info(f"Starting {self.service_name} service...")
        await self._initialize_dependencies()
        await self._setup_service_specific_components()
        self.logger.info(f"{self.service_name} service started successfully")

    async def shutdown(self):
        """Service shutdown lifecycle hook"""
        self.logger.info(f"Shutting down {self.service_name} service...")
        await self._cleanup_resources()
        self.logger.info(f"{self.service_name} service shutdown complete")

    async def _initialize_dependencies(self):
        """Override in subclasses for dependency initialization"""
        pass

    async def _setup_service_specific_components(self):
        """Override in subclasses for service-specific setup"""
        pass

    async def _cleanup_resources(self):
        """Override in subclasses for resource cleanup"""
        pass
```

#### **Plugin Architecture for Custom Analyzers**
```python
# Plugin system for custom OSINT analyzers
class OSINTAnalyzerPlugin:
    """
    Base class for OSINT analyzer plugins
    Enables developers to create custom analyzers that integrate seamlessly
    """

    def __init__(self, plugin_name: str, version: str):
        self.plugin_name = plugin_name
        self.version = version
        self.capabilities = self._define_capabilities()

    def _define_capabilities(self) -> List[str]:
        """Override to define analyzer capabilities"""
        raise NotImplementedError

    async def analyze(self, input_data: Dict) -> AnalysisResult:
        """Main analysis method - override in implementations"""
        raise NotImplementedError

    async def validate_input(self, input_data: Dict) -> bool:
        """Input validation - override for custom validation logic"""
        return True

    def get_metadata(self) -> Dict:
        """Plugin metadata for registration and discovery"""
        return {
            'name': self.plugin_name,
            'version': self.version,
            'capabilities': self.capabilities,
            'input_schema': self._get_input_schema(),
            'output_schema': self._get_output_schema()
        }

# Example custom analyzer implementation
class CustomThreatAnalyzer(OSINTAnalyzerPlugin):
    """
    Example custom threat analyzer implementation
    Demonstrates how to create domain-specific analyzers
    """

    def __init__(self):
        super().__init__("custom_threat_analyzer", "1.0.0")

    def _define_capabilities(self) -> List[str]:
        return ["threat_detection", "ioc_analysis", "behavioral_analysis"]

    async def analyze(self, input_data: Dict) -> AnalysisResult:
        """Custom threat analysis implementation"""
        # Implement custom analysis logic
        threat_indicators = await self._extract_threat_indicators(input_data)
        behavioral_patterns = await self._analyze_behavioral_patterns(input_data)
        risk_score = await self._calculate_risk_score(threat_indicators, behavioral_patterns)

        return AnalysisResult(
            threat_indicators=threat_indicators,
            behavioral_patterns=behavioral_patterns,
            risk_score=risk_score,
            confidence=self._calculate_confidence(threat_indicators)
        )

    async def _extract_threat_indicators(self, input_data: Dict) -> List[ThreatIndicator]:
        """Extract threat indicators from input data"""
        # Custom implementation
        pass

    async def _analyze_behavioral_patterns(self, input_data: Dict) -> List[BehavioralPattern]:
        """Analyze behavioral patterns for threat assessment"""
        # Custom implementation
        pass
```

### **Database Integration Patterns**

#### **Multi-Database Access Layer**
```python
# Database abstraction layer for BEV platform
class BEVDatabaseManager:
    """
    Unified database access layer for BEV platform
    Manages connections to PostgreSQL, Neo4j, Redis, and Elasticsearch
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.postgres = None
        self.neo4j = None
        self.redis = None
        self.elasticsearch = None

    async def initialize(self):
        """Initialize all database connections"""
        self.postgres = await self._initialize_postgresql()
        self.neo4j = await self._initialize_neo4j()
        self.redis = await self._initialize_redis()
        self.elasticsearch = await self._initialize_elasticsearch()

    async def _initialize_postgresql(self) -> AsyncPostgreSQLConnection:
        """Initialize PostgreSQL connection with pgvector extension"""
        engine = create_async_engine(
            f"postgresql+asyncpg://{self.config.postgres.user}:"
            f"{self.config.postgres.password}@{self.config.postgres.host}:"
            f"{self.config.postgres.port}/{self.config.postgres.database}"
        )
        return AsyncSession(engine)

    async def _initialize_neo4j(self) -> Neo4jConnection:
        """Initialize Neo4j connection for graph operations"""
        driver = AsyncGraphDatabase.driver(
            f"bolt://{self.config.neo4j.host}:{self.config.neo4j.port}",
            auth=(self.config.neo4j.user, self.config.neo4j.password)
        )
        return driver

    async def _initialize_redis(self) -> RedisConnection:
        """Initialize Redis connection for caching and sessions"""
        return aioredis.from_url(
            f"redis://{self.config.redis.host}:{self.config.redis.port}",
            password=self.config.redis.password
        )

    async def _initialize_elasticsearch(self) -> ElasticsearchConnection:
        """Initialize Elasticsearch connection for search operations"""
        return AsyncElasticsearch([
            f"http://{self.config.elasticsearch.host}:{self.config.elasticsearch.port}"
        ])

# Database operation patterns
class OSINTDataRepository:
    """
    Repository pattern for OSINT data operations
    Provides high-level interface for OSINT data management
    """

    def __init__(self, db_manager: BEVDatabaseManager):
        self.db = db_manager

    async def store_threat_intelligence(self, intelligence: ThreatIntelligence) -> str:
        """Store threat intelligence with appropriate database selection"""
        # Store structured data in PostgreSQL
        intelligence_id = await self._store_in_postgresql(intelligence)

        # Store relationships in Neo4j
        await self._store_relationships_in_neo4j(intelligence_id, intelligence.relationships)

        # Cache frequently accessed data in Redis
        await self._cache_in_redis(intelligence_id, intelligence.summary)

        # Index for search in Elasticsearch
        await self._index_in_elasticsearch(intelligence_id, intelligence.searchable_content)

        return intelligence_id

    async def query_related_threats(self, threat_id: str, relationship_types: List[str]) -> List[ThreatIntelligence]:
        """Query related threats using Neo4j graph traversal"""
        query = """
        MATCH (t:Threat {id: $threat_id})-[r]->(related:Threat)
        WHERE type(r) IN $relationship_types
        RETURN related
        """
        results = await self.db.neo4j.run(query, threat_id=threat_id, relationship_types=relationship_types)
        return [self._convert_to_threat_intelligence(record) for record in results]
```

## Custom Development Patterns

### **Creating Custom OSINT Analyzers**

#### **Analyzer Development Template**
```python
# Template for creating custom OSINT analyzers
class CustomOSINTAnalyzer(OSINTAnalyzerPlugin):
    """
    Template for custom OSINT analyzer development
    Follow this pattern for creating domain-specific analyzers
    """

    def __init__(self, analyzer_name: str):
        super().__init__(analyzer_name, "1.0.0")
        self.ai_models = self._initialize_ai_models()
        self.data_sources = self._initialize_data_sources()

    def _define_capabilities(self) -> List[str]:
        """Define what this analyzer can do"""
        return [
            "domain_specific_analysis",      # Replace with actual capabilities
            "threat_pattern_recognition",
            "behavioral_profiling"
        ]

    async def analyze(self, input_data: Dict) -> AnalysisResult:
        """Main analysis workflow"""
        try:
            # Step 1: Validate and preprocess input
            validated_data = await self.validate_and_preprocess(input_data)

            # Step 2: Perform domain-specific analysis
            analysis_results = await self._perform_analysis(validated_data)

            # Step 3: Apply AI enhancement
            enhanced_results = await self._apply_ai_enhancement(analysis_results)

            # Step 4: Generate actionable intelligence
            actionable_intelligence = await self._generate_intelligence(enhanced_results)

            return AnalysisResult(
                raw_results=analysis_results,
                enhanced_results=enhanced_results,
                actionable_intelligence=actionable_intelligence,
                confidence=self._calculate_confidence(enhanced_results),
                metadata=self._generate_metadata()
            )

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            return AnalysisResult(error=str(e), confidence=0.0)

    async def _perform_analysis(self, data: Dict) -> Dict:
        """Implement domain-specific analysis logic here"""
        # Example: Analyze network infrastructure
        infrastructure_analysis = await self._analyze_infrastructure(data)

        # Example: Analyze behavioral patterns
        behavioral_analysis = await self._analyze_behavior_patterns(data)

        # Example: Correlate with threat intelligence
        threat_correlation = await self._correlate_threat_intelligence(data)

        return {
            'infrastructure': infrastructure_analysis,
            'behavioral': behavioral_analysis,
            'threat_correlation': threat_correlation
        }

    async def _apply_ai_enhancement(self, results: Dict) -> Dict:
        """Apply AI models for enhanced analysis"""
        # Use pre-trained models for enhancement
        enhanced_results = {}

        for analysis_type, data in results.items():
            if analysis_type in self.ai_models:
                model = self.ai_models[analysis_type]
                enhanced_data = await model.enhance(data)
                enhanced_results[analysis_type] = enhanced_data
            else:
                enhanced_results[analysis_type] = data

        return enhanced_results

    def _get_input_schema(self) -> Dict:
        """Define expected input format"""
        return {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Analysis target"},
                "analysis_type": {"type": "string", "enum": self.capabilities},
                "parameters": {"type": "object", "description": "Analysis parameters"}
            },
            "required": ["target", "analysis_type"]
        }

    def _get_output_schema(self) -> Dict:
        """Define output format"""
        return {
            "type": "object",
            "properties": {
                "analysis_results": {"type": "object"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "metadata": {"type": "object"}
            }
        }
```

#### **Analyzer Registration and Discovery**
```python
# Analyzer registration system
class AnalyzerRegistry:
    """
    Registry for OSINT analyzer plugins
    Manages plugin discovery, registration, and lifecycle
    """

    def __init__(self):
        self.registered_analyzers = {}
        self.analyzer_metadata = {}

    def register_analyzer(self, analyzer: OSINTAnalyzerPlugin) -> bool:
        """Register a new analyzer plugin"""
        try:
            # Validate analyzer
            if not self._validate_analyzer(analyzer):
                return False

            # Register analyzer
            self.registered_analyzers[analyzer.plugin_name] = analyzer
            self.analyzer_metadata[analyzer.plugin_name] = analyzer.get_metadata()

            self.logger.info(f"Registered analyzer: {analyzer.plugin_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register analyzer {analyzer.plugin_name}: {str(e)}")
            return False

    def discover_analyzers(self, plugin_directory: str) -> List[OSINTAnalyzerPlugin]:
        """Discover analyzer plugins from directory"""
        discovered_analyzers = []

        for plugin_file in Path(plugin_directory).glob("*_analyzer.py"):
            try:
                analyzer = self._load_analyzer_from_file(plugin_file)
                if self.register_analyzer(analyzer):
                    discovered_analyzers.append(analyzer)

            except Exception as e:
                self.logger.error(f"Failed to load analyzer from {plugin_file}: {str(e)}")

        return discovered_analyzers

    def get_analyzer(self, analyzer_name: str) -> Optional[OSINTAnalyzerPlugin]:
        """Get registered analyzer by name"""
        return self.registered_analyzers.get(analyzer_name)

    def list_analyzers(self) -> List[Dict]:
        """List all registered analyzers with metadata"""
        return list(self.analyzer_metadata.values())
```

### **AI Model Integration Patterns**

#### **Custom AI Model Integration**
```python
# AI model integration for custom capabilities
class BEVAIModelIntegration:
    """
    Integration layer for AI models in BEV platform
    Supports custom model integration with GPU optimization
    """

    def __init__(self, gpu_manager: GPUManager):
        self.gpu_manager = gpu_manager
        self.loaded_models = {}
        self.model_cache = {}

    async def load_custom_model(self, model_config: ModelConfig) -> AIModel:
        """Load custom AI model with GPU optimization"""
        if model_config.name in self.loaded_models:
            return self.loaded_models[model_config.name]

        # Allocate GPU memory for model
        gpu_allocation = await self.gpu_manager.allocate_memory(
            model_config.memory_requirement,
            priority=model_config.priority
        )

        try:
            # Load model with GPU allocation
            model = await self._load_model_with_gpu(model_config, gpu_allocation)

            # Cache model for reuse
            self.loaded_models[model_config.name] = model
            self.model_cache[model_config.name] = {
                'model': model,
                'gpu_allocation': gpu_allocation,
                'last_used': datetime.now()
            }

            return model

        except Exception as e:
            # Release GPU allocation on failure
            await self.gpu_manager.release_memory(gpu_allocation)
            raise

    async def inference_with_batching(self, model_name: str, input_data: List[Dict]) -> List[Dict]:
        """Perform inference with intelligent batching for performance"""
        model = self.loaded_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not loaded")

        # Optimize batch size based on GPU memory and input size
        optimal_batch_size = await self._calculate_optimal_batch_size(model, input_data)

        # Process in optimized batches
        results = []
        for batch_start in range(0, len(input_data), optimal_batch_size):
            batch_end = min(batch_start + optimal_batch_size, len(input_data))
            batch_data = input_data[batch_start:batch_end]

            batch_results = await model.inference(batch_data)
            results.extend(batch_results)

        return results

    async def _calculate_optimal_batch_size(self, model: AIModel, sample_data: List[Dict]) -> int:
        """Calculate optimal batch size based on GPU memory and data characteristics"""
        # Get available GPU memory
        available_memory = await self.gpu_manager.get_available_memory()

        # Estimate memory per sample
        sample_memory = await self._estimate_sample_memory(model, sample_data[0])

        # Calculate optimal batch size with safety margin
        safety_margin = 0.8  # Use 80% of available memory
        optimal_batch_size = int((available_memory * safety_margin) / sample_memory)

        return max(1, min(optimal_batch_size, len(sample_data)))
```

## Testing Frameworks and Quality Assurance

### **Comprehensive Testing Strategy**

#### **Unit Testing Framework**
```python
# Unit testing framework for BEV components
import pytest
from unittest.mock import AsyncMock, MagicMock

class TestOSINTAnalyzer:
    """
    Example unit test suite for OSINT analyzers
    Demonstrates testing patterns for BEV components
    """

    @pytest.fixture
    async def analyzer(self):
        """Create analyzer instance for testing"""
        config = {
            'ai_models': {'test_model': 'mock'},
            'data_sources': {'test_source': 'mock'}
        }
        analyzer = CustomOSINTAnalyzer("test_analyzer")
        analyzer.ai_models = {'test_model': AsyncMock()}
        analyzer.data_sources = {'test_source': AsyncMock()}
        return analyzer

    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing"""
        return {
            'target': 'test.example.com',
            'analysis_type': 'domain_analysis',
            'parameters': {'depth': 2}
        }

    async def test_analyze_success(self, analyzer, sample_input_data):
        """Test successful analysis workflow"""
        # Mock dependencies
        analyzer._perform_analysis = AsyncMock(return_value={'test': 'data'})
        analyzer._apply_ai_enhancement = AsyncMock(return_value={'enhanced': 'data'})
        analyzer._generate_intelligence = AsyncMock(return_value={'intelligence': 'data'})

        # Execute analysis
        result = await analyzer.analyze(sample_input_data)

        # Verify results
        assert result.confidence > 0
        assert result.error is None
        assert 'enhanced' in result.enhanced_results

    async def test_analyze_invalid_input(self, analyzer):
        """Test analysis with invalid input"""
        invalid_input = {'invalid': 'data'}

        result = await analyzer.analyze(invalid_input)

        assert result.confidence == 0.0
        assert result.error is not None

    async def test_ai_enhancement(self, analyzer):
        """Test AI enhancement functionality"""
        test_data = {'infrastructure': {'domains': ['test.com']}}

        # Mock AI model
        mock_model = AsyncMock()
        mock_model.enhance.return_value = {'enhanced_domains': ['test.com']}
        analyzer.ai_models = {'infrastructure': mock_model}

        result = await analyzer._apply_ai_enhancement(test_data)

        assert 'infrastructure' in result
        assert 'enhanced_domains' in result['infrastructure']
        mock_model.enhance.assert_called_once()

    @pytest.mark.performance
    async def test_performance_requirements(self, analyzer, sample_input_data):
        """Test performance requirements"""
        import time

        start_time = time.time()
        result = await analyzer.analyze(sample_input_data)
        end_time = time.time()

        # Verify performance requirements
        analysis_time = end_time - start_time
        assert analysis_time < 30.0  # Analysis should complete within 30 seconds
        assert result.confidence > 0.7  # Confidence should be high for test data
```

#### **Integration Testing Framework**
```python
# Integration testing for multi-service workflows
class TestMultiServiceIntegration:
    """
    Integration tests for multi-service BEV workflows
    Tests end-to-end functionality across services
    """

    @pytest.fixture(scope="class")
    async def test_environment(self):
        """Setup test environment with required services"""
        # Start test containers
        test_containers = await self._start_test_containers()

        # Initialize test data
        await self._initialize_test_data()

        yield test_containers

        # Cleanup
        await self._cleanup_test_environment(test_containers)

    async def test_osint_investigation_workflow(self, test_environment):
        """Test complete OSINT investigation workflow"""
        # Submit investigation request
        investigation_request = {
            'target': 'test-threat-actor.example.com',
            'investigation_type': 'comprehensive',
            'analyzers': ['domain_analyzer', 'threat_intel_analyzer']
        }

        # Execute investigation
        investigation_id = await self._submit_investigation(investigation_request)
        assert investigation_id is not None

        # Wait for completion
        result = await self._wait_for_investigation_completion(investigation_id, timeout=60)

        # Verify results
        assert result.status == 'completed'
        assert len(result.findings) > 0
        assert result.confidence > 0.5

    async def test_cross_node_communication(self, test_environment):
        """Test communication between THANOS and ORACLE1 nodes"""
        # Test THANOS → ORACLE1 monitoring data flow
        metrics_sent = await self._send_test_metrics_from_thanos()
        received_metrics = await self._query_metrics_from_oracle1()

        assert metrics_sent == received_metrics

        # Test ORACLE1 → THANOS alerting
        test_alert = await self._trigger_test_alert_from_oracle1()
        alert_received = await self._check_alert_received_by_thanos()

        assert alert_received
        assert test_alert.alert_id == alert_received.alert_id

    async def test_ai_model_coordination(self, test_environment):
        """Test AI model coordination across nodes"""
        # Submit analysis requiring multiple AI models
        analysis_request = {
            'type': 'multi_modal_analysis',
            'data': {
                'text': 'Sample threat intelligence text',
                'network_data': {'ips': ['192.168.1.1']},
                'behavioral_data': {'patterns': ['login_anomaly']}
            }
        }

        result = await self._submit_multi_modal_analysis(analysis_request)

        # Verify AI model coordination
        assert 'text_analysis' in result
        assert 'network_analysis' in result
        assert 'behavioral_analysis' in result
        assert result.overall_confidence > 0.6
```

### **Performance Testing and Benchmarking**

#### **Performance Test Suite**
```python
# Performance testing framework
class TestPlatformPerformance:
    """
    Performance tests for BEV platform components
    Validates performance requirements and benchmarks
    """

    @pytest.mark.performance
    async def test_concurrent_analysis_performance(self):
        """Test performance under concurrent analysis load"""
        # Create multiple concurrent analysis requests
        analysis_requests = [
            self._create_analysis_request(f"target_{i}")
            for i in range(100)
        ]

        start_time = time.time()

        # Execute concurrent analyses
        results = await asyncio.gather(*[
            self._execute_analysis(request)
            for request in analysis_requests
        ])

        end_time = time.time()

        # Verify performance requirements
        total_time = end_time - start_time
        average_time_per_analysis = total_time / len(results)

        assert total_time < 120.0  # Complete within 2 minutes
        assert average_time_per_analysis < 5.0  # Average < 5 seconds per analysis
        assert all(result.success for result in results)  # All analyses successful

    @pytest.mark.performance
    async def test_gpu_memory_utilization(self):
        """Test GPU memory utilization efficiency"""
        # Monitor GPU memory before test
        initial_gpu_memory = await self._get_gpu_memory_usage()

        # Execute GPU-intensive operations
        for i in range(10):
            await self._execute_gpu_intensive_analysis()

        # Monitor GPU memory after test
        final_gpu_memory = await self._get_gpu_memory_usage()

        # Verify memory efficiency
        memory_increase = final_gpu_memory - initial_gpu_memory
        assert memory_increase < 1024  # Less than 1GB memory leak
        assert final_gpu_memory < 8192  # Stay within RTX 3080 limits

    @pytest.mark.performance
    async def test_database_query_performance(self):
        """Test database query performance requirements"""
        test_queries = [
            "SELECT * FROM threat_intelligence WHERE confidence > 0.8 LIMIT 100",
            "MATCH (t:Threat)-[:RELATES_TO]-(r:ThreatActor) RETURN t, r LIMIT 50",
            "GET threat_indicators_cache",
            "POST /api/search {'query': 'APT', 'filters': ['high_confidence']}"
        ]

        for query in test_queries:
            start_time = time.time()
            result = await self._execute_query(query)
            end_time = time.time()

            query_time = end_time - start_time
            assert query_time < 1.0  # All queries < 1 second
            assert result is not None
```

## Operations and Maintenance

### **Daily Operations Procedures**

#### **Health Monitoring and Alerting**
```bash
#!/bin/bash
# daily-operations-checklist.sh - Daily operations checklist for BEV platform

echo "🔍 BEV Platform Daily Operations Checklist"
echo "Date: $(date)"
echo "Operator: $USER"
echo "=================================================="

# System health overview
check_system_health() {
    echo "🏥 System Health Overview"

    # Node status
    echo "📊 Node Status:"
    echo "  🤖 STARLORD: $(check_starlord_status)"
    echo "  🏛️ THANOS: $(check_thanos_status)"
    echo "  🔍 ORACLE1: $(check_oracle1_status)"

    # Service health
    echo "🔧 Service Health:"
    check_critical_services

    # Resource utilization
    echo "📈 Resource Utilization:"
    check_resource_utilization

    echo ""
}

# Check critical services across all nodes
check_critical_services() {
    local services=(
        "100.122.12.54:80:OSINT Platform"
        "100.122.12.54:3010:MCP Server"
        "100.122.12.54:5432:PostgreSQL"
        "100.122.12.54:6379:Redis"
        "100.96.197.84:3000:Grafana"
        "100.96.197.84:9090:Prometheus"
        "100.96.197.84:8200:Vault"
    )

    for service in "${services[@]}"; do
        IFS=':' read -r host port name <<< "$service"
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "  ✅ $name ($host:$port)"
        else
            echo "  ❌ $name ($host:$port) - NOT RESPONDING"
        fi
    done
}

# Resource utilization monitoring
check_resource_utilization() {
    # GPU utilization
    echo "  🎮 GPU Status:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
                  --format=csv,noheader | while read line; do
            echo "    $line"
        done
    else
        echo "    No GPU detected on this node"
    fi

    # Memory usage
    echo "  💾 Memory Usage:"
    free -h | grep -E "Mem:|Swap:" | while read line; do
        echo "    $line"
    done

    # Disk usage
    echo "  💿 Disk Usage:"
    df -h | grep -E "/$|/var|/home" | while read line; do
        echo "    $line"
    done
}

# Performance metrics collection
collect_performance_metrics() {
    echo "📊 Performance Metrics Collection"

    # Database performance
    echo "🗄️ Database Performance:"
    check_database_performance

    # API response times
    echo "🌐 API Response Times:"
    check_api_response_times

    # Investigation throughput
    echo "🔍 Investigation Throughput:"
    check_investigation_metrics

    echo ""
}

# Database performance monitoring
check_database_performance() {
    # PostgreSQL performance
    if nc -z 100.122.12.54 5432; then
        echo "  📊 PostgreSQL:"
        local pg_stats=$(docker exec thanos_postgres psql -U researcher -d osint_primary -c \
            "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';" \
            -t | tr -d ' ')
        echo "    Active connections: $pg_stats"
    fi

    # Redis performance
    if nc -z 100.122.12.54 6379; then
        echo "  🔴 Redis:"
        local redis_info=$(docker exec thanos_redis redis-cli info memory | grep used_memory_human)
        echo "    $redis_info"
    fi

    # Neo4j performance
    if nc -z 100.122.12.54 7474; then
        echo "  🕸️ Neo4j:"
        local neo4j_stats=$(curl -s -u neo4j:BevGraphMaster2024 \
            http://100.122.12.54:7474/db/data/cypher \
            -H "Content-Type: application/json" \
            -d '{"query":"MATCH (n) RETURN count(n) as node_count"}' \
            | jq -r '.data[0][0]' 2>/dev/null || echo "N/A")
        echo "    Total nodes: $neo4j_stats"
    fi
}

# API response time monitoring
check_api_response_times() {
    local endpoints=(
        "http://100.122.12.54/api/health:OSINT API"
        "http://100.122.12.54:3010/mcp/health:MCP Server"
        "http://100.96.197.84:3000/api/health:Grafana"
        "http://100.96.197.84:9090/-/healthy:Prometheus"
    )

    for endpoint in "${endpoints[@]}"; do
        IFS=':' read -r url name <<< "$endpoint"
        local start_time=$(date +%s%N)
        if curl -f -s "$url" > /dev/null; then
            local end_time=$(date +%s%N)
            local response_time=$(((end_time - start_time) / 1000000))
            echo "  ✅ $name: ${response_time}ms"
        else
            echo "  ❌ $name: TIMEOUT/ERROR"
        fi
    done
}

# Security monitoring
check_security_status() {
    echo "🔒 Security Status Check"

    # Vault status
    echo "🔐 Vault Status:"
    if curl -f -s http://100.96.197.84:8200/v1/sys/health | jq -r .sealed | grep -q "false"; then
        echo "  ✅ Vault: Unsealed and healthy"
    else
        echo "  ⚠️ Vault: Sealed or unhealthy - requires attention"
    fi

    # Tailscale VPN status
    echo "🔗 VPN Status:"
    if tailscale status | grep -q "100.122.12.54\|100.96.197.84"; then
        echo "  ✅ Tailscale VPN: Connected to all nodes"
    else
        echo "  ⚠️ Tailscale VPN: Some nodes disconnected"
    fi

    # Certificate expiry check
    echo "🏆 Certificate Status:"
    check_certificate_expiry

    echo ""
}

# Certificate expiry monitoring
check_certificate_expiry() {
    local cert_endpoints=(
        "100.122.12.54:443"
        "100.96.197.84:8200"
    )

    for endpoint in "${cert_endpoints[@]}"; do
        local cert_info=$(echo | openssl s_client -connect "$endpoint" 2>/dev/null | \
                         openssl x509 -noout -dates 2>/dev/null | grep notAfter)
        if [ -n "$cert_info" ]; then
            echo "  📜 $endpoint: $cert_info"
        else
            echo "  ℹ️ $endpoint: No SSL certificate or connection failed"
        fi
    done
}

# Generate daily report
generate_daily_report() {
    echo "📋 Daily Operations Report"

    local report_file="daily_report_$(date +%Y%m%d).txt"

    {
        echo "BEV Platform Daily Operations Report"
        echo "Generated: $(date)"
        echo "Operator: $USER"
        echo "=================================="
        echo ""

        check_system_health
        collect_performance_metrics
        check_security_status

        echo "🎯 Action Items:"
        echo "  - Review any failed services and restart if necessary"
        echo "  - Monitor resource utilization trends"
        echo "  - Verify backup completion (check backup logs)"
        echo "  - Update threat intelligence feeds if scheduled"
        echo ""

        echo "📊 Monitoring Dashboard: http://100.96.197.84:3000/"
        echo "🔍 OSINT Platform: http://100.122.12.54/"
        echo "🔒 Vault Interface: http://100.96.197.84:8200/"

    } | tee "$report_file"

    echo "📄 Daily report saved to: $report_file"
}

# Main execution
main() {
    check_system_health
    collect_performance_metrics
    check_security_status
    generate_daily_report

    echo "✅ Daily operations checklist completed!"
    echo "📊 For detailed monitoring: http://100.96.197.84:3000/"
    echo "🚨 Alert status: Check AlertManager at http://100.96.197.84:9093/"
}

main "$@"
```

### **Backup and Recovery Procedures**

#### **Comprehensive Backup Strategy**
```bash
#!/bin/bash
# comprehensive-backup-system.sh - Complete backup and recovery system

echo "💾 BEV Platform Comprehensive Backup System"

# Backup configuration
BACKUP_BASE_DIR="/backup/bev"
RETENTION_DAYS=30
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup directories
create_backup_structure() {
    echo "📁 Creating backup directory structure..."

    local backup_dirs=(
        "$BACKUP_BASE_DIR/databases/$TIMESTAMP"
        "$BACKUP_BASE_DIR/configurations/$TIMESTAMP"
        "$BACKUP_BASE_DIR/vault/$TIMESTAMP"
        "$BACKUP_BASE_DIR/logs/$TIMESTAMP"
        "$BACKUP_BASE_DIR/ai_models/$TIMESTAMP"
    )

    for dir in "${backup_dirs[@]}"; do
        mkdir -p "$dir"
    done

    echo "✅ Backup directory structure created"
}

# Database backup procedures
backup_databases() {
    echo "🗄️ Backing up databases..."

    local db_backup_dir="$BACKUP_BASE_DIR/databases/$TIMESTAMP"

    # PostgreSQL backup
    echo "📊 Backing up PostgreSQL..."
    docker exec thanos_postgres pg_dump -U researcher -d osint_primary \
        | gzip > "$db_backup_dir/postgresql_osint_primary.sql.gz"

    # Backup PostgreSQL configuration
    docker exec thanos_postgres cat /var/lib/postgresql/data/postgresql.conf \
        > "$db_backup_dir/postgresql.conf"

    # Redis backup
    echo "🔴 Backing up Redis..."
    docker exec thanos_redis redis-cli BGSAVE
    sleep 10  # Wait for background save to complete
    docker cp thanos_redis:/data/dump.rdb "$db_backup_dir/redis_dump.rdb"

    # Neo4j backup
    echo "🕸️ Backing up Neo4j..."
    docker exec thanos_neo4j neo4j-admin database dump \
        --database=neo4j --to-path=/tmp/backup
    docker cp thanos_neo4j:/tmp/backup/neo4j.dump "$db_backup_dir/neo4j_database.dump"

    # Elasticsearch backup (if available)
    if nc -z 100.122.12.54 9200; then
        echo "🔍 Backing up Elasticsearch..."
        curl -X PUT "100.122.12.54:9200/_snapshot/bev_backup_repo" \
             -H 'Content-Type: application/json' \
             -d '{"type": "fs", "settings": {"location": "/tmp/elasticsearch_backup"}}'

        curl -X PUT "100.122.12.54:9200/_snapshot/bev_backup_repo/snapshot_$TIMESTAMP"
        sleep 30  # Wait for snapshot completion

        docker cp elasticsearch:/tmp/elasticsearch_backup "$db_backup_dir/elasticsearch_snapshot"
    fi

    echo "✅ Database backups completed"
}

# Configuration backup
backup_configurations() {
    echo "⚙️ Backing up configurations..."

    local config_backup_dir="$BACKUP_BASE_DIR/configurations/$TIMESTAMP"

    # Docker compose files
    cp docker-compose-*.yml "$config_backup_dir/"

    # Environment files
    cp .env.* "$config_backup_dir/"

    # Configuration directories
    tar -czf "$config_backup_dir/config_directory.tar.gz" config/

    # Service configurations
    if [ -d "monitoring/config" ]; then
        tar -czf "$config_backup_dir/monitoring_config.tar.gz" monitoring/config/
    fi

    # SSL certificates (if any)
    if [ -d "certs" ]; then
        tar -czf "$config_backup_dir/certificates.tar.gz" certs/
    fi

    echo "✅ Configuration backups completed"
}

# Vault backup (requires unsealed vault)
backup_vault() {
    echo "🔐 Backing up Vault..."

    local vault_backup_dir="$BACKUP_BASE_DIR/vault/$TIMESTAMP"

    # Check if Vault is unsealed
    if curl -f -s http://100.96.197.84:8200/v1/sys/health | jq -r .sealed | grep -q "false"; then
        # Create Vault snapshot
        docker exec oracle1_vault vault operator raft snapshot save /tmp/vault-snapshot.snap

        # Copy snapshot
        docker cp oracle1_vault:/tmp/vault-snapshot.snap "$vault_backup_dir/"

        # Backup Vault configuration
        docker exec oracle1_vault cat /vault/config/vault.hcl > "$vault_backup_dir/vault.hcl"

        echo "✅ Vault backup completed"
    else
        echo "⚠️ Vault is sealed - skipping Vault backup"
        echo "ℹ️ To backup Vault, unseal it first and re-run backup"
    fi
}

# Application logs backup
backup_logs() {
    echo "📋 Backing up application logs..."

    local logs_backup_dir="$BACKUP_BASE_DIR/logs/$TIMESTAMP"

    # Docker container logs
    docker-compose -f docker-compose-thanos-unified.yml logs --no-color > "$logs_backup_dir/thanos_services.log"
    docker-compose -f docker-compose-oracle1-unified.yml logs --no-color > "$logs_backup_dir/oracle1_services.log"

    # System logs
    if [ -d "/var/log/bev" ]; then
        tar -czf "$logs_backup_dir/system_logs.tar.gz" /var/log/bev/
    fi

    # Application-specific logs
    if [ -d "logs" ]; then
        tar -czf "$logs_backup_dir/application_logs.tar.gz" logs/
    fi

    echo "✅ Log backups completed"
}

# AI models backup
backup_ai_models() {
    echo "🤖 Backing up AI models..."

    local models_backup_dir="$BACKUP_BASE_DIR/ai_models/$TIMESTAMP"

    # Custom trained models
    if [ -d "models" ]; then
        tar -czf "$models_backup_dir/custom_models.tar.gz" models/
    fi

    # Model configurations
    if [ -f "model_configs.json" ]; then
        cp model_configs.json "$models_backup_dir/"
    fi

    # Model cache (if preserving)
    if [ -d ".model_cache" ]; then
        echo "ℹ️ Model cache found - backing up (this may take time)..."
        tar -czf "$models_backup_dir/model_cache.tar.gz" .model_cache/
    fi

    echo "✅ AI models backup completed"
}

# Cleanup old backups
cleanup_old_backups() {
    echo "🧹 Cleaning up old backups..."

    # Remove backups older than retention period
    find "$BACKUP_BASE_DIR" -type d -name "*_*" -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null

    # Report cleanup results
    local remaining_backups=$(find "$BACKUP_BASE_DIR" -type d -name "*_*" | wc -l)
    echo "✅ Cleanup completed - $remaining_backups backups retained"
}

# Backup verification
verify_backup() {
    echo "🔍 Verifying backup integrity..."

    local backup_dir="$BACKUP_BASE_DIR"
    local verification_log="$backup_dir/verification_$TIMESTAMP.log"

    {
        echo "Backup Verification Report"
        echo "Generated: $(date)"
        echo "=========================="
        echo ""

        # Check database backups
        echo "Database Backups:"
        if [ -f "$backup_dir/databases/$TIMESTAMP/postgresql_osint_primary.sql.gz" ]; then
            local pg_size=$(stat -c%s "$backup_dir/databases/$TIMESTAMP/postgresql_osint_primary.sql.gz")
            echo "  ✅ PostgreSQL: $(($pg_size / 1024 / 1024))MB"
        else
            echo "  ❌ PostgreSQL: Missing"
        fi

        if [ -f "$backup_dir/databases/$TIMESTAMP/redis_dump.rdb" ]; then
            local redis_size=$(stat -c%s "$backup_dir/databases/$TIMESTAMP/redis_dump.rdb")
            echo "  ✅ Redis: $(($redis_size / 1024 / 1024))MB"
        else
            echo "  ❌ Redis: Missing"
        fi

        # Check configuration backups
        echo ""
        echo "Configuration Backups:"
        if [ -f "$backup_dir/configurations/$TIMESTAMP/config_directory.tar.gz" ]; then
            echo "  ✅ Configuration files backed up"
        else
            echo "  ❌ Configuration files missing"
        fi

        # Check Vault backup
        echo ""
        echo "Vault Backup:"
        if [ -f "$backup_dir/vault/$TIMESTAMP/vault-snapshot.snap" ]; then
            echo "  ✅ Vault snapshot created"
        else
            echo "  ⚠️ Vault snapshot not available (vault may be sealed)"
        fi

        echo ""
        echo "Total backup size: $(du -sh $backup_dir/databases/$TIMESTAMP $backup_dir/configurations/$TIMESTAMP | awk '{sum+=$1} END {print sum}')"

    } | tee "$verification_log"

    echo "✅ Backup verification completed: $verification_log"
}

# Main backup execution
main() {
    echo "🚀 Starting comprehensive backup process..."

    create_backup_structure
    backup_databases
    backup_configurations
    backup_vault
    backup_logs
    backup_ai_models
    verify_backup
    cleanup_old_backups

    echo ""
    echo "✅ Comprehensive backup completed successfully!"
    echo "📁 Backup location: $BACKUP_BASE_DIR/$TIMESTAMP"
    echo "📊 Backup size: $(du -sh $BACKUP_BASE_DIR/$TIMESTAMP | cut -f1)"
    echo "🔍 Verification log: $BACKUP_BASE_DIR/verification_$TIMESTAMP.log"
    echo ""
    echo "🔄 To restore from this backup, use: ./restore-backup.sh $TIMESTAMP"
}

main "$@"
```

### **Emergency Recovery Procedures**

#### **Disaster Recovery Playbook**
```bash
#!/bin/bash
# disaster-recovery-playbook.sh - Emergency recovery procedures

echo "🚨 BEV Platform Disaster Recovery Playbook"
echo "Emergency contact: [Your emergency contact information]"

# Recovery scenario selection
select_recovery_scenario() {
    echo "🎯 Select recovery scenario:"
    echo "1) Single service failure"
    echo "2) Database corruption"
    echo "3) Node failure (THANOS/ORACLE1)"
    echo "4) Complete platform failure"
    echo "5) Security incident response"

    read -p "Enter scenario number (1-5): " scenario

    case $scenario in
        1) recover_single_service ;;
        2) recover_database_corruption ;;
        3) recover_node_failure ;;
        4) recover_complete_platform ;;
        5) security_incident_response ;;
        *) echo "Invalid selection" && exit 1 ;;
    esac
}

# Single service recovery
recover_single_service() {
    echo "🔧 Single Service Recovery Procedure"

    read -p "Enter failing service name: " service_name

    echo "📋 Recovery steps for $service_name:"
    echo "1. Check service logs: docker-compose logs $service_name"
    echo "2. Check resource usage: docker stats $service_name"
    echo "3. Restart service: docker-compose restart $service_name"
    echo "4. If restart fails, rebuild: docker-compose up -d --build $service_name"
    echo "5. Verify service health after restart"

    read -p "Execute automatic recovery? (y/n): " auto_recover

    if [ "$auto_recover" = "y" ]; then
        echo "🔄 Executing automatic recovery..."

        # Get service logs
        echo "📋 Collecting service logs..."
        docker-compose logs --tail=100 "$service_name" > "recovery_${service_name}_$(date +%Y%m%d_%H%M%S).log"

        # Restart service
        echo "🔄 Restarting service..."
        docker-compose restart "$service_name"

        # Wait and verify
        sleep 30
        if docker-compose ps "$service_name" | grep -q "Up"; then
            echo "✅ Service $service_name recovered successfully"
        else
            echo "❌ Service recovery failed - manual intervention required"
            echo "🔍 Check logs: recovery_${service_name}_*.log"
        fi
    fi
}

# Database corruption recovery
recover_database_corruption() {
    echo "🗄️ Database Corruption Recovery Procedure"

    echo "⚠️ CRITICAL: This procedure will restore from backup"
    echo "📝 Backup timestamps available:"
    ls -la /backup/bev/databases/ | grep "^d" | tail -5

    read -p "Enter backup timestamp to restore (YYYYMMDD_HHMMSS): " backup_timestamp

    if [ ! -d "/backup/bev/databases/$backup_timestamp" ]; then
        echo "❌ Backup not found: $backup_timestamp"
        exit 1
    fi

    echo "🚨 DANGER: This will overwrite current database data"
    read -p "Type 'CONFIRM' to proceed: " confirmation

    if [ "$confirmation" != "CONFIRM" ]; then
        echo "Recovery cancelled"
        exit 1
    fi

    echo "🔄 Executing database recovery..."

    # Stop services
    echo "🛑 Stopping database services..."
    docker-compose stop bev_postgres bev_redis bev_neo4j

    # Restore PostgreSQL
    echo "📊 Restoring PostgreSQL..."
    docker-compose start bev_postgres
    sleep 30
    gunzip -c "/backup/bev/databases/$backup_timestamp/postgresql_osint_primary.sql.gz" | \
        docker exec -i thanos_postgres psql -U researcher -d osint_primary

    # Restore Redis
    echo "🔴 Restoring Redis..."
    docker-compose stop bev_redis
    docker cp "/backup/bev/databases/$backup_timestamp/redis_dump.rdb" thanos_redis:/data/dump.rdb
    docker-compose start bev_redis

    # Restore Neo4j
    echo "🕸️ Restoring Neo4j..."
    docker-compose stop bev_neo4j
    docker cp "/backup/bev/databases/$backup_timestamp/neo4j_database.dump" thanos_neo4j:/tmp/
    docker-compose start bev_neo4j
    sleep 30
    docker exec thanos_neo4j neo4j-admin database load --from-path=/tmp neo4j

    echo "✅ Database recovery completed"
    echo "🔍 Verify data integrity and restart dependent services"
}

# Node failure recovery
recover_node_failure() {
    echo "🖥️ Node Failure Recovery Procedure"

    echo "Select failed node:"
    echo "1) THANOS (Primary OSINT)"
    echo "2) ORACLE1 (Monitoring)"
    echo "3) STARLORD (AI Companion)"

    read -p "Enter node number (1-3): " node_choice

    case $node_choice in
        1) recover_thanos_node ;;
        2) recover_oracle1_node ;;
        3) recover_starlord_node ;;
        *) echo "Invalid selection" && exit 1 ;;
    esac
}

# THANOS node recovery
recover_thanos_node() {
    echo "🏛️ THANOS Node Recovery"

    echo "🚨 THANOS node failure - Primary OSINT services down"
    echo "📋 Recovery procedure:"
    echo "1. Verify hardware status (GPU, storage, network)"
    echo "2. Restore from latest backup if necessary"
    echo "3. Redeploy THANOS services"
    echo "4. Verify cross-node connectivity"

    read -p "Execute THANOS recovery? (y/n): " proceed

    if [ "$proceed" = "y" ]; then
        echo "🔄 Recovering THANOS node..."

        # Deploy THANOS services
        ./deploy-thanos-node.sh

        # Verify connectivity
        echo "🔗 Testing cross-node connectivity..."
        if ping -c 3 100.122.12.54; then
            echo "✅ THANOS node network connectivity restored"
        else
            echo "❌ THANOS node network connectivity failed"
        fi

        # Test services
        echo "🔍 Testing OSINT services..."
        if curl -f -s http://100.122.12.54/api/health; then
            echo "✅ OSINT services restored"
        else
            echo "❌ OSINT services not responding"
        fi
    fi
}

# Security incident response
security_incident_response() {
    echo "🔒 Security Incident Response Procedure"

    echo "⚠️ SECURITY INCIDENT DETECTED"
    echo "📋 Immediate actions:"
    echo "1. Isolate affected systems"
    echo "2. Preserve evidence"
    echo "3. Assess scope of compromise"
    echo "4. Initiate incident response plan"

    read -p "Execute emergency isolation? (y/n): " isolate

    if [ "$isolate" = "y" ]; then
        echo "🚨 Executing emergency isolation..."

        # Stop external network access
        echo "🔒 Isolating network access..."
        sudo iptables -A OUTPUT -j DROP
        sudo iptables -A INPUT -j DROP

        # Stop all services
        echo "🛑 Stopping all services..."
        docker-compose -f docker-compose-thanos-unified.yml stop
        docker-compose -f docker-compose-oracle1-unified.yml stop

        # Create forensic snapshot
        echo "🔍 Creating forensic snapshot..."
        timestamp=$(date +%Y%m%d_%H%M%S)
        mkdir -p "/forensics/incident_$timestamp"

        # Collect system state
        docker ps -a > "/forensics/incident_$timestamp/docker_containers.txt"
        docker images > "/forensics/incident_$timestamp/docker_images.txt"
        netstat -tulpn > "/forensics/incident_$timestamp/network_connections.txt"
        ps aux > "/forensics/incident_$timestamp/running_processes.txt"

        echo "✅ Emergency isolation completed"
        echo "📁 Forensic data: /forensics/incident_$timestamp/"
        echo "📞 Contact security team immediately"
    fi
}

# Main recovery menu
main() {
    echo "🆘 BEV Platform Emergency Recovery System"
    echo "========================================"
    echo ""
    echo "⚠️ WARNING: These procedures may result in data loss"
    echo "📞 For critical incidents, contact the emergency response team"
    echo ""

    select_recovery_scenario
}

main "$@"
```

## Troubleshooting Guide

### **Common Issues and Resolutions**

#### **Service Connectivity Issues**
```bash
# Troubleshooting service connectivity
troubleshoot_connectivity() {
    echo "🔗 Service Connectivity Troubleshooting"

    # Test basic connectivity
    local services=(
        "100.122.12.54:80:OSINT Platform"
        "100.122.12.54:3010:MCP Server"
        "100.122.12.54:5432:PostgreSQL"
        "100.96.197.84:3000:Grafana"
        "100.96.197.84:9090:Prometheus"
    )

    for service in "${services[@]}"; do
        IFS=':' read -r host port name <<< "$service"

        echo "Testing $name ($host:$port)..."

        if nc -z "$host" "$port"; then
            echo "  ✅ Port reachable"

            # Test HTTP services
            if [ "$port" = "80" ] || [ "$port" = "3000" ] || [ "$port" = "3010" ] || [ "$port" = "9090" ]; then
                if curl -f -s "http://$host:$port" > /dev/null; then
                    echo "  ✅ HTTP responding"
                else
                    echo "  ⚠️ Port open but HTTP not responding"
                fi
            fi
        else
            echo "  ❌ Port not reachable"
            echo "    Troubleshooting steps:"
            echo "    1. Check if service is running: docker-compose ps"
            echo "    2. Check Docker networks: docker network ls"
            echo "    3. Check firewall rules: sudo iptables -L"
            echo "    4. Check Tailscale VPN: tailscale status"
        fi
        echo ""
    done
}

# GPU troubleshooting
troubleshoot_gpu() {
    echo "🎮 GPU Troubleshooting"

    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status:"
        nvidia-smi
        echo ""

        # Check CUDA availability
        if command -v nvcc &> /dev/null; then
            echo "🔧 CUDA Version:"
            nvcc --version
        else
            echo "⚠️ CUDA not available - some AI features may not work"
        fi

        # Check Docker GPU access
        echo "🐳 Docker GPU Access:"
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi; then
            echo "✅ Docker can access GPU"
        else
            echo "❌ Docker cannot access GPU"
            echo "Troubleshooting steps:"
            echo "1. Install nvidia-container-toolkit"
            echo "2. Restart Docker: sudo systemctl restart docker"
            echo "3. Check Docker daemon configuration"
        fi
    else
        echo "❌ NVIDIA drivers not installed or GPU not available"
        echo "Install NVIDIA drivers for GPU acceleration"
    fi
}
```

## Conclusion

The BEV AI Assistant Platform Developer Guide and Operations Manual provides comprehensive coverage of development workflows, operational procedures, and maintenance protocols for the revolutionary AI-powered cybersecurity research platform. This guide enables developers to extend platform capabilities while ensuring operational teams can maintain enterprise-grade service levels.

### **Key Developer Resources**

1. **Development Environment**: Complete setup procedures for all development scenarios
2. **Architecture Patterns**: Service-oriented architecture with plugin system for extensibility
3. **Custom Analyzer Development**: Framework for creating domain-specific OSINT analyzers
4. **AI Model Integration**: Patterns for integrating custom AI models with GPU optimization
5. **Testing Frameworks**: Comprehensive testing strategies for quality assurance

### **Operational Excellence**

1. **Daily Operations**: Automated health monitoring and performance validation procedures
2. **Backup and Recovery**: Comprehensive backup strategies with disaster recovery playbooks
3. **Security Monitoring**: Continuous security assessment and incident response protocols
4. **Performance Optimization**: Resource utilization monitoring and optimization procedures
5. **Troubleshooting**: Systematic problem resolution methodologies

### **Platform Extensibility**

The BEV platform is designed for extensibility and customization:
- Plugin architecture for custom analyzers and tools
- AI model integration framework for specialized capabilities
- Standardized service patterns for consistent development
- Comprehensive testing frameworks for quality assurance
- Enterprise-grade operational procedures for production deployment

**The BEV Developer Guide represents the foundation for extending and operating the world's most advanced AI-powered cybersecurity research platform.**

---

*For specific implementation examples, configuration templates, and advanced procedures, refer to the individual sections and code examples throughout this guide.*