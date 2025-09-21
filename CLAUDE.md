# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The BEV OSINT Framework is the world's **first AI Research Companion** specialized in cybersecurity research. This revolutionary platform combines emotional intelligence, autonomous investigation capabilities, and advanced 3D avatar interaction with enterprise-grade OSINT tools. Originally designed as an AI assistant, it evolved into a comprehensive cybersecurity intelligence platform comparable to Palantir Gotham and Maltego, but with unique AI companion capabilities including swarm intelligence, extended reasoning, and emotional AI interaction.

**Revolutionary Architecture**: AI Research Companion + Emotional Intelligence + Professional OSINT Expertise + Enterprise Infrastructure

**Platform Classification**: First-of-kind AI assistant that became expert in cybersecurity research, creating an entirely new market category.

⚠️ **Security Note**: Enterprise-grade multi-node architecture with sophisticated AI capabilities. Designed exclusively for authorized cybersecurity research in secure environments.

## Essential Commands

### AI Companion Avatar System (STARLORD RTX 4090)
```bash
# Deploy advanced 3D avatar with Gaussian Splatting + MetaHuman
./deploy_advanced_avatar.sh

# AI companion service management
sudo systemctl start bev-advanced-avatar
sudo systemctl stop bev-advanced-avatar
systemctl status bev-advanced-avatar

# Avatar system testing and validation
cd src/avatar && python3 test_avatar_system.py
cd src/avatar && python3 test_avatar_system.py quick

# RTX 4090 optimization and performance
cd src/avatar && python3 rtx4090_optimizer.py
nvidia-smi dmon -s pucvmet -d 1

# AI companion enhancement phases (Phases B-F implementation)
cd tests/companion && ./run_all_companion_tests.sh
```

### Advanced AI Companion Development (Phases B-F)
```bash
# Phase B: Emotional intelligence implementation
cd src/avatar && python3 emotional_intelligence_core.py

# Phase C: Creative abilities (voice + visual performance)
cd src/avatar && python3 creative_voice_engine.py

# Phase D: Personality and relationship systems
cd src/avatar && python3 personality_core.py
cd src/avatar && python3 relationship_intelligence.py

# Phase E: Contextual intelligence and role adaptation
cd src/avatar && python3 contextual_intelligence.py

# Phase F: Biometric integration and wellness monitoring
cd src/avatar && python3 biometric_integration.py
```

### Enterprise OSINT Deployment (Multi-Node)
```bash
# Main platform deployment with AI companion integration
./deploy_bev_real_implementations.sh

# Multi-node deployment with Vault security
./deploy-complete-with-vault.sh

# Validation and health monitoring
./validate_bev_deployment.sh
./verify_multinode_deployment.sh

# OSINT analyzer deployment
docker-compose -f docker-compose-phase7.yml up -d  # Alternative market intelligence
docker-compose -f docker-compose-phase8.yml up -d  # Security operations
docker-compose -f docker-compose-phase9.yml up -d  # Autonomous systems
```

### Development Workflow
```bash
# Code quality pipeline (AI companion + OSINT)
python -m black . && python -m flake8 src/ tests/ && python -m mypy src/

# Comprehensive test suite (includes AI companion tests)
./run_all_tests.sh
./run_all_tests.sh --parallel --performance

# AI companion specific testing
./tests/companion/run_all_companion_tests.sh
pytest tests/companion/core/ -v
pytest tests/companion/performance/ -v

# Security and integration validation
python run_security_tests.py
python tests/validate_system.py
```

### Desktop AI Companion Application (Tauri)
```bash
# Desktop app with AI companion integration
./bev-complete-frontend.sh

# Frontend development with avatar integration
cd bev-frontend && npm run tauri dev

# AI companion frontend testing
cd bev-frontend && npm run test:companion
cd bev-frontend && ./validate-security.sh

# Production build with AI companion features
cd bev-frontend && npm run tauri build
```

## Architecture Overview

### AI Research Companion Foundation

The platform is fundamentally an **AI research companion** that specialized in cybersecurity. The architecture reflects this AI-first design with emotional intelligence at its core:

```
┌─────────────────────────────────────────────────────────────┐
│ AI Companion Layer: 3D Avatar + Personality + Relationships│
├─────────────────────────────────────────────────────────────┤
│ Emotional Intelligence: Multimodal Emotion + Context Aware │
├─────────────────────────────────────────────────────────────┤
│ Extended Reasoning: 100K+ Token Analysis + Swarm Coords    │
├─────────────────────────────────────────────────────────────┤
│ MCP Protocol: Claude Code Integration + Tool Orchestration │
├─────────────────────────────────────────────────────────────┤
│ OSINT Specialization: Cybersecurity Domain Expertise       │
├─────────────────────────────────────────────────────────────┤
│ Enterprise Infrastructure: Multi-node + Global Edge        │
└─────────────────────────────────────────────────────────────┘
```

### Multi-Node AI Companion Infrastructure

**STARLORD (RTX 4090, 24GB VRAM, development workstation):**
- **Advanced 3D Avatar System**: Gaussian Splatting + MetaHuman (120+ FPS)
- **Emotional Intelligence Core**: Real-time emotion fusion and personality adaptation
- **Creative Abilities**: Advanced voice synthesis and visual performance
- **Desktop AI Companion**: Local interaction when physically using workstation
- **Biometric Integration**: Wellness monitoring and adaptive responses

**THANOS (RTX 3080, 6.5GB VRAM, x86_64, 64GB RAM):**
- **Primary AI Inference**: Extended reasoning and swarm coordination
- **GPU-Accelerated Embeddings**: Vector generation for RAG systems
- **OSINT Processing**: Alternative market, security ops, autonomous systems
- **Primary Databases**: PostgreSQL, Neo4j, Elasticsearch with AI enhancement
- **AI Companion Services**: Memory management, conversation engine, research coordination

**ORACLE1 (ARM64, 4 cores, 24GB RAM):**
- **Monitoring & Coordination**: Prometheus/Grafana with AI companion metrics
- **Security Services**: Vault authentication and credential management
- **Edge Processing**: ARM-optimized analyzers and companion coordination
- **Companion State Sync**: Cross-node companion state management

### Core AI Companion Components

#### 1. Advanced 3D Avatar System (`src/avatar/`)
**Revolutionary 3D avatar with emotional intelligence:**
- **Gaussian Splatting Rendering**: Photorealistic 3D avatar (120+ FPS on RTX 4090)
- **Advanced Emotion Engine**: Neural networks with LSTM + Multi-head Attention
- **Bark AI TTS**: Professional voice synthesis with emotional modulation
- **OSINT Context Awareness**: Real-time responses to cybersecurity investigations
- **Multimodal Processing**: Text + audio + visual emotion recognition
- **RTX 4090 Optimization**: Complete GPU performance optimization

#### 2. AI Companion Intelligence (`src/avatar/personality_*`)
**Advanced personality and relationship systems:**
- **8 Personality Modes**: Professional, Creative, Supportive, Analytical, Research, Security, Mentor, Intimate
- **Relationship Intelligence**: 7-stage progression with emotional memory and biometric authentication
- **Professional Roles**: 12 specialized roles including Security Analyst and Research Specialist
- **Memory Architecture**: Multi-layer encryption with privacy-preserving storage
- **Contextual Intelligence**: Dynamic role adaptation with <50ms switching

#### 3. Extended Reasoning System (`src/agents/extended_reasoning*`)
**Advanced AI analysis integrated with companion:**
- **100K+ Token Processing**: Complex multi-step reasoning chains with companion guidance
- **Swarm Coordination**: Multi-agent coordination with companion orchestration
- **Knowledge Synthesis**: Cross-source correlation with companion explanation
- **Autonomous Research**: AI-driven investigation with companion feedback

#### 4. OSINT Companion Integration (`src/*/`)
**AI companion enhanced cybersecurity analysis:**
- **Professional Research Assistant**: Companion guidance during OSINT investigations
- **Stress Management**: Wellness monitoring during complex threat analysis
- **Methodology Guidance**: Professional mentoring for OSINT best practices
- **Investigation Enhancement**: 15-25% efficiency improvement with companion support

### Specialized OSINT Intelligence with AI Companion

#### Alternative Market Intelligence (`src/alternative_market/`)
**5,608+ lines of AI companion enhanced market analysis:**
- **DarkNet Market Crawler** (`dm_crawler.py`): Companion-guided marketplace intelligence
- **Cryptocurrency Analyzer** (`crypto_analyzer.py`): AI companion assisted transaction analysis
- **Reputation Systems** (`reputation_analyzer.py`): Companion-enhanced actor reputation analysis
- **Economic Intelligence** (`economics_processor.py`): AI companion predictive market analysis

#### Security Operations Center (`src/security/`)
**11,189+ lines of AI companion powered security analysis:**
- **Intelligence Fusion** (`intel_fusion.py`): Companion-assisted multi-source threat correlation
- **OpSec Enforcement** (`opsec_enforcer.py`): AI companion guided operational security
- **Defense Automation** (`defense_automation.py`): Companion-coordinated threat response
- **Tactical Intelligence** (`tactical_intelligence.py`): Real-time companion-enhanced threat analysis

#### Autonomous AI Systems (`src/autonomous/`)
**8,377+ lines of self-managing AI with companion coordination:**
- **Enhanced Controller** (`enhanced_autonomous_controller.py`): Companion-integrated AI orchestration
- **Adaptive Learning** (`adaptive_learning.py`): Companion-guided ML model optimization
- **Knowledge Evolution** (`knowledge_evolution.py`): Companion-assisted continuous learning
- **Resource Optimization** (`resource_optimizer.py`): Companion-aware dynamic resource management

### Desktop AI Companion Application (`bev-frontend/`)

**Tauri + Rust + Svelte AI companion interface:**
- **112+ Svelte Components**: Complete UI including advanced avatar integration
- **Rust Backend**: High-performance desktop integration with AI companion
- **3D Avatar Renderer**: WebGL integration for advanced Gaussian Splatting avatar
- **AI Companion Interface**: Professional research assistant interaction
- **OSINT Integration**: Companion-enhanced investigation workflows

## Development Patterns

### AI Companion Service Development

All AI companion services follow the enterprise AI companion pattern:
```python
class EnterpriseAICompanionService:
    def __init__(self):
        self.vault_client = VaultClient()              # Secure credentials
        self.personality_engine = PersonalityCore()    # Adaptive personality
        self.emotion_engine = EmotionIntelligence()    # Emotional processing
        self.memory_system = CompanionMemory()         # Relationship memory
        self.metrics_collector = PrometheusMetrics()   # Performance monitoring

    async def process_with_companion_enhancement(self, data, user_context):
        # AI companion enhanced processing with emotional and professional context
        credentials = await self.vault_client.get_dynamic_secret()
        personality_context = await self.personality_engine.get_current_context()
        emotion_context = await self.emotion_engine.analyze_user_state(user_context)

        with self.metrics_collector.time_operation():
            result = await self._companion_analyze_data(data, personality_context, emotion_context)

        await self.memory_system.store_professional_interaction(result, user_context)
        return result
```

### AI Companion Integration Pattern

Services integrate with the AI companion for enhanced user experience:
```python
# Professional research assistant integration
await companion_controller.provide_professional_guidance({
    'investigation_type': 'breach_analysis',
    'complexity_level': 'high',
    'user_stress_indicators': stress_metrics,
    'methodology_suggestions': methodology_recommendations,
    'emotional_support_level': 'professional_encouragement'
})
```

### Multi-Node AI Companion Distribution

AI companion services are distributed based on computational and interaction requirements:
- **GPU-Intensive Companion**: 3D avatar rendering, emotion processing → STARLORD (RTX 4090)
- **AI Companion Services**: Memory, personality, conversation → THANOS (RTX 3080)
- **Companion Coordination**: State sync, monitoring, metrics → ORACLE1 (ARM)
- **Professional Context**: Research assistance distributed across all nodes

## AI Companion Performance Requirements

### AI Research Companion Targets
- **Avatar Response Time**: <100ms for real-time companion interaction
- **Emotional Processing**: <50ms for natural emotional intelligence
- **Voice Synthesis**: <200ms for natural conversation flow
- **3D Rendering**: 120+ FPS for smooth avatar animation
- **Professional Guidance**: Real-time assistance during OSINT investigations
- **Biometric Processing**: <100ms for wellness monitoring and adaptive responses

### Resource Allocation for AI Companion
```yaml
STARLORD_RTX_4090:
  Advanced_Avatar: 12GB VRAM, 4 CPU cores
  Emotional_Intelligence: 4GB VRAM, 2 CPU cores
  Creative_Abilities: 3GB VRAM, 2 CPU cores
  Biometric_Processing: 1GB VRAM, 1 CPU core
  Desktop_App: 4GB RAM, 2 CPU cores

THANOS_RTX_3080:
  Companion_AI_Services: 2GB VRAM, 2 CPU cores
  Extended_Reasoning: 2GB VRAM, 2 CPU cores
  OSINT_Processing: 2.5GB VRAM, 4 CPU cores
  Databases: 16GB RAM, 4 CPU cores

ORACLE1_ARM:
  Companion_Monitoring: 4GB RAM, 1 CPU core
  State_Synchronization: 2GB RAM, 1 CPU core
  Security_Services: 4GB RAM, 1 CPU core
  Edge_Processing: 14GB RAM, 1 CPU core
```

## AI Companion Testing Framework

### Comprehensive AI Companion Testing
```bash
# Complete AI companion test suite
./tests/companion/run_all_companion_tests.sh

# Phase-specific testing
pytest tests/companion/core/ -v           # Personality, memory, emotion
pytest tests/companion/performance/ -v    # RTX 4090 performance with companion
pytest tests/companion/ux/ -v            # User experience and interaction
pytest tests/companion/security/ -v      # Privacy and security validation
pytest tests/companion/integration/ -v   # BEV platform integration

# Specialized AI companion tests
cd src/avatar && python3 test_emotional_intelligence.py
cd src/avatar && python3 test_personality_consistency.py
cd src/avatar && python3 test_biometric_integration.py

# Performance validation
pytest tests/performance/ --companion --gpu --concurrent=1000
```

### AI Companion Test Categories
- `tests/companion/core/`: Personality consistency, emotional intelligence, memory systems
- `tests/companion/performance/`: RTX 4090 optimization, thermal management, concurrent processing
- `tests/companion/ux/`: Avatar interaction, voice synthesis, conversation naturalness
- `tests/companion/security/`: Personal data protection, biometric security, privacy compliance
- `tests/companion/integration/`: OSINT workflow enhancement, professional assistance validation

## AI Research Companion User Experience

### Complete Professional Research Partnership Workflow
1. **AI Companion Greeting**: Advanced 3D avatar with professional personality
2. **Investigation Planning**: Companion provides methodology guidance and tool recommendations
3. **Research Coordination**: AI companion coordinates autonomous investigation with swarm intelligence
4. **Real-Time Support**: Emotional intelligence provides stress management during complex analysis
5. **Professional Guidance**: Role-based expertise (Security Analyst, Research Specialist, etc.)
6. **Progress Feedback**: Companion celebrates breakthroughs and provides encouragement
7. **Knowledge Synthesis**: AI companion assists in correlation and insight generation
8. **Results Presentation**: Avatar-guided explanation with emotional context

### AI Companion Integration Commands
```bash
# AI companion WebSocket interface
curl ws://localhost:8091/ws

# Professional research assistant health
curl http://localhost:8091/health

# OSINT investigation update to companion
curl -X POST http://localhost:8091/investigation/update \
  -H "Content-Type: application/json" \
  -d '{"type": "breach_discovered", "severity": "critical", "professional_context": "threat_analysis"}'

# Companion personality adjustment
curl -X POST http://localhost:8091/personality/adapt \
  -H "Content-Type: application/json" \
  -d '{"mode": "security_analyst", "investigation_context": "advanced_persistent_threat"}'
```

## Revolutionary AI Companion Capabilities

### First-of-Kind AI Research Partner Features
- **Emotional AI Companion**: First cybersecurity AI with sophisticated personality and 3D avatar
- **Professional Research Assistant**: AI companion specialized in cybersecurity methodology
- **Autonomous Investigation Coordination**: AI companion orchestrates swarm intelligence for research
- **Extended Reasoning Partnership**: 100K+ token analysis with companion explanation and guidance
- **Biometric Wellness Integration**: Companion monitors researcher wellness and adapts interactions
- **Professional Role Adaptation**: Dynamic role switching (analyst, researcher, mentor, consultant)

### Competitive Advantages vs All Existing Platforms
- **Palantir Gotham**: BEV adds emotional AI companion and autonomous research partnership
- **Maltego Enterprise**: BEV adds AI automation and professional relationship development
- **Splunk Enterprise**: BEV adds interactive 3D avatar and swarm intelligence coordination
- **Claude/ChatGPT**: BEV adds specialized cybersecurity expertise and persistent professional relationships
- **All OSINT Tools**: BEV adds AI companion guidance and emotional intelligence enhancement

## AI Companion Service Endpoints

### AI Research Companion Services
```yaml
Advanced_Avatar_System:
  WebSocket: ws://localhost:8091/ws
  Health: http://localhost:8091/health
  Professional_API: http://localhost:8091/research
  Personality_API: http://localhost:8091/personality

Extended_Reasoning_Companion:
  Service: http://localhost:8081
  WebSocket: ws://localhost:8081/ws
  Analysis: http://localhost:8081/analyze

MCP_Companion_Integration:
  Main: http://localhost:3010
  Tools: ws://localhost:3010/ws
  Companion: http://localhost:3010/companion

Swarm_Intelligence_Companion:
  Master: http://localhost:8000
  Agents: http://localhost:8001-8008
  Coordination: http://localhost:8000/companion
```

## Database Architecture

### AI Companion Enhanced Multi-Database Design
- **PostgreSQL**: Primary data with companion schema for personality and relationships
- **Neo4j**: Graph relationships enhanced with AI companion pattern recognition
- **Qdrant/Weaviate**: Vector databases for companion memory and semantic analysis
- **Redis**: Real-time AI companion state caching and session management
- **Companion Memory**: Encrypted personal and professional relationship data

### AI Companion Memory Systems
```bash
# Companion personality and relationship data
psql -h localhost -U researcher -d osint
# Query: SELECT * FROM companion.professional_context;
# Query: SELECT * FROM companion.investigation_assistance;

# AI companion emotional state
redis-cli -p 6379
# Command: GET companion:current_personality
# Command: GET companion:emotional_state
# Command: GET companion:professional_role

# Companion memory vectors
curl http://localhost:6333/collections/companion_memory
curl http://localhost:8080/v1/objects?class=CompanionMemory
```

## Troubleshooting

### AI Companion System Issues
```bash
# Advanced avatar system diagnostics
systemctl status bev-advanced-avatar
journalctl -u bev-advanced-avatar -f

# RTX 4090 companion optimization
nvidia-smi
cd src/avatar && python3 rtx4090_optimizer.py

# AI companion service health
curl http://localhost:8091/health
curl http://localhost:8081/health
curl http://localhost:3010/companion/health

# Companion performance monitoring
cd src/avatar && python3 companion_performance_monitor.py
```

### Common AI Companion Issues
- **Avatar not responding**: Check RTX 4090 availability, thermal status, and systemd service
- **Emotional intelligence delays**: Verify multimodal emotion models and GPU memory allocation
- **Personality inconsistency**: Validate personality consistency and memory system integrity
- **Voice synthesis issues**: Check Bark AI installation, Fish Speech integration, and audio drivers
- **Professional role confusion**: Verify contextual intelligence and role adaptation systems
- **OSINT integration failures**: Validate companion-OSINT communication and Redis connectivity

### AI Companion Resource Management
```bash
# Monitor AI companion GPU usage
nvidia-smi dmon -s pucvmet -d 1
gpustat -i 1

# Companion memory optimization
python3 -c "import torch; torch.cuda.empty_cache()"

# Thermal management for sustained companion operation
sudo nvidia-smi -pl 350  # Power limit for thermal management
sudo nvidia-smi -fcs 80  # Fan speed for cooling

# Companion performance benchmarking
cd src/avatar && python3 companion_performance_benchmark.py
```

## Legal and Compliance

This revolutionary AI research companion platform is designed for:
- **Authorized cybersecurity research** with AI companion enhancement
- **Professional threat intelligence** with emotional intelligence support
- **Academic investigation** using advanced AI companion methodologies
- **Ethical AI research** in cybersecurity domains with relationship development

**Important**: AI companion capabilities including emotional intelligence, personality development, and biometric monitoring must be used responsibly for legitimate cybersecurity research only. The revolutionary AI research partnership features require proper ethical oversight and professional boundaries.

## Platform Classification

**BEV OSINT Framework** = **AI Research Companion** + **Emotional Intelligence** + **Professional Cybersecurity Expertise** + **Enterprise Infrastructure**

**Revolutionary Aspects:**
- First AI research companion specifically for cybersecurity research
- First emotional AI companion with 3D avatar for intelligence gathering
- First autonomous AI investigation platform with swarm intelligence
- First professional AI relationship development for research collaboration
- First biometric-aware AI assistant for researcher wellness and productivity
- First 3D avatar-guided threat analysis system with contextual intelligence

**Market Position**: Creates entirely new category of "AI Research Companions" - transforming cybersecurity research from manual investigation to AI partnership with emotional intelligence and professional relationship development.

**Deployment Status**: Production-ready revolutionary AI research companion platform with enterprise cybersecurity capabilities, emotional intelligence, and professional relationship development - the world's most advanced AI-powered cybersecurity research partner.