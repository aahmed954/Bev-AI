# BEV OSINT Framework Documentation

## 🎯 Documentation Hub

Welcome to the comprehensive documentation for the BEV OSINT Framework - a sophisticated intelligence gathering and analysis platform designed for cybersecurity research and threat analysis.

## 🚀 Quick Start

### For First-Time Users
1. **[Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md)** - Start here for role-based navigation
2. **[Knowledge Base](KNOWLEDGE_BASE.md)** - Central hub with cross-references and quick lookup
3. **[System Deployment](#deployment-guide)** - Get the system running quickly

### For Returning Users
- **[API Reference](API_REFERENCE.md)** - Complete API documentation
- **[Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)** - Solve common issues
- **[Common Tasks](#common-operations)** - Frequently needed procedures

## 📚 Documentation Organization

### 🔰 Getting Started
| Document | Purpose | Audience |
|----------|---------|----------|
| [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) | Navigation by role and complexity | All users |
| [Knowledge Base](KNOWLEDGE_BASE.md) | Cross-referenced information hub | All users |
| **This README** | Overview and quick navigation | New users |

### 🏗️ Technical Documentation
| Document | Purpose | Audience |
|----------|---------|----------|
| [API Reference](API_REFERENCE.md) | Complete API documentation | Developers, Integrators |
| [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) | Architecture and system components | Architects, Operators |
| [Integration Workflows](comprehensive/BEV_INTEGRATION_WORKFLOWS.md) | Development and integration patterns | Developers |

### 👥 User Guides
| Document | Purpose | Audience |
|----------|---------|----------|
| [OSINT Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) | Investigation workflows and methodologies | Analysts, Researchers |
| [Operational Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) | Security procedures and compliance | Security teams, Operators |
| [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) | Operations and problem resolution | Operators, Support teams |

## 🎯 Choose Your Path

### I'm a Security Researcher / OSINT Analyst
**Goal**: Conduct intelligence investigations and threat analysis

**Start Here**: [OSINT Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md)

**Key Resources**:
- Investigation methodologies and workflows
- Custom analyzer usage and configuration
- Operational security procedures
- Report generation and case management

### I'm a System Operator / DevOps Engineer  
**Goal**: Deploy, monitor, and maintain the BEV system

**Start Here**: [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)

**Key Resources**:
- System deployment and configuration
- Performance monitoring and optimization
- Backup and recovery procedures
- Security operations and incident response

### I'm a Developer / Integrator
**Goal**: Integrate with or extend the BEV framework

**Start Here**: [API Reference](API_REFERENCE.md)

**Key Resources**:
- REST API and WebSocket documentation
- Custom analyzer development
- Integration patterns and examples
- SDK usage and client libraries

### I'm a Security Professional / Compliance Officer
**Goal**: Ensure secure and compliant operations

**Start Here**: [Operational Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md)

**Key Resources**:
- Security architecture and controls
- Compliance frameworks and procedures
- Risk management and threat modeling
- Incident response and recovery

## 🚀 Deployment Guide

### Prerequisites
- **Operating System**: Ubuntu 20.04+ or compatible Linux
- **Hardware**: 16GB RAM minimum (32GB recommended), 500GB SSD
- **Software**: Docker 20.10+, Docker Compose v2
- **Network**: Internet access and Tor proxy capabilities

### Quick Deployment
```bash
# 1. Clone and navigate to project
cd /path/to/bev-framework

# 2. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 3. Deploy everything
chmod +x deploy_everything.sh
./deploy_everything.sh

# 4. Validate deployment
./validate_bev_deployment.sh
```

### Access Points After Deployment
| Service | URL | Purpose |
|---------|-----|---------|
| **IntelOwl Dashboard** | http://localhost | Main OSINT interface |
| **Cytoscape Visualization** | http://localhost/cytoscape | Network graphs |
| **Grafana Monitoring** | http://localhost:3000 | System monitoring |
| **API Documentation** | http://localhost:3010/docs | Interactive API docs |

**📋 Full deployment details**: [Component Catalog - Deployment Section](comprehensive/BEV_COMPONENT_CATALOG.md#deployment-guide)

## 📊 Common Operations

### Daily Operations
```bash
# Check system health
./validate_bev_deployment.sh

# Monitor system performance
# Visit: http://localhost:3000 (Grafana)

# View system logs
docker-compose -f docker-compose.complete.yml logs -f

# Run system tests
./run_all_tests.sh --quick
```

### Maintenance Tasks
```bash
# Restart specific service
docker-compose -f docker-compose.complete.yml restart service_name

# Update system
git pull && ./deploy_everything.sh

# Backup databases
./scripts/backup_databases.sh

# Clean up resources
docker system prune -f
```

### Investigation Operations
```bash
# Access IntelOwl for analysis
# Visit: http://localhost

# Query via API
curl -H "Authorization: Bearer <token>" \
     http://localhost:3010/tools/call \
     -d '{"name": "osint_collector", "arguments": {"target": "example.com"}}'

# Access graph database
# Visit: http://localhost:7474 (neo4j/BevGraphMaster2024)
```

**📋 Complete procedures**: [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) and [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)

## 🎓 Learning Resources

### Beginner Path
1. **System Overview**: [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md#system-overview)
2. **Basic Usage**: [Documentation Index - Quick Start](comprehensive/BEV_DOCUMENTATION_INDEX.md#quick-start-guides)
3. **First Investigation**: [Analyst Guide - Basic Workflows](comprehensive/BEV_ANALYST_GUIDE.md#basic-investigation-workflows)

### Intermediate Path
1. **Advanced Analysis**: [Analyst Guide - Advanced Techniques](comprehensive/BEV_ANALYST_GUIDE.md#advanced-analysis-techniques)
2. **API Integration**: [Integration Workflows](comprehensive/BEV_INTEGRATION_WORKFLOWS.md)
3. **Performance Optimization**: [Troubleshooting - Performance](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#performance-optimization)

### Expert Path
1. **Custom Development**: [Integration Guide - Custom Analyzers](comprehensive/BEV_INTEGRATION_WORKFLOWS.md#custom-analyzer-development)
2. **Security Architecture**: [Security Guide - Advanced Topics](comprehensive/BEV_OPERATIONAL_SECURITY.md#advanced-security-topics)
3. **System Architecture**: [Component Catalog - Deep Dive](comprehensive/BEV_COMPONENT_CATALOG.md#detailed-architecture)

## 🔍 Quick Reference

### Essential Commands
| Task | Command | Documentation |
|------|---------|---------------|
| **Deploy System** | `./deploy_everything.sh` | [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) |
| **Health Check** | `./validate_bev_deployment.sh` | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) |
| **Run Tests** | `./run_all_tests.sh` | [Project Root] |
| **View Logs** | `docker-compose logs -f` | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) |

### Key Endpoints
| Service | URL | Credentials |
|---------|-----|-------------|
| IntelOwl | http://localhost | No authentication |
| Grafana | http://localhost:3000 | admin/admin |
| Neo4j | http://localhost:7474 | neo4j/BevGraphMaster2024 |
| API Server | http://localhost:3010 | Token required |

### Critical Files
| File | Purpose | Documentation |
|------|---------|---------------|
| `.env` | Configuration and API keys | [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) |
| `docker-compose.complete.yml` | Service orchestration | [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) |
| `validate_bev_deployment.sh` | System validation | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) |

## 🚨 Need Help?

### Common Issues
| Problem | Solution | Documentation |
|---------|----------|---------------|
| **Services won't start** | Check Docker and run health scripts | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#service-startup-issues) |
| **Poor performance** | Monitor resources and optimize | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#performance-issues) |
| **API authentication** | Verify tokens and permissions | [API Reference](API_REFERENCE.md#authentication) |
| **Investigation workflows** | Follow analyst procedures | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) |

### Emergency Procedures
- **System Down**: [Emergency Recovery](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#emergency-procedures)
- **Security Incident**: [Incident Response](comprehensive/BEV_OPERATIONAL_SECURITY.md#incident-response)
- **Data Loss**: [Backup Recovery](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#backup-recovery)

### Get Support
1. **Check Documentation**: Use [Knowledge Base](KNOWLEDGE_BASE.md) for cross-referenced information
2. **Review Troubleshooting**: [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) covers common issues
3. **Consult API Docs**: [API Reference](API_REFERENCE.md) for technical integration questions
4. **Follow Security Procedures**: [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) for security-related issues

## ⚠️ Important Notes

### Security Considerations
- **No Authentication**: System runs without authentication for single-user deployment
- **Private Networks Only**: Never expose to public internet
- **Authorized Use Only**: For legitimate security research and OSINT operations
- **Legal Compliance**: Follow all applicable laws and regulations

### System Requirements
- **Performance**: System optimized for high-throughput analysis
- **Resources**: Significant CPU/RAM requirements for complex analyses
- **Storage**: Large storage needs for investigation data and graphs
- **Network**: Tor integration for anonymized data collection

### Support Scope
- **Technical Documentation**: Comprehensive guides and references provided
- **Community Support**: GitHub issues and discussions available
- **Security Research**: Designed for authorized cybersecurity research
- **Legal Responsibility**: Users responsible for compliance and legal use

---

## 🔗 Documentation Map

```
docs/
├── README.md (This file)                          # Documentation hub and navigation
├── KNOWLEDGE_BASE.md                              # Cross-referenced information hub
├── API_REFERENCE.md                               # Complete API documentation
└── comprehensive/
    ├── BEV_DOCUMENTATION_INDEX.md                 # Navigation by role and complexity
    ├── BEV_COMPONENT_CATALOG.md                   # Architecture and system components
    ├── BEV_ANALYST_GUIDE.md                       # Investigation workflows
    ├── BEV_OPERATIONAL_SECURITY.md                # Security procedures
    ├── BEV_INTEGRATION_WORKFLOWS.md               # Development integration
    └── BEV_TROUBLESHOOTING_GUIDE.md               # Operations and problem solving
```

**🎯 Start with [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) for role-based navigation, or [Knowledge Base](KNOWLEDGE_BASE.md) for comprehensive cross-referenced information.**