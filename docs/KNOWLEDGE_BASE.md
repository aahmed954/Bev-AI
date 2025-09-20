# BEV OSINT Framework - Knowledge Base

## 🎯 Navigation Hub

This knowledge base provides comprehensive cross-referenced information for the BEV OSINT Framework. Use this as your central navigation point to find specific information across all documentation.

## 📚 Documentation Structure

### 🏗️ Architecture & Technical Reference
- **[API Reference](API_REFERENCE.md)** - Complete API documentation with examples
- **[Architecture Overview](comprehensive/BEV_COMPONENT_CATALOG.md)** - System architecture and components
- **[Integration Workflows](comprehensive/BEV_INTEGRATION_WORKFLOWS.md)** - Development and integration patterns

### 👥 User Guides by Role
- **[OSINT Analysts](comprehensive/BEV_ANALYST_GUIDE.md)** - Investigation workflows and methodologies
- **[System Operators](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)** - Operations and troubleshooting
- **[Security Teams](comprehensive/BEV_OPERATIONAL_SECURITY.md)** - Security architecture and procedures

### 🚀 Quick Start Resources
- **[Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md)** - Complete documentation navigation
- **[Deployment Guide](#deployment-quick-reference)** - Fast deployment procedures
- **[Common Tasks](#common-tasks-reference)** - Frequently needed operations

## 🔍 Quick Reference Tables

### Service Access Points
| Service | URL | Credentials | Purpose |
|---------|-----|-------------|---------|
| IntelOwl Dashboard | http://localhost | None (No auth) | Main OSINT interface |
| Cytoscape Graph | http://localhost/cytoscape | None | Network visualization |
| Neo4j Browser | http://localhost:7474 | neo4j/BevGraphMaster2024 | Graph database |
| Grafana Monitoring | http://localhost:3000 | admin/admin | System monitoring |
| Prometheus Metrics | http://localhost:9090 | None | Metrics collection |
| RabbitMQ Management | http://localhost:15672 | guest/guest | Message queue |
| MCP Server API | http://localhost:3010 | Token required | OSINT API server |

### Database Connections
| Database | Connection String | Purpose |
|----------|-------------------|---------|
| PostgreSQL | `postgresql://bev:BevOSINT2024@localhost:5432/osint` | Primary data storage |
| Neo4j | `bolt://localhost:7687` | Graph relationships |
| Redis | `redis://:BevCacheMaster@localhost:6379` | Caching and sessions |
| Elasticsearch | `http://localhost:9200` | Search and analytics |

### Custom Analyzers
| Analyzer | Purpose | Sources | Configuration |
|----------|---------|---------|---------------|
| BreachDatabaseAnalyzer | Breach database searches | Dehashed, Snusbase, WeLeakInfo | API keys required |
| DarknetMarketAnalyzer | Darknet market intelligence | AlphaBay, White House, Torrez | Tor proxy required |
| CryptoTrackerAnalyzer | Cryptocurrency analysis | Bitcoin, Ethereum explorers | Blockchain APIs |
| SocialMediaAnalyzer | Social media OSINT | Instagram, Twitter, LinkedIn | Platform APIs |

## 🚀 Deployment Quick Reference

### System Requirements
- **OS**: Ubuntu 20.04+ or compatible Linux
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 500GB SSD minimum
- **Docker**: 20.10+ with Docker Compose
- **Network**: Tor proxy capabilities

### Essential Commands
```bash
# Complete system deployment
./deploy_everything.sh

# System validation
./validate_bev_deployment.sh

# Health check
./scripts/health_check.sh

# Run all tests
./run_all_tests.sh

# Stop all services
docker-compose -f docker-compose.complete.yml down
```

### Configuration Files
- **Environment**: `.env` - API keys and configuration
- **Services**: `docker-compose.complete.yml` - Service orchestration
- **IntelOwl**: `intelowl/` - Custom analyzers configuration
- **Testing**: `tests/test_config.yaml` - Test parameters

## 📋 Common Tasks Reference

### Investigation Workflows
| Task | Documentation | Key Steps |
|------|---------------|-----------|
| Email Investigation | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md#email-investigations) | 1. Breach search 2. Social analysis 3. Correlation |
| Domain Analysis | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md#domain-investigations) | 1. WHOIS lookup 2. Subdomain enum 3. Infrastructure mapping |
| Cryptocurrency Tracking | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md#cryptocurrency-investigations) | 1. Address analysis 2. Transaction flow 3. Exchange identification |
| Threat Actor Attribution | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md#attribution-analysis) | 1. TTP analysis 2. Infrastructure overlap 3. Behavioral patterns |

### System Operations
| Task | Documentation | Commands |
|------|---------------|----------|
| Service Health Check | [Troubleshooting](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#health-monitoring) | `./validate_bev_deployment.sh` |
| Performance Monitoring | [Troubleshooting](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#performance-monitoring) | Access Grafana dashboard |
| Database Backup | [Troubleshooting](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#backup-procedures) | `./scripts/backup_databases.sh` |
| Log Analysis | [Troubleshooting](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#log-analysis) | `docker-compose logs -f` |

### Development Tasks
| Task | Documentation | Key Resources |
|------|---------------|---------------|
| Custom Analyzer Development | [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md#custom-analyzers) | Analyzer templates and examples |
| API Integration | [API Reference](API_REFERENCE.md) | WebSocket and REST endpoints |
| Testing New Features | [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md#testing-patterns) | Test frameworks and patterns |
| Security Implementation | [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) | Security patterns and validation |

## 🔗 Cross-Reference Matrix

### By Use Case
| Use Case | Primary Docs | Supporting Docs | Prerequisites |
|----------|--------------|-----------------|---------------|
| **New Installation** | [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) | Component Catalog, Security Guide | System requirements, Docker |
| **Daily Operations** | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) | API Reference, Troubleshooting | User training, access credentials |
| **System Administration** | [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) | Security Guide, Component Catalog | Admin access, monitoring setup |
| **Custom Development** | [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md) | API Reference, Component Catalog | Development environment |
| **Security Audit** | [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) | Troubleshooting, Component Catalog | Security clearance, audit tools |

### By Component
| Component | Primary Docs | Integration Docs | Troubleshooting |
|-----------|--------------|------------------|----------------|
| **IntelOwl Platform** | [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) | [Service Issues](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#intelowl-issues) |
| **Custom Analyzers** | [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md) | [API Reference](API_REFERENCE.md) | [Analyzer Debugging](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#analyzer-issues) |
| **MCP Server** | [API Reference](API_REFERENCE.md) | [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md) | [API Issues](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#api-issues) |
| **Graph Database** | [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) | [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) | [Database Issues](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#database-issues) |
| **Security Framework** | [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) | [All Guides] | [Security Incidents](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#security-incidents) |

## 🎓 Learning Paths

### For OSINT Analysts
1. **Start**: [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) → Quick Start for Analysts
2. **Learn**: [Analyst Guide](comprehensive/BEV_ANALYST_GUIDE.md) → Investigation methodologies
3. **Practice**: [API Reference](API_REFERENCE.md) → Tool usage examples
4. **Advanced**: [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) → Operational security

### For System Operators
1. **Start**: [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) → Operator Quick Start
2. **Deploy**: [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) → Architecture understanding
3. **Monitor**: [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md) → Operations procedures
4. **Secure**: [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) → Security operations

### For Developers
1. **Start**: [API Reference](API_REFERENCE.md) → Technical overview
2. **Integrate**: [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md) → Development patterns
3. **Extend**: [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) → Architecture deep dive
4. **Secure**: [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md) → Security patterns

## 🚨 Emergency Procedures

### System Down
1. **Check**: `docker-compose ps` - Verify service status
2. **Restart**: `./deploy_everything.sh` - Full system restart
3. **Validate**: `./validate_bev_deployment.sh` - Confirm recovery
4. **Document**: [Incident Response](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#incident-response)

### Security Incident
1. **Isolate**: [Emergency Isolation](comprehensive/BEV_OPERATIONAL_SECURITY.md#emergency-procedures)
2. **Assess**: [Incident Assessment](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#security-incidents)
3. **Respond**: [Response Procedures](comprehensive/BEV_OPERATIONAL_SECURITY.md#incident-response)
4. **Recovery**: [System Recovery](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#recovery-procedures)

### Performance Issues
1. **Monitor**: Access Grafana dashboard at http://localhost:3000
2. **Diagnose**: [Performance Troubleshooting](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md#performance-issues)
3. **Optimize**: [Performance Tuning](comprehensive/BEV_COMPONENT_CATALOG.md#performance-optimization)
4. **Validate**: Run performance tests with `./run_all_tests.sh --performance`

## 📞 Support Resources

### Documentation Issues
- **Missing Information**: Check cross-references in related documents
- **Outdated Procedures**: Validate against current system version
- **Technical Questions**: Refer to appropriate technical documentation

### System Issues
- **Service Problems**: [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)
- **Security Concerns**: [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md)
- **Integration Issues**: [Integration Guide](comprehensive/BEV_INTEGRATION_WORKFLOWS.md)

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **Security Disclosure**: Follow responsible disclosure procedures
- **Contributing**: See contribution guidelines in project root

---

## 🎯 Quick Search Guide

**Need to find something specific?** Use these search patterns:

- **Configuration**: Look in [Component Catalog](comprehensive/BEV_COMPONENT_CATALOG.md) or `.env` files
- **Procedures**: Check [Documentation Index](comprehensive/BEV_DOCUMENTATION_INDEX.md) by role
- **Troubleshooting**: Start with [Troubleshooting Guide](comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)
- **API Details**: Reference [API Documentation](API_REFERENCE.md)
- **Security**: Consult [Security Guide](comprehensive/BEV_OPERATIONAL_SECURITY.md)
- **Workflows**: Review [relevant role guide](#user-guides-by-role)

**Remember**: This is a research framework designed for authorized security research and OSINT operations only. Always follow applicable laws, regulations, and ethical guidelines.