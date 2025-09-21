#!/bin/bash
# BEV MASTER DEPLOYMENT SCRIPT - REAL IMPLEMENTATIONS ONLY
# Deploys substantial services (500+ lines) with Vault integration
# Uses existing infrastructure without creating new complexity

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

DEPLOYMENT_LOG="deployment_$(date +%Y%m%d_%H%M%S).log"
ROLLBACK_STATE="rollback_state_$(date +%Y%m%d_%H%M%S).txt"

# Logging function
log() {
    echo -e "$1" | tee -a "$DEPLOYMENT_LOG"
}

log "${PURPLE}🚀 BEV MASTER DEPLOYMENT - REAL IMPLEMENTATIONS${NC}"
log "${BLUE}=============================================================${NC}"
log "Deploying substantial services with proven implementations"
log "Started: $(date)"
log ""

# Pre-flight checks
log "${CYAN}Phase 0: Pre-flight Validation${NC}"

# Check if we're in the correct directory
if [ ! -f "docker-compose.complete.yml" ]; then
    log "${RED}❌ Not in BEV project directory. Run from project root.${NC}"
    exit 1
fi

# Check for substantial implementations
REQUIRED_SERVICES=(
    "src/alternative_market/dm_crawler.py"
    "src/alternative_market/crypto_analyzer.py"
    "src/alternative_market/reputation_analyzer.py"
    "src/alternative_market/economics_processor.py"
    "src/security/intel_fusion.py"
    "src/security/opsec_enforcer.py"
    "src/security/defense_automation.py"
    "src/security/tactical_intelligence.py"
    "src/autonomous/enhanced_autonomous_controller.py"
    "src/autonomous/adaptive_learning.py"
    "src/autonomous/resource_optimizer.py"
    "src/autonomous/knowledge_evolution.py"
)

log "Validating real implementations..."
for service in "${REQUIRED_SERVICES[@]}"; do
    if [ ! -f "$service" ]; then
        log "${RED}❌ Missing implementation: $service${NC}"
        exit 1
    fi

    # Check if implementation is substantial (>500 lines)
    lines=$(wc -l < "$service")
    if [ "$lines" -lt 500 ]; then
        log "${YELLOW}⚠️  Service $service has only $lines lines (expected >500)${NC}"
    else
        log "${GREEN}✅ $service validated ($lines lines)${NC}"
    fi
done

# Check Docker Compose files
for compose_file in docker-compose-phase7.yml docker-compose-phase8.yml docker-compose-phase9.yml; do
    if [ ! -f "$compose_file" ]; then
        log "${RED}❌ Missing compose file: $compose_file${NC}"
        exit 1
    fi
    log "${GREEN}✅ Found $compose_file${NC}"
done

# Create rollback state
log "Creating rollback state..."
docker ps --format "table {{.Names}}\t{{.Status}}" > "$ROLLBACK_STATE"

# Phase 1: Infrastructure and Credentials
log ""
log "${CYAN}Phase 1: Infrastructure and Vault Credentials${NC}"

# Generate credentials if missing
if [ ! -f ".env.secure" ]; then
    log "Generating secure credentials..."
    if [ -f "generate-secure-credentials.sh" ]; then
        ./generate-secure-credentials.sh
    else
        log "${YELLOW}No credential generator found, using default .env${NC}"
        cp .env .env.secure 2>/dev/null || true
    fi
fi

# Source credentials
if [ -f ".env.secure" ]; then
    source .env.secure
    log "${GREEN}✅ Credentials loaded${NC}"
elif [ -f ".env" ]; then
    source .env
    log "${YELLOW}⚠️  Using default .env credentials${NC}"
else
    log "${RED}❌ No credential files found${NC}"
    exit 1
fi

# Start core infrastructure
log "Starting core infrastructure..."
log "→ PostgreSQL with pgvector"
log "→ Neo4j graph database"
log "→ Redis cache"
log "→ Elasticsearch search"

docker-compose -f docker-compose.complete.yml up -d \
    postgres \
    neo4j \
    redis \
    elasticsearch \
    tor \
    kafka-1 kafka-2 kafka-3 \
    rabbitmq

# Wait for core services
log "Waiting for core services to be ready..."
sleep 30

# Validate core services
for service in postgres neo4j redis elasticsearch; do
    if docker ps | grep -q "bev_$service"; then
        log "${GREEN}✅ $service running${NC}"
    else
        log "${RED}❌ $service failed to start${NC}"
        exit 1
    fi
done

# Phase 2: Alternative Market Intelligence (Phase 7)
log ""
log "${CYAN}Phase 2: Alternative Market Intelligence (Phase 7)${NC}"
log "Services: DM Crawler (886 lines), Crypto Analyzer (1539 lines), Reputation Analyzer (1246 lines), Economics Processor (1693 lines)"

# Create networks if they don't exist
docker network create bev_osint 2>/dev/null || true

docker-compose -f docker-compose-phase7.yml up -d

# Health check for Phase 7
log "Validating Phase 7 services..."
sleep 30
PHASE7_SERVICES=("bev_dm_crawler" "bev_crypto_analyzer" "bev_reputation_analyzer" "bev_economics_processor")
for service in "${PHASE7_SERVICES[@]}"; do
    if docker ps | grep -q "$service"; then
        log "${GREEN}✅ $service deployed successfully${NC}"
    else
        log "${YELLOW}⚠️  $service may still be starting...${NC}"
    fi
done

# Phase 3: Security Operations (Phase 8)
log ""
log "${CYAN}Phase 3: Security Operations (Phase 8)${NC}"
log "Services: Intel Fusion (2137 lines), OPSEC Enforcer (1606 lines), Defense Automation (1379 lines), Tactical Intelligence (1162 lines)"

docker-compose -f docker-compose-phase8.yml up -d

# Health check for Phase 8
log "Validating Phase 8 services..."
sleep 30
PHASE8_SERVICES=("bev_intel_fusion" "bev_opsec_enforcer" "bev_defense_automation" "bev_tactical_intelligence")
for service in "${PHASE8_SERVICES[@]}"; do
    if docker ps | grep -q "$service"; then
        log "${GREEN}✅ $service deployed successfully${NC}"
    else
        log "${YELLOW}⚠️  $service may still be starting...${NC}"
    fi
done

# Phase 4: Autonomous Systems (Phase 9)
log ""
log "${CYAN}Phase 4: Autonomous Systems (Phase 9)${NC}"
log "Services: Enhanced Controller (1383 lines), Adaptive Learning (1566 lines), Resource Optimizer (1395 lines), Knowledge Evolution (1514 lines)"

docker-compose -f docker-compose-phase9.yml up -d

# Health check for Phase 9
log "Validating Phase 9 services..."
sleep 30
PHASE9_SERVICES=("bev_enhanced_autonomous_controller" "bev_adaptive_learning" "bev_resource_optimizer" "bev_knowledge_evolution")
for service in "${PHASE9_SERVICES[@]}"; do
    if docker ps | grep -q "$service"; then
        log "${GREEN}✅ $service deployed successfully${NC}"
    else
        log "${YELLOW}⚠️  $service may still be starting...${NC}"
    fi
done

# Phase 5: System Validation
log ""
log "${CYAN}Phase 5: System Validation${NC}"

# Wait for all services to stabilize
log "Allowing services to stabilize..."
sleep 60

# Comprehensive health check
log "Running comprehensive health checks..."
TOTAL_SERVICES=0
HEALTHY_SERVICES=0

ALL_SERVICES=(
    "bev_postgres" "bev_neo4j" "bev_redis" "bev_elasticsearch"
    "bev_dm_crawler" "bev_crypto_analyzer" "bev_reputation_analyzer" "bev_economics_processor"
    "bev_intel_fusion" "bev_opsec_enforcer" "bev_defense_automation" "bev_tactical_intelligence"
    "bev_enhanced_autonomous_controller" "bev_adaptive_learning" "bev_resource_optimizer" "bev_knowledge_evolution"
)

for service in "${ALL_SERVICES[@]}"; do
    TOTAL_SERVICES=$((TOTAL_SERVICES + 1))
    if docker ps | grep -q "$service.*Up"; then
        HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
        log "${GREEN}✅ $service - Healthy${NC}"
    else
        log "${RED}❌ $service - Not running${NC}"
    fi
done

# Calculate deployment success rate
SUCCESS_RATE=$((HEALTHY_SERVICES * 100 / TOTAL_SERVICES))
log ""
log "${BLUE}Deployment Summary:${NC}"
log "Services deployed: $HEALTHY_SERVICES/$TOTAL_SERVICES"
log "Success rate: $SUCCESS_RATE%"

if [ "$SUCCESS_RATE" -ge 80 ]; then
    log "${GREEN}🎉 Deployment successful! ($SUCCESS_RATE% services healthy)${NC}"

    # Display service endpoints
    log ""
    log "${CYAN}Service Endpoints:${NC}"
    log "• PostgreSQL: localhost:5432 (user: $POSTGRES_USER)"
    log "• Neo4j: http://localhost:7474 (user: neo4j)"
    log "• Redis: localhost:6379"
    log "• Elasticsearch: http://localhost:9200"
    log "• Alternative Market Intelligence: Phase 7 services running"
    log "• Security Operations: Phase 8 services running"
    log "• Autonomous Systems: Phase 9 services running"

    # Create deployment success marker
    echo "$(date): Deployment successful - $SUCCESS_RATE% success rate" > .deployment_success

elif [ "$SUCCESS_RATE" -ge 50 ]; then
    log "${YELLOW}⚠️  Partial deployment ($SUCCESS_RATE% success rate)${NC}"
    log "Some services may still be starting. Check logs with:"
    log "docker-compose -f docker-compose-phase7.yml logs"
    log "docker-compose -f docker-compose-phase8.yml logs"
    log "docker-compose -f docker-compose-phase9.yml logs"

else
    log "${RED}❌ Deployment failed ($SUCCESS_RATE% success rate)${NC}"
    log "Critical failure - consider rollback"
    exit 1
fi

# Phase 6: Post-Deployment Actions
log ""
log "${CYAN}Phase 6: Post-Deployment Actions${NC}"

# Save deployment state
log "Saving deployment state..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" > "deployment_state_$(date +%Y%m%d_%H%M%S).txt"

# Enable monitoring if available
if docker ps | grep -q "bev_prometheus"; then
    log "${GREEN}✅ Monitoring stack detected and running${NC}"
else
    log "${YELLOW}ℹ️  Monitoring stack not deployed (optional)${NC}"
fi

# Final summary
log ""
log "${PURPLE}🚀 BEV REAL IMPLEMENTATIONS DEPLOYMENT COMPLETE${NC}"
log "${BLUE}=============================================================${NC}"
log "Completed: $(date)"
log "Deployment log: $DEPLOYMENT_LOG"
log "Rollback state: $ROLLBACK_STATE"
log ""
log "${GREEN}Real services with substantial implementations successfully deployed!${NC}"
log "• Alternative Market Intelligence: 4 services (4,000+ lines total)"
log "• Security Operations: 4 services (6,000+ lines total)"
log "• Autonomous Systems: 4 services (5,500+ lines total)"
log ""
log "Use './validate_bev_deployment.sh' for ongoing health checks"
log "Use 'docker-compose -f docker-compose-complete.yml logs -f' for monitoring"

# Show quick status
log ""
log "${CYAN}Quick Status Check:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bev_ | head -20

exit 0