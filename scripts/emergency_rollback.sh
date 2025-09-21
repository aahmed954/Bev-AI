#!/bin/bash
# Emergency Rollback Script for BEV Distributed Deployment
# Provides graduated rollback strategies for different failure scenarios

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

echo -e "${PURPLE}🚨 BEV EMERGENCY ROLLBACK SYSTEM${NC}"
echo -e "${BLUE}=================================${NC}"
echo "Date: $(date)"
echo ""

# Create rollback directory with timestamp
ROLLBACK_DIR="/tmp/bev-rollback-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ROLLBACK_DIR"
log "Rollback session directory: $ROLLBACK_DIR"

# Function to backup current state
backup_current_state() {
    log "${BLUE}📁 Backing up current deployment state...${NC}"

    # Backup Thanos state
    if ssh -o ConnectTimeout=5 thanos "echo 'Connected'" > /dev/null 2>&1; then
        log "Backing up Thanos service states..."
        ssh thanos "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" > "$ROLLBACK_DIR/thanos_services_before.txt"
        ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml config" > "$ROLLBACK_DIR/thanos_config_backup.yml" 2>/dev/null || true
    fi

    # Backup Oracle1 state
    if ssh -o ConnectTimeout=5 oracle1 "echo 'Connected'" > /dev/null 2>&1; then
        log "Backing up Oracle1 service states..."
        ssh oracle1 "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'" > "$ROLLBACK_DIR/oracle1_services_before.txt"
        ssh oracle1 "docker-compose -f /opt/bev/docker-compose-oracle1-unified.yml config" > "$ROLLBACK_DIR/oracle1_config_backup.yml" 2>/dev/null || true
    fi

    # Backup Starlord state
    log "Backing up Starlord service states..."
    docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' > "$ROLLBACK_DIR/starlord_services_before.txt"

    log "${GREEN}✅ Current state backed up to $ROLLBACK_DIR${NC}"
}

# Function for selective service restart
selective_restart() {
    local node=$1
    local services=$2

    log "${YELLOW}🔄 Selective restart on $node: $services${NC}"

    case $node in
        "thanos")
            for service in $services; do
                log "Restarting Thanos service: bev_$service"
                ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml restart $service" || true
                sleep 5
            done
            ;;
        "oracle1")
            for service in $services; do
                log "Restarting Oracle1 service: bev_$service"
                ssh oracle1 "docker-compose -f /opt/bev/docker-compose-oracle1-unified.yml restart $service" || true
                sleep 5
            done
            ;;
        "starlord")
            for service in $services; do
                log "Restarting Starlord service: bev_$service"
                docker-compose -f docker-compose-development.yml restart $service || true
                sleep 5
            done
            ;;
    esac
}

# Function for database recovery
database_recovery() {
    log "${BLUE}🗄 Database Recovery Mode${NC}"

    # Stop all dependent services first
    log "Stopping dependent services..."
    ssh thanos "docker stop \$(docker ps -q --filter 'name=bev_' | grep -v postgres | grep -v neo4j | grep -v redis)" || true
    sleep 10

    # Restart core databases in order
    log "Restarting PostgreSQL..."
    ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml restart postgres"
    sleep 30

    log "Restarting Neo4j..."
    ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml restart neo4j"
    sleep 30

    log "Restarting Redis..."
    ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml restart redis"
    sleep 15

    # Restart dependent services
    log "Restarting dependent services..."
    ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml up -d"

    log "${GREEN}✅ Database recovery completed${NC}"
}

# Function for full node reset
full_node_reset() {
    local node=$1

    log "${RED}🔥 Full reset for node: $node${NC}"

    case $node in
        "thanos")
            log "Stopping all Thanos services..."
            ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml down" || true
            sleep 10

            log "Removing orphaned containers..."
            ssh thanos "docker container prune -f" || true

            log "Restarting Thanos deployment..."
            ssh thanos "cd /opt/bev && docker-compose -f docker-compose-thanos-unified.yml up -d"
            ;;
        "oracle1")
            log "Stopping all Oracle1 services..."
            ssh oracle1 "docker-compose -f /opt/bev/docker-compose-oracle1-unified.yml down" || true
            sleep 10

            log "Removing orphaned containers..."
            ssh oracle1 "docker container prune -f" || true

            log "Restarting Oracle1 deployment..."
            ssh oracle1 "cd /opt/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d"
            ;;
        "starlord")
            log "Stopping all Starlord services..."
            docker-compose -f docker-compose-development.yml down || true
            sleep 10

            log "Removing orphaned containers..."
            docker container prune -f || true

            log "Restarting Starlord deployment..."
            docker-compose -f docker-compose-development.yml up -d
            ;;
    esac
}

# Function for emergency stop all
emergency_stop_all() {
    log "${RED}🛑 EMERGENCY STOP ALL SERVICES${NC}"

    # Stop all nodes in parallel
    {
        log "Stopping Thanos services..."
        ssh thanos "docker-compose -f /opt/bev/docker-compose-thanos-unified.yml down" || true
    } &

    {
        log "Stopping Oracle1 services..."
        ssh oracle1 "docker-compose -f /opt/bev/docker-compose-oracle1-unified.yml down" || true
    } &

    {
        log "Stopping Starlord services..."
        docker-compose -f docker-compose-development.yml down || true
        pkill -f "npm run dev" || true
    } &

    wait

    log "${GREEN}✅ All services stopped${NC}"
}

# Function to validate recovery
validate_recovery() {
    log "${BLUE}🏥 Validating recovery...${NC}"

    # Run distributed health check
    if [ -f "./scripts/health_check_distributed.sh" ]; then
        ./scripts/health_check_distributed.sh
        HEALTH_STATUS=$?

        if [ $HEALTH_STATUS -eq 0 ]; then
            log "${GREEN}✅ Recovery validation passed - system healthy${NC}"
            return 0
        else
            log "${YELLOW}⚠️ Recovery validation partial - some issues remain${NC}"
            return 1
        fi
    else
        log "${YELLOW}⚠️ Health check script not found - manual validation required${NC}"
        return 2
    fi
}

# Function to show rollback menu
show_rollback_menu() {
    echo ""
    echo -e "${CYAN}🔧 ROLLBACK OPTIONS:${NC}"
    echo "1. Selective Service Restart (least disruptive)"
    echo "2. Database Recovery (database issues)"
    echo "3. Full Node Reset (node-specific issues)"
    echo "4. Emergency Stop All (critical failure)"
    echo "5. Custom Recovery (guided troubleshooting)"
    echo "6. Status Check (assessment only)"
    echo "7. Exit"
    echo ""
}

# Function for custom recovery
custom_recovery() {
    echo -e "${CYAN}🔍 CUSTOM RECOVERY MODE${NC}"
    echo "This mode provides guided troubleshooting options."
    echo ""

    # Check which nodes are accessible
    echo "Node accessibility check:"
    THANOS_OK=false
    ORACLE1_OK=false

    if ssh -o ConnectTimeout=5 thanos "echo 'Connected'" > /dev/null 2>&1; then
        echo -e "Thanos: ${GREEN}✅ Accessible${NC}"
        THANOS_OK=true
    else
        echo -e "Thanos: ${RED}❌ Not accessible${NC}"
    fi

    if ssh -o ConnectTimeout=5 oracle1 "echo 'Connected'" > /dev/null 2>&1; then
        echo -e "Oracle1: ${GREEN}✅ Accessible${NC}"
        ORACLE1_OK=true
    else
        echo -e "Oracle1: ${RED}❌ Not accessible${NC}"
    fi

    echo -e "Starlord: ${GREEN}✅ Local node${NC}"
    echo ""

    # Provide targeted recommendations
    if [ "$THANOS_OK" = false ]; then
        echo -e "${RED}🚨 CRITICAL: Thanos node not accessible${NC}"
        echo "Recommended actions:"
        echo "1. Check Tailscale VPN connectivity"
        echo "2. Verify Thanos node is powered on"
        echo "3. Check SSH key authentication"
        echo "4. Try manual SSH: ssh thanos"
        return 1
    fi

    if [ "$ORACLE1_OK" = false ]; then
        echo -e "${YELLOW}⚠️ WARNING: Oracle1 node not accessible${NC}"
        echo "Impact: Monitoring and security services unavailable"
        echo "System can operate without Oracle1 but with reduced capabilities"
    fi

    # Check service health on accessible nodes
    if [ "$THANOS_OK" = true ]; then
        echo "Checking Thanos critical services..."
        THANOS_CRITICAL_ISSUES=()

        if ! ssh thanos "docker exec bev_postgres pg_isready -U researcher" > /dev/null 2>&1; then
            THANOS_CRITICAL_ISSUES+=("PostgreSQL")
        fi

        if ! ssh thanos "curl -s http://localhost:7474" > /dev/null 2>&1; then
            THANOS_CRITICAL_ISSUES+=("Neo4j")
        fi

        if ! ssh thanos "docker exec bev_redis redis-cli ping" > /dev/null 2>&1; then
            THANOS_CRITICAL_ISSUES+=("Redis")
        fi

        if [ ${#THANOS_CRITICAL_ISSUES[@]} -gt 0 ]; then
            echo -e "${RED}🚨 Thanos critical service issues: ${THANOS_CRITICAL_ISSUES[*]}${NC}"
            echo "Recommendation: Run database recovery (option 2)"
        else
            echo -e "${GREEN}✅ Thanos critical services healthy${NC}"
        fi
    fi

    echo ""
    echo "Based on analysis, recommended recovery action:"
    if [ ${#THANOS_CRITICAL_ISSUES[@]} -gt 0 ]; then
        echo -e "${YELLOW}→ Database Recovery (option 2)${NC}"
    elif [ "$ORACLE1_OK" = false ]; then
        echo -e "${YELLOW}→ Oracle1 Node Reset (option 3)${NC}"
    else
        echo -e "${GREEN}→ Selective Service Restart (option 1)${NC}"
    fi
}

# Main execution
backup_current_state

# Interactive menu
while true; do
    show_rollback_menu
    echo -n "Select rollback option (1-7): "
    read -r choice

    case $choice in
        1)
            echo -n "Enter node (thanos/oracle1/starlord): "
            read -r node
            echo -n "Enter services to restart (space-separated): "
            read -r services
            selective_restart "$node" "$services"
            validate_recovery
            ;;
        2)
            database_recovery
            validate_recovery
            ;;
        3)
            echo -n "Enter node to reset (thanos/oracle1/starlord): "
            read -r node
            full_node_reset "$node"
            sleep 60  # Wait for services to start
            validate_recovery
            ;;
        4)
            echo -e "${RED}⚠️ This will stop ALL services. Are you sure? (yes/no):${NC}"
            read -r confirm
            if [ "$confirm" = "yes" ]; then
                emergency_stop_all
            else
                echo "Emergency stop cancelled."
            fi
            ;;
        5)
            custom_recovery
            ;;
        6)
            validate_recovery
            ;;
        7)
            log "Exiting rollback system. Recovery session saved to: $ROLLBACK_DIR"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please select 1-7.${NC}"
            ;;
    esac

    echo ""
    echo -e "${BLUE}Press Enter to continue or Ctrl+C to exit...${NC}"
    read -r
done