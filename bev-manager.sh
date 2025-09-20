#!/bin/bash
# BEV DISTRIBUTED SYSTEM MANAGER
# Easy control of all nodes

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

show_status() {
    echo -e "${PURPLE}BEV DISTRIBUTED SYSTEM STATUS${NC}"
    echo "================================"
    
    echo -e "${BLUE}THANOS (Local High-Performance):${NC}"
    ssh starlord@thanos "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep bev" 2>/dev/null || echo "  No services running"
    
    echo -e "${BLUE}ORACLE1 (ARM Cloud VM):${NC}"
    ssh starlord@oracle1 "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep bev" 2>/dev/null || echo "  No services running"
    
    echo -e "${BLUE}STARLORD (Local Workstation):${NC}"
    docker ps --format 'table {{.Names}}\t{{.Status}}' | grep bev 2>/dev/null || echo "  No services running"
}

start_all() {
    echo -e "${CYAN}Starting all nodes...${NC}"
    
    ssh starlord@thanos "cd /opt/bev && docker-compose -f docker-compose-thanos.yml up -d"
    ssh starlord@oracle1 "cd /opt/bev && docker-compose -f docker-compose-oracle1.yml up -d"
    sudo systemctl start bev-local
    
    echo -e "${GREEN}✅ All nodes started${NC}"
}

stop_all() {
    echo -e "${YELLOW}Stopping all nodes...${NC}"
    
    ssh starlord@thanos "cd /opt/bev && docker-compose -f docker-compose-thanos.yml down"
    ssh starlord@oracle1 "cd /opt/bev && docker-compose -f docker-compose-oracle1.yml down"
    sudo systemctl stop bev-local
    
    echo -e "${GREEN}✅ All nodes stopped${NC}"
}

update_all() {
    echo -e "${CYAN}Updating all nodes from GitHub...${NC}"
    
    # Update local first
    git pull origin main
    
    # Update Thanos
    ssh starlord@thanos "cd /opt/bev && git pull origin main && docker-compose -f docker-compose-thanos.yml up -d"
    
    # Update Oracle1
    ssh starlord@oracle1 "cd /opt/bev && git pull origin main && docker-compose -f docker-compose-oracle1.yml up -d"
    
    echo -e "${GREEN}✅ All nodes updated${NC}"
}

logs_node() {
    case $1 in
        thanos)
            ssh starlord@thanos "cd /opt/bev && docker-compose -f docker-compose-thanos.yml logs -f --tail=100"
            ;;
        oracle1)
            ssh starlord@oracle1 "cd /opt/bev && docker-compose -f docker-compose-oracle1.yml logs -f --tail=100"
            ;;
        local)
            docker-compose -f docker-compose-starlord.yml logs -f --tail=100
            ;;
        *)
            echo "Usage: $0 logs [thanos|oracle1|local]"
            ;;
    esac
}

case "$1" in
    status)
        show_status
        ;;
    start)
        start_all
        ;;
    stop)
        stop_all
        ;;
    update)
        update_all
        ;;
    logs)
        logs_node $2
        ;;
    *)
        echo -e "${PURPLE}BEV Distributed System Manager${NC}"
        echo ""
        echo "Usage: $0 {status|start|stop|update|logs [node]}"
        echo ""
        echo "Commands:"
        echo "  status  - Show status of all nodes"
        echo "  start   - Start services on all nodes"
        echo "  stop    - Stop services on all nodes"
        echo "  update  - Pull from GitHub and restart all nodes"
        echo "  logs    - View logs (thanos|oracle1|local)"
        echo ""
        echo "Quick access:"
        echo "  Databases:  http://thanos:5432 (PostgreSQL)"
        echo "              http://thanos:7474 (Neo4j)"
        echo "  Monitoring: http://oracle1:3000 (Grafana)"
        echo "  Dev UI:     http://localhost:5173 (when started)"
        ;;
esac
