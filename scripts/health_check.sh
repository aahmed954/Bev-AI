#!/bin/bash
# Health check for Bev system

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}    BEV SYSTEM HEALTH CHECK     ${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

HEALTH_SCORE=0
MAX_SCORE=10

# Function to check service
check_service() {
    local name=$1
    local check_command=$2
    local port=$3
    
    echo -n "Checking $name... "
    
    if eval $check_command > /dev/null 2>&1; then
        echo -e "${GREEN}✓ RUNNING${NC}"
        if [ ! -z "$port" ]; then
            echo "  └─ Port $port is accessible"
        fi
        return 0
    else
        echo -e "${RED}✗ DOWN${NC}"
        return 1
    fi
}

# Check databases
echo -e "${YELLOW}DATABASE SERVICES:${NC}"
if check_service "PostgreSQL" "pg_isready -h localhost -p 5432" "5432"; then
    ((HEALTH_SCORE++))
fi

if check_service "Neo4j" "curl -s http://localhost:7474" "7474"; then
    ((HEALTH_SCORE++))
fi

if check_service "Redis" "redis-cli ping" "6379"; then
    ((HEALTH_SCORE++))
fi

if check_service "Elasticsearch" "curl -s http://localhost:9200" "9200"; then
    ((HEALTH_SCORE++))
fi

echo ""

# Check agents
echo -e "${YELLOW}AGENT SERVICES:${NC}"

# Research Coordinator
if [ -f /tmp/bev_research.pid ]; then
    PID=$(cat /tmp/bev_research.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "Research Coordinator... ${GREEN}✓ RUNNING${NC} (PID: $PID)"
        ((HEALTH_SCORE++))
    else
        echo -e "Research Coordinator... ${RED}✗ DOWN${NC}"
    fi
else
    echo -e "Research Coordinator... ${RED}✗ NOT STARTED${NC}"
fi

# Code Optimizer
if [ -f /tmp/bev_code.pid ]; then
    PID=$(cat /tmp/bev_code.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "Code Optimizer... ${GREEN}✓ RUNNING${NC} (PID: $PID)"
        ((HEALTH_SCORE++))
    else
        echo -e "Code Optimizer... ${RED}✗ DOWN${NC}"
    fi
else
    echo -e "Code Optimizer... ${RED}✗ NOT STARTED${NC}"
fi

# Memory Manager
if [ -f /tmp/bev_memory.pid ]; then
    PID=$(cat /tmp/bev_memory.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "Memory Manager... ${GREEN}✓ RUNNING${NC} (PID: $PID)"
        ((HEALTH_SCORE++))
    else
        echo -e "Memory Manager... ${RED}✗ DOWN${NC}"
    fi
else
    echo -e "Memory Manager... ${RED}✗ NOT STARTED${NC}"
fi

# Tool Coordinator
if [ -f /tmp/bev_tool.pid ]; then
    PID=$(cat /tmp/bev_tool.pid)
    if kill -0 $PID 2>/dev/null; then
        echo -e "Tool Coordinator... ${GREEN}✓ RUNNING${NC} (PID: $PID)"
        ((HEALTH_SCORE++))
    else
        echo -e "Tool Coordinator... ${RED}✗ DOWN${NC}"
    fi
else
    echo -e "Tool Coordinator... ${RED}✗ NOT STARTED${NC}"
fi

echo ""

# Check monitoring services
echo -e "${YELLOW}MONITORING SERVICES:${NC}"
if check_service "Prometheus" "curl -s http://localhost:9090" "9090"; then
    ((HEALTH_SCORE++))
fi

if check_service "Grafana" "curl -s http://localhost:3000" "3000"; then
    ((HEALTH_SCORE++))
fi

echo ""

# Check disk space
echo -e "${YELLOW}SYSTEM RESOURCES:${NC}"
DISK_USAGE=$(df -h /home/starlord/Bev | awk 'NR==2 {print $5}' | sed 's/%//')
echo -n "Disk Usage: "
if [ $DISK_USAGE -lt 80 ]; then
    echo -e "${GREEN}$DISK_USAGE%${NC}"
elif [ $DISK_USAGE -lt 90 ]; then
    echo -e "${YELLOW}$DISK_USAGE% (Warning)${NC}"
else
    echo -e "${RED}$DISK_USAGE% (Critical)${NC}"
fi

# Check memory
MEM_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
echo -n "Memory Usage: "
if [ $MEM_USAGE -lt 80 ]; then
    echo -e "${GREEN}$MEM_USAGE%${NC}"
elif [ $MEM_USAGE -lt 90 ]; then
    echo -e "${YELLOW}$MEM_USAGE% (Warning)${NC}"
else
    echo -e "${RED}$MEM_USAGE% (Critical)${NC}"
fi

# Check CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print int(100 - $1)}')
echo -n "CPU Usage: "
if [ $CPU_USAGE -lt 80 ]; then
    echo -e "${GREEN}$CPU_USAGE%${NC}"
elif [ $CPU_USAGE -lt 90 ]; then
    echo -e "${YELLOW}$CPU_USAGE% (Warning)${NC}"
else
    echo -e "${RED}$CPU_USAGE% (Critical)${NC}"
fi

echo ""

# Check logs for errors
echo -e "${YELLOW}LOG ANALYSIS:${NC}"
if [ -d "/home/starlord/Bev/logs" ]; then
    ERROR_COUNT=$(find /home/starlord/Bev/logs -name "*.log" -mtime -1 -exec grep -l "ERROR" {} \; 2>/dev/null | wc -l)
    if [ $ERROR_COUNT -eq 0 ]; then
        echo -e "Recent Errors: ${GREEN}None${NC}"
    else
        echo -e "Recent Errors: ${YELLOW}$ERROR_COUNT file(s) with errors${NC}"
        echo "  Run 'tail -f logs/*.log | grep ERROR' to view"
    fi
else
    echo "Log directory not found"
fi

echo ""

# Overall health score
echo -e "${BLUE}================================${NC}"
echo -n "OVERALL HEALTH: "

HEALTH_PERCENTAGE=$((HEALTH_SCORE * 100 / MAX_SCORE))

if [ $HEALTH_PERCENTAGE -ge 80 ]; then
    echo -e "${GREEN}EXCELLENT${NC} ($HEALTH_SCORE/$MAX_SCORE services running)"
elif [ $HEALTH_PERCENTAGE -ge 60 ]; then
    echo -e "${YELLOW}DEGRADED${NC} ($HEALTH_SCORE/$MAX_SCORE services running)"
else
    echo -e "${RED}CRITICAL${NC} ($HEALTH_SCORE/$MAX_SCORE services running)"
fi

echo -e "${BLUE}================================${NC}"

# Recommendations
if [ $HEALTH_SCORE -lt $MAX_SCORE ]; then
    echo ""
    echo -e "${YELLOW}RECOMMENDATIONS:${NC}"
    
    if ! check_service "PostgreSQL" "pg_isready -h localhost -p 5432" > /dev/null 2>&1; then
        echo "  • Start PostgreSQL: sudo systemctl start postgresql"
    fi
    
    if ! check_service "Neo4j" "curl -s http://localhost:7474" > /dev/null 2>&1; then
        echo "  • Start Neo4j: neo4j start"
    fi
    
    if ! check_service "Redis" "redis-cli ping" > /dev/null 2>&1; then
        echo "  • Start Redis: redis-server"
    fi
    
    if [ ! -f /tmp/bev_research.pid ] || ! kill -0 $(cat /tmp/bev_research.pid 2>/dev/null) 2>/dev/null; then
        echo "  • Start agents: ./scripts/start_agents.sh"
    fi
fi

exit $((MAX_SCORE - HEALTH_SCORE))
