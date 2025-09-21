#!/bin/bash
# Stop all Bev agents

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Bev Agent Swarm...${NC}"

# Function to stop agent
stop_agent() {
    local pid_file=$1
    local agent_name=$2
    
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        if kill -0 $PID 2>/dev/null; then
            echo -n "Stopping $agent_name (PID: $PID)... "
            kill $PID
            sleep 2
            
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID
            fi
            
            echo -e "${GREEN}STOPPED${NC}"
        else
            echo "$agent_name was not running"
        fi
        rm -f "$pid_file"
    else
        echo "$agent_name PID file not found"
    fi
}

# Stop all agents
stop_agent "/tmp/bev_research.pid" "Research Coordinator"
stop_agent "/tmp/bev_code.pid" "Code Optimizer"
stop_agent "/tmp/bev_memory.pid" "Memory Manager"
stop_agent "/tmp/bev_tool.pid" "Tool Coordinator"

echo ""
echo -e "${GREEN}All agents stopped successfully!${NC}"
echo ""
echo "To restart agents, run: ./scripts/start_agents.sh"
