#!/bin/bash
# Start all Bev agents

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Bev Agent Swarm...${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Start databases if not running
echo -e "${YELLOW}Checking database services...${NC}"

# Check PostgreSQL
if ! pg_isready -h localhost -p 5432; then
    echo -e "${RED}PostgreSQL is not running! Please start it first.${NC}"
    exit 1
fi

# Check Neo4j
if ! curl -s http://localhost:7474 > /dev/null; then
    echo -e "${RED}Neo4j is not running! Please start it first.${NC}"
    exit 1
fi

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}Redis is not running! Please start it first.${NC}"
    exit 1
fi

echo -e "${GREEN}All database services are running!${NC}"

# Start agents in background
echo -e "${YELLOW}Starting Research Coordinator...${NC}"
python src/agents/research_coordinator.py &
RESEARCH_PID=$!
echo -e "${GREEN}Research Coordinator started (PID: $RESEARCH_PID)${NC}"

echo -e "${YELLOW}Starting Code Optimizer...${NC}"
python src/agents/code_optimizer.py &
CODE_PID=$!
echo -e "${GREEN}Code Optimizer started (PID: $CODE_PID)${NC}"

echo -e "${YELLOW}Starting Memory Manager...${NC}"
python src/agents/memory_manager.py &
MEMORY_PID=$!
echo -e "${GREEN}Memory Manager started (PID: $MEMORY_PID)${NC}"

echo -e "${YELLOW}Starting Tool Coordinator...${NC}"
python src/agents/tool_coordinator.py &
TOOL_PID=$!
echo -e "${GREEN}Tool Coordinator started (PID: $TOOL_PID)${NC}"

# Save PIDs to file
echo $RESEARCH_PID > /tmp/bev_research.pid
echo $CODE_PID > /tmp/bev_code.pid
echo $MEMORY_PID > /tmp/bev_memory.pid
echo $TOOL_PID > /tmp/bev_tool.pid

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Bev Agent Swarm is running!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Agent PIDs:"
echo "  Research Coordinator: $RESEARCH_PID"
echo "  Code Optimizer: $CODE_PID"
echo "  Memory Manager: $MEMORY_PID"
echo "  Tool Coordinator: $TOOL_PID"
echo ""
echo "To stop agents, run: ./scripts/stop_agents.sh"
echo "To check health, run: ./scripts/health_check.sh"
echo ""
echo -e "${YELLOW}Monitoring logs at ./logs/${NC}"

# Keep script running and monitor agents
while true; do
    sleep 30
    
    # Check if agents are still running
    if ! kill -0 $RESEARCH_PID 2>/dev/null; then
        echo -e "${RED}Research Coordinator crashed! Restarting...${NC}"
        python src/agents/research_coordinator.py &
        RESEARCH_PID=$!
        echo $RESEARCH_PID > /tmp/bev_research.pid
    fi
    
    if ! kill -0 $CODE_PID 2>/dev/null; then
        echo -e "${RED}Code Optimizer crashed! Restarting...${NC}"
        python src/agents/code_optimizer.py &
        CODE_PID=$!
        echo $CODE_PID > /tmp/bev_code.pid
    fi
    
    if ! kill -0 $MEMORY_PID 2>/dev/null; then
        echo -e "${RED}Memory Manager crashed! Restarting...${NC}"
        python src/agents/memory_manager.py &
        MEMORY_PID=$!
        echo $MEMORY_PID > /tmp/bev_memory.pid
    fi
    
    if ! kill -0 $TOOL_PID 2>/dev/null; then
        echo -e "${RED}Tool Coordinator crashed! Restarting...${NC}"
        python src/agents/tool_coordinator.py &
        TOOL_PID=$!
        echo $TOOL_PID > /tmp/bev_tool.pid
    fi
done
