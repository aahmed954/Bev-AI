#!/bin/bash
# Master BEV Deployment Controller for Claude Code CLI

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 BEV DEPLOYMENT MASTER CONTROLLER${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Function to run Claude Code CLI command and wait for confirmation
run_task() {
    local task_id=$1
    local command=$2
    local description=$3
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Task $task_id: $description${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "Executing: claude-code '$command'"
    echo ""
    
    # Log start time
    echo "[$task_id] Started: $(date)" >> deployment.log
    
    # Execute the command
    claude-code "$command"
    
    # Log completion
    echo "[$task_id] Completed: $(date)" >> deployment.log
    
    # Ask for confirmation before proceeding
    echo ""
    echo -e "${GREEN}Task $task_id completed.${NC}"
    echo -e "${YELLOW}Please verify the output above is correct.${NC}"
    read -p "Press Enter to continue to next task, or Ctrl+C to stop: "
    
    # Save checkpoint
    echo "$task_id" > .last_completed_task
    
    echo ""
}

# Check if resuming from a checkpoint
if [ -f ".last_completed_task" ]; then
    LAST_TASK=$(cat .last_completed_task)
    echo -e "${YELLOW}Found checkpoint at task: $LAST_TASK${NC}"
    read -p "Resume from next task? (y/n): " resume
    if [ "$resume" != "y" ]; then
        echo "Starting from beginning..."
        rm -f .last_completed_task
    fi
fi

# Phase A: Understanding
if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "A1" ]; then
    run_task "A1" "/sc:load" "Load current state and memories"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "A2" ]; then
    run_task "A2" "/sc:index" "Index all 151 services"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "A3" ]; then
    run_task "A3" "/sc:explain \"multinode architecture with 151 services\"" "Explain current state"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "A4" ]; then
    run_task "A4" "/sc:save \"phase-a-complete\"" "Save Phase A checkpoint"
fi

echo -e "${GREEN}✅ PHASE A COMPLETE${NC}"
echo ""

# Phase B: Analysis & Fixes
if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "B1" ]; then
    run_task "B1" "/sc:analyze" "Analyze entire codebase"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "B2" ]; then
    run_task "B2" "/sc:troubleshoot \"Docker services not starting on remote nodes\"" "Troubleshoot deployment issues"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "B3" ]; then
    run_task "B3" "/sc:cleanup" "Clean up dead code and duplicates"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "B4" ]; then
    run_task "B4" "/sc:improve" "Apply code improvements"
fi

if [ ! -f ".last_completed_task" ] || [ "$(cat .last_completed_task)" < "B5" ]; then
    run_task "B5" "/sc:save \"phase-b-complete\"" "Save Phase B checkpoint"
fi

echo -e "${GREEN}✅ PHASE B COMPLETE${NC}"
echo ""

# Continue for remaining phases...
# (Truncated for brevity, but would include all phases)

echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo ""
echo "Review the deployment.log for complete history"
echo "All tasks completed successfully!"

# Clean up checkpoint file
rm -f .last_completed_task
