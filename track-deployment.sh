#!/bin/bash
# BEV Deployment Progress Tracker
# Forces Claude Code CLI to complete each task individually

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

PROGRESS_FILE="deployment_progress.txt"
TASKS_FILE="BEV_DEPLOYMENT_TASKS.md"

# Initialize progress file if doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
    echo "PHASE=A" > "$PROGRESS_FILE"
    echo "TASK=1" >> "$PROGRESS_FILE"
    echo "COMPLETED=" >> "$PROGRESS_FILE"
fi

source "$PROGRESS_FILE"

# Function to mark task complete
mark_complete() {
    local task=$1
    COMPLETED="${COMPLETED}${task},"
    echo "PHASE=$PHASE" > "$PROGRESS_FILE"
    echo "TASK=$TASK" >> "$PROGRESS_FILE"
    echo "COMPLETED=$COMPLETED" >> "$PROGRESS_FILE"
    echo -e "${GREEN}✅ Task $task completed${NC}"
}

# Function to check if task is done
is_complete() {
    local task=$1
    if [[ "$COMPLETED" == *"$task,"* ]]; then
        return 0
    else
        return 1
    fi
}

# Display current status
show_status() {
    echo -e "${PURPLE}BEV DEPLOYMENT PROGRESS${NC}"
    echo "========================"
    echo -e "Current Phase: ${YELLOW}$PHASE${NC}"
    echo -e "Current Task: ${YELLOW}$TASK${NC}"
    echo ""
    echo "Completed Tasks:"
    IFS=',' read -ra TASKS <<< "$COMPLETED"
    for task in "${TASKS[@]}"; do
        if [ ! -z "$task" ]; then
            echo -e "  ${GREEN}✓${NC} $task"
        fi
    done
    echo ""
}

# Get next task
get_next_task() {
    case "$PHASE-$TASK" in
        "A-1")
            if ! is_complete "A1"; then
                echo -e "${CYAN}Task A1: Load current state${NC}"
                echo "Run: claude-code '/sc:load'"
                echo "Then run: $0 complete A1"
            else
                TASK=2
                get_next_task
            fi
            ;;
        "A-2")
            if ! is_complete "A2"; then
                echo -e "${CYAN}Task A2: Index project${NC}"
                echo "Run: claude-code '/sc:index'"
                echo "Then run: $0 complete A2"
            else
                TASK=3
                get_next_task
            fi
            ;;
        "A-3")
            if ! is_complete "A3"; then
                echo -e "${CYAN}Task A3: Explain current state${NC}"
                echo "Run: claude-code '/sc:explain \"current state\"'"
                echo "Then run: $0 complete A3"
            else
                TASK=4
                get_next_task
            fi
            ;;
        "A-4")
            if ! is_complete "A4"; then
                echo -e "${CYAN}Task A4: Save Phase A${NC}"
                echo "Run: claude-code '/sc:save \"phase-a\"'"
                echo "Then run: $0 complete A4"
            else
                echo -e "${GREEN}Phase A Complete! Moving to Phase B${NC}"
                PHASE=B
                TASK=1
                get_next_task
            fi
            ;;
        "B-1")
            if ! is_complete "B1"; then
                echo -e "${CYAN}Task B1: Analyze codebase${NC}"
                echo "Run: claude-code '/sc:analyze'"
                echo "Then run: $0 complete B1"
            else
                TASK=2
                get_next_task
            fi
            ;;
        # ... Continue for all tasks
        *)
            echo -e "${GREEN}All tasks complete!${NC}"
            ;;
    esac
}

# Main logic
case "$1" in
    "complete")
        if [ -z "$2" ]; then
            echo "Usage: $0 complete <task_id>"
            exit 1
        fi
        mark_complete "$2"
        show_status
        get_next_task
        ;;
    "status")
        show_status
        get_next_task
        ;;
    "reset")
        echo "PHASE=A" > "$PROGRESS_FILE"
        echo "TASK=1" >> "$PROGRESS_FILE"
        echo "COMPLETED=" >> "$PROGRESS_FILE"
        echo -e "${YELLOW}Progress reset${NC}"
        ;;
    "skip")
        # For emergency skipping (use carefully!)
        echo -e "${RED}WARNING: Skipping current task${NC}"
        TASK=$((TASK + 1))
        echo "PHASE=$PHASE" > "$PROGRESS_FILE"
        echo "TASK=$TASK" >> "$PROGRESS_FILE"
        echo "COMPLETED=$COMPLETED" >> "$PROGRESS_FILE"
        get_next_task
        ;;
    *)
        show_status
        get_next_task
        echo ""
        echo "Commands:"
        echo "  $0 status      - Show current progress"
        echo "  $0 complete X  - Mark task X as complete"
        echo "  $0 reset       - Start over"
        echo "  $0 skip        - Skip current task (emergency)"
        ;;
esac
