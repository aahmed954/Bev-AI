#!/bin/bash

echo "🔍 BEV Completeness Check"
echo "========================="

MISSING=()

# Check critical files
FILES=(
    "src/agents/agent_protocol.py"
    "docker/message-queue/docker-compose-messaging.yml"
    "src/infrastructure/message_queue_manager.py"
    "dags/data_lake_medallion_dag.py"
    "deploy/multi_node_orchestrator.py"
    ".env.example"
    "deploy_complete_system.sh"
)

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING+=("$file")
        echo "❌ Missing: $file"
    else
        echo "✅ Found: $file"
    fi
done

if [ ${#MISSING[@]} -eq 0 ]; then
    echo -e "\n✨ ALL CRITICAL FILES PRESENT! ✨"
    echo "Ready to run: ./deploy_complete_system.sh"
else
    echo -e "\n⚠️  Missing ${#MISSING[@]} files"
    echo "Create the missing files before deployment"
fi
