#!/bin/bash
# OSINT Collection Script for Oracle1

cd /opt/bev-osint

# Activate Python environment
source venv/bin/activate

# Run OSINT collection
python3 -c "
import asyncio
from src.agents.research_coordinator import ResearchCoordinator

async def collect():
    coordinator = ResearchCoordinator()
    await coordinator.collect_osint_batch([
        'cybersecurity threats',
        'cryptocurrency markets',
        'dark web activity'
    ])

asyncio.run(collect())
" >> /var/log/bev/osint-collection.log 2>&1

# Index results in Elasticsearch
curl -X POST "localhost:9200/osint-$(date +%Y%m%d)/_doc" \
  -H "Content-Type: application/json" \
  -d "@/tmp/osint_results.json"
