#!/bin/bash
# Run this after fixes to prepare for deployment to nodes

echo "Preparing project for deployment to THANOS and ORACLE1..."

# Create archive excluding unnecessary files
tar -czf bev-deployment.tar.gz \
    --exclude='.git' \
    --exclude='node_modules' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='backups' \
    --exclude='logs' \
    .

echo "Deployment package created: bev-deployment.tar.gz"
echo ""
echo "To deploy to nodes:"
echo "  1. scp bev-deployment.tar.gz starlord@thanos:/opt/"
echo "  2. scp bev-deployment.tar.gz starlord@oracle1:/opt/"
echo "  3. SSH to each node and extract: tar -xzf /opt/bev-deployment.tar.gz -C /opt/bev/"
