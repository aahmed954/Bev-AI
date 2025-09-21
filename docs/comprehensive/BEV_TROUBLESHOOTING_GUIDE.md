# BEV OSINT Framework - Troubleshooting & Incident Response Guide

## Overview

This comprehensive guide provides systematic troubleshooting procedures, incident response workflows, and problem resolution strategies for the BEV OSINT Framework. It covers common issues, diagnostic procedures, and recovery methods for maintaining operational effectiveness.

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Service-Specific Troubleshooting](#service-specific-troubleshooting)
3. [Network and Connectivity Issues](#network-and-connectivity-issues)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Security Incident Response](#security-incident-response)
6. [Data Recovery Procedures](#data-recovery-procedures)
7. [Common Error Scenarios](#common-error-scenarios)
8. [Monitoring and Alerting](#monitoring-and-alerting)
9. [Emergency Procedures](#emergency-procedures)

---

## Quick Diagnostic Commands

### System Health Check Script
```bash
#!/bin/bash
# BEV System Health Check
# Usage: ./health_check.sh [--verbose] [--fix]

echo "🏥 BEV OSINT Framework Health Check"
echo "=================================="
echo "Timestamp: $(date)"
echo ""

VERBOSE=false
AUTO_FIX=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --fix|-f)
            AUTO_FIX=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_docker_status() {
    echo "🐳 Docker Services Status"
    echo "------------------------"
    
    if ! docker --version >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker not installed or not running${NC}"
        return 1
    fi
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker daemon not running${NC}"
        if [[ $AUTO_FIX == true ]]; then
            echo "🔧 Attempting to start Docker..."
            sudo systemctl start docker
        fi
        return 1
    fi
    
    echo -e "${GREEN}✅ Docker daemon running${NC}"
    
    # Check container status
    containers=$(docker ps --format "table {{.Names}}\t{{.Status}}")
    echo "$containers"
    
    # Count running vs total containers
    running_count=$(docker ps --format "{{.Names}}" | wc -l)
    total_count=$(docker ps -a --format "{{.Names}}" | wc -l)
    
    echo "Running: $running_count/$total_count containers"
    
    if [[ $running_count -lt $total_count ]]; then
        echo -e "${YELLOW}⚠️ Some containers are not running${NC}"
        
        if [[ $AUTO_FIX == true ]]; then
            echo "🔧 Attempting to start stopped containers..."
            docker-compose up -d
        fi
    fi
}

check_network_connectivity() {
    echo "🌐 Network Connectivity"
    echo "----------------------"
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Internet connectivity${NC}"
    else
        echo -e "${RED}❌ No internet connectivity${NC}"
        return 1
    fi
    
    # Check Tor connectivity
    if command -v tor >/dev/null 2>&1; then
        tor_result=$(timeout 10 curl -s -x socks5://127.0.0.1:9050 \
            http://check.torproject.org/ 2>/dev/null | grep -o "Congratulations" || echo "Failed")
        
        if [[ $tor_result == "Congratulations" ]]; then
            echo -e "${GREEN}✅ Tor proxy working${NC}"
        else
            echo -e "${RED}❌ Tor proxy not working${NC}"
            
            if [[ $AUTO_FIX == true ]]; then
                echo "🔧 Restarting Tor service..."
                docker-compose restart bev_tor
            fi
        fi
    else
        echo -e "${YELLOW}⚠️ Tor not installed${NC}"
    fi
    
    # Check key service endpoints
    services=("bev_postgres:5432" "bev_redis:6379" "bev_neo4j:7687")
    
    for service in "${services[@]}"; do
        host=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            echo -e "${GREEN}✅ $service reachable${NC}"
        else
            echo -e "${RED}❌ $service not reachable${NC}"
        fi
    done
}

check_database_status() {
    echo "🗄️ Database Status"
    echo "-----------------"
    
    # PostgreSQL
    if docker exec bev_postgres pg_isready -U bev >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL responding${NC}"
        
        # Check database connections
        connections=$(docker exec bev_postgres psql -U bev -d osint -t -c \
            "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ')
        echo "   Active connections: $connections"
        
        if [[ $VERBOSE == true ]]; then
            # Check database sizes
            docker exec bev_postgres psql -U bev -d osint -c \
                "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database;"
        fi
    else
        echo -e "${RED}❌ PostgreSQL not responding${NC}"
        
        if [[ $AUTO_FIX == true ]]; then
            echo "🔧 Restarting PostgreSQL..."
            docker-compose restart bev_postgres
        fi
    fi
    
    # Redis
    if docker exec bev_redis redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis responding${NC}"
        
        # Check Redis memory usage
        memory_info=$(docker exec bev_redis redis-cli info memory | grep used_memory_human)
        echo "   $memory_info"
    else
        echo -e "${RED}❌ Redis not responding${NC}"
        
        if [[ $AUTO_FIX == true ]]; then
            echo "🔧 Restarting Redis..."
            docker-compose restart bev_redis
        fi
    fi
    
    # Neo4j
    if docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024 \
        "RETURN 1" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Neo4j responding${NC}"
        
        if [[ $VERBOSE == true ]]; then
            # Check database statistics
            node_count=$(docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024 \
                "MATCH (n) RETURN count(n)" 2>/dev/null | tail -1)
            echo "   Total nodes: $node_count"
        fi
    else
        echo -e "${RED}❌ Neo4j not responding${NC}"
        
        if [[ $AUTO_FIX == true ]]; then
            echo "🔧 Restarting Neo4j..."
            docker-compose restart bev_neo4j
        fi
    fi
}

check_disk_space() {
    echo "💾 Disk Space"
    echo "-------------"
    
    # Check overall disk usage
    df -h / | tail -1 | while read filesystem size used avail percent mountpoint; do
        usage_num=$(echo $percent | tr -d '%')
        
        if [[ $usage_num -lt 80 ]]; then
            echo -e "${GREEN}✅ Root filesystem: $percent used${NC}"
        elif [[ $usage_num -lt 90 ]]; then
            echo -e "${YELLOW}⚠️ Root filesystem: $percent used${NC}"
        else
            echo -e "${RED}❌ Root filesystem: $percent used (CRITICAL)${NC}"
        fi
    done
    
    # Check Docker volumes
    docker system df --format "table {{.Type}}\t{{.Size}}\t{{.Reclaimable}}"
    
    # Check log files
    log_size=$(du -sh logs/ 2>/dev/null | cut -f1 || echo "N/A")
    echo "Log files size: $log_size"
    
    # Clean up if requested and needed
    if [[ $AUTO_FIX == true ]]; then
        echo "🔧 Cleaning up Docker system..."
        docker system prune -f
    fi
}

check_resource_usage() {
    echo "📊 Resource Usage"
    echo "-----------------"
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    echo "CPU Usage: ${cpu_usage}%"
    
    # Memory usage
    memory_info=$(free -h | awk 'NR==2{printf "Memory Usage: %s/%s (%.2f%%)", $3,$2,$3*100/$2 }')
    echo "$memory_info"
    
    # Top processes by CPU
    if [[ $VERBOSE == true ]]; then
        echo "Top CPU processes:"
        ps aux --sort=-%cpu | head -6
    fi
}

# Run all checks
echo "Starting comprehensive health check..."
echo ""

check_docker_status
echo ""

check_network_connectivity
echo ""

check_database_status
echo ""

check_disk_space
echo ""

check_resource_usage
echo ""

echo "Health check completed at $(date)"
```

### Quick Status Commands
```bash
# Quick service status check
alias bev-status='docker-compose ps && docker stats --no-stream'

# Check all service health endpoints
alias bev-health='curl -s http://localhost/health | jq .'

# View recent logs from all services
alias bev-logs='docker-compose logs --tail=50 -f'

# Check Tor circuit status
alias tor-status='curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/'

# Database quick check
alias db-check='docker exec bev_postgres pg_isready -U bev'
```

---

## Service-Specific Troubleshooting

### PostgreSQL Issues

#### Common PostgreSQL Problems
```bash
# Problem: Connection refused
# Symptoms: "psql: error: connection to server at 'bev_postgres' failed"

diagnose_postgres_connection() {
    echo "🔍 Diagnosing PostgreSQL connection issues..."
    
    # Check if container is running
    if ! docker ps | grep -q bev_postgres; then
        echo "❌ PostgreSQL container not running"
        echo "🔧 Solution: docker-compose up -d bev_postgres"
        return 1
    fi
    
    # Check container logs
    echo "📋 Recent PostgreSQL logs:"
    docker logs bev_postgres --tail=20
    
    # Check port binding
    port_check=$(docker port bev_postgres 5432 2>/dev/null)
    if [[ -z "$port_check" ]]; then
        echo "❌ Port 5432 not exposed"
        echo "🔧 Check docker-compose.yml port configuration"
    else
        echo "✅ Port binding: $port_check"
    fi
    
    # Check database process inside container
    db_process=$(docker exec bev_postgres ps aux | grep postgres)
    if [[ -z "$db_process" ]]; then
        echo "❌ PostgreSQL process not running inside container"
        echo "🔧 Solution: Restart container with proper initialization"
    else
        echo "✅ PostgreSQL process running"
    fi
    
    # Test connection
    if docker exec bev_postgres pg_isready -U bev; then
        echo "✅ PostgreSQL accepting connections"
    else
        echo "❌ PostgreSQL not accepting connections"
        echo "🔧 Check authentication configuration"
    fi
}

# Problem: Out of disk space
fix_postgres_disk_space() {
    echo "🔧 Cleaning PostgreSQL disk space..."
    
    # Check database sizes
    docker exec bev_postgres psql -U bev -d osint -c \
        "SELECT datname, pg_size_pretty(pg_database_size(datname)) FROM pg_database;"
    
    # Clean old logs
    docker exec bev_postgres find /var/lib/postgresql/data/log -name "*.log" -mtime +7 -delete
    
    # Vacuum databases
    docker exec bev_postgres psql -U bev -d osint -c "VACUUM FULL;"
    
    # Restart service
    docker-compose restart bev_postgres
}

# Problem: Too many connections
fix_postgres_connections() {
    echo "🔧 Resolving PostgreSQL connection limit..."
    
    # Show current connections
    docker exec bev_postgres psql -U bev -d osint -c \
        "SELECT count(*) as active_connections FROM pg_stat_activity;"
    
    # Kill idle connections
    docker exec bev_postgres psql -U bev -d osint -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
         WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"
    
    # Increase max connections (temporary)
    docker exec bev_postgres psql -U bev -d osint -c \
        "ALTER SYSTEM SET max_connections = 200;"
    
    # Restart to apply changes
    docker-compose restart bev_postgres
}
```

### Redis Issues

#### Redis Troubleshooting
```bash
# Problem: Redis memory issues
diagnose_redis_memory() {
    echo "🔍 Diagnosing Redis memory usage..."
    
    # Check Redis info
    docker exec bev_redis redis-cli info memory
    
    # Check memory configuration
    max_memory=$(docker exec bev_redis redis-cli config get maxmemory)
    echo "Max memory setting: $max_memory"
    
    # Check eviction policy
    eviction_policy=$(docker exec bev_redis redis-cli config get maxmemory-policy)
    echo "Eviction policy: $eviction_policy"
    
    # Check key statistics
    docker exec bev_redis redis-cli info keyspace
}

fix_redis_memory() {
    echo "🔧 Fixing Redis memory issues..."
    
    # Set memory limit (2GB)
    docker exec bev_redis redis-cli config set maxmemory 2gb
    
    # Set eviction policy
    docker exec bev_redis redis-cli config set maxmemory-policy allkeys-lru
    
    # Force garbage collection
    docker exec bev_redis redis-cli debug restart
    
    # Restart service
    docker-compose restart bev_redis
}

# Problem: Redis connection timeouts
fix_redis_timeouts() {
    echo "🔧 Fixing Redis timeout issues..."
    
    # Increase timeout settings
    docker exec bev_redis redis-cli config set timeout 300
    
    # Check for slow queries
    docker exec bev_redis redis-cli slowlog get 10
    
    # Restart service
    docker-compose restart bev_redis
}
```

### Neo4j Issues

#### Neo4j Troubleshooting
```bash
# Problem: Neo4j authentication failures
fix_neo4j_auth() {
    echo "🔧 Fixing Neo4j authentication..."
    
    # Reset Neo4j password
    docker exec bev_neo4j neo4j-admin set-initial-password BevGraph2024
    
    # Restart service
    docker-compose restart bev_neo4j
    
    # Wait for startup
    sleep 30
    
    # Test connection
    docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024 "RETURN 1"
}

# Problem: Neo4j performance issues
optimize_neo4j_performance() {
    echo "🔧 Optimizing Neo4j performance..."
    
    # Check heap size
    docker exec bev_neo4j cat /var/lib/neo4j/conf/neo4j.conf | grep heap
    
    # Restart with increased memory
    docker-compose stop bev_neo4j
    
    # Update heap size in environment
    export NEO4J_dbms_memory_heap_initial_size=2G
    export NEO4J_dbms_memory_heap_max_size=4G
    
    docker-compose up -d bev_neo4j
    
    # Wait for startup
    sleep 60
    
    # Verify configuration
    docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024 \
        "CALL dbms.listConfig() YIELD name, value WHERE name CONTAINS 'heap' RETURN name, value"
}
```

### Tor Proxy Issues

#### Tor Connectivity Troubleshooting
```bash
# Problem: Tor circuits not working
diagnose_tor_issues() {
    echo "🔍 Diagnosing Tor connectivity..."
    
    # Check Tor service status
    if ! docker ps | grep -q bev_tor; then
        echo "❌ Tor container not running"
        docker-compose up -d bev_tor
        sleep 10
    fi
    
    # Check Tor logs
    echo "📋 Recent Tor logs:"
    docker logs bev_tor --tail=20
    
    # Test SOCKS5 proxy
    if timeout 10 curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/ | grep -q "Congratulations"; then
        echo "✅ Tor proxy working"
    else
        echo "❌ Tor proxy not working"
        return 1
    fi
    
    # Check circuit status
    circuit_count=$(docker exec bev_tor tor --hash-password "" 2>/dev/null | wc -l)
    echo "Circuit status check..."
}

fix_tor_circuits() {
    echo "🔧 Fixing Tor circuits..."
    
    # Restart Tor service
    docker-compose restart bev_tor
    
    # Wait for circuit establishment
    sleep 30
    
    # Force new circuit
    if command -v tor >/dev/null 2>&1; then
        echo "AUTHENTICATE" | nc 127.0.0.1 9051
        echo "SIGNAL NEWNYM" | nc 127.0.0.1 9051
    fi
    
    # Test connectivity
    timeout 15 curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/
}
```

---

## Network and Connectivity Issues

### Network Diagnostic Procedures

#### Container Network Issues
```bash
# Diagnose container networking
diagnose_container_network() {
    echo "🔍 Diagnosing container network issues..."
    
    # Check Docker networks
    echo "Docker networks:"
    docker network ls
    
    # Inspect BEV network
    echo "BEV network details:"
    docker network inspect bev_network
    
    # Check container connectivity
    echo "Testing inter-container connectivity..."
    
    # Test database connections
    docker exec bev_postgres nc -z bev_redis 6379 && echo "✅ PostgreSQL → Redis" || echo "❌ PostgreSQL → Redis"
    docker exec bev_postgres nc -z bev_neo4j 7687 && echo "✅ PostgreSQL → Neo4j" || echo "❌ PostgreSQL → Neo4j"
    
    # Test from application containers
    if docker ps | grep -q bev_intelowl; then
        docker exec bev_intelowl nc -z bev_postgres 5432 && echo "✅ IntelOwl → PostgreSQL" || echo "❌ IntelOwl → PostgreSQL"
        docker exec bev_intelowl nc -z bev_redis 6379 && echo "✅ IntelOwl → Redis" || echo "❌ IntelOwl → Redis"
    fi
}

# Fix network connectivity
fix_container_network() {
    echo "🔧 Fixing container network issues..."
    
    # Restart networking
    docker-compose down
    docker network prune -f
    docker-compose up -d
    
    # Wait for services to start
    sleep 60
    
    # Verify connectivity
    diagnose_container_network
}
```

#### External Connectivity Issues
```bash
# Test external connectivity
test_external_connectivity() {
    echo "🔍 Testing external connectivity..."
    
    # Test direct internet access
    if ping -c 3 8.8.8.8 >/dev/null 2>&1; then
        echo "✅ Direct internet access"
    else
        echo "❌ No direct internet access"
        return 1
    fi
    
    # Test DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        echo "✅ DNS resolution working"
    else
        echo "❌ DNS resolution failed"
    fi
    
    # Test HTTPS connectivity
    if curl -s https://httpbin.org/ip >/dev/null 2>&1; then
        echo "✅ HTTPS connectivity"
    else
        echo "❌ HTTPS connectivity failed"
    fi
    
    # Test through Tor
    if timeout 15 curl -s -x socks5://127.0.0.1:9050 https://httpbin.org/ip >/dev/null 2>&1; then
        echo "✅ Tor connectivity"
    else
        echo "❌ Tor connectivity failed"
    fi
}
```

---

## Performance Troubleshooting

### Performance Monitoring Script
```bash
#!/bin/bash
# Performance monitoring and optimization

monitor_performance() {
    echo "📊 BEV Performance Monitor"
    echo "========================="
    echo "Timestamp: $(date)"
    echo ""
    
    # System resources
    echo "💻 System Resources:"
    echo "-------------------"
    
    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    echo "CPU Usage: ${cpu_usage}%"
    
    # Memory usage
    memory_usage=$(free | awk 'NR==2{printf "%.2f", $3*100/$2 }')
    echo "Memory Usage: ${memory_usage}%"
    
    # Disk I/O
    disk_io=$(iostat -d 1 2 | tail -1 | awk '{print $3, $4}')
    echo "Disk I/O (read/write): $disk_io"
    
    # Network usage
    network_rx=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
    network_tx=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
    echo "Network (RX/TX): $network_rx / $network_tx bytes"
    
    echo ""
    
    # Container resources
    echo "🐳 Container Resources:"
    echo "----------------------"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    
    echo ""
    
    # Database performance
    echo "🗄️ Database Performance:"
    echo "-----------------------"
    
    # PostgreSQL connections
    pg_connections=$(docker exec bev_postgres psql -U bev -d osint -t -c \
        "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ')
    echo "PostgreSQL connections: $pg_connections"
    
    # PostgreSQL slow queries
    slow_queries=$(docker exec bev_postgres psql -U bev -d osint -t -c \
        "SELECT count(*) FROM pg_stat_statements WHERE mean_exec_time > 1000;" 2>/dev/null | tr -d ' ')
    echo "PostgreSQL slow queries: $slow_queries"
    
    # Redis memory usage
    redis_memory=$(docker exec bev_redis redis-cli info memory | grep used_memory_human | cut -d: -f2)
    echo "Redis memory usage: $redis_memory"
    
    # Neo4j transaction count
    neo4j_tx=$(docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024 \
        "CALL dbms.listTransactions()" 2>/dev/null | wc -l)
    echo "Neo4j active transactions: $neo4j_tx"
}

# Performance optimization
optimize_performance() {
    echo "🚀 Optimizing BEV Performance"
    echo "=============================="
    
    # Clean up Docker system
    echo "🧹 Cleaning Docker system..."
    docker system prune -f
    docker volume prune -f
    
    # Optimize PostgreSQL
    echo "🔧 Optimizing PostgreSQL..."
    docker exec bev_postgres psql -U bev -d osint -c "VACUUM ANALYZE;"
    
    # Clean Redis memory
    echo "🧹 Cleaning Redis memory..."
    docker exec bev_redis redis-cli flushdb
    
    # Restart services for fresh start
    echo "🔄 Restarting services..."
    docker-compose restart bev_postgres bev_redis bev_neo4j
    
    # Wait for services to stabilize
    sleep 60
    
    echo "✅ Performance optimization complete"
}
```

### Database Performance Tuning
```sql
-- PostgreSQL performance tuning queries

-- Check slow queries
SELECT query, calls, total_exec_time, mean_exec_time, rows
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;

-- Vacuum and analyze all tables
VACUUM ANALYZE;

-- Update table statistics
ANALYZE;
```

---

## Security Incident Response

### Security Incident Classification
```yaml
Incident_Severity_Levels:
  CRITICAL:
    description: "System compromise, data breach, or legal issues"
    response_time: "< 15 minutes"
    escalation: "Immediate notification to security team"
    actions: ["Isolate system", "Preserve evidence", "Legal consultation"]
    
  HIGH:
    description: "Unauthorized access attempts, service disruption"
    response_time: "< 1 hour"
    escalation: "Notify operations team"
    actions: ["Investigate", "Contain threat", "Update security measures"]
    
  MEDIUM:
    description: "Performance degradation, configuration issues"
    response_time: "< 4 hours"
    escalation: "Log and assign to team member"
    actions: ["Monitor", "Plan remediation", "Update procedures"]
    
  LOW:
    description: "Minor anomalies, information gathering attempts"
    response_time: "< 24 hours"
    escalation: "Document for trend analysis"
    actions: ["Document", "Monitor patterns", "Scheduled review"]
```

### Incident Response Playbooks

#### Data Breach Response
```bash
#!/bin/bash
# Data breach incident response playbook

data_breach_response() {
    echo "🚨 DATA BREACH INCIDENT RESPONSE"
    echo "================================"
    echo "Initiated at: $(date)"
    
    # 1. Immediate containment
    echo "🔒 STEP 1: Immediate Containment"
    echo "-------------------------------"
    
    # Stop all services
    echo "Stopping all BEV services..."
    docker-compose down
    
    # Block external network access
    echo "Implementing network isolation..."
    iptables -A INPUT -j DROP
    iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT  # Keep SSH for admin
    iptables -A OUTPUT -j DROP
    
    # 2. Evidence preservation
    echo "📸 STEP 2: Evidence Preservation"
    echo "--------------------------------"
    
    timestamp=$(date +"%Y%m%d_%H%M%S")
    evidence_dir="/backup/incident_evidence_$timestamp"
    mkdir -p "$evidence_dir"
    
    # Preserve system state
    echo "Capturing system state..."
    ps aux > "$evidence_dir/running_processes.txt"
    netstat -tuln > "$evidence_dir/network_connections.txt"
    lsof > "$evidence_dir/open_files.txt"
    
    # Preserve logs
    echo "Preserving logs..."
    cp -r logs/ "$evidence_dir/application_logs/"
    journalctl --since="24 hours ago" > "$evidence_dir/system_logs.txt"
    
    # Database snapshots
    echo "Creating database snapshots..."
    if docker ps | grep -q bev_postgres; then
        docker exec bev_postgres pg_dump osint > "$evidence_dir/postgres_dump.sql"
    fi
    
    # Memory dump (if available)
    if command -v memdump >/dev/null 2>&1; then
        echo "Creating memory dump..."
        memdump > "$evidence_dir/memory_dump.bin"
    fi
    
    # 3. Impact assessment
    echo "📊 STEP 3: Impact Assessment"
    echo "----------------------------"
    
    # Check for data exfiltration
    echo "Checking for potential data exfiltration..."
    
    # Analyze network logs for unusual traffic
    grep -E "(POST|PUT)" logs/access.log | tail -100 > "$evidence_dir/recent_uploads.log"
    
    # Check database access logs
    grep -E "(SELECT|COPY|EXPORT)" logs/postgres.log | tail -100 > "$evidence_dir/db_access.log"
    
    # 4. Legal notification checklist
    echo "⚖️ STEP 4: Legal Notification"
    echo "-----------------------------"
    echo "LEGAL NOTIFICATION CHECKLIST:"
    echo "- [ ] Notify legal team immediately"
    echo "- [ ] Document incident timeline"
    echo "- [ ] Preserve attorney-client privilege"
    echo "- [ ] Consider law enforcement notification"
    echo "- [ ] Prepare breach notification letters"
    echo "- [ ] Review insurance coverage"
    
    # 5. Recovery planning
    echo "🔧 STEP 5: Recovery Planning"
    echo "----------------------------"
    echo "RECOVERY CHECKLIST:"
    echo "- [ ] Identify attack vector"
    echo "- [ ] Patch vulnerabilities"
    echo "- [ ] Reset all credentials"
    echo "- [ ] Restore from clean backups"
    echo "- [ ] Implement additional security measures"
    echo "- [ ] Plan staged service restoration"
    
    echo ""
    echo "🎯 Incident response initiated successfully"
    echo "Evidence preserved in: $evidence_dir"
    echo "Next steps: Follow legal notification and recovery checklists"
}
```

#### Unauthorized Access Response
```bash
#!/bin/bash
# Unauthorized access incident response

unauthorized_access_response() {
    echo "🛡️ UNAUTHORIZED ACCESS RESPONSE"
    echo "==============================="
    echo "Initiated at: $(date)"
    
    # 1. Assess the scope
    echo "🔍 STEP 1: Scope Assessment"
    echo "---------------------------"
    
    # Check active sessions
    echo "Checking active user sessions..."
    who -a > /tmp/active_sessions.txt
    cat /tmp/active_sessions.txt
    
    # Check recent login attempts
    echo "Recent login attempts:"
    last -10
    
    # Check failed authentication attempts
    echo "Failed authentication attempts:"
    grep "authentication failure" /var/log/auth.log | tail -20
    
    # 2. Immediate response actions
    echo "🚨 STEP 2: Immediate Response"
    echo "----------------------------"
    
    # Kill suspicious sessions
    echo "Reviewing active sessions for termination..."
    # Manual review required - don't automatically kill sessions
    
    # Reset passwords
    echo "Generating new passwords..."
    ./generate_secrets.sh
    
    # Rotate API keys
    echo "Rotating API keys..."
    # Implementation depends on specific API key system
    
    # 3. System hardening
    echo "🔧 STEP 3: System Hardening"
    echo "---------------------------"
    
    # Update firewall rules
    echo "Updating firewall rules..."
    
    # Block suspicious IPs (requires manual input)
    echo "Review and block suspicious IP addresses:"
    echo "Recent connections:"
    netstat -an | grep :22 | grep ESTABLISHED
    
    # 4. Monitoring enhancement
    echo "📊 STEP 4: Enhanced Monitoring"
    echo "------------------------------"
    
    # Enable detailed logging
    echo "Enabling enhanced logging..."
    
    # Set up real-time monitoring
    echo "Setting up real-time monitoring..."
    
    # Configure alerts
    echo "Configuring security alerts..."
    
    echo ""
    echo "✅ Unauthorized access response actions completed"
    echo "Manual review required for session termination and IP blocking"
}
```

### Security Monitoring Scripts
```bash
#!/bin/bash
# Continuous security monitoring

security_monitor() {
    echo "🛡️ Security Monitoring Dashboard"
    echo "==============================="
    
    while true; do
        clear
        echo "Security Status - $(date)"
        echo "========================"
        
        # Failed login attempts
        failed_logins=$(grep "authentication failure" /var/log/auth.log | grep "$(date +%b\ %d)" | wc -l)
        echo "Failed logins today: $failed_logins"
        
        # Suspicious network activity
        suspicious_connections=$(netstat -an | grep -E ":(1337|4444|5555|6666|7777)" | wc -l)
        echo "Suspicious network connections: $suspicious_connections"
        
        # Docker container anomalies
        stopped_containers=$(docker ps -a | grep -v Up | wc -l)
        echo "Stopped containers: $stopped_containers"
        
        # Disk space monitoring
        disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
        echo "Disk usage: $disk_usage%"
        
        # Tor circuit health
        if timeout 5 curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/ | grep -q "Congratulations"; then
            echo "Tor status: ✅ Working"
        else
            echo "Tor status: ❌ Failed"
        fi
        
        # Alert conditions
        if [[ $failed_logins -gt 10 ]]; then
            echo "🚨 ALERT: High number of failed logins"
        fi
        
        if [[ $suspicious_connections -gt 0 ]]; then
            echo "🚨 ALERT: Suspicious network connections detected"
        fi
        
        if [[ $disk_usage -gt 90 ]]; then
            echo "🚨 ALERT: Disk space critically low"
        fi
        
        sleep 30
    done
}
```

---

## Data Recovery Procedures

### Backup Recovery Scripts
```bash
#!/bin/bash
# BEV data recovery procedures

restore_from_backup() {
    local backup_date="$1"
    local component="$2"
    
    if [[ -z "$backup_date" || -z "$component" ]]; then
        echo "Usage: restore_from_backup <YYYY-MM-DD> <component>"
        echo "Components: postgres, neo4j, redis, config, all"
        return 1
    fi
    
    echo "🔄 BEV Data Recovery"
    echo "==================="
    echo "Backup date: $backup_date"
    echo "Component: $component"
    echo ""
    
    backup_dir="/backup/$backup_date"
    
    if [[ ! -d "$backup_dir" ]]; then
        echo "❌ Backup directory not found: $backup_dir"
        return 1
    fi
    
    case $component in
        postgres)
            restore_postgres_backup "$backup_dir"
            ;;
        neo4j)
            restore_neo4j_backup "$backup_dir"
            ;;
        redis)
            restore_redis_backup "$backup_dir"
            ;;
        config)
            restore_config_backup "$backup_dir"
            ;;
        all)
            restore_postgres_backup "$backup_dir"
            restore_neo4j_backup "$backup_dir"
            restore_redis_backup "$backup_dir"
            restore_config_backup "$backup_dir"
            ;;
        *)
            echo "❌ Unknown component: $component"
            return 1
            ;;
    esac
}

restore_postgres_backup() {
    local backup_dir="$1"
    
    echo "🗄️ Restoring PostgreSQL backup..."
    
    # Stop PostgreSQL service
    docker-compose stop bev_postgres
    
    # Remove old data
    docker volume rm bev_postgres_data 2>/dev/null || true
    
    # Start PostgreSQL
    docker-compose up -d bev_postgres
    
    # Wait for PostgreSQL to be ready
    echo "Waiting for PostgreSQL to start..."
    sleep 30
    
    # Restore databases
    for db_file in "$backup_dir"/postgres_*.sql; do
        if [[ -f "$db_file" ]]; then
            db_name=$(basename "$db_file" .sql | sed 's/postgres_//')
            echo "Restoring database: $db_name"
            
            # Create database if it doesn't exist
            docker exec bev_postgres createdb -U bev "$db_name" 2>/dev/null || true
            
            # Restore data
            docker exec -i bev_postgres psql -U bev -d "$db_name" < "$db_file"
        fi
    done
    
    echo "✅ PostgreSQL restoration complete"
}

restore_neo4j_backup() {
    local backup_dir="$1"
    
    echo "📊 Restoring Neo4j backup..."
    
    # Stop Neo4j service
    docker-compose stop bev_neo4j
    
    # Remove old data
    docker volume rm bev_neo4j_data 2>/dev/null || true
    
    # Restore backup
    if [[ -f "$backup_dir/neo4j_backup.dump" ]]; then
        # Start Neo4j
        docker-compose up -d bev_neo4j
        
        # Wait for Neo4j to start
        echo "Waiting for Neo4j to start..."
        sleep 60
        
        # Load backup
        docker exec bev_neo4j neo4j-admin load --from="/backup/neo4j_backup.dump" --database=neo4j --force
        
        # Restart Neo4j
        docker-compose restart bev_neo4j
        
        echo "✅ Neo4j restoration complete"
    else
        echo "❌ Neo4j backup file not found"
    fi
}

restore_redis_backup() {
    local backup_dir="$1"
    
    echo "🔄 Restoring Redis backup..."
    
    # Stop Redis service
    docker-compose stop bev_redis
    
    # Restore RDB file if it exists
    if [[ -f "$backup_dir/redis_dump.rdb" ]]; then
        # Copy backup to Redis data directory
        docker run --rm -v bev_redis_data:/data -v "$backup_dir":/backup \
            redis:alpine cp /backup/redis_dump.rdb /data/dump.rdb
        
        echo "✅ Redis backup file restored"
    else
        echo "⚠️ Redis backup file not found, starting with empty database"
    fi
    
    # Start Redis service
    docker-compose up -d bev_redis
    
    echo "✅ Redis restoration complete"
}

restore_config_backup() {
    local backup_dir="$1"
    
    echo "⚙️ Restoring configuration backup..."
    
    # Restore environment file
    if [[ -f "$backup_dir/.env" ]]; then
        cp "$backup_dir/.env" .env
        echo "✅ Environment configuration restored"
    fi
    
    # Restore Docker Compose configuration
    if [[ -f "$backup_dir/docker-compose.complete.yml" ]]; then
        cp "$backup_dir/docker-compose.complete.yml" docker-compose.complete.yml
        echo "✅ Docker Compose configuration restored"
    fi
    
    # Restore certificates
    if [[ -d "$backup_dir/certs" ]]; then
        cp -r "$backup_dir/certs" ./
        echo "✅ Certificates restored"
    fi
    
    echo "✅ Configuration restoration complete"
}

# Point-in-time recovery for PostgreSQL
postgres_point_in_time_recovery() {
    local recovery_time="$1"
    
    if [[ -z "$recovery_time" ]]; then
        echo "Usage: postgres_point_in_time_recovery 'YYYY-MM-DD HH:MM:SS'"
        return 1
    fi
    
    echo "⏰ PostgreSQL Point-in-Time Recovery"
    echo "Recovery time: $recovery_time"
    
    # Stop PostgreSQL
    docker-compose stop bev_postgres
    
    # Create recovery configuration
    cat > recovery.conf << EOF
restore_command = 'cp /backup/wal_archive/%f %p'
recovery_target_time = '$recovery_time'
recovery_target_action = 'promote'
EOF
    
    # Copy recovery config to data directory
    docker run --rm -v bev_postgres_data:/data -v $(pwd):/host \
        postgres:13 cp /host/recovery.conf /data/recovery.conf
    
    # Start PostgreSQL in recovery mode
    docker-compose up -d bev_postgres
    
    echo "✅ Point-in-time recovery initiated"
    echo "Monitor logs for recovery completion"
}
```

---

## Common Error Scenarios

### Error Code Reference

#### Database Connection Errors
```bash
# Error: "psql: error: connection to server failed"
# Solution:
check_postgres_connection() {
    echo "Checking PostgreSQL connection..."
    
    # Verify container is running
    if ! docker ps | grep -q bev_postgres; then
        echo "Starting PostgreSQL container..."
        docker-compose up -d bev_postgres
        sleep 30
    fi
    
    # Test connection
    if docker exec bev_postgres pg_isready -U bev; then
        echo "✅ PostgreSQL connection successful"
    else
        echo "❌ PostgreSQL connection failed"
        echo "Checking logs..."
        docker logs bev_postgres --tail=20
    fi
}

# Error: "Redis connection refused"
# Solution:
check_redis_connection() {
    echo "Checking Redis connection..."
    
    if ! docker ps | grep -q bev_redis; then
        echo "Starting Redis container..."
        docker-compose up -d bev_redis
        sleep 10
    fi
    
    if docker exec bev_redis redis-cli ping | grep -q PONG; then
        echo "✅ Redis connection successful"
    else
        echo "❌ Redis connection failed"
        docker logs bev_redis --tail=20
    fi
}
```

#### Memory and Resource Errors
```bash
# Error: "Out of memory"
# Solution:
handle_memory_errors() {
    echo "Handling memory errors..."
    
    # Check current memory usage
    free -h
    
    # Check for memory-intensive processes
    ps aux --sort=-%mem | head -10
    
    # Clear system caches
    sync && echo 3 > /proc/sys/vm/drop_caches
    
    # Restart memory-intensive services
    docker-compose restart bev_postgres bev_neo4j bev_elasticsearch
    
    # Check swap usage
    swapon -s
}

# Error: "Disk space full"
# Solution:
handle_disk_space_errors() {
    echo "Handling disk space errors..."
    
    # Check disk usage
    df -h
    
    # Clean up logs
    find logs/ -name "*.log" -mtime +7 -delete
    
    # Clean Docker system
    docker system prune -f
    docker volume prune -f
    
    # Clean up old backups
    find /backup/ -mtime +30 -delete
    
    # Vacuum databases
    docker exec bev_postgres psql -U bev -d osint -c "VACUUM FULL;"
}
```

#### Network and Connectivity Errors
```bash
# Error: "Tor circuit failed"
# Solution:
fix_tor_circuit_errors() {
    echo "Fixing Tor circuit errors..."
    
    # Restart Tor service
    docker-compose restart bev_tor
    
    # Wait for circuit establishment
    sleep 30
    
    # Test new circuit
    if timeout 15 curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/ | grep -q "Congratulations"; then
        echo "✅ Tor circuit working"
    else
        echo "❌ Tor circuit still failing"
        
        # Check Tor logs
        docker logs bev_tor --tail=20
        
        # Try manual circuit reset
        echo "AUTHENTICATE" | nc 127.0.0.1 9051
        echo "SIGNAL NEWNYM" | nc 127.0.0.1 9051
    fi
}

# Error: "DNS resolution failed"
# Solution:
fix_dns_errors() {
    echo "Fixing DNS resolution errors..."
    
    # Check current DNS configuration
    cat /etc/resolv.conf
    
    # Test DNS servers
    nslookup google.com 8.8.8.8
    nslookup google.com 1.1.1.1
    
    # Restart systemd-resolved
    sudo systemctl restart systemd-resolved
    
    # Flush DNS cache
    sudo systemctl flush-dns
}
```

---

## Monitoring and Alerting

### Automated Monitoring Setup
```bash
#!/bin/bash
# Automated monitoring and alerting system

setup_monitoring() {
    echo "📊 Setting up BEV monitoring system..."
    
    # Create monitoring directory
    mkdir -p monitoring/
    
    # Create alert configuration
    cat > monitoring/alert_config.yaml << EOF
alerts:
  high_cpu:
    threshold: 80
    duration: 300
    action: "email"
  
  high_memory:
    threshold: 85
    duration: 180
    action: "email"
  
  disk_space:
    threshold: 90
    duration: 60
    action: "email,sms"
  
  failed_logins:
    threshold: 10
    duration: 3600
    action: "email,slack"
  
  tor_failure:
    threshold: 1
    duration: 60
    action: "email"
  
  database_down:
    threshold: 1
    duration: 30
    action: "email,sms,slack"

notification_channels:
  email:
    smtp_server: "smtp.example.com"
    smtp_port: 587
    username: "alerts@example.com"
    password: "secure_password"
    recipients: ["admin@example.com"]
  
  slack:
    webhook_url: "https://hooks.slack.com/services/..."
    channel: "#bev-alerts"
  
  sms:
    provider: "twilio"
    account_sid: "your_account_sid"
    auth_token: "your_auth_token"
    from_number: "+1234567890"
    to_numbers: ["+1987654321"]
EOF
    
    # Create monitoring script
    cat > monitoring/monitor.py << 'EOF'
#!/usr/bin/env python3
import time
import subprocess
import json
import yaml
import smtplib
import requests
from email.mime.text import MimeText
from datetime import datetime

class BEVMonitor:
    def __init__(self, config_file='alert_config.yaml'):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.alerts = {}
    
    def check_system_metrics(self):
        metrics = {}
        
        # CPU usage
        cpu_cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | awk -F'%' '{print $1}'"
        metrics['cpu'] = float(subprocess.check_output(cpu_cmd, shell=True).decode().strip())
        
        # Memory usage
        mem_cmd = "free | awk 'NR==2{printf \"%.2f\", $3*100/$2 }'"
        metrics['memory'] = float(subprocess.check_output(mem_cmd, shell=True).decode().strip())
        
        # Disk usage
        disk_cmd = "df / | awk 'NR==2 {print $5}' | sed 's/%//'"
        metrics['disk'] = float(subprocess.check_output(disk_cmd, shell=True).decode().strip())
        
        return metrics
    
    def check_service_health(self):
        services = {}
        
        # Check Docker containers
        container_cmd = "docker ps --format '{{.Names}}:{{.Status}}'"
        containers = subprocess.check_output(container_cmd, shell=True).decode().strip().split('\n')
        
        for container in containers:
            if ':' in container:
                name, status = container.split(':', 1)
                services[name] = 'up' if 'Up' in status else 'down'
        
        # Check Tor connectivity
        try:
            tor_cmd = "timeout 10 curl -s -x socks5://127.0.0.1:9050 http://check.torproject.org/"
            tor_result = subprocess.check_output(tor_cmd, shell=True).decode()
            services['tor'] = 'up' if 'Congratulations' in tor_result else 'down'
        except:
            services['tor'] = 'down'
        
        return services
    
    def check_security_events(self):
        events = {}
        
        # Failed login attempts
        try:
            failed_logins_cmd = "grep 'authentication failure' /var/log/auth.log | grep '$(date +%b\\ %d)' | wc -l"
            events['failed_logins'] = int(subprocess.check_output(failed_logins_cmd, shell=True).decode().strip())
        except:
            events['failed_logins'] = 0
        
        return events
    
    def send_alert(self, alert_type, message, channels):
        timestamp = datetime.now().isoformat()
        
        for channel in channels:
            if channel == 'email':
                self.send_email_alert(alert_type, message, timestamp)
            elif channel == 'slack':
                self.send_slack_alert(alert_type, message, timestamp)
            elif channel == 'sms':
                self.send_sms_alert(alert_type, message, timestamp)
    
    def send_email_alert(self, alert_type, message, timestamp):
        config = self.config['notification_channels']['email']
        
        msg = MimeText(f"""
BEV OSINT Framework Alert

Alert Type: {alert_type}
Timestamp: {timestamp}
Message: {message}

Please check the system immediately.
        """)
        
        msg['Subject'] = f'BEV Alert: {alert_type}'
        msg['From'] = config['username']
        msg['To'] = ', '.join(config['recipients'])
        
        try:
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['username'], config['password'])
                server.send_message(msg)
            print(f"Email alert sent for {alert_type}")
        except Exception as e:
            print(f"Failed to send email alert: {e}")
    
    def run_monitoring_loop(self):
        print("Starting BEV monitoring...")
        
        while True:
            try:
                # Check system metrics
                metrics = self.check_system_metrics()
                
                # Check for CPU alerts
                if metrics['cpu'] > self.config['alerts']['high_cpu']['threshold']:
                    self.send_alert('HIGH_CPU', f"CPU usage: {metrics['cpu']}%", ['email'])
                
                # Check for memory alerts
                if metrics['memory'] > self.config['alerts']['high_memory']['threshold']:
                    self.send_alert('HIGH_MEMORY', f"Memory usage: {metrics['memory']}%", ['email'])
                
                # Check for disk space alerts
                if metrics['disk'] > self.config['alerts']['disk_space']['threshold']:
                    self.send_alert('DISK_SPACE', f"Disk usage: {metrics['disk']}%", ['email', 'sms'])
                
                # Check service health
                services = self.check_service_health()
                
                for service, status in services.items():
                    if status == 'down':
                        self.send_alert('SERVICE_DOWN', f"Service {service} is down", ['email', 'slack'])
                
                # Check security events
                events = self.check_security_events()
                
                if events['failed_logins'] > self.config['alerts']['failed_logins']['threshold']:
                    self.send_alert('SECURITY_ALERT', f"Failed logins: {events['failed_logins']}", ['email', 'slack'])
                
                print(f"Monitor check completed at {datetime.now()}")
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    monitor = BEVMonitor()
    monitor.run_monitoring_loop()
EOF
    
    chmod +x monitoring/monitor.py
    echo "✅ Monitoring system configured"
}
```

---

## Emergency Procedures

### Emergency Shutdown
```bash
#!/bin/bash
# Emergency shutdown procedures

emergency_shutdown() {
    echo "🚨 EMERGENCY SHUTDOWN INITIATED"
    echo "=============================="
    echo "Timestamp: $(date)"
    
    # Create emergency log
    echo "Creating emergency shutdown log..."
    echo "Emergency shutdown initiated at $(date)" > emergency_shutdown.log
    
    # Stop all BEV services immediately
    echo "Stopping all BEV services..."
    docker-compose down --timeout 30
    
    # Force stop any remaining containers
    echo "Force stopping any remaining containers..."
    docker stop $(docker ps -q) 2>/dev/null || true
    
    # Block all network traffic except SSH
    echo "Implementing network isolation..."
    iptables -F
    iptables -A INPUT -p tcp --dport 22 -j ACCEPT
    iptables -A INPUT -i lo -j ACCEPT
    iptables -A OUTPUT -o lo -j ACCEPT
    iptables -A INPUT -j DROP
    iptables -A OUTPUT -j DROP
    
    # Create system state snapshot
    echo "Creating system state snapshot..."
    ps aux > emergency_shutdown_processes.log
    netstat -tuln > emergency_shutdown_network.log
    df -h > emergency_shutdown_disk.log
    
    echo "✅ Emergency shutdown completed"
    echo "System is now isolated and secured"
    echo "Manual intervention required for restart"
}

# Emergency restart procedure
emergency_restart() {
    echo "🔄 EMERGENCY RESTART PROCEDURE"
    echo "============================="
    echo "Timestamp: $(date)"
    
    # Pre-restart checks
    echo "Performing pre-restart security checks..."
    
    # Check for security issues
    if grep -q "authentication failure" /var/log/auth.log; then
        echo "⚠️ Warning: Authentication failures detected"
        echo "Review security before proceeding"
        read -p "Continue with restart? (y/N): " confirm
        if [[ $confirm != "y" ]]; then
            echo "Restart aborted"
            return 1
        fi
    fi
    
    # Reset network rules
    echo "Resetting network configuration..."
    iptables -F
    iptables -P INPUT ACCEPT
    iptables -P OUTPUT ACCEPT
    iptables -P FORWARD ACCEPT
    
    # Regenerate secrets
    echo "Regenerating security credentials..."
    ./generate_secrets.sh
    
    # Start services in stages
    echo "Starting core services..."
    docker-compose up -d bev_postgres bev_redis bev_neo4j
    
    # Wait for databases
    echo "Waiting for databases to initialize..."
    sleep 60
    
    # Start application services
    echo "Starting application services..."
    docker-compose up -d
    
    # Verify system health
    echo "Verifying system health..."
    sleep 30
    ./health_check.sh
    
    echo "✅ Emergency restart completed"
}
```

---

*Last Updated: 2025-09-19*
*Framework Version: BEV OSINT v2.0*
*Classification: INTERNAL*
*Document Version: 1.0*