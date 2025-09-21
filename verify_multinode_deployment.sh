#!/bin/bash
# BEV Multi-Node Deployment Verification Script
# Comprehensive checks for all deployed services

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
THANOS_HOST="thanos"
ORACLE1_HOST="oracle1"
THANOS_IP="100.122.12.54"
ORACLE1_IP="100.96.197.84"
STARLORD_IP="100.122.12.35"
VAULT_ADDR="http://$STARLORD_IP:8200"

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}           BEV DEPLOYMENT VERIFICATION                         ${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to check service
check_service() {
    local host=$1
    local service=$2
    local port=$3
    local description=$4
    
    echo -n -e "Checking $description on $host... "
    
    if nc -z -w2 $host $port 2>/dev/null; then
        echo -e "${GREEN}✅ RUNNING${NC}"
        return 0
    else
        echo -e "${RED}❌ NOT ACCESSIBLE${NC}"
        return 1
    fi
}

# Function to check Docker container
check_container() {
    local host=$1
    local container=$2
    local description=$3
    
    echo -n -e "Checking $description on $host... "
    
    if ssh $host "docker ps | grep -q $container" 2>/dev/null; then
        echo -e "${GREEN}✅ RUNNING${NC}"
        return 0
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        return 1
    fi
}

# Function to test database connection
test_database() {
    local host=$1
    local port=$2
    local db_type=$3
    
    echo -n -e "Testing $db_type connection... "
    
    case $db_type in
        postgres)
            if PGPASSWORD=$(vault kv get -field=postgres_password bev/database 2>/dev/null) \
               psql -h $host -p $port -U bev -d bev_db -c "SELECT 1" &>/dev/null; then
                echo -e "${GREEN}✅ CONNECTED${NC}"
            else
                echo -e "${RED}❌ CONNECTION FAILED${NC}"
            fi
            ;;
        neo4j)
            if curl -s http://$host:$port &>/dev/null; then
                echo -e "${GREEN}✅ ACCESSIBLE${NC}"
            else
                echo -e "${RED}❌ NOT ACCESSIBLE${NC}"
            fi
            ;;
        redis)
            if redis-cli -h $host -p $port ping &>/dev/null; then
                echo -e "${GREEN}✅ CONNECTED${NC}"
            else
                echo -e "${RED}❌ CONNECTION FAILED${NC}"
            fi
            ;;
    esac
}

# Phase 1: Check Vault
echo -e "${CYAN}Phase 1: Vault Status${NC}"
echo "----------------------------------------"

# Check Vault container
check_container localhost vault "Vault Container"

# Check Vault API
check_service localhost vault 8200 "Vault API"

# Check Vault status
echo -n "Checking Vault seal status... "
if vault status &>/dev/null; then
    echo -e "${GREEN}✅ UNSEALED${NC}"
else
    echo -e "${RED}❌ SEALED OR UNAVAILABLE${NC}"
fi

# Check Vault authentication
echo -n "Checking Vault authentication... "
if [ -n "$VAULT_TOKEN" ] && vault token lookup &>/dev/null; then
    echo -e "${GREEN}✅ AUTHENTICATED${NC}"
else
    echo -e "${RED}❌ NOT AUTHENTICATED${NC}"
fi

echo ""

# Phase 2: Check Thanos Services
echo -e "${CYAN}Phase 2: Thanos Node Services${NC}"
echo "----------------------------------------"

# Check containers on Thanos
containers_thanos=(
    "bev_postgres"
    "bev_neo4j"
    "bev_elasticsearch"
    "bev_influxdb"
    "bev_zookeeper"
    "bev_kafka_1"
    "bev_rabbitmq_1"
    "bev_intelowl_django"
    "bev_autonomous_coordinator"
)

for container in "${containers_thanos[@]}"; do
    check_container $THANOS_HOST $container "$container"
done

# Check service ports on Thanos
echo ""
echo "Checking service accessibility:"
check_service $THANOS_IP postgres 5432 "PostgreSQL"
check_service $THANOS_IP neo4j 7474 "Neo4j Browser"
check_service $THANOS_IP neo4j 7687 "Neo4j Bolt"
check_service $THANOS_IP elasticsearch 9200 "Elasticsearch"
check_service $THANOS_IP influxdb 8086 "InfluxDB"
check_service $THANOS_IP kafka 19092 "Kafka Broker 1"
check_service $THANOS_IP rabbitmq 5672 "RabbitMQ"
check_service $THANOS_IP rabbitmq 15672 "RabbitMQ Management"

echo ""

# Phase 3: Check Oracle1 Services
echo -e "${CYAN}Phase 3: Oracle1 Node Services${NC}"
echo "----------------------------------------"

# Check containers on Oracle1
containers_oracle1=(
    "bev_redis"
    "bev_prometheus"
    "bev_grafana"
    "bev_consul"
    "bev_tor"
    "bev_proxy_manager"
    "bev_breach_analyzer"
    "bev_crypto_analyzer"
    "bev_social_analyzer"
)

for container in "${containers_oracle1[@]}"; do
    check_container $ORACLE1_HOST $container "$container"
done

# Check service ports on Oracle1
echo ""
echo "Checking service accessibility:"
check_service $ORACLE1_IP redis 6379 "Redis"
check_service $ORACLE1_IP prometheus 9090 "Prometheus"
check_service $ORACLE1_IP grafana 3000 "Grafana"
check_service $ORACLE1_IP consul 8500 "Consul UI"
check_service $ORACLE1_IP tor 9050 "Tor Proxy"
check_service $ORACLE1_IP nginx 80 "Nginx Proxy"

echo ""

# Phase 4: Test Cross-Node Connectivity
echo -e "${CYAN}Phase 4: Cross-Node Connectivity${NC}"
echo "----------------------------------------"

# Test Thanos -> Oracle1
echo -n "Testing Thanos -> Oracle1... "
if ssh $THANOS_HOST "ping -c 1 $ORACLE1_IP" &>/dev/null; then
    echo -e "${GREEN}✅ CONNECTED${NC}"
else
    echo -e "${RED}❌ NOT CONNECTED${NC}"
fi

# Test Oracle1 -> Thanos
echo -n "Testing Oracle1 -> Thanos... "
if ssh $ORACLE1_HOST "ping -c 1 $THANOS_IP" &>/dev/null; then
    echo -e "${GREEN}✅ CONNECTED${NC}"
else
    echo -e "${RED}❌ NOT CONNECTED${NC}"
fi

# Test Oracle1 -> Vault (via Starlord)
echo -n "Testing Oracle1 -> Vault... "
if ssh $ORACLE1_HOST "curl -s $VAULT_ADDR/v1/sys/health" &>/dev/null; then
    echo -e "${GREEN}✅ ACCESSIBLE${NC}"
else
    echo -e "${RED}❌ NOT ACCESSIBLE${NC}"
fi

echo ""

# Phase 5: Database Connectivity Tests
echo -e "${CYAN}Phase 5: Database Connectivity${NC}"
echo "----------------------------------------"

test_database $THANOS_IP 5432 postgres
test_database $THANOS_IP 7474 neo4j
test_database $ORACLE1_IP 6379 redis

echo ""

# Phase 6: Service Health Checks
echo -e "${CYAN}Phase 6: Service Health Checks${NC}"
echo "----------------------------------------"

# Check Elasticsearch health
echo -n "Elasticsearch cluster health... "
health=$(curl -s http://$THANOS_IP:9200/_cluster/health | jq -r '.status' 2>/dev/null || echo "unknown")
case $health in
    green)
        echo -e "${GREEN}✅ GREEN${NC}"
        ;;
    yellow)
        echo -e "${YELLOW}⚠️  YELLOW${NC}"
        ;;
    red)
        echo -e "${RED}❌ RED${NC}"
        ;;
    *)
        echo -e "${RED}❌ UNKNOWN${NC}"
        ;;
esac

# Check Kafka cluster
echo -n "Kafka cluster status... "
if ssh $THANOS_HOST "docker exec bev_kafka_1 kafka-broker-api-versions --bootstrap-server localhost:9092" &>/dev/null; then
    echo -e "${GREEN}✅ HEALTHY${NC}"
else
    echo -e "${RED}❌ UNHEALTHY${NC}"
fi

# Check RabbitMQ cluster
echo -n "RabbitMQ cluster status... "
if ssh $THANOS_HOST "docker exec bev_rabbitmq_1 rabbitmqctl cluster_status" &>/dev/null; then
    echo -e "${GREEN}✅ HEALTHY${NC}"
else
    echo -e "${RED}❌ UNHEALTHY${NC}"
fi

echo ""

# Phase 7: GPU Availability (Thanos)
echo -e "${CYAN}Phase 7: GPU Resources (Thanos)${NC}"
echo "----------------------------------------"

echo -n "Checking NVIDIA GPU... "
if ssh $THANOS_HOST "nvidia-smi" &>/dev/null; then
    gpu_info=$(ssh $THANOS_HOST "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" 2>/dev/null)
    echo -e "${GREEN}✅ AVAILABLE${NC}"
    echo "  GPU: $gpu_info"
else
    echo -e "${RED}❌ NOT AVAILABLE${NC}"
fi

echo ""

# Phase 8: Monitoring Stack
echo -e "${CYAN}Phase 8: Monitoring Stack${NC}"
echo "----------------------------------------"

# Check Prometheus targets
echo -n "Prometheus targets... "
targets=$(curl -s http://$ORACLE1_IP:9090/api/v1/targets | jq '.data.activeTargets | length' 2>/dev/null || echo 0)
echo -e "${GREEN}$targets active targets${NC}"

# Check Grafana datasources
echo -n "Grafana datasources... "
if curl -s http://admin:admin@$ORACLE1_IP:3000/api/datasources &>/dev/null; then
    echo -e "${GREEN}✅ CONFIGURED${NC}"
else
    echo -e "${YELLOW}⚠️  CHECK CONFIGURATION${NC}"
fi

echo ""

# Summary
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}                    VERIFICATION SUMMARY                       ${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"

# Count successes and failures
total_checks=0
failed_checks=0

# Quick summary of critical services
critical_services=(
    "Vault:$STARLORD_IP:8200"
    "PostgreSQL:$THANOS_IP:5432"
    "Neo4j:$THANOS_IP:7474"
    "Redis:$ORACLE1_IP:6379"
    "Prometheus:$ORACLE1_IP:9090"
    "Grafana:$ORACLE1_IP:3000"
)

echo ""
echo -e "${CYAN}Critical Services Status:${NC}"
for service in "${critical_services[@]}"; do
    IFS=':' read -r name host port <<< "$service"
    echo -n "  • $name: "
    if nc -z -w2 $host $port 2>/dev/null; then
        echo -e "${GREEN}OPERATIONAL${NC}"
    else
        echo -e "${RED}DOWN${NC}"
        ((failed_checks++))
    fi
    ((total_checks++))
done

echo ""
if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ ALL CRITICAL SERVICES OPERATIONAL${NC}"
    echo -e "${GREEN}BEV deployment verification completed successfully!${NC}"
else
    echo -e "${RED}⚠️  $failed_checks/$total_checks CRITICAL SERVICES NEED ATTENTION${NC}"
    echo -e "${YELLOW}Please check the failed services and review logs${NC}"
fi

echo ""
echo -e "${CYAN}Access URLs:${NC}"
echo "  • Vault UI:       http://$STARLORD_IP:8200/ui"
echo "  • Neo4j Browser:  http://$THANOS_IP:7474"
echo "  • Grafana:        http://$ORACLE1_IP:3000"
echo "  • Prometheus:     http://$ORACLE1_IP:9090"
echo "  • Consul:         http://$ORACLE1_IP:8500"
echo "  • RabbitMQ:       http://$THANOS_IP:15672"
echo ""
echo "Verification completed at $(date)"
