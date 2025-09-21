#!/bin/bash
# Verify all services are accessible from remote nodes

echo "🔗 VERIFYING SERVICE BINDINGS ACROSS NODES"
echo "=========================================="

# Test Thanos services from Oracle1
echo "Testing Thanos services from Oracle1:"
ssh oracle1 "
  echo -n 'PostgreSQL: ' && nc -z thanos 5432 && echo '✅' || echo '❌'
  echo -n 'Neo4j HTTP: ' && nc -z thanos 7474 && echo '✅' || echo '❌'
  echo -n 'Elasticsearch: ' && nc -z thanos 9200 && echo '✅' || echo '❌'
  echo -n 'Kafka: ' && nc -z thanos 9092 && echo '✅' || echo '❌'
"

# Test Oracle1 services from Thanos
echo ""
echo "Testing Oracle1 services from Thanos:"
ssh thanos "
  echo -n 'Prometheus: ' && nc -z oracle1 9090 && echo '✅' || echo '❌'
  echo -n 'Grafana: ' && nc -z oracle1 3000 && echo '✅' || echo '❌'
  echo -n 'Vault: ' && nc -z oracle1 8200 && echo '✅' || echo '❌'
  echo -n 'Redis: ' && nc -z oracle1 6379 && echo '✅' || echo '❌'
"

# Test Starlord services from both nodes
echo ""
echo "Testing Starlord services from remote nodes:"
ssh thanos "echo -n 'Frontend from Thanos: ' && nc -z starlord 5173 && echo '✅' || echo '❌'"
ssh oracle1 "echo -n 'Frontend from Oracle1: ' && nc -z starlord 5173 && echo '✅' || echo '❌'"

echo ""
echo "✅ Service binding verification complete!"
