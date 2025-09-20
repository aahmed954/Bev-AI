#!/bin/bash

#################################################################
# BEV Integration Test Suite
#
# End-to-end workflow testing for the complete BEV system
# Tests document processing, research, analysis pipelines
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORTS_DIR/integration_tests_$TIMESTAMP.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
fi

# Utility functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

pass_test() {
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
    log_success "$1"
}

fail_test() {
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
    log_error "$1"
}

# Test setup
setup_test_environment() {
    log_info "Setting up integration test environment..."

    mkdir -p "$REPORTS_DIR"
    mkdir -p "$SCRIPT_DIR/test_data"

    # Create test documents
    create_test_documents
}

create_test_documents() {
    local test_data_dir="$SCRIPT_DIR/test_data"

    # Create a test PDF document
    cat > "$test_data_dir/test_document.txt" << 'EOF'
Subject: Intelligence Report - Project Phoenix
Classification: CONFIDENTIAL
Date: 2024-01-15

Executive Summary:
Project Phoenix involves cyber intelligence gathering on suspected threat actors
operating in the telecommunications sector. The following entities have been
identified as persons of interest:

1. John Smith (email: j.smith@example.com)
   - Phone: +1-555-0123
   - IP Address: 192.168.1.100
   - Bitcoin Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa

2. Corporate Entity: TechCorp Solutions
   - Domain: techcorp-solutions.com
   - Registration: 2023-05-20
   - Registrar: NameCheap Inc.

Network Infrastructure:
- Server IP: 203.0.113.5
- DNS Records: ns1.techcorp-solutions.com, ns2.techcorp-solutions.com
- SSL Certificate: Let's Encrypt (expires 2024-06-15)

Recommendations:
Continue monitoring the identified assets and expand investigation
to include social media profiles and financial transactions.

End of Report
EOF

    # Create a test image with text
    cat > "$test_data_dir/test_image_content.txt" << 'EOF'
CLASSIFIED DOCUMENT
EYES ONLY

Target Assessment Report
Subject: Alex Johnson
DOB: 1985-03-12
SSN: 123-45-6789
Address: 123 Main Street, Anytown, ST 12345

Associated Contacts:
- Sarah Johnson (spouse): sarah.j@email.com
- Mike Thompson (associate): mike.t@company.org

Digital Footprints:
- GitHub: alexj_dev
- LinkedIn: alex-johnson-security
- Twitter: @alex_sec_analyst

Risk Level: MEDIUM
Next Review: 2024-02-15
EOF

    log_info "Created test documents for processing"
}

# OCR Service Integration Tests
test_ocr_service_integration() {
    log_info "Testing OCR service integration..."

    local test_file="$SCRIPT_DIR/test_data/test_image_content.txt"
    local ocr_endpoint="http://localhost:8001/api/v1/ocr/process"

    # Test OCR service availability
    if curl -s -f "http://localhost:8001/health" > /dev/null; then
        pass_test "OCR service is accessible"
    else
        fail_test "OCR service is not accessible"
        return 1
    fi

    # Test OCR processing (simulated with text file)
    local response=$(curl -s -X POST -F "file=@$test_file" "$ocr_endpoint" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$response" | jq -e '.text' > /dev/null 2>&1; then
        pass_test "OCR service processed document successfully"

        # Check if extracted text contains expected entities
        local extracted_text=$(echo "$response" | jq -r '.text')
        if echo "$extracted_text" | grep -q "Alex Johnson"; then
            pass_test "OCR extracted target name correctly"
        else
            fail_test "OCR failed to extract target name"
        fi

        if echo "$extracted_text" | grep -q "123-45-6789"; then
            pass_test "OCR extracted SSN correctly"
        else
            fail_test "OCR failed to extract SSN"
        fi
    else
        fail_test "OCR service failed to process document"
    fi
}

# Document Analyzer Integration Tests
test_document_analyzer_integration() {
    log_info "Testing document analyzer integration..."

    local test_file="$SCRIPT_DIR/test_data/test_document.txt"

    # Test document analysis pipeline
    local analysis_result=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"document_path\": \"$test_file\", \"analysis_types\": [\"entity_extraction\", \"sentiment_analysis\", \"classification\"]}" \
        "http://localhost:8001/api/v1/analyze" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$analysis_result" | jq -e '.entities' > /dev/null 2>&1; then
        pass_test "Document analyzer processed document"

        # Check entity extraction
        local entities=$(echo "$analysis_result" | jq -r '.entities[] | .text' 2>/dev/null || echo "")

        if echo "$entities" | grep -q "John Smith"; then
            pass_test "Document analyzer extracted person entity"
        else
            fail_test "Document analyzer failed to extract person entity"
        fi

        if echo "$entities" | grep -q "j.smith@example.com"; then
            pass_test "Document analyzer extracted email entity"
        else
            fail_test "Document analyzer failed to extract email entity"
        fi

        if echo "$entities" | grep -q "192.168.1.100"; then
            pass_test "Document analyzer extracted IP address"
        else
            fail_test "Document analyzer failed to extract IP address"
        fi
    else
        fail_test "Document analyzer failed to process document"
    fi
}

# IntelOwl Integration Tests
test_intelowl_integration() {
    log_info "Testing IntelOwl integration..."

    # Test IntelOwl job submission
    local job_data='{
        "observable_name": "192.168.1.100",
        "observable_classification": "ip",
        "analyzers_requested": ["VirusTotal", "AbuseIPDB", "Shodan"]
    }'

    local job_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -H "Authorization: Token ${INTELOWL_API_TOKEN:-dummy_token}" \
        -d "$job_data" \
        "http://localhost:8000/api/jobs" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$job_response" | jq -e '.id' > /dev/null 2>&1; then
        local job_id=$(echo "$job_response" | jq -r '.id')
        pass_test "IntelOwl job submitted successfully (ID: $job_id)"

        # Wait for job completion
        sleep 5

        # Check job status
        local job_status=$(curl -s -H "Authorization: Token ${INTELOWL_API_TOKEN:-dummy_token}" \
            "http://localhost:8000/api/jobs/$job_id" 2>/dev/null | jq -r '.status // "unknown"')

        if [[ "$job_status" == "completed" || "$job_status" == "running" ]]; then
            pass_test "IntelOwl job processing correctly (status: $job_status)"
        else
            fail_test "IntelOwl job failed (status: $job_status)"
        fi
    else
        fail_test "IntelOwl job submission failed"
    fi
}

# Swarm Intelligence Integration Tests
test_swarm_intelligence_integration() {
    log_info "Testing swarm intelligence integration..."

    # Test swarm master coordination
    local swarm_task='{
        "task_type": "intelligence_gathering",
        "target": "techcorp-solutions.com",
        "parameters": {
            "depth": "moderate",
            "focus_areas": ["whois", "dns", "subdomains", "ssl_certificates"]
        }
    }'

    local swarm_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$swarm_task" \
        "http://localhost:8002/api/v1/tasks" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$swarm_response" | jq -e '.task_id' > /dev/null 2>&1; then
        local task_id=$(echo "$swarm_response" | jq -r '.task_id')
        pass_test "Swarm intelligence task created (ID: $task_id)"

        # Check task distribution
        sleep 3
        local task_status=$(curl -s "http://localhost:8002/api/v1/tasks/$task_id/status" 2>/dev/null | jq -r '.status // "unknown"')

        if [[ "$task_status" == "in_progress" || "$task_status" == "completed" ]]; then
            pass_test "Swarm task distributed and processing"
        else
            fail_test "Swarm task distribution failed"
        fi
    else
        fail_test "Swarm intelligence task creation failed"
    fi

    # Test agent coordination
    local agent_status=$(curl -s "http://localhost:8002/api/v1/agents/status" 2>/dev/null | jq -r '.active_agents // 0')

    if [[ "$agent_status" -gt 0 ]]; then
        pass_test "Swarm agents are active ($agent_status agents)"
    else
        fail_test "No active swarm agents found"
    fi
}

# Memory Manager Integration Tests
test_memory_manager_integration() {
    log_info "Testing memory manager integration..."

    # Test memory storage
    local memory_data='{
        "content": "Intelligence report on Project Phoenix activities",
        "content_type": "analysis_result",
        "tags": ["project_phoenix", "cyber_intelligence", "telecommunications"],
        "metadata": {
            "source": "integration_test",
            "confidence": 0.85,
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }'

    local memory_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$memory_data" \
        "http://localhost:8005/api/v1/memory/store" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$memory_response" | jq -e '.memory_id' > /dev/null 2>&1; then
        local memory_id=$(echo "$memory_response" | jq -r '.memory_id')
        pass_test "Memory stored successfully (ID: $memory_id)"

        # Test memory retrieval
        local retrieved_memory=$(curl -s "http://localhost:8005/api/v1/memory/$memory_id" 2>/dev/null)

        if echo "$retrieved_memory" | jq -e '.content' > /dev/null 2>&1; then
            pass_test "Memory retrieved successfully"
        else
            fail_test "Memory retrieval failed"
        fi

        # Test semantic search
        local search_results=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"query": "Project Phoenix cyber intelligence", "limit": 5}' \
            "http://localhost:8005/api/v1/memory/search" 2>/dev/null)

        if echo "$search_results" | jq -e '.results[0]' > /dev/null 2>&1; then
            pass_test "Memory semantic search working"
        else
            fail_test "Memory semantic search failed"
        fi
    else
        fail_test "Memory storage failed"
    fi
}

# Research Coordinator Integration Tests
test_research_coordinator_integration() {
    log_info "Testing research coordinator integration..."

    # Test research task creation
    local research_task='{
        "research_type": "person_investigation",
        "target": "John Smith",
        "parameters": {
            "email": "j.smith@example.com",
            "phone": "+1-555-0123",
            "social_media": true,
            "financial_records": false,
            "background_check": true
        }
    }'

    local research_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$research_task" \
        "http://localhost:8004/api/v1/research/tasks" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$research_response" | jq -e '.task_id' > /dev/null 2>&1; then
        local task_id=$(echo "$research_response" | jq -r '.task_id')
        pass_test "Research task created (ID: $task_id)"

        # Check task execution
        sleep 5
        local task_progress=$(curl -s "http://localhost:8004/api/v1/research/tasks/$task_id/progress" 2>/dev/null | jq -r '.completion_percentage // 0')

        if [[ "$task_progress" -gt 0 ]]; then
            pass_test "Research task is progressing ($task_progress% complete)"
        else
            fail_test "Research task not progressing"
        fi

        # Test parallel research streams
        local streams=$(curl -s "http://localhost:8004/api/v1/research/tasks/$task_id/streams" 2>/dev/null | jq -r '.active_streams // 0')

        if [[ "$streams" -gt 0 ]]; then
            pass_test "Parallel research streams active ($streams streams)"
        else
            fail_test "No parallel research streams active"
        fi
    else
        fail_test "Research task creation failed"
    fi
}

# Security Integration Tests
test_security_integration() {
    log_info "Testing security service integration..."

    # Test Guardian enforcer
    local security_event='{
        "event_type": "suspicious_activity",
        "severity": "medium",
        "source_ip": "192.168.1.100",
        "details": {
            "action": "unauthorized_access_attempt",
            "target_service": "intelowl-django",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    }'

    local guardian_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$security_event" \
        "http://localhost:8008/api/v1/security/events" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$guardian_response" | jq -e '.event_id' > /dev/null 2>&1; then
        pass_test "Security event logged with Guardian"

        # Check automated response
        sleep 2
        local response_actions=$(curl -s "http://localhost:8008/api/v1/security/responses" 2>/dev/null | jq -r '.active_responses // 0')

        if [[ "$response_actions" -gt 0 ]]; then
            pass_test "Automated security response triggered"
        else
            fail_test "No automated security response"
        fi
    else
        fail_test "Security event logging failed"
    fi

    # Test IDS integration
    local ids_alerts=$(curl -s "http://localhost:8010/api/v1/alerts" 2>/dev/null | jq -r '.total_alerts // 0')
    log_info "IDS has $ids_alerts total alerts"

    # Test Vault secret access
    if command -v vault &> /dev/null; then
        export VAULT_ADDR="http://localhost:8200"
        export VAULT_TOKEN="${VAULT_ROOT_TOKEN:-}"

        if vault status &> /dev/null; then
            pass_test "Vault is accessible and unsealed"

            # Test secret storage and retrieval
            if vault kv put secret/test/integration key=value &> /dev/null; then
                if vault kv get -field=key secret/test/integration | grep -q "value"; then
                    pass_test "Vault secret storage and retrieval working"
                    vault kv delete secret/test/integration &> /dev/null
                else
                    fail_test "Vault secret retrieval failed"
                fi
            else
                fail_test "Vault secret storage failed"
            fi
        else
            fail_test "Vault is not accessible or sealed"
        fi
    else
        log_info "Vault CLI not available, skipping detailed Vault tests"
    fi
}

# Data Pipeline Integration Tests
test_data_pipeline_integration() {
    log_info "Testing data pipeline integration..."

    # Test complete data flow: OCR -> Analysis -> Storage -> Search
    local test_document="$SCRIPT_DIR/test_data/test_document.txt"

    # Step 1: Document ingestion
    log_info "Step 1: Document ingestion"
    local ingestion_response=$(curl -s -X POST -F "file=@$test_document" \
        "http://localhost:8001/api/v1/ingest" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$ingestion_response" | jq -e '.document_id' > /dev/null 2>&1; then
        local document_id=$(echo "$ingestion_response" | jq -r '.document_id')
        pass_test "Document ingested successfully (ID: $document_id)"

        # Step 2: Wait for processing
        log_info "Step 2: Waiting for document processing"
        sleep 10

        # Step 3: Check analysis results
        log_info "Step 3: Checking analysis results"
        local analysis_results=$(curl -s "http://localhost:8001/api/v1/documents/$document_id/analysis" 2>/dev/null)

        if echo "$analysis_results" | jq -e '.entities' > /dev/null 2>&1; then
            pass_test "Document analysis completed"

            # Step 4: Verify storage in graph database
            log_info "Step 4: Verifying storage in graph database"
            local neo4j_query='{"statements":[{"statement":"MATCH (d:Document {id: $doc_id}) RETURN d","parameters":{"doc_id":"'$document_id'"}}]}'

            local neo4j_response=$(curl -s -X POST \
                -H "Content-Type: application/json" \
                -H "Authorization: Basic $(echo -n "${NEO4J_USER:-neo4j}:${NEO4J_PASSWORD:-password}" | base64)" \
                -d "$neo4j_query" \
                "http://localhost:7474/db/data/transaction/commit" 2>/dev/null)

            if echo "$neo4j_response" | jq -e '.results[0].data[0]' > /dev/null 2>&1; then
                pass_test "Document stored in graph database"
            else
                fail_test "Document not found in graph database"
            fi

            # Step 5: Verify searchability in Elasticsearch
            log_info "Step 5: Verifying searchability in Elasticsearch"
            sleep 5  # Wait for indexing

            local es_search=$(curl -s -X GET \
                "http://localhost:9200/documents/_search?q=Project%20Phoenix" 2>/dev/null)

            if echo "$es_search" | jq -e '.hits.hits[0]' > /dev/null 2>&1; then
                pass_test "Document searchable in Elasticsearch"
            else
                fail_test "Document not searchable in Elasticsearch"
            fi
        else
            fail_test "Document analysis failed"
        fi
    else
        fail_test "Document ingestion failed"
    fi
}

# Autonomous System Integration Tests
test_autonomous_integration() {
    log_info "Testing autonomous system integration..."

    # Test autonomous controller decision making
    local decision_request='{
        "scenario": "threat_detected",
        "context": {
            "threat_level": "medium",
            "affected_services": ["intelowl-django"],
            "source_ip": "192.168.1.100",
            "confidence": 0.75
        },
        "available_actions": ["block_ip", "increase_monitoring", "notify_admin", "no_action"]
    }'

    local decision_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$decision_request" \
        "http://localhost:8013/api/v1/decisions" 2>/dev/null || echo '{"error": "request failed"}')

    if echo "$decision_response" | jq -e '.recommended_action' > /dev/null 2>&1; then
        local action=$(echo "$decision_response" | jq -r '.recommended_action')
        local confidence=$(echo "$decision_response" | jq -r '.confidence')
        pass_test "Autonomous controller made decision: $action (confidence: $confidence)"
    else
        fail_test "Autonomous controller decision making failed"
    fi

    # Test Live2D avatar integration
    local avatar_status=$(curl -s "http://localhost:8015/api/v1/avatar/status" 2>/dev/null | jq -r '.status // "unknown"')

    if [[ "$avatar_status" == "active" ]]; then
        pass_test "Live2D avatar is active"

        # Test avatar communication
        local avatar_message='{
            "message": "System integration test in progress",
            "emotion": "neutral",
            "gesture": "explaining"
        }'

        local avatar_response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d "$avatar_message" \
            "http://localhost:8015/api/v1/avatar/speak" 2>/dev/null)

        if echo "$avatar_response" | jq -e '.message_id' > /dev/null 2>&1; then
            pass_test "Live2D avatar communication working"
        else
            fail_test "Live2D avatar communication failed"
        fi
    else
        fail_test "Live2D avatar is not active (status: $avatar_status)"
    fi
}

# Monitoring Integration Tests
test_monitoring_integration() {
    log_info "Testing monitoring system integration..."

    # Test Prometheus metrics collection
    local prometheus_metrics=$(curl -s "http://localhost:9090/api/v1/query?query=up" 2>/dev/null | jq -r '.data.result | length')

    if [[ "$prometheus_metrics" -gt 0 ]]; then
        pass_test "Prometheus collecting metrics from $prometheus_metrics targets"
    else
        fail_test "Prometheus not collecting any metrics"
    fi

    # Test Grafana dashboard access
    local grafana_dashboards=$(curl -s "http://localhost:3001/api/search?type=dash-db" 2>/dev/null | jq -r '. | length' || echo "0")

    if [[ "$grafana_dashboards" -gt 0 ]]; then
        pass_test "Grafana has $grafana_dashboards dashboards available"
    else
        fail_test "No Grafana dashboards found"
    fi

    # Test InfluxDB data storage
    local influxdb_status=$(curl -s "http://localhost:8086/health" 2>/dev/null | jq -r '.status // "unknown"')

    if [[ "$influxdb_status" == "pass" ]]; then
        pass_test "InfluxDB is healthy and storing metrics"
    else
        fail_test "InfluxDB health check failed (status: $influxdb_status)"
    fi
}

# Message Queue Integration Tests
test_message_queue_integration() {
    log_info "Testing message queue integration..."

    # Test RabbitMQ management API
    local rabbitmq_auth="${RABBITMQ_USER:-admin}:${RABBITMQ_PASSWORD:-password}"
    local rabbitmq_queues=$(curl -s -u "$rabbitmq_auth" "http://localhost:15672/api/queues" 2>/dev/null | jq -r '. | length' || echo "0")

    if [[ "$rabbitmq_queues" -gt 0 ]]; then
        pass_test "RabbitMQ has $rabbitmq_queues queues configured"
    else
        fail_test "No RabbitMQ queues found"
    fi

    # Test Kafka message flow
    if command -v kafka-console-producer.sh &> /dev/null; then
        local test_topic="integration_test_$(date +%s)"
        local test_message="Integration test message at $(date)"

        # Produce message
        echo "$test_message" | kafka-console-producer.sh --broker-list localhost:19092 --topic "$test_topic" 2>/dev/null &
        sleep 2

        # Consume message
        local consumed_message=$(timeout 5 kafka-console-consumer.sh --bootstrap-server localhost:19092 --topic "$test_topic" --from-beginning --max-messages 1 2>/dev/null || echo "")

        if [[ "$consumed_message" == "$test_message" ]]; then
            pass_test "Kafka message flow working correctly"
        else
            fail_test "Kafka message flow not working"
        fi

        # Cleanup
        kafka-topics.sh --bootstrap-server localhost:19092 --delete --topic "$test_topic" 2>/dev/null || true
    else
        log_info "Kafka CLI tools not available, skipping message flow test"
    fi
}

# Network Security Integration Tests
test_network_security_integration() {
    log_info "Testing network security integration..."

    # Test Tor proxy functionality
    local tor_check=$(curl -s --socks5 localhost:9050 "http://httpbin.org/ip" 2>/dev/null | jq -r '.origin // "unknown"' || echo "unknown")

    if [[ "$tor_check" != "unknown" && "$tor_check" != "" ]]; then
        pass_test "Tor proxy is routing traffic correctly"
        log_info "Tor exit IP: $tor_check"
    else
        fail_test "Tor proxy not working correctly"
    fi

    # Test traffic analyzer
    local traffic_stats=$(curl -s "http://localhost:8011/api/v1/stats" 2>/dev/null | jq -r '.packets_analyzed // 0')

    if [[ "$traffic_stats" -gt 0 ]]; then
        pass_test "Traffic analyzer processing packets ($traffic_stats analyzed)"
    else
        fail_test "Traffic analyzer not processing packets"
    fi

    # Test anomaly detector
    local anomaly_alerts=$(curl -s "http://localhost:8012/api/v1/anomalies" 2>/dev/null | jq -r '.total_anomalies // 0')
    log_info "Anomaly detector has detected $anomaly_alerts anomalies"
}

# Cleanup test environment
cleanup_test_environment() {
    log_info "Cleaning up test environment..."

    # Remove test data
    rm -rf "$SCRIPT_DIR/test_data" 2>/dev/null || true

    # Clean up any test resources in services
    # This would include removing test documents, clearing test queues, etc.

    log_info "Cleanup completed"
}

# Main test execution
run_integration_tests() {
    log_info "Starting BEV integration test suite..."

    setup_test_environment

    # Core integration tests
    test_ocr_service_integration
    test_document_analyzer_integration
    test_intelowl_integration

    # Intelligence system tests
    test_swarm_intelligence_integration
    test_memory_manager_integration
    test_research_coordinator_integration

    # Security system tests
    test_security_integration

    # Complete workflow tests
    test_data_pipeline_integration
    test_autonomous_integration

    # Infrastructure tests
    test_monitoring_integration
    test_message_queue_integration
    test_network_security_integration

    cleanup_test_environment
}

# Generate integration test report
generate_integration_report() {
    local report_file="$REPORTS_DIR/integration_test_report_$TIMESTAMP.html"

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Integration Test Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }
        .stat-box.passed { border-left: 5px solid #28a745; }
        .stat-box.failed { border-left: 5px solid #dc3545; }
        .summary { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BEV Integration Test Report</h1>
            <div>Generated on: $(date)</div>
        </div>

        <div class="stats">
            <div class="stat-box passed">
                <h3>$PASSED_TESTS</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box failed">
                <h3>$FAILED_TESTS</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box">
                <h3>$TOTAL_TESTS</h3>
                <p>Total</p>
            </div>
        </div>

        <div class="summary">
            <h3>Integration Test Summary</h3>
            <p>Success Rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%</p>
            <p>System Integration Status: $( [[ $FAILED_TESTS -eq 0 ]] && echo "FULLY INTEGRATED" || echo "INTEGRATION ISSUES DETECTED" )</p>
            <p>Detailed logs: <code>$LOG_FILE</code></p>
        </div>
    </div>
</body>
</html>
EOF

    log_info "Integration test report generated: $report_file"
}

# Main function
main() {
    mkdir -p "$REPORTS_DIR"

    run_integration_tests
    generate_integration_report

    log_info "Integration testing completed!"
    log_info "Results: $PASSED_TESTS passed, $FAILED_TESTS failed"
    log_info "Success rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"

    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Execute if run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi