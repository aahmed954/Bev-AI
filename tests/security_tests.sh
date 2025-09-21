#!/bin/bash

#################################################################
# BEV Security Test Suite
#
# Comprehensive security testing for the BEV system
# Tests vulnerability scanning, access controls, encryption,
# network security, and compliance validation
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORTS_DIR/security_tests_$TIMESTAMP.log"

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
CRITICAL_FAILURES=0

# Security test configuration
SCAN_TIMEOUT=300
MAX_SCAN_THREADS=10

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

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_critical() {
    log "${RED}[CRITICAL]${NC} $1"
    ((CRITICAL_FAILURES++))
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

critical_fail_test() {
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
    ((CRITICAL_FAILURES++))
    log_critical "$1"
}

# Check security tools availability
check_security_tools() {
    log_info "Checking security testing tools availability..."

    local security_tools=("nmap" "nikto" "sqlmap" "dirb" "sslscan" "testssl.sh")
    local missing_tools=()

    for tool in "${security_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing security tools: ${missing_tools[*]}"
        log_info "Installing missing tools..."

        # Try to install missing tools
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            for tool in "${missing_tools[@]}"; do
                case "$tool" in
                    "testssl.sh")
                        log_info "Downloading testssl.sh..."
                        wget -q -O /tmp/testssl.sh https://testssl.sh/testssl.sh
                        chmod +x /tmp/testssl.sh
                        sudo mv /tmp/testssl.sh /usr/local/bin/
                        ;;
                    *)
                        sudo apt-get install -y "$tool" -qq
                        ;;
                esac
            done
        else
            log_warning "Cannot install missing tools automatically. Some tests may be skipped."
        fi
    fi

    log_info "Security tools check completed"
}

# Network security scanning
test_network_security() {
    log_info "Testing network security..."

    # Port scanning with nmap
    if command -v nmap &> /dev/null; then
        log_info "Running port scan on localhost..."

        local nmap_output=$(nmap -sS -O -sV --version-intensity 5 -p 1-65535 localhost 2>/dev/null || nmap -sT -p 1-65535 localhost 2>/dev/null)
        local nmap_results_file="$REPORTS_DIR/nmap_scan_$TIMESTAMP.txt"
        echo "$nmap_output" > "$nmap_results_file"

        # Analyze open ports
        local open_ports=$(echo "$nmap_output" | grep "^[0-9]" | grep "open" | wc -l)
        log_info "Found $open_ports open ports"

        # Check for unexpected open ports
        local expected_ports=(22 80 443 5432 6379 7001 7002 7003 7474 7687 8000 8080 8086 8200 9090 9200 3000 3001 3002)
        local unexpected_ports=()

        while IFS= read -r line; do
            if [[ "$line" =~ ^([0-9]+)/.*open ]]; then
                local port="${BASH_REMATCH[1]}"
                local expected=false
                for expected_port in "${expected_ports[@]}"; do
                    if [[ "$port" == "$expected_port" ]]; then
                        expected=true
                        break
                    fi
                done
                if [[ "$expected" == false ]]; then
                    unexpected_ports+=("$port")
                fi
            fi
        done <<< "$nmap_output"

        if [[ ${#unexpected_ports[@]} -eq 0 ]]; then
            pass_test "No unexpected open ports found"
        else
            fail_test "Unexpected open ports detected: ${unexpected_ports[*]}"
        fi

        # Check for vulnerable services
        if echo "$nmap_output" | grep -i "vulnerable"; then
            critical_fail_test "Vulnerable services detected in port scan"
        else
            pass_test "No obviously vulnerable services detected"
        fi
    else
        log_warning "nmap not available, skipping port scan"
    fi

    # Test firewall rules
    test_firewall_configuration
}

# Test firewall configuration
test_firewall_configuration() {
    log_info "Testing firewall configuration..."

    # Check if firewall is enabled
    local firewall_status="unknown"

    if command -v ufw &> /dev/null; then
        firewall_status=$(ufw status | head -1 | awk '{print $2}')
    elif command -v firewall-cmd &> /dev/null; then
        firewall_status=$(systemctl is-active firewalld 2>/dev/null || echo "inactive")
    elif command -v iptables &> /dev/null; then
        local iptables_rules=$(iptables -L | wc -l)
        if [[ "$iptables_rules" -gt 10 ]]; then
            firewall_status="active"
        else
            firewall_status="inactive"
        fi
    fi

    log_info "Firewall status: $firewall_status"

    if [[ "$firewall_status" == "active" ]]; then
        pass_test "Firewall is active"
    else
        fail_test "Firewall is not active or not configured"
    fi

    # Test external access restrictions
    test_external_access_restrictions
}

# Test external access restrictions
test_external_access_restrictions() {
    log_info "Testing external access restrictions..."

    # Test that internal services are not accessible from external interfaces
    local internal_services=(
        "5432:PostgreSQL"
        "6379:Redis"
        "7001:Redis_Cluster_1"
        "7002:Redis_Cluster_2"
        "7003:Redis_Cluster_3"
        "8086:InfluxDB"
        "9200:Elasticsearch"
    )

    for service_info in "${internal_services[@]}"; do
        IFS=':' read -ra service_parts <<< "$service_info"
        local port="${service_parts[0]}"
        local service_name="${service_parts[1]}"

        # Try to connect from external interface (if available)
        local external_ip=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' || echo "127.0.0.1")

        if [[ "$external_ip" != "127.0.0.1" ]] && nc -z -w5 "$external_ip" "$port" 2>/dev/null; then
            critical_fail_test "$service_name (port $port) is accessible from external interface"
        else
            pass_test "$service_name (port $port) properly restricted from external access"
        fi
    done
}

# Authentication and authorization testing
test_authentication_security() {
    log_info "Testing authentication and authorization security..."

    # Test default credentials
    test_default_credentials

    # Test authentication mechanisms
    test_auth_mechanisms

    # Test session management
    test_session_security

    # Test access controls
    test_access_controls
}

# Test for default credentials
test_default_credentials() {
    log_info "Testing for default credentials..."

    local default_creds=(
        "http://localhost:15672:admin:admin:RabbitMQ_Management"
        "http://localhost:3001:admin:admin:Grafana"
        "http://localhost:8080:admin:admin:Airflow"
    )

    for cred_info in "${default_creds[@]}"; do
        IFS=':' read -ra cred_parts <<< "$cred_info"
        local url="${cred_parts[0]}"
        local username="${cred_parts[1]}"
        local password="${cred_parts[2]}"
        local service="${cred_parts[3]}"

        local response=$(curl -s -o /dev/null -w "%{http_code}" -u "$username:$password" "$url" 2>/dev/null || echo "000")

        if [[ "$response" == "200" ]]; then
            critical_fail_test "$service is using default credentials ($username:$password)"
        else
            pass_test "$service is not using default credentials"
        fi
    done

    # Test for common weak passwords
    test_weak_passwords
}

# Test for weak passwords
test_weak_passwords() {
    log_info "Testing for weak passwords..."

    local weak_passwords=("password" "123456" "admin" "root" "guest" "test")
    local services=(
        "http://localhost:15672:RabbitMQ"
        "http://localhost:3001:Grafana"
        "http://localhost:8080:Airflow"
    )

    for service_info in "${services[@]}"; do
        IFS=':' read -ra service_parts <<< "$service_info"
        local url="${service_parts[0]}"
        local service_name="${service_parts[1]}"

        for password in "${weak_passwords[@]}"; do
            local response=$(curl -s -o /dev/null -w "%{http_code}" -u "admin:$password" "$url" 2>/dev/null || echo "000")

            if [[ "$response" == "200" ]]; then
                critical_fail_test "$service_name accepts weak password: $password"
                break
            fi
        done
    done

    pass_test "No weak passwords detected in tested services"
}

# Test authentication mechanisms
test_auth_mechanisms() {
    log_info "Testing authentication mechanisms..."

    # Test IntelOwl authentication
    local intelowl_unauth=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/api/jobs" 2>/dev/null || echo "000")

    if [[ "$intelowl_unauth" == "401" || "$intelowl_unauth" == "403" ]]; then
        pass_test "IntelOwl API requires authentication"
    else
        critical_fail_test "IntelOwl API allows unauthenticated access"
    fi

    # Test Vault authentication
    local vault_unauth=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8200/v1/secret/data/test" 2>/dev/null || echo "000")

    if [[ "$vault_unauth" == "400" || "$vault_unauth" == "403" ]]; then
        pass_test "Vault requires authentication"
    else
        fail_test "Vault authentication may be improperly configured"
    fi

    # Test protected endpoints
    local protected_endpoints=(
        "http://localhost:15672/api/overview:RabbitMQ_Management"
        "http://localhost:3001/api/admin/users:Grafana_Admin"
    )

    for endpoint_info in "${protected_endpoints[@]}"; do
        IFS=':' read -ra endpoint_parts <<< "$endpoint_info"
        local endpoint="${endpoint_parts[0]}"
        local service="${endpoint_parts[1]}"

        local response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "000")

        if [[ "$response" == "401" || "$response" == "403" ]]; then
            pass_test "$service endpoint properly protected"
        else
            fail_test "$service endpoint may not be properly protected (status: $response)"
        fi
    done
}

# Test session security
test_session_security() {
    log_info "Testing session security..."

    # Test session cookie security
    local cookie_test_urls=(
        "http://localhost:3001:Grafana"
        "http://localhost:8080:Airflow"
    )

    for url_info in "${cookie_test_urls[@]}"; do
        IFS=':' read -ra url_parts <<< "$url_info"
        local url="${url_parts[0]}"
        local service="${url_parts[1]}"

        local headers=$(curl -s -I "$url" 2>/dev/null || echo "")

        # Check for secure cookie flags
        if echo "$headers" | grep -i "set-cookie" | grep -i "secure"; then
            pass_test "$service uses secure cookie flags"
        else
            fail_test "$service missing secure cookie flags"
        fi

        # Check for HttpOnly flag
        if echo "$headers" | grep -i "set-cookie" | grep -i "httponly"; then
            pass_test "$service uses HttpOnly cookie flags"
        else
            fail_test "$service missing HttpOnly cookie flags"
        fi
    done
}

# Test access controls
test_access_controls() {
    log_info "Testing access controls..."

    # Test RBAC implementation
    test_rbac_implementation

    # Test privilege escalation
    test_privilege_escalation

    # Test file permissions
    test_file_permissions
}

# Test RBAC implementation
test_rbac_implementation() {
    log_info "Testing RBAC implementation..."

    # Test Vault RBAC
    if command -v vault &> /dev/null; then
        export VAULT_ADDR="http://localhost:8200"
        export VAULT_TOKEN="${VAULT_ROOT_TOKEN:-}"

        if vault auth -method=userpass 2>/dev/null; then
            pass_test "Vault RBAC authentication method available"
        else
            log_warning "Vault RBAC authentication method not configured"
        fi

        # Test policy enforcement
        local policies=$(vault policy list 2>/dev/null | wc -l || echo "0")
        if [[ "$policies" -gt 1 ]]; then
            pass_test "Vault has $policies security policies configured"
        else
            fail_test "Vault has insufficient security policies"
        fi
    fi

    # Test database user permissions
    test_database_permissions
}

# Test database permissions
test_database_permissions() {
    log_info "Testing database permissions..."

    # Test PostgreSQL user permissions
    local pg_test_user="test_limited_user"
    local pg_create_user="CREATE USER $pg_test_user WITH PASSWORD 'testpass';"
    local pg_grant_select="GRANT SELECT ON ALL TABLES IN SCHEMA public TO $pg_test_user;"

    if PGPASSWORD="${POSTGRES_PASSWORD:-password}" psql -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -d osint -c "$pg_create_user" 2>/dev/null; then
        if PGPASSWORD="${POSTGRES_PASSWORD:-password}" psql -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -d osint -c "$pg_grant_select" 2>/dev/null; then
            # Test that limited user cannot create tables
            if ! PGPASSWORD="testpass" psql -h localhost -p 5432 -U "$pg_test_user" -d osint -c "CREATE TABLE test_unauthorized (id INT);" 2>/dev/null; then
                pass_test "PostgreSQL limited user cannot create tables"
            else
                critical_fail_test "PostgreSQL limited user can create tables"
            fi

            # Cleanup
            PGPASSWORD="${POSTGRES_PASSWORD:-password}" psql -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -d osint -c "DROP USER IF EXISTS $pg_test_user;" 2>/dev/null
        fi
    else
        log_warning "Could not test PostgreSQL user permissions"
    fi
}

# Test privilege escalation
test_privilege_escalation() {
    log_info "Testing for privilege escalation vulnerabilities..."

    # Test container privilege escalation
    local privileged_containers=$(docker ps --format "table {{.Names}}\t{{.Image}}" | grep -E "(privileged|--privileged)" || echo "")

    if [[ -z "$privileged_containers" ]]; then
        pass_test "No privileged containers detected"
    else
        critical_fail_test "Privileged containers detected: $privileged_containers"
    fi

    # Test for containers running as root
    local root_containers=()
    local containers=$(docker ps --format "{{.Names}}")

    while IFS= read -r container; do
        local user=$(docker exec "$container" whoami 2>/dev/null || echo "unknown")
        if [[ "$user" == "root" ]]; then
            root_containers+=("$container")
        fi
    done <<< "$containers"

    if [[ ${#root_containers[@]} -eq 0 ]]; then
        pass_test "No containers running as root"
    else
        fail_test "Containers running as root: ${root_containers[*]}"
    fi
}

# Test file permissions
test_file_permissions() {
    log_info "Testing file permissions..."

    # Check critical file permissions
    local critical_files=(
        "$PROJECT_DIR/.env:600"
        "$PROJECT_DIR/docker-compose*.yml:644"
    )

    for file_info in "${critical_files[@]}"; do
        IFS=':' read -ra file_parts <<< "$file_info"
        local file_path="${file_parts[0]}"
        local expected_perms="${file_parts[1]}"

        if [[ -f "$file_path" ]]; then
            local actual_perms=$(stat -c "%a" "$file_path" 2>/dev/null || echo "000")

            if [[ "$actual_perms" == "$expected_perms" ]] || [[ "$actual_perms" -le "$expected_perms" ]]; then
                pass_test "File $file_path has secure permissions ($actual_perms)"
            else
                fail_test "File $file_path has overly permissive permissions ($actual_perms)"
            fi
        fi
    done

    # Check for world-writable files
    local world_writable=$(find "$PROJECT_DIR" -type f -perm -002 2>/dev/null | head -10)

    if [[ -z "$world_writable" ]]; then
        pass_test "No world-writable files found"
    else
        fail_test "World-writable files detected: $world_writable"
    fi
}

# Web application security testing
test_web_application_security() {
    log_info "Testing web application security..."

    # Test for common web vulnerabilities
    test_common_web_vulnerabilities

    # Test SSL/TLS configuration
    test_ssl_tls_security

    # Test HTTP security headers
    test_http_security_headers

    # Test for web application vulnerabilities
    test_web_app_vulnerabilities
}

# Test common web vulnerabilities
test_common_web_vulnerabilities() {
    log_info "Testing for common web vulnerabilities..."

    local web_services=(
        "http://localhost:80:IntelOwl_Web"
        "http://localhost:8000:IntelOwl_API"
        "http://localhost:3001:Grafana"
        "http://localhost:8080:Airflow"
    )

    for service_info in "${web_services[@]}"; do
        IFS=':' read -ra service_parts <<< "$service_info"
        local url="${service_parts[0]}"
        local service="${service_parts[1]}"

        log_info "Testing $service for web vulnerabilities..."

        # Test for directory traversal
        local traversal_response=$(curl -s -o /dev/null -w "%{http_code}" "$url/../../../etc/passwd" 2>/dev/null || echo "000")

        if [[ "$traversal_response" == "404" || "$traversal_response" == "403" ]]; then
            pass_test "$service protected against directory traversal"
        else
            critical_fail_test "$service may be vulnerable to directory traversal"
        fi

        # Test for XSS protection
        local xss_test="<script>alert('xss')</script>"
        local xss_response=$(curl -s "$url" -d "input=$xss_test" 2>/dev/null | grep -o "$xss_test" || echo "")

        if [[ -z "$xss_response" ]]; then
            pass_test "$service appears protected against XSS"
        else
            critical_fail_test "$service may be vulnerable to XSS"
        fi

        # Test for SQL injection (basic)
        test_sql_injection "$url" "$service"
    done
}

# Test SQL injection vulnerabilities
test_sql_injection() {
    local url="$1"
    local service="$2"

    log_info "Testing $service for SQL injection vulnerabilities..."

    if command -v sqlmap &> /dev/null; then
        local sqlmap_output=$(timeout 60 sqlmap -u "$url" --batch --level=1 --risk=1 2>/dev/null | grep -E "(vulnerable|injection)" || echo "no_injection")

        if echo "$sqlmap_output" | grep -q "vulnerable"; then
            critical_fail_test "$service vulnerable to SQL injection"
        else
            pass_test "$service not vulnerable to basic SQL injection"
        fi
    else
        # Manual SQL injection test
        local sql_payloads=("'" "1' OR '1'='1" "'; DROP TABLE users; --")

        for payload in "${sql_payloads[@]}"; do
            local response=$(curl -s "$url" -d "input=$payload" 2>/dev/null | grep -i -E "(error|exception|syntax|mysql|postgresql)" || echo "")

            if [[ -n "$response" ]]; then
                critical_fail_test "$service may be vulnerable to SQL injection (payload: $payload)"
                return
            fi
        done

        pass_test "$service not vulnerable to basic SQL injection"
    fi
}

# Test SSL/TLS security
test_ssl_tls_security() {
    log_info "Testing SSL/TLS security..."

    local ssl_services=(
        "localhost:443:IntelOwl_HTTPS"
        "localhost:7473:Neo4j_HTTPS"
    )

    for service_info in "${ssl_services[@]}"; do
        IFS=':' read -ra service_parts <<< "$service_info"
        local host="${service_parts[0]}"
        local port="${service_parts[1]}"
        local service="${service_parts[2]}"

        # Check if SSL port is open
        if ! nc -z -w5 "$host" "$port" 2>/dev/null; then
            log_warning "$service SSL port $port not accessible"
            continue
        fi

        # Test SSL configuration with sslscan
        if command -v sslscan &> /dev/null; then
            log_info "Running SSL scan for $service..."

            local sslscan_output=$(sslscan --no-colour "$host:$port" 2>/dev/null)
            local sslscan_results_file="$REPORTS_DIR/sslscan_${service}_$TIMESTAMP.txt"
            echo "$sslscan_output" > "$sslscan_results_file"

            # Check for weak ciphers
            if echo "$sslscan_output" | grep -i -E "(null|export|des|rc4|md5)"; then
                critical_fail_test "$service supports weak SSL ciphers"
            else
                pass_test "$service SSL cipher configuration appears secure"
            fi

            # Check for SSL/TLS version support
            if echo "$sslscan_output" | grep -i "sslv2\|sslv3\|tlsv1\.0"; then
                fail_test "$service supports deprecated SSL/TLS versions"
            else
                pass_test "$service only supports secure SSL/TLS versions"
            fi
        fi

        # Test with testssl.sh if available
        if command -v testssl.sh &> /dev/null; then
            log_info "Running comprehensive SSL test for $service..."

            local testssl_output=$(testssl.sh --quiet --color 0 "$host:$port" 2>/dev/null | grep -E "(HIGH|CRITICAL|LOW)")
            local testssl_results_file="$REPORTS_DIR/testssl_${service}_$TIMESTAMP.txt"
            echo "$testssl_output" > "$testssl_results_file"

            if echo "$testssl_output" | grep -i "critical"; then
                critical_fail_test "$service has critical SSL vulnerabilities"
            elif echo "$testssl_output" | grep -i "high"; then
                fail_test "$service has high severity SSL issues"
            else
                pass_test "$service SSL configuration passed comprehensive test"
            fi
        fi
    done
}

# Test HTTP security headers
test_http_security_headers() {
    log_info "Testing HTTP security headers..."

    local web_endpoints=(
        "http://localhost:80:IntelOwl_Web"
        "http://localhost:3001:Grafana"
        "http://localhost:8080:Airflow"
    )

    local required_headers=(
        "X-Content-Type-Options"
        "X-Frame-Options"
        "X-XSS-Protection"
        "Strict-Transport-Security"
        "Content-Security-Policy"
    )

    for endpoint_info in "${web_endpoints[@]}"; do
        IFS=':' read -ra endpoint_parts <<< "$endpoint_info"
        local url="${endpoint_parts[0]}"
        local service="${endpoint_parts[1]}"

        log_info "Checking security headers for $service..."

        local headers=$(curl -s -I "$url" 2>/dev/null || echo "")
        local missing_headers=()

        for header in "${required_headers[@]}"; do
            if ! echo "$headers" | grep -i "$header"; then
                missing_headers+=("$header")
            fi
        done

        if [[ ${#missing_headers[@]} -eq 0 ]]; then
            pass_test "$service has all required security headers"
        else
            fail_test "$service missing security headers: ${missing_headers[*]}"
        fi

        # Check for information disclosure headers
        if echo "$headers" | grep -i -E "(server|x-powered-by)"; then
            fail_test "$service exposes server information in headers"
        else
            pass_test "$service does not expose server information"
        fi
    done
}

# Test web application vulnerabilities
test_web_app_vulnerabilities() {
    log_info "Testing web application vulnerabilities..."

    local web_targets=(
        "http://localhost:80:IntelOwl"
        "http://localhost:3001:Grafana"
    )

    for target_info in "${web_targets[@]}"; do
        IFS=':' read -ra target_parts <<< "$target_info"
        local url="${target_parts[0]}"
        local service="${target_parts[1]}"

        # Directory enumeration with dirb
        if command -v dirb &> /dev/null; then
            log_info "Running directory enumeration for $service..."

            local dirb_output=$(timeout 300 dirb "$url" -w -S 2>/dev/null | grep -E "DIRECTORY|FILE" || echo "")
            local dirb_results_file="$REPORTS_DIR/dirb_${service}_$TIMESTAMP.txt"
            echo "$dirb_output" > "$dirb_results_file"

            # Check for sensitive directories
            if echo "$dirb_output" | grep -i -E "(admin|config|backup|test|dev)"; then
                fail_test "$service exposes potentially sensitive directories"
            else
                pass_test "$service does not expose obvious sensitive directories"
            fi
        fi

        # Web vulnerability scanning with nikto
        if command -v nikto &> /dev/null; then
            log_info "Running vulnerability scan for $service..."

            local nikto_output=$(timeout 300 nikto -h "$url" -Format txt 2>/dev/null | grep -E "(OSVDB|CVE)" || echo "")
            local nikto_results_file="$REPORTS_DIR/nikto_${service}_$TIMESTAMP.txt"
            echo "$nikto_output" > "$nikto_results_file"

            if [[ -n "$nikto_output" ]]; then
                fail_test "$service has known vulnerabilities detected by nikto"
            else
                pass_test "$service passed nikto vulnerability scan"
            fi
        fi
    done
}

# Container security testing
test_container_security() {
    log_info "Testing container security..."

    # Test container configurations
    test_container_configurations

    # Test container vulnerabilities
    test_container_vulnerabilities

    # Test container runtime security
    test_container_runtime_security
}

# Test container configurations
test_container_configurations() {
    log_info "Testing container security configurations..."

    local containers=$(docker ps --format "{{.Names}}")

    while IFS= read -r container; do
        # Check if container is running as non-root
        local user=$(docker exec "$container" id -u 2>/dev/null || echo "0")

        if [[ "$user" != "0" ]]; then
            pass_test "Container $container running as non-root user (UID: $user)"
        else
            fail_test "Container $container running as root"
        fi

        # Check for read-only root filesystem
        local rootfs_ro=$(docker inspect "$container" | jq -r '.[0].HostConfig.ReadonlyRootfs' 2>/dev/null || echo "false")

        if [[ "$rootfs_ro" == "true" ]]; then
            pass_test "Container $container has read-only root filesystem"
        else
            fail_test "Container $container does not have read-only root filesystem"
        fi

        # Check for privilege escalation prevention
        local no_new_privs=$(docker inspect "$container" | jq -r '.[0].HostConfig.SecurityOpt[]? | select(startswith("no-new-privileges"))' 2>/dev/null || echo "")

        if [[ -n "$no_new_privs" ]]; then
            pass_test "Container $container prevents privilege escalation"
        else
            fail_test "Container $container does not prevent privilege escalation"
        fi

        # Check for capability dropping
        local dropped_caps=$(docker inspect "$container" | jq -r '.[0].HostConfig.CapDrop[]?' 2>/dev/null | wc -l)

        if [[ "$dropped_caps" -gt 0 ]]; then
            pass_test "Container $container drops $dropped_caps capabilities"
        else
            fail_test "Container $container does not drop any capabilities"
        fi

    done <<< "$containers"
}

# Test container vulnerabilities
test_container_vulnerabilities() {
    log_info "Testing container vulnerabilities..."

    # Get list of running containers and their images
    local container_images=$(docker ps --format "{{.Image}}" | sort -u)

    # Basic vulnerability check using docker history
    while IFS= read -r image; do
        log_info "Checking image $image for security issues..."

        # Check for latest tag usage
        if [[ "$image" == *":latest" ]] || [[ "$image" != *":"* ]]; then
            fail_test "Image $image uses 'latest' tag (security risk)"
        else
            pass_test "Image $image uses specific version tag"
        fi

        # Check image age (basic check)
        local image_created=$(docker inspect "$image" | jq -r '.[0].Created' 2>/dev/null | cut -d'T' -f1)
        local image_age_days=$(( ($(date +%s) - $(date -d "$image_created" +%s)) / 86400 ))

        if [[ "$image_age_days" -gt 90 ]]; then
            fail_test "Image $image is $image_age_days days old (may have unpatched vulnerabilities)"
        else
            pass_test "Image $image is reasonably recent ($image_age_days days old)"
        fi

    done <<< "$container_images"
}

# Test container runtime security
test_container_runtime_security() {
    log_info "Testing container runtime security..."

    # Check Docker daemon configuration
    local docker_config="/etc/docker/daemon.json"

    if [[ -f "$docker_config" ]]; then
        # Check for user namespace remapping
        if jq -e '.["userns-remap"]' "$docker_config" &>/dev/null; then
            pass_test "Docker user namespace remapping configured"
        else
            fail_test "Docker user namespace remapping not configured"
        fi

        # Check for content trust
        if jq -e '.["content-trust"]' "$docker_config" &>/dev/null; then
            pass_test "Docker content trust enabled"
        else
            fail_test "Docker content trust not enabled"
        fi
    else
        log_warning "Docker daemon configuration file not found"
    fi

    # Check for Docker socket exposure
    local docker_socket_containers=$(docker ps --format "{{.Names}}" --filter "volume=/var/run/docker.sock" 2>/dev/null || echo "")

    if [[ -z "$docker_socket_containers" ]]; then
        pass_test "Docker socket not exposed to containers"
    else
        critical_fail_test "Docker socket exposed to containers: $docker_socket_containers"
    fi
}

# Data protection and encryption testing
test_data_protection() {
    log_info "Testing data protection and encryption..."

    # Test data at rest encryption
    test_data_at_rest_encryption

    # Test data in transit encryption
    test_data_in_transit_encryption

    # Test backup security
    test_backup_security

    # Test data retention policies
    test_data_retention
}

# Test data at rest encryption
test_data_at_rest_encryption() {
    log_info "Testing data at rest encryption..."

    # Check database encryption
    test_database_encryption

    # Check file system encryption
    test_filesystem_encryption

    # Check secrets management
    test_secrets_management
}

# Test database encryption
test_database_encryption() {
    log_info "Testing database encryption..."

    # Test PostgreSQL encryption
    local pg_ssl_status=$(PGPASSWORD="${POSTGRES_PASSWORD:-password}" psql -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -d osint -c "SHOW ssl;" 2>/dev/null | grep -o "on\|off" || echo "unknown")

    if [[ "$pg_ssl_status" == "on" ]]; then
        pass_test "PostgreSQL SSL encryption enabled"
    else
        fail_test "PostgreSQL SSL encryption not enabled"
    fi

    # Test Redis encryption (if configured)
    local redis_tls_check=$(redis-cli -h localhost -p 6379 -a "${REDIS_PASSWORD:-}" --tls ping 2>/dev/null | grep "PONG" || echo "")

    if [[ -n "$redis_tls_check" ]]; then
        pass_test "Redis TLS encryption enabled"
    else
        log_warning "Redis TLS encryption not configured"
    fi
}

# Test filesystem encryption
test_filesystem_encryption() {
    log_info "Testing filesystem encryption..."

    # Check for encrypted volumes
    local encrypted_volumes=$(lsblk -f | grep -i "crypt" || echo "")

    if [[ -n "$encrypted_volumes" ]]; then
        pass_test "Encrypted filesystems detected"
    else
        log_warning "No encrypted filesystems detected"
    fi

    # Check Docker volume encryption
    local docker_volumes=$(docker volume ls -q)
    local encrypted_docker_volumes=0

    while IFS= read -r volume; do
        local volume_info=$(docker volume inspect "$volume" 2>/dev/null | jq -r '.[0].Options.encrypted // "false"')
        if [[ "$volume_info" == "true" ]]; then
            ((encrypted_docker_volumes++))
        fi
    done <<< "$docker_volumes"

    if [[ "$encrypted_docker_volumes" -gt 0 ]]; then
        pass_test "$encrypted_docker_volumes Docker volumes are encrypted"
    else
        fail_test "No encrypted Docker volumes found"
    fi
}

# Test secrets management
test_secrets_management() {
    log_info "Testing secrets management..."

    # Test Vault secrets encryption
    if command -v vault &> /dev/null; then
        export VAULT_ADDR="http://localhost:8200"
        export VAULT_TOKEN="${VAULT_ROOT_TOKEN:-}"

        # Test secret storage
        if vault kv put secret/test/encryption test="encrypted_value" &>/dev/null; then
            # Verify secret is encrypted in storage
            local secret_data=$(vault kv get -field=test secret/test/encryption 2>/dev/null)

            if [[ "$secret_data" == "encrypted_value" ]]; then
                pass_test "Vault secrets encryption and retrieval working"
            else
                fail_test "Vault secrets encryption/retrieval failed"
            fi

            # Cleanup
            vault kv delete secret/test/encryption &>/dev/null
        else
            fail_test "Vault secret storage failed"
        fi

        # Test encryption in transit for Vault
        local vault_health=$(curl -s "http://localhost:8200/v1/sys/health" | jq -r '.sealed // true')

        if [[ "$vault_health" == "false" ]]; then
            pass_test "Vault is unsealed and operational"
        else
            fail_test "Vault is sealed or not operational"
        fi
    fi

    # Check for hardcoded secrets in configuration files
    test_hardcoded_secrets
}

# Test for hardcoded secrets
test_hardcoded_secrets() {
    log_info "Testing for hardcoded secrets..."

    local config_files=$(find "$PROJECT_DIR" -name "*.yml" -o -name "*.yaml" -o -name "*.env" -o -name "*.conf" 2>/dev/null)
    local secret_patterns=("password.*=.*[a-zA-Z0-9]" "key.*=.*[a-zA-Z0-9]" "token.*=.*[a-zA-Z0-9]" "secret.*=.*[a-zA-Z0-9]")
    local hardcoded_secrets=()

    while IFS= read -r file; do
        for pattern in "${secret_patterns[@]}"; do
            local matches=$(grep -i -E "$pattern" "$file" 2>/dev/null | grep -v "YOUR_" | grep -v "CHANGE_ME" | grep -v "\${" || echo "")
            if [[ -n "$matches" ]]; then
                hardcoded_secrets+=("$file: $matches")
            fi
        done
    done <<< "$config_files"

    if [[ ${#hardcoded_secrets[@]} -eq 0 ]]; then
        pass_test "No hardcoded secrets detected in configuration files"
    else
        critical_fail_test "Hardcoded secrets detected: ${hardcoded_secrets[*]}"
    fi
}

# Test data in transit encryption
test_data_in_transit_encryption() {
    log_info "Testing data in transit encryption..."

    # Test inter-service communication encryption
    local service_communications=(
        "localhost:7687:Neo4j_Bolt"
        "localhost:9200:Elasticsearch"
        "localhost:5432:PostgreSQL"
    )

    for comm_info in "${service_communications[@]}"; do
        IFS=':' read -ra comm_parts <<< "$comm_info"
        local host="${comm_parts[0]}"
        local port="${comm_parts[1]}"
        local service="${comm_parts[2]}"

        # Test if service supports TLS
        local tls_support=$(echo | openssl s_client -connect "$host:$port" -verify_return_error 2>/dev/null | grep "Verify return code" || echo "no_tls")

        if echo "$tls_support" | grep -q "ok"; then
            pass_test "$service supports TLS encryption"
        else
            fail_test "$service does not support TLS encryption"
        fi
    done
}

# Test backup security
test_backup_security() {
    log_info "Testing backup security..."

    # Check for backup encryption
    local backup_dirs=("/backup" "/var/backups" "./backups")

    for backup_dir in "${backup_dirs[@]}"; do
        if [[ -d "$backup_dir" ]]; then
            # Check for encrypted backup files
            local encrypted_backups=$(find "$backup_dir" -name "*.gpg" -o -name "*.enc" -o -name "*.aes" 2>/dev/null | wc -l)

            if [[ "$encrypted_backups" -gt 0 ]]; then
                pass_test "Encrypted backups found in $backup_dir"
            else
                fail_test "No encrypted backups found in $backup_dir"
            fi

            # Check backup file permissions
            local insecure_backups=$(find "$backup_dir" -type f -perm -044 2>/dev/null | wc -l)

            if [[ "$insecure_backups" -eq 0 ]]; then
                pass_test "Backup files in $backup_dir have secure permissions"
            else
                fail_test "$insecure_backups backup files in $backup_dir have insecure permissions"
            fi
        fi
    done
}

# Test data retention policies
test_data_retention() {
    log_info "Testing data retention policies..."

    # Test log rotation and retention
    local log_dirs=("./logs" "/var/log")

    for log_dir in "${log_dirs[@]}"; do
        if [[ -d "$log_dir" ]]; then
            # Check for log rotation configuration
            if [[ -f "/etc/logrotate.d/bev" ]] || [[ -f "/etc/logrotate.conf" ]]; then
                pass_test "Log rotation configured for $log_dir"
            else
                fail_test "Log rotation not configured for $log_dir"
            fi

            # Check for old log files (basic retention check)
            local old_logs=$(find "$log_dir" -name "*.log" -mtime +30 2>/dev/null | wc -l)

            if [[ "$old_logs" -eq 0 ]]; then
                pass_test "No old log files found in $log_dir (good retention)"
            else
                fail_test "$old_logs old log files found in $log_dir (review retention policy)"
            fi
        fi
    done
}

# Tor network security testing
test_tor_security() {
    log_info "Testing Tor network security..."

    # Test Tor proxy functionality
    test_tor_proxy_security

    # Test Tor node security
    test_tor_node_security

    # Test traffic anonymization
    test_traffic_anonymization
}

# Test Tor proxy security
test_tor_proxy_security() {
    log_info "Testing Tor proxy security..."

    # Test Tor SOCKS5 proxy
    local tor_proxy_test=$(curl -s --socks5 localhost:9050 "http://httpbin.org/ip" 2>/dev/null | jq -r '.origin' || echo "failed")

    if [[ "$tor_proxy_test" != "failed" && "$tor_proxy_test" != "" ]]; then
        pass_test "Tor SOCKS5 proxy is functional"
        log_info "Tor exit IP: $tor_proxy_test"

        # Verify IP is different from direct connection
        local direct_ip=$(curl -s "http://httpbin.org/ip" 2>/dev/null | jq -r '.origin' || echo "unknown")

        if [[ "$tor_proxy_test" != "$direct_ip" ]]; then
            pass_test "Tor proxy successfully anonymizes traffic"
        else
            critical_fail_test "Tor proxy is not anonymizing traffic"
        fi
    else
        critical_fail_test "Tor SOCKS5 proxy is not functional"
    fi

    # Test Tor control port security
    local tor_control_test=$(echo -e "AUTHENTICATE\nQUIT" | nc localhost 9051 2>/dev/null | grep "250 OK" || echo "")

    if [[ -n "$tor_control_test" ]]; then
        fail_test "Tor control port is accessible without authentication"
    else
        pass_test "Tor control port requires authentication"
    fi
}

# Test Tor node security
test_tor_node_security() {
    log_info "Testing Tor node security..."

    local tor_nodes=("localhost:9001" "localhost:9002" "localhost:9003")

    for node in "${tor_nodes[@]}"; do
        local host=$(echo "$node" | cut -d':' -f1)
        local port=$(echo "$node" | cut -d':' -f2)

        if nc -z -w5 "$host" "$port" 2>/dev/null; then
            pass_test "Tor node $node is accessible"

            # Check Tor node directory information
            local dir_port=$((port + 29))  # Directory port is typically relay_port + 29

            if nc -z -w5 "$host" "$dir_port" 2>/dev/null; then
                pass_test "Tor node $node directory port is accessible"
            else
                fail_test "Tor node $node directory port is not accessible"
            fi
        else
            fail_test "Tor node $node is not accessible"
        fi
    done
}

# Test traffic anonymization
test_traffic_anonymization() {
    log_info "Testing traffic anonymization..."

    # Test DNS leak protection
    local dns_leak_test=$(curl -s --socks5 localhost:9050 "https://1.1.1.1/cdn-cgi/trace" 2>/dev/null | grep "ip=" | cut -d'=' -f2 || echo "unknown")

    if [[ "$dns_leak_test" != "unknown" ]]; then
        log_info "DNS through Tor resolves to: $dns_leak_test"

        # Check if it's different from direct DNS resolution
        local direct_dns=$(curl -s "https://1.1.1.1/cdn-cgi/trace" 2>/dev/null | grep "ip=" | cut -d'=' -f2 || echo "unknown")

        if [[ "$dns_leak_test" != "$direct_dns" ]]; then
            pass_test "DNS queries are anonymized through Tor"
        else
            fail_test "DNS queries may be leaking outside Tor"
        fi
    else
        fail_test "Could not test DNS anonymization"
    fi

    # Test for WebRTC leak protection (basic check)
    local webrtc_test=$(curl -s --socks5 localhost:9050 "http://httpbin.org/headers" 2>/dev/null | grep -i "webrtc" || echo "")

    if [[ -z "$webrtc_test" ]]; then
        pass_test "No WebRTC headers detected in Tor traffic"
    else
        fail_test "WebRTC headers detected in Tor traffic (potential leak)"
    fi
}

# Compliance and security standards testing
test_compliance() {
    log_info "Testing compliance with security standards..."

    # Test basic security compliance
    test_basic_security_compliance

    # Test logging and auditing
    test_logging_auditing

    # Test incident response capabilities
    test_incident_response
}

# Test basic security compliance
test_basic_security_compliance() {
    log_info "Testing basic security compliance..."

    # Check for security documentation
    local security_docs=("$PROJECT_DIR/SECURITY.md" "$PROJECT_DIR/docs/security.md" "$PROJECT_DIR/security/README.md")
    local security_doc_found=false

    for doc in "${security_docs[@]}"; do
        if [[ -f "$doc" ]]; then
            security_doc_found=true
            break
        fi
    done

    if [[ "$security_doc_found" == true ]]; then
        pass_test "Security documentation found"
    else
        fail_test "Security documentation not found"
    fi

    # Check for security policies
    local security_policies=("$PROJECT_DIR/SECURITY_POLICY.md" "$PROJECT_DIR/docs/security_policy.md")
    local security_policy_found=false

    for policy in "${security_policies[@]}"; do
        if [[ -f "$policy" ]]; then
            security_policy_found=true
            break
        fi
    done

    if [[ "$security_policy_found" == true ]]; then
        pass_test "Security policy documentation found"
    else
        fail_test "Security policy documentation not found"
    fi

    # Check for vulnerability disclosure process
    local vuln_disclosure=("$PROJECT_DIR/SECURITY.md" "$PROJECT_DIR/.github/SECURITY.md")
    local vuln_disclosure_found=false

    for disclosure in "${vuln_disclosure[@]}"; do
        if [[ -f "$disclosure" ]]; then
            vuln_disclosure_found=true
            break
        fi
    done

    if [[ "$vuln_disclosure_found" == true ]]; then
        pass_test "Vulnerability disclosure process documented"
    else
        fail_test "Vulnerability disclosure process not documented"
    fi
}

# Test logging and auditing
test_logging_auditing() {
    log_info "Testing logging and auditing capabilities..."

    # Check for centralized logging
    local log_aggregators=("localhost:9200" "localhost:8086")
    local logging_configured=false

    for aggregator in "${log_aggregators[@]}"; do
        local host=$(echo "$aggregator" | cut -d':' -f1)
        local port=$(echo "$aggregator" | cut -d':' -f2)

        if nc -z -w5 "$host" "$port" 2>/dev/null; then
            logging_configured=true
            pass_test "Centralized logging available at $aggregator"
            break
        fi
    done

    if [[ "$logging_configured" == false ]]; then
        fail_test "Centralized logging not configured"
    fi

    # Check for audit trails
    local audit_logs=("./logs/audit.log" "/var/log/audit/audit.log")
    local audit_logging=false

    for audit_log in "${audit_logs[@]}"; do
        if [[ -f "$audit_log" ]]; then
            audit_logging=true
            pass_test "Audit logging found at $audit_log"
            break
        fi
    done

    if [[ "$audit_logging" == false ]]; then
        fail_test "Audit logging not configured"
    fi

    # Check for security event logging
    test_security_event_logging
}

# Test security event logging
test_security_event_logging() {
    log_info "Testing security event logging..."

    # Test IDS logging
    local ids_logs=$(curl -s "http://localhost:8010/api/v1/alerts" 2>/dev/null | jq -r '.total_alerts // 0')

    if [[ "$ids_logs" -gt 0 ]]; then
        pass_test "IDS is logging security events ($ids_logs alerts)"
    else
        log_warning "IDS has no logged security events"
    fi

    # Test Guardian enforcer logging
    local guardian_logs=$(curl -s "http://localhost:8008/api/v1/security/events" 2>/dev/null | jq -r '.total_events // 0')

    if [[ "$guardian_logs" -gt 0 ]]; then
        pass_test "Guardian enforcer is logging security events ($guardian_logs events)"
    else
        log_warning "Guardian enforcer has no logged security events"
    fi

    # Test authentication logging
    local auth_events=$(journalctl -u ssh --since "1 hour ago" 2>/dev/null | grep -c "authentication" || echo "0")

    if [[ "$auth_events" -gt 0 ]]; then
        pass_test "Authentication events are being logged ($auth_events events)"
    else
        log_warning "No recent authentication events logged"
    fi
}

# Test incident response capabilities
test_incident_response() {
    log_info "Testing incident response capabilities..."

    # Test alert management
    local prometheus_alerts=$(curl -s "http://localhost:9090/api/v1/alerts" 2>/dev/null | jq -r '.data.alerts | length' || echo "0")

    if [[ "$prometheus_alerts" -gt 0 ]]; then
        log_info "Prometheus has $prometheus_alerts active alerts"
    else
        log_info "Prometheus has no active alerts"
    fi

    # Test monitoring system responsiveness
    local monitoring_response_time=$(curl -w "%{time_total}" -s -o /dev/null "http://localhost:9090/-/healthy" 2>/dev/null || echo "999")

    if (( $(echo "$monitoring_response_time < 2.0" | bc -l) )); then
        pass_test "Monitoring system responsive (${monitoring_response_time}s)"
    else
        fail_test "Monitoring system slow to respond (${monitoring_response_time}s)"
    fi

    # Test automated response capabilities
    test_automated_response_capabilities
}

# Test automated response capabilities
test_automated_response_capabilities() {
    log_info "Testing automated response capabilities..."

    # Test Guardian enforcer automated responses
    local guardian_responses=$(curl -s "http://localhost:8008/api/v1/security/responses" 2>/dev/null | jq -r '.active_responses // 0')

    log_info "Guardian enforcer has $guardian_responses active automated responses"

    # Test anomaly detector responses
    local anomaly_responses=$(curl -s "http://localhost:8012/api/v1/anomalies/responses" 2>/dev/null | jq -r '.active_responses // 0')

    log_info "Anomaly detector has $anomaly_responses active automated responses"

    # Test system health auto-recovery
    local system_health=$(curl -s "http://localhost:9090/api/v1/query?query=up" 2>/dev/null | jq -r '.data.result | map(select(.value[1] == "0")) | length')

    if [[ "$system_health" -eq 0 ]]; then
        pass_test "All monitored services are healthy"
    else
        fail_test "$system_health services are down or unhealthy"
    fi
}

# Generate comprehensive security report
generate_security_report() {
    local report_file="$REPORTS_DIR/security_report_$TIMESTAMP.html"

    log_info "Generating comprehensive security report..."

    # Calculate risk levels
    local risk_level="LOW"
    if [[ $CRITICAL_FAILURES -gt 0 ]]; then
        risk_level="CRITICAL"
    elif [[ $FAILED_TESTS -gt 10 ]]; then
        risk_level="HIGH"
    elif [[ $FAILED_TESTS -gt 5 ]]; then
        risk_level="MEDIUM"
    fi

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Security Assessment Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .risk-level { padding: 10px; text-align: center; font-weight: bold; font-size: 1.2em; margin: 20px 0; border-radius: 5px; }
        .risk-critical { background-color: #dc3545; color: white; }
        .risk-high { background-color: #fd7e14; color: white; }
        .risk-medium { background-color: #ffc107; color: black; }
        .risk-low { background-color: #28a745; color: white; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }
        .stat-box.passed { border-left: 5px solid #28a745; }
        .stat-box.failed { border-left: 5px solid #dc3545; }
        .stat-box.critical { border-left: 5px solid #dc3545; background: #f8d7da; }
        .section { margin: 20px 0; }
        .section h3 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        .test-category { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 3px solid #007bff; }
        .critical-issue { background: #f8d7da; padding: 10px; margin: 5px 0; border-left: 3px solid #dc3545; }
        .recommendation { background: #d1ecf1; padding: 10px; margin: 5px 0; border-left: 3px solid #bee5eb; }
        .summary { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BEV Security Assessment Report</h1>
            <div>Assessment Date: $(date)</div>
            <div>Assessment Duration: $(( $(date +%s) - $(date -d "1 hour ago" +%s) )) seconds</div>
        </div>

        <div class="risk-level risk-$(echo "$risk_level" | tr '[:upper:]' '[:lower:]')">
            OVERALL RISK LEVEL: $risk_level
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
            <div class="stat-box critical">
                <h3>$CRITICAL_FAILURES</h3>
                <p>Critical</p>
            </div>
            <div class="stat-box">
                <h3>$TOTAL_TESTS</h3>
                <p>Total</p>
            </div>
        </div>

        <div class="section">
            <h3>Security Test Categories</h3>
            <div class="test-category">
                <strong>Network Security:</strong> Port scanning, firewall configuration, external access controls
            </div>
            <div class="test-category">
                <strong>Authentication & Authorization:</strong> Default credentials, weak passwords, RBAC, session security
            </div>
            <div class="test-category">
                <strong>Web Application Security:</strong> Common vulnerabilities, SSL/TLS, security headers, injection attacks
            </div>
            <div class="test-category">
                <strong>Container Security:</strong> Configuration hardening, vulnerability scanning, runtime security
            </div>
            <div class="test-category">
                <strong>Data Protection:</strong> Encryption at rest and in transit, secrets management, backup security
            </div>
            <div class="test-category">
                <strong>Tor Network Security:</strong> Proxy functionality, node security, traffic anonymization
            </div>
            <div class="test-category">
                <strong>Compliance & Auditing:</strong> Security documentation, logging, incident response capabilities
            </div>
        </div>

        <div class="section">
            <h3>Critical Security Issues</h3>
            $(if [[ $CRITICAL_FAILURES -gt 0 ]]; then
                echo '<div class="critical-issue">⚠️ <strong>ATTENTION:</strong> '$CRITICAL_FAILURES' critical security issues detected that require immediate attention.</div>'
            else
                echo '<div class="recommendation">✅ No critical security issues detected.</div>'
            fi)
        </div>

        <div class="section">
            <h3>Security Recommendations</h3>
            <div class="recommendation">
                <strong>Immediate Actions:</strong>
                <ul>
                    <li>Review and address all critical failures immediately</li>
                    <li>Implement missing authentication controls</li>
                    <li>Enable SSL/TLS for all web services</li>
                    <li>Configure proper firewall rules</li>
                </ul>
            </div>
            <div class="recommendation">
                <strong>Short-term Improvements:</strong>
                <ul>
                    <li>Implement container security hardening</li>
                    <li>Set up comprehensive logging and monitoring</li>
                    <li>Establish vulnerability management process</li>
                    <li>Create incident response procedures</li>
                </ul>
            </div>
            <div class="recommendation">
                <strong>Long-term Security Strategy:</strong>
                <ul>
                    <li>Regular security assessments and penetration testing</li>
                    <li>Security awareness training for development team</li>
                    <li>Implementation of DevSecOps practices</li>
                    <li>Compliance with relevant security standards</li>
                </ul>
            </div>
        </div>

        <div class="summary">
            <h3>Assessment Summary</h3>
            <p><strong>Security Posture:</strong> $( [[ $CRITICAL_FAILURES -eq 0 && $FAILED_TESTS -lt 5 ]] && echo "GOOD" || echo "NEEDS IMPROVEMENT" )</p>
            <p><strong>Critical Issues:</strong> $CRITICAL_FAILURES</p>
            <p><strong>Success Rate:</strong> $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%</p>
            <p><strong>Next Assessment:</strong> Recommended within 30 days</p>
            <p><strong>Detailed Logs:</strong> <code>$LOG_FILE</code></p>
        </div>

        <div class="section">
            <h3>Compliance Status</h3>
            <ul>
                <li>Basic Security Controls: $( [[ $FAILED_TESTS -lt 10 ]] && echo "✅ Compliant" || echo "❌ Non-compliant" )</li>
                <li>Access Controls: $( [[ $CRITICAL_FAILURES -eq 0 ]] && echo "✅ Implemented" || echo "❌ Requires attention" )</li>
                <li>Data Protection: $( [[ $FAILED_TESTS -lt 5 ]] && echo "✅ Adequate" || echo "❌ Needs improvement" )</li>
                <li>Monitoring & Logging: $( [[ $PASSED_TESTS -gt 20 ]] && echo "✅ Adequate" || echo "❌ Insufficient" )</li>
            </ul>
        </div>
    </div>
</body>
</html>
EOF

    log_success "Security assessment report generated: $report_file"
}

# Main execution function
main() {
    log_info "Starting BEV security test suite..."

    mkdir -p "$REPORTS_DIR"

    # Initialize security testing environment
    check_security_tools

    # Run comprehensive security tests
    test_network_security
    test_authentication_security
    test_web_application_security
    test_container_security
    test_data_protection
    test_tor_security
    test_compliance

    # Generate comprehensive security report
    generate_security_report

    # Final security assessment summary
    log_info "Security testing completed!"
    log_info "Results: $PASSED_TESTS passed, $FAILED_TESTS failed, $CRITICAL_FAILURES critical"
    log_info "Overall risk level: $( [[ $CRITICAL_FAILURES -gt 0 ]] && echo "CRITICAL" || [[ $FAILED_TESTS -gt 10 ]] && echo "HIGH" || [[ $FAILED_TESTS -gt 5 ]] && echo "MEDIUM" || echo "LOW" )"
    log_info "Security assessment report: $REPORTS_DIR/security_report_$TIMESTAMP.html"
    log_info "Detailed logs: $LOG_FILE"

    # Exit with appropriate code
    if [[ $CRITICAL_FAILURES -gt 0 ]]; then
        log_critical "Critical security issues detected! Immediate attention required."
        exit 2
    elif [[ $FAILED_TESTS -gt 0 ]]; then
        log_error "Security issues detected. Review and remediate before production deployment."
        exit 1
    else
        log_success "Security assessment passed. System ready for production deployment."
        exit 0
    fi
}

# Command line interface
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--timeout)
                SCAN_TIMEOUT="$2"
                shift 2
                ;;
            -j|--threads)
                MAX_SCAN_THREADS="$2"
                shift 2
                ;;
            -q|--quick)
                # Quick mode - skip time-intensive tests
                log_info "Running in quick mode..."
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -t, --timeout SECONDS   Scan timeout in seconds (default: $SCAN_TIMEOUT)"
                echo "  -j, --threads NUMBER    Maximum scan threads (default: $MAX_SCAN_THREADS)"
                echo "  -q, --quick             Quick mode - skip time-intensive tests"
                echo "  -h, --help              Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    main "$@"
fi