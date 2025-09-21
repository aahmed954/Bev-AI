#!/bin/bash

# Dependency Chain Validation Module
# Gate 4: Service dependencies and startup order validation

# Service dependency mapping
declare -A SERVICE_DEPENDENCIES=(
    ["postgres"]=""
    ["redis"]=""
    ["elasticsearch"]=""
    ["neo4j"]=""
    ["rabbitmq"]=""
    ["vault"]=""
    ["tor"]=""
    ["intelowl_uwsgi"]="postgres redis rabbitmq elasticsearch"
    ["intelowl_nginx"]="intelowl_uwsgi"
    ["intelowl_celery_beat"]="postgres redis rabbitmq"
    ["intelowl_celery_worker"]="postgres redis rabbitmq elasticsearch"
    ["grafana"]="prometheus"
    ["prometheus"]="postgres redis neo4j"
    ["mcp_server"]="postgres redis neo4j"
)

# Health check endpoints
declare -A HEALTH_ENDPOINTS=(
    ["postgres"]="pg_isready"
    ["redis"]="redis-cli ping"
    ["elasticsearch"]="http://localhost:9200/_cluster/health"
    ["neo4j"]="http://localhost:7474/db/manage/server/core/available"
    ["rabbitmq"]="http://localhost:15672/api/aliveness-test/%2F"
    ["grafana"]="http://localhost:3000/api/health"
    ["prometheus"]="http://localhost:9090/-/healthy"
    ["intelowl_nginx"]="http://localhost/api/me"
    ["mcp_server"]="http://localhost:3010/health"
)

# Service startup timeouts (seconds)
declare -A SERVICE_TIMEOUTS=(
    ["postgres"]="30"
    ["redis"]="15"
    ["elasticsearch"]="60"
    ["neo4j"]="45"
    ["rabbitmq"]="30"
    ["vault"]="20"
    ["tor"]="25"
    ["intelowl_uwsgi"]="60"
    ["intelowl_nginx"]="30"
    ["intelowl_celery_beat"]="45"
    ["intelowl_celery_worker"]="45"
    ["grafana"]="30"
    ["prometheus"]="40"
    ["mcp_server"]="30"
)

dependency_validation() {
    log "INFO" "Validating service dependencies and startup order"
    local validation_passed=true

    # Test 1: Dependency chain validation
    if ! validate_dependency_chain; then
        validation_passed=false
    fi

    # Test 2: Service health check validation
    if ! validate_health_checks; then
        validation_passed=false
    fi

    # Test 3: Network routing validation
    if ! validate_network_routing; then
        validation_passed=false
    fi

    # Test 4: Security policy validation
    if ! validate_security_policies; then
        validation_passed=false
    fi

    # Test 5: Service initialization validation
    if ! validate_service_initialization; then
        validation_passed=false
    fi

    return $(if [[ "$validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_dependency_chain() {
    log "INFO" "Validating service dependency chains"
    local dependency_validation_passed=true

    # Build startup order based on dependencies
    local startup_order=($(get_startup_order))

    log "INFO" "Calculated startup order: ${startup_order[*]}"

    # Validate each service's dependencies
    for service in "${!SERVICE_DEPENDENCIES[@]}"; do
        local dependencies="${SERVICE_DEPENDENCIES[$service]}"

        if [[ -n "$dependencies" ]]; then
            log "INFO" "Validating dependencies for $service: $dependencies"

            IFS=' ' read -ra DEPS <<< "$dependencies"
            for dep in "${DEPS[@]}"; do
                if [[ -n "${SERVICE_DEPENDENCIES[$dep]}" ]] || [[ "$dep" =~ ^(postgres|redis|elasticsearch|neo4j|rabbitmq|vault|tor)$ ]]; then
                    log "SUCCESS" "Dependency $dep -> $service is valid"
                else
                    log "ERROR" "Invalid dependency: $dep (required by $service)"
                    dependency_validation_passed=false
                fi
            done
        else
            log "SUCCESS" "Service $service has no dependencies"
        fi
    done

    # Check for circular dependencies
    if detect_circular_dependencies; then
        log "ERROR" "Circular dependencies detected in service configuration"
        dependency_validation_passed=false
    else
        log "SUCCESS" "No circular dependencies found"
    fi

    return $(if [[ "$dependency_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_health_checks() {
    log "INFO" "Validating service health check configurations"
    local health_check_passed=true

    for service in "${!HEALTH_ENDPOINTS[@]}"; do
        local endpoint="${HEALTH_ENDPOINTS[$service]}"

        log "INFO" "Validating health check for $service: $endpoint"

        case "$endpoint" in
            http*)
                # Test HTTP endpoint format
                if curl -s --connect-timeout 5 "$endpoint" >/dev/null 2>&1; then
                    log "SUCCESS" "Health endpoint $endpoint is accessible"
                else
                    log "WARN" "Health endpoint $endpoint is not accessible (service may not be running)"
                fi
                ;;
            "pg_isready")
                if command -v pg_isready >/dev/null 2>&1; then
                    log "SUCCESS" "PostgreSQL health check command available"
                else
                    log "WARN" "pg_isready command not found"
                fi
                ;;
            "redis-cli ping")
                if command -v redis-cli >/dev/null 2>&1; then
                    log "SUCCESS" "Redis health check command available"
                else
                    log "WARN" "redis-cli command not found"
                fi
                ;;
            *)
                log "WARN" "Unknown health check type for $service: $endpoint"
                ;;
        esac
    done

    return $(if [[ "$health_check_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_network_routing() {
    log "INFO" "Validating network routing and connectivity"
    local network_validation_passed=true

    # Test internal service communication
    local service_routes=(
        "intelowl_uwsgi:postgres:5432"
        "intelowl_uwsgi:redis:6379"
        "intelowl_uwsgi:elasticsearch:9200"
        "intelowl_celery_worker:rabbitmq:5672"
        "mcp_server:neo4j:7687"
        "grafana:prometheus:9090"
    )

    for route_info in "${service_routes[@]}"; do
        IFS=':' read -r source target port <<< "$route_info"

        log "INFO" "Validating route: $source -> $target:$port"

        # Test port connectivity (when services are running)
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            log "SUCCESS" "Target port $port is available for $target"
        else
            log "INFO" "Target port $port not currently open (service $target not running)"
        fi
    done

    # Validate Docker network configuration
    if docker network ls | grep -q "bev"; then
        log "SUCCESS" "BEV Docker network exists"
    else
        log "INFO" "BEV Docker network will be created during deployment"
    fi

    return $(if [[ "$network_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_security_policies() {
    log "INFO" "Validating security policies and access controls"
    local security_validation_passed=true

    # Check firewall configuration
    if command -v ufw >/dev/null 2>&1; then
        local ufw_status=$(ufw status | head -n1)
        if [[ "$ufw_status" == *"active"* ]]; then
            log "SUCCESS" "UFW firewall is active"

            # Check if BEV ports are properly configured
            local bev_ports=("80" "443" "7474" "3000" "9090")
            for port in "${bev_ports[@]}"; do
                if ufw status | grep -q "$port"; then
                    log "SUCCESS" "Firewall rule exists for port $port"
                else
                    log "WARN" "No firewall rule for port $port"
                fi
            done
        else
            log "WARN" "UFW firewall is inactive"
        fi
    else
        log "WARN" "UFW firewall not installed"
    fi

    # Check AppArmor/SELinux status
    if command -v aa-status >/dev/null 2>&1; then
        if aa-status --enabled >/dev/null 2>&1; then
            log "SUCCESS" "AppArmor is enabled"
        else
            log "INFO" "AppArmor is not enabled"
        fi
    elif command -v getenforce >/dev/null 2>&1; then
        local selinux_status=$(getenforce)
        log "INFO" "SELinux status: $selinux_status"
    fi

    # Validate file permissions
    local sensitive_files=(
        ".env:600"
        "vault/config.json:600"
        "vault/credentials/:700"
    )

    for file_info in "${sensitive_files[@]}"; do
        IFS=':' read -r file expected_perm <<< "$file_info"

        if [[ -e "$file" ]]; then
            local actual_perm=$(stat -c "%a" "$file" 2>/dev/null)
            if [[ "$actual_perm" == "$expected_perm" ]]; then
                log "SUCCESS" "File permissions correct for $file ($actual_perm)"
            else
                log "WARN" "File permissions for $file: $actual_perm (expected: $expected_perm)"

                if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                    if chmod "$expected_perm" "$file" 2>/dev/null; then
                        log "SUCCESS" "Fixed permissions for $file"
                    else
                        log "ERROR" "Failed to fix permissions for $file"
                        security_validation_passed=false
                    fi
                fi
            fi
        fi
    done

    return $(if [[ "$security_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_service_initialization() {
    log "INFO" "Validating service initialization requirements"
    local init_validation_passed=true

    # Check database initialization scripts
    local db_init_requirements=(
        "database/postgres/01_create_databases.sql"
        "database/postgres/02_create_users.sql"
        "database/postgres/03_setup_permissions.sql"
        "database/neo4j/01_create_constraints.cypher"
        "database/neo4j/02_create_indexes.cypher"
    )

    for init_script in "${db_init_requirements[@]}"; do
        if [[ -f "$init_script" ]]; then
            log "SUCCESS" "Database initialization script found: $init_script"
        else
            log "WARN" "Database initialization script missing: $init_script"
        fi
    done

    # Check IntelOwl configuration
    local intelowl_configs=(
        "intelowl/configuration/analyzer_config.json"
        "intelowl/configuration/connector_config.json"
        "intelowl/configuration/generic.env"
    )

    for config in "${intelowl_configs[@]}"; do
        if [[ -f "$config" ]]; then
            log "SUCCESS" "IntelOwl configuration found: $config"
        else
            log "ERROR" "IntelOwl configuration missing: $config"
            init_validation_passed=false
        fi
    done

    # Check custom analyzer availability
    local custom_analyzers=(
        "intelowl/custom_analyzers/breach_database.py"
        "intelowl/custom_analyzers/darknet_market.py"
        "intelowl/custom_analyzers/crypto_tracker.py"
        "intelowl/custom_analyzers/social_media.py"
    )

    for analyzer in "${custom_analyzers[@]}"; do
        if [[ -f "$analyzer" ]]; then
            log "SUCCESS" "Custom analyzer found: $analyzer"
        else
            log "ERROR" "Custom analyzer missing: $analyzer"
            init_validation_passed=false
        fi
    done

    return $(if [[ "$init_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

# Helper functions

get_startup_order() {
    local visited=()
    local result=()

    # Topological sort of dependencies
    for service in "${!SERVICE_DEPENDENCIES[@]}"; do
        if [[ ! " ${visited[*]} " =~ " ${service} " ]]; then
            visit_service "$service" visited result
        fi
    done

    echo "${result[@]}"
}

visit_service() {
    local service="$1"
    local -n visited_ref=$2
    local -n result_ref=$3

    visited_ref+=("$service")

    local dependencies="${SERVICE_DEPENDENCIES[$service]}"
    if [[ -n "$dependencies" ]]; then
        IFS=' ' read -ra DEPS <<< "$dependencies"
        for dep in "${DEPS[@]}"; do
            if [[ ! " ${visited_ref[*]} " =~ " ${dep} " ]]; then
                visit_service "$dep" visited_ref result_ref
            fi
        done
    fi

    result_ref+=("$service")
}

detect_circular_dependencies() {
    # Simple cycle detection using DFS
    local visiting=()
    local visited=()

    for service in "${!SERVICE_DEPENDENCIES[@]}"; do
        if [[ ! " ${visited[*]} " =~ " ${service} " ]]; then
            if has_cycle "$service" visiting visited; then
                return 0  # Circular dependency found
            fi
        fi
    done

    return 1  # No circular dependencies
}

has_cycle() {
    local service="$1"
    local -n visiting_ref=$2
    local -n visited_ref=$3

    if [[ " ${visiting_ref[*]} " =~ " ${service} " ]]; then
        return 0  # Cycle detected
    fi

    if [[ " ${visited_ref[*]} " =~ " ${service} " ]]; then
        return 1  # Already processed
    fi

    visiting_ref+=("$service")

    local dependencies="${SERVICE_DEPENDENCIES[$service]}"
    if [[ -n "$dependencies" ]]; then
        IFS=' ' read -ra DEPS <<< "$dependencies"
        for dep in "${DEPS[@]}"; do
            if has_cycle "$dep" visiting_ref visited_ref; then
                return 0
            fi
        done
    fi

    # Remove from visiting and add to visited
    local new_visiting=()
    for item in "${visiting_ref[@]}"; do
        if [[ "$item" != "$service" ]]; then
            new_visiting+=("$item")
        fi
    done
    visiting_ref=("${new_visiting[@]}")
    visited_ref+=("$service")

    return 1
}