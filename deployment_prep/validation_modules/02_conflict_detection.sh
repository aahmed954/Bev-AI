#!/bin/bash

# Service Conflict Detection Module
# Gate 2: Detect and resolve service conflicts

# BEV service ports
declare -A BEV_PORTS=(
    ["intelowl"]="80,443"
    ["neo4j"]="7474,7687"
    ["postgres"]="5432"
    ["redis"]="6379"
    ["elasticsearch"]="9200,9300"
    ["grafana"]="3000"
    ["prometheus"]="9090"
    ["rabbitmq"]="5672,15672"
    ["mcp_server"]="3010"
    ["tor"]="9050,9051"
)

# Common container name patterns
BEV_CONTAINER_PATTERNS=(
    "bev_*"
    "intelowl_*"
    "*_neo4j"
    "*_postgres"
    "*_redis"
    "*_elasticsearch"
    "*_grafana"
    "*_prometheus"
)

conflict_detection() {
    log "INFO" "Detecting service conflicts"
    local conflicts_found=false

    # Test 1: Port conflicts
    if ! validate_port_availability; then
        conflicts_found=true
    fi

    # Test 2: Container conflicts
    if ! validate_container_conflicts; then
        conflicts_found=true
    fi

    # Test 3: Volume conflicts
    if ! validate_volume_conflicts; then
        conflicts_found=true
    fi

    # Test 4: Network conflicts
    if ! validate_network_conflicts; then
        conflicts_found=true
    fi

    # Test 5: Process conflicts
    if ! validate_process_conflicts; then
        conflicts_found=true
    fi

    return $(if [[ "$conflicts_found" == "false" ]]; then echo 0; else echo 1; fi)
}

validate_port_availability() {
    log "INFO" "Checking port availability"
    local port_conflicts=false

    for service in "${!BEV_PORTS[@]}"; do
        local ports="${BEV_PORTS[$service]}"
        IFS=',' read -ra PORT_ARRAY <<< "$ports"

        for port in "${PORT_ARRAY[@]}"; do
            if check_port_in_use "$port"; then
                local process_info=$(get_port_process_info "$port")
                log "ERROR" "Port $port required by $service is in use by: $process_info"

                if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                    if attempt_port_conflict_resolution "$port" "$service"; then
                        log "SUCCESS" "Port $port conflict resolved automatically"
                    else
                        log "ERROR" "Failed to resolve port $port conflict automatically"
                        port_conflicts=true
                    fi
                else
                    log "WARN" "Port $port conflict detected. Use --auto-fix to attempt resolution"
                    port_conflicts=true
                fi
            else
                log "SUCCESS" "Port $port available for $service"
            fi
        done
    done

    return $(if [[ "$port_conflicts" == "false" ]]; then echo 0; else echo 1; fi)
}

validate_container_conflicts() {
    log "INFO" "Checking container name conflicts"
    local container_conflicts=false

    # Get list of running containers
    local running_containers=$(docker ps --format "{{.Names}}")

    for pattern in "${BEV_CONTAINER_PATTERNS[@]}"; do
        local matching_containers=$(echo "$running_containers" | grep -E "^${pattern//\*/.*}$" || true)

        if [[ -n "$matching_containers" ]]; then
            log "WARN" "Found containers matching BEV pattern '$pattern':"
            while IFS= read -r container; do
                log "WARN" "  - $container"

                if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                    if attempt_container_conflict_resolution "$container"; then
                        log "SUCCESS" "Container $container stopped automatically"
                    else
                        log "ERROR" "Failed to stop container $container automatically"
                        container_conflicts=true
                    fi
                else
                    container_conflicts=true
                fi
            done <<< "$matching_containers"
        fi
    done

    return $(if [[ "$container_conflicts" == "false" ]]; then echo 0; else echo 1; fi)
}

validate_volume_conflicts() {
    log "INFO" "Checking volume conflicts"
    local volume_conflicts=false

    # Check for BEV-related volumes
    local bev_volumes=$(docker volume ls --format "{{.Name}}" | grep -E "(bev|intelowl|neo4j|postgres)" || true)

    if [[ -n "$bev_volumes" ]]; then
        log "WARN" "Found existing BEV-related volumes:"
        while IFS= read -r volume; do
            log "WARN" "  - $volume"

            # Check if volume is in use
            local containers_using_volume=$(docker ps -a --filter volume="$volume" --format "{{.Names}}" || true)
            if [[ -n "$containers_using_volume" ]]; then
                log "WARN" "Volume $volume is in use by: $containers_using_volume"
                volume_conflicts=true
            else
                if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                    if docker volume rm "$volume" >/dev/null 2>&1; then
                        log "SUCCESS" "Unused volume $volume removed automatically"
                    else
                        log "ERROR" "Failed to remove volume $volume"
                        volume_conflicts=true
                    fi
                fi
            fi
        done <<< "$bev_volumes"
    else
        log "SUCCESS" "No conflicting volumes found"
    fi

    return $(if [[ "$volume_conflicts" == "false" ]]; then echo 0; else echo 1; fi)
}

validate_network_conflicts() {
    log "INFO" "Checking Docker network conflicts"
    local network_conflicts=false

    # Check for BEV-related networks
    local bev_networks=$(docker network ls --format "{{.Name}}" | grep -E "(bev|intelowl)" || true)

    if [[ -n "$bev_networks" ]]; then
        log "WARN" "Found existing BEV-related networks:"
        while IFS= read -r network; do
            log "WARN" "  - $network"

            if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                if docker network rm "$network" >/dev/null 2>&1; then
                    log "SUCCESS" "Network $network removed automatically"
                else
                    log "ERROR" "Failed to remove network $network (may be in use)"
                    network_conflicts=true
                fi
            else
                network_conflicts=true
            fi
        done <<< "$bev_networks"
    else
        log "SUCCESS" "No conflicting networks found"
    fi

    return $(if [[ "$network_conflicts" == "false" ]]; then echo 0; else echo 1; fi)
}

validate_process_conflicts() {
    log "INFO" "Checking system process conflicts"
    local process_conflicts=false

    # Check for processes that might conflict with BEV services
    local conflicting_processes=(
        "postgres:PostgreSQL"
        "redis-server:Redis"
        "neo4j:Neo4j"
        "elasticsearch:Elasticsearch"
        "grafana-server:Grafana"
        "prometheus:Prometheus"
    )

    for process_info in "${conflicting_processes[@]}"; do
        IFS=':' read -r process_name service_name <<< "$process_info"

        if pgrep -f "$process_name" >/dev/null; then
            local pid=$(pgrep -f "$process_name")
            log "WARN" "$service_name process running (PID: $pid)"

            # Don't auto-stop system processes - require manual intervention
            log "WARN" "Manual intervention required: stop $service_name before deployment"
            process_conflicts=true
        else
            log "SUCCESS" "No conflicting $service_name process found"
        fi
    done

    return $(if [[ "$process_conflicts" == "false" ]]; then echo 0; else echo 1; fi)
}

# Helper functions for conflict detection

check_port_in_use() {
    local port="$1"
    netstat -tuln 2>/dev/null | grep -q ":$port " || ss -tuln 2>/dev/null | grep -q ":$port "
}

get_port_process_info() {
    local port="$1"
    local process_info=$(netstat -tulnp 2>/dev/null | grep ":$port " | awk '{print $7}' | head -n1)

    if [[ -z "$process_info" ]]; then
        process_info=$(ss -tulnp 2>/dev/null | grep ":$port " | sed 's/.*pid=\([0-9]*\).*/\1/' | head -n1)
        if [[ -n "$process_info" ]] && [[ "$process_info" =~ ^[0-9]+$ ]]; then
            process_info="PID $process_info ($(ps -p "$process_info" -o comm= 2>/dev/null || echo 'unknown'))"
        fi
    fi

    echo "${process_info:-unknown}"
}

# Conflict resolution functions

attempt_port_conflict_resolution() {
    local port="$1"
    local service="$2"

    log "INFO" "Attempting to resolve port $port conflict for $service"

    # Try to identify and stop the conflicting process
    local pid=$(netstat -tulnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 | head -n1)

    if [[ -z "$pid" ]]; then
        pid=$(ss -tulnp 2>/dev/null | grep ":$port " | sed 's/.*pid=\([0-9]*\).*/\1/' | head -n1)
    fi

    if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]]; then
        local process_name=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")

        # Only auto-kill non-critical processes
        local safe_to_kill_processes=("nginx" "apache2" "httpd" "docker-proxy")

        for safe_process in "${safe_to_kill_processes[@]}"; do
            if [[ "$process_name" == *"$safe_process"* ]]; then
                log "INFO" "Stopping process $process_name (PID: $pid) using port $port"
                if kill -TERM "$pid" 2>/dev/null; then
                    sleep 2
                    if ! kill -0 "$pid" 2>/dev/null; then
                        return 0
                    else
                        kill -KILL "$pid" 2>/dev/null
                        return $?
                    fi
                fi
            fi
        done

        log "WARN" "Process $process_name (PID: $pid) using port $port requires manual intervention"
        return 1
    fi

    return 1
}

attempt_container_conflict_resolution() {
    local container="$1"

    log "INFO" "Attempting to stop conflicting container: $container"

    # Get container status
    local container_status=$(docker ps --filter name="$container" --format "{{.Status}}" 2>/dev/null || true)

    if [[ -n "$container_status" ]]; then
        # Try graceful stop first
        if docker stop "$container" >/dev/null 2>&1; then
            return 0
        else
            # Force kill if graceful stop fails
            docker kill "$container" >/dev/null 2>&1
            return $?
        fi
    fi

    return 0
}