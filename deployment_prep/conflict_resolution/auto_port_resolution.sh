#!/bin/bash

# Automated Port Conflict Resolution Module
# Safely resolves port conflicts where possible

# Safe processes to automatically terminate
SAFE_PROCESSES=(
    "nginx"
    "apache2"
    "httpd"
    "docker-proxy"
    "python3.*SimpleHTTPServer"
    "python.*http.server"
    "node.*express"
    "node.*http-server"
    "ruby.*webrick"
    "php.*built-in"
)

# Critical processes that require manual intervention
CRITICAL_PROCESSES=(
    "postgres"
    "mysql"
    "mongodb"
    "redis-server"
    "elasticsearch"
    "neo4j"
    "systemd"
    "ssh"
    "sshd"
)

resolve_port_conflicts() {
    local conflicts_resolved=0
    local conflicts_failed=0

    log "INFO" "Starting automated port conflict resolution"

    for service in "${!BEV_PORTS[@]}"; do
        local ports="${BEV_PORTS[$service]}"
        IFS=',' read -ra PORT_ARRAY <<< "$ports"

        for port in "${PORT_ARRAY[@]}"; do
            if check_port_in_use "$port"; then
                log "WARN" "Port $port conflict detected for $service"

                if resolve_single_port_conflict "$port" "$service"; then
                    log "SUCCESS" "Port $port conflict resolved for $service"
                    ((conflicts_resolved++))
                else
                    log "ERROR" "Failed to resolve port $port conflict for $service"
                    ((conflicts_failed++))
                fi
            fi
        done
    done

    log "INFO" "Port conflict resolution summary:"
    log "INFO" "  - Resolved: $conflicts_resolved"
    log "INFO" "  - Failed: $conflicts_failed"

    return $(if [[ $conflicts_failed -eq 0 ]]; then echo 0; else echo 1; fi)
}

resolve_single_port_conflict() {
    local port="$1"
    local service="$2"

    log "INFO" "Attempting to resolve port $port conflict"

    # Get process information
    local process_info=$(get_detailed_port_process_info "$port")
    local pid=$(echo "$process_info" | cut -d':' -f1)
    local process_name=$(echo "$process_info" | cut -d':' -f2)
    local command_line=$(echo "$process_info" | cut -d':' -f3)

    if [[ -z "$pid" ]] || [[ "$pid" == "unknown" ]]; then
        log "ERROR" "Cannot identify process using port $port"
        return 1
    fi

    log "INFO" "Port $port is used by PID $pid ($process_name): $command_line"

    # Check if process is safe to terminate
    if is_safe_process "$process_name" "$command_line"; then
        log "INFO" "Process $process_name is safe to terminate"

        if terminate_process_gracefully "$pid" "$process_name"; then
            # Wait for port to be released
            local attempts=0
            while check_port_in_use "$port" && [[ $attempts -lt 10 ]]; do
                sleep 1
                ((attempts++))
            done

            if ! check_port_in_use "$port"; then
                log "SUCCESS" "Port $port is now available"
                return 0
            else
                log "ERROR" "Port $port still in use after terminating process"
                return 1
            fi
        else
            log "ERROR" "Failed to terminate process $process_name (PID: $pid)"
            return 1
        fi

    elif is_critical_process "$process_name"; then
        log "ERROR" "Process $process_name is critical - manual intervention required"
        log "ERROR" "Please manually stop the service using port $port before deployment"
        return 1

    else
        # Unknown process - require confirmation
        log "WARN" "Unknown process type: $process_name"

        if [[ "$FORCE_MODE" == "true" ]]; then
            log "WARN" "Force mode enabled - attempting to terminate unknown process"

            if terminate_process_gracefully "$pid" "$process_name"; then
                local attempts=0
                while check_port_in_use "$port" && [[ $attempts -lt 10 ]]; do
                    sleep 1
                    ((attempts++))
                done

                if ! check_port_in_use "$port"; then
                    log "SUCCESS" "Port $port is now available (force mode)"
                    return 0
                else
                    log "ERROR" "Port $port still in use after forced termination"
                    return 1
                fi
            else
                log "ERROR" "Failed to terminate unknown process (force mode)"
                return 1
            fi
        else
            log "ERROR" "Manual intervention required for unknown process: $process_name"
            log "ERROR" "Use --force to attempt automatic termination"
            return 1
        fi
    fi
}

is_safe_process() {
    local process_name="$1"
    local command_line="$2"

    for safe_pattern in "${SAFE_PROCESSES[@]}"; do
        if [[ "$process_name" =~ $safe_pattern ]] || [[ "$command_line" =~ $safe_pattern ]]; then
            return 0
        fi
    done

    return 1
}

is_critical_process() {
    local process_name="$1"

    for critical_pattern in "${CRITICAL_PROCESSES[@]}"; do
        if [[ "$process_name" =~ $critical_pattern ]]; then
            return 0
        fi
    done

    return 1
}

terminate_process_gracefully() {
    local pid="$1"
    local process_name="$2"

    log "INFO" "Attempting graceful termination of $process_name (PID: $pid)"

    # First try SIGTERM
    if kill -TERM "$pid" 2>/dev/null; then
        log "INFO" "Sent SIGTERM to process $pid"

        # Wait up to 10 seconds for graceful shutdown
        local attempts=0
        while kill -0 "$pid" 2>/dev/null && [[ $attempts -lt 10 ]]; do
            sleep 1
            ((attempts++))
        done

        if ! kill -0 "$pid" 2>/dev/null; then
            log "SUCCESS" "Process $pid terminated gracefully"
            return 0
        else
            log "WARN" "Process $pid did not respond to SIGTERM, trying SIGKILL"

            # Force kill with SIGKILL
            if kill -KILL "$pid" 2>/dev/null; then
                sleep 2
                if ! kill -0 "$pid" 2>/dev/null; then
                    log "SUCCESS" "Process $pid terminated forcefully"
                    return 0
                else
                    log "ERROR" "Process $pid survived SIGKILL"
                    return 1
                fi
            else
                log "ERROR" "Failed to send SIGKILL to process $pid"
                return 1
            fi
        fi
    else
        log "ERROR" "Failed to send SIGTERM to process $pid"
        return 1
    fi
}

get_detailed_port_process_info() {
    local port="$1"
    local pid=""
    local process_name=""
    local command_line=""

    # Try netstat first
    local netstat_info=$(netstat -tulnp 2>/dev/null | grep ":$port ")
    if [[ -n "$netstat_info" ]]; then
        pid=$(echo "$netstat_info" | awk '{print $7}' | cut -d'/' -f1 | head -n1)
        process_name=$(echo "$netstat_info" | awk '{print $7}' | cut -d'/' -f2 | head -n1)
    fi

    # Try ss if netstat didn't work
    if [[ -z "$pid" ]] || [[ "$pid" == "-" ]]; then
        local ss_info=$(ss -tulnp 2>/dev/null | grep ":$port ")
        if [[ -n "$ss_info" ]]; then
            pid=$(echo "$ss_info" | grep -o 'pid=[0-9]*' | cut -d'=' -f2 | head -n1)
            if [[ -n "$pid" ]]; then
                process_name=$(ps -p "$pid" -o comm= 2>/dev/null)
            fi
        fi
    fi

    # Get command line if we have PID
    if [[ -n "$pid" ]] && [[ "$pid" =~ ^[0-9]+$ ]]; then
        command_line=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 100)
        if [[ -z "$process_name" ]]; then
            process_name=$(ps -p "$pid" -o comm= 2>/dev/null)
        fi
    fi

    # Return formatted information
    if [[ -n "$pid" ]]; then
        echo "$pid:${process_name:-unknown}:${command_line:-unknown}"
    else
        echo "unknown:unknown:unknown"
    fi
}

# Docker-specific conflict resolution
resolve_docker_port_conflicts() {
    log "INFO" "Resolving Docker-specific port conflicts"

    # Stop containers using BEV ports
    local conflicting_containers=$(docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}" | tail -n +2)

    if [[ -n "$conflicting_containers" ]]; then
        while IFS= read -r container_line; do
            local container_id=$(echo "$container_line" | awk '{print $1}')
            local container_name=$(echo "$container_line" | awk '{print $2}')
            local container_ports=$(echo "$container_line" | awk '{print $3}')

            # Check if container uses any BEV ports
            for service in "${!BEV_PORTS[@]}"; do
                local ports="${BEV_PORTS[$service]}"
                IFS=',' read -ra PORT_ARRAY <<< "$ports"

                for port in "${PORT_ARRAY[@]}"; do
                    if [[ "$container_ports" == *":$port->"* ]] || [[ "$container_ports" == *":$port/"* ]]; then
                        log "WARN" "Container $container_name uses BEV port $port"

                        if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                            log "INFO" "Stopping conflicting container: $container_name"

                            if docker stop "$container_id" >/dev/null 2>&1; then
                                log "SUCCESS" "Stopped container $container_name"
                            else
                                log "ERROR" "Failed to stop container $container_name"
                            fi
                        fi
                    fi
                done
            done
        done <<< "$conflicting_containers"
    fi
}

# Network service conflict resolution
resolve_system_service_conflicts() {
    log "INFO" "Checking for conflicting system services"

    # Services that commonly conflict with BEV
    local conflicting_services=(
        "nginx:80,443"
        "apache2:80,443"
        "postgresql:5432"
        "redis:6379"
        "elasticsearch:9200,9300"
        "neo4j:7474,7687"
    )

    for service_info in "${conflicting_services[@]}"; do
        IFS=':' read -r service_name service_ports <<< "$service_info"

        if systemctl is-active --quiet "$service_name" 2>/dev/null; then
            log "WARN" "System service $service_name is running and may conflict"

            IFS=',' read -ra PORT_ARRAY <<< "$service_ports"
            local has_conflict=false

            for port in "${PORT_ARRAY[@]}"; do
                # Check if this port is needed by BEV
                for bev_service in "${!BEV_PORTS[@]}"; do
                    local bev_ports="${BEV_PORTS[$bev_service]}"
                    if [[ "$bev_ports" == *"$port"* ]]; then
                        has_conflict=true
                        break
                    fi
                done

                if [[ "$has_conflict" == "true" ]]; then
                    break
                fi
            done

            if [[ "$has_conflict" == "true" ]]; then
                log "ERROR" "Service $service_name conflicts with BEV deployment"
                log "ERROR" "Manual intervention required: sudo systemctl stop $service_name"

                if [[ "$FORCE_MODE" == "true" ]]; then
                    log "WARN" "Force mode: attempting to stop $service_name"
                    if sudo systemctl stop "$service_name" 2>/dev/null; then
                        log "SUCCESS" "Stopped system service $service_name"
                    else
                        log "ERROR" "Failed to stop system service $service_name"
                    fi
                fi
            else
                log "SUCCESS" "Service $service_name does not conflict with BEV ports"
            fi
        fi
    done
}