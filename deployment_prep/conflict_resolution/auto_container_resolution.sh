#!/bin/bash

# Automated Container Conflict Resolution Module
# Manages Docker container conflicts and cleanup

resolve_container_conflicts() {
    log "INFO" "Starting automated container conflict resolution"

    local conflicts_resolved=0
    local conflicts_failed=0

    # Stop conflicting containers
    if stop_conflicting_containers; then
        ((conflicts_resolved++))
    else
        ((conflicts_failed++))
    fi

    # Clean up unused resources
    if cleanup_unused_docker_resources; then
        ((conflicts_resolved++))
    else
        ((conflicts_failed++))
    fi

    # Resolve volume conflicts
    if resolve_volume_conflicts; then
        ((conflicts_resolved++))
    else
        ((conflicts_failed++))
    fi

    # Resolve network conflicts
    if resolve_network_conflicts; then
        ((conflicts_resolved++))
    else
        ((conflicts_failed++))
    fi

    log "INFO" "Container conflict resolution summary:"
    log "INFO" "  - Categories resolved: $conflicts_resolved"
    log "INFO" "  - Categories failed: $conflicts_failed"

    return $(if [[ $conflicts_failed -eq 0 ]]; then echo 0; else echo 1; fi)
}

stop_conflicting_containers() {
    log "INFO" "Stopping conflicting containers"

    local containers_stopped=0
    local containers_failed=0

    # Get list of running containers
    local running_containers=$(docker ps --format "{{.ID}}\t{{.Names}}\t{{.Image}}")

    if [[ -z "$running_containers" ]]; then
        log "SUCCESS" "No running containers found"
        return 0
    fi

    while IFS=$'\t' read -r container_id container_name container_image; do
        if is_conflicting_container "$container_name" "$container_image"; then
            log "WARN" "Found conflicting container: $container_name ($container_image)"

            if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                if stop_container_safely "$container_id" "$container_name"; then
                    log "SUCCESS" "Stopped conflicting container: $container_name"
                    ((containers_stopped++))
                else
                    log "ERROR" "Failed to stop container: $container_name"
                    ((containers_failed++))
                fi
            else
                log "WARN" "Container $container_name requires manual intervention (use --auto-fix)"
                ((containers_failed++))
            fi
        fi
    done <<< "$running_containers"

    log "INFO" "Container stop summary: $containers_stopped stopped, $containers_failed failed"

    return $(if [[ $containers_failed -eq 0 ]]; then echo 0; else echo 1; fi)
}

is_conflicting_container() {
    local container_name="$1"
    local container_image="$2"

    # Check against BEV container patterns
    for pattern in "${BEV_CONTAINER_PATTERNS[@]}"; do
        if [[ "$container_name" =~ ${pattern//\*/.*} ]]; then
            return 0
        fi
    done

    # Check against known conflicting images
    local conflicting_images=(
        "postgres"
        "redis"
        "elasticsearch"
        "neo4j"
        "nginx"
        "grafana"
        "prometheus"
        "rabbitmq"
        "intelowl"
    )

    for image_pattern in "${conflicting_images[@]}"; do
        if [[ "$container_image" == *"$image_pattern"* ]]; then
            return 0
        fi
    done

    return 1
}

stop_container_safely() {
    local container_id="$1"
    local container_name="$2"

    log "INFO" "Attempting to stop container $container_name gracefully"

    # Get container information
    local container_status=$(docker inspect "$container_id" --format='{{.State.Status}}' 2>/dev/null)

    if [[ "$container_status" != "running" ]]; then
        log "INFO" "Container $container_name is not running"
        return 0
    fi

    # Check if container has important data
    if container_has_persistent_data "$container_id"; then
        log "WARN" "Container $container_name may have persistent data"

        if [[ "$FORCE_MODE" == "true" ]]; then
            log "WARN" "Force mode: stopping container with potential data loss"
        else
            log "ERROR" "Container has persistent data - use --force to override"
            return 1
        fi
    fi

    # Try graceful stop first
    if docker stop --time=30 "$container_id" >/dev/null 2>&1; then
        log "SUCCESS" "Container $container_name stopped gracefully"
        return 0
    else
        log "WARN" "Graceful stop failed, attempting force stop"

        if docker kill "$container_id" >/dev/null 2>&1; then
            log "SUCCESS" "Container $container_name force stopped"
            return 0
        else
            log "ERROR" "Failed to stop container $container_name"
            return 1
        fi
    fi
}

container_has_persistent_data() {
    local container_id="$1"

    # Check for mounted volumes
    local volume_mounts=$(docker inspect "$container_id" --format='{{range .Mounts}}{{.Type}}:{{.Source}}:{{.Destination}} {{end}}' 2>/dev/null)

    if [[ -n "$volume_mounts" ]]; then
        # Check for non-temporary mounts
        if echo "$volume_mounts" | grep -qE "(volume|bind).*/(var|home|opt|data)"; then
            return 0
        fi
    fi

    # Check for database-related containers
    local container_image=$(docker inspect "$container_id" --format='{{.Config.Image}}' 2>/dev/null)

    if [[ "$container_image" == *"postgres"* ]] || [[ "$container_image" == *"mysql"* ]] ||
       [[ "$container_image" == *"mongodb"* ]] || [[ "$container_image" == *"neo4j"* ]] ||
       [[ "$container_image" == *"elasticsearch"* ]]; then
        return 0
    fi

    return 1
}

cleanup_unused_docker_resources() {
    log "INFO" "Cleaning up unused Docker resources"

    local cleanup_success=true

    # Remove stopped containers
    if [[ "$AUTO_FIX_MODE" == "true" ]]; then
        log "INFO" "Removing stopped containers"

        local stopped_containers=$(docker ps -a --filter status=exited --format "{{.ID}}" | head -20)
        if [[ -n "$stopped_containers" ]]; then
            if docker rm $stopped_containers >/dev/null 2>&1; then
                log "SUCCESS" "Removed stopped containers"
            else
                log "WARN" "Failed to remove some stopped containers"
            fi
        else
            log "SUCCESS" "No stopped containers to remove"
        fi

        # Remove dangling images
        log "INFO" "Removing dangling images"
        local dangling_images=$(docker images -f dangling=true -q | head -10)
        if [[ -n "$dangling_images" ]]; then
            if docker rmi $dangling_images >/dev/null 2>&1; then
                log "SUCCESS" "Removed dangling images"
            else
                log "WARN" "Failed to remove some dangling images"
            fi
        else
            log "SUCCESS" "No dangling images to remove"
        fi

        # Clean up unused networks (except default ones)
        log "INFO" "Removing unused networks"
        local unused_networks=$(docker network ls --filter "dangling=true" --format "{{.ID}}" | head -10)
        if [[ -n "$unused_networks" ]]; then
            if docker network rm $unused_networks >/dev/null 2>&1; then
                log "SUCCESS" "Removed unused networks"
            else
                log "WARN" "Failed to remove some unused networks"
            fi
        else
            log "SUCCESS" "No unused networks to remove"
        fi
    else
        log "INFO" "Auto-fix mode disabled - skipping resource cleanup"
    fi

    return $(if [[ "$cleanup_success" == "true" ]]; then echo 0; else echo 1; fi)
}

resolve_volume_conflicts() {
    log "INFO" "Resolving Docker volume conflicts"

    # Find BEV-related volumes
    local bev_volumes=$(docker volume ls --format "{{.Name}}" | grep -E "(bev|intelowl|neo4j|postgres|redis|elasticsearch)" || true)

    if [[ -z "$bev_volumes" ]]; then
        log "SUCCESS" "No conflicting volumes found"
        return 0
    fi

    log "WARN" "Found existing BEV-related volumes:"
    echo "$bev_volumes" | while IFS= read -r volume; do
        log "WARN" "  - $volume"
    done

    if [[ "$AUTO_FIX_MODE" == "true" ]]; then
        # Check which volumes are in use
        echo "$bev_volumes" | while IFS= read -r volume; do
            local containers_using_volume=$(docker ps -a --filter volume="$volume" --format "{{.Names}}" || true)

            if [[ -n "$containers_using_volume" ]]; then
                log "WARN" "Volume $volume is in use by: $containers_using_volume"

                if [[ "$FORCE_MODE" == "true" ]]; then
                    log "WARN" "Force mode: removing containers using volume $volume"

                    echo "$containers_using_volume" | while IFS= read -r container; do
                        docker rm -f "$container" >/dev/null 2>&1 || true
                    done

                    if docker volume rm "$volume" >/dev/null 2>&1; then
                        log "SUCCESS" "Removed volume $volume (force mode)"
                    else
                        log "ERROR" "Failed to remove volume $volume"
                    fi
                else
                    log "WARN" "Volume $volume in use - use --force to remove"
                fi
            else
                if docker volume rm "$volume" >/dev/null 2>&1; then
                    log "SUCCESS" "Removed unused volume $volume"
                else
                    log "ERROR" "Failed to remove volume $volume"
                fi
            fi
        done
    else
        log "WARN" "Auto-fix mode disabled - manual volume cleanup required"
        log "INFO" "To remove volumes manually: docker volume rm <volume_name>"
        return 1
    fi

    return 0
}

resolve_network_conflicts() {
    log "INFO" "Resolving Docker network conflicts"

    # Find BEV-related networks
    local bev_networks=$(docker network ls --format "{{.Name}}" | grep -E "(bev|intelowl)" || true)

    if [[ -z "$bev_networks" ]]; then
        log "SUCCESS" "No conflicting networks found"
        return 0
    fi

    log "WARN" "Found existing BEV-related networks:"
    echo "$bev_networks" | while IFS= read -r network; do
        log "WARN" "  - $network"
    done

    if [[ "$AUTO_FIX_MODE" == "true" ]]; then
        echo "$bev_networks" | while IFS= read -r network; do
            # Skip default networks
            if [[ "$network" == "bridge" ]] || [[ "$network" == "host" ]] || [[ "$network" == "none" ]]; then
                continue
            fi

            # Check if network is in use
            local containers_using_network=$(docker network inspect "$network" --format='{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || true)

            if [[ -n "$containers_using_network" ]]; then
                log "WARN" "Network $network is in use by: $containers_using_network"

                if [[ "$FORCE_MODE" == "true" ]]; then
                    log "WARN" "Force mode: disconnecting containers from network $network"

                    # Disconnect containers
                    echo "$containers_using_network" | tr ' ' '\n' | while IFS= read -r container; do
                        if [[ -n "$container" ]]; then
                            docker network disconnect "$network" "$container" >/dev/null 2>&1 || true
                        fi
                    done

                    if docker network rm "$network" >/dev/null 2>&1; then
                        log "SUCCESS" "Removed network $network (force mode)"
                    else
                        log "ERROR" "Failed to remove network $network"
                    fi
                else
                    log "WARN" "Network $network in use - use --force to remove"
                fi
            else
                if docker network rm "$network" >/dev/null 2>&1; then
                    log "SUCCESS" "Removed unused network $network"
                else
                    log "ERROR" "Failed to remove network $network"
                fi
            fi
        done
    else
        log "WARN" "Auto-fix mode disabled - manual network cleanup required"
        log "INFO" "To remove networks manually: docker network rm <network_name>"
        return 1
    fi

    return 0
}

# Create backup of container configurations before stopping
backup_container_configurations() {
    local backup_dir="$1"

    log "INFO" "Backing up container configurations"

    mkdir -p "$backup_dir/containers"

    # Get running containers
    local running_containers=$(docker ps --format "{{.ID}}\t{{.Names}}")

    if [[ -n "$running_containers" ]]; then
        while IFS=$'\t' read -r container_id container_name; do
            if is_conflicting_container "$container_name" ""; then
                log "INFO" "Backing up configuration for container: $container_name"

                # Save container inspect output
                docker inspect "$container_id" > "$backup_dir/containers/${container_name}_inspect.json" 2>/dev/null

                # Save container environment
                docker exec "$container_id" env > "$backup_dir/containers/${container_name}_env.txt" 2>/dev/null || true

                # Save container processes
                docker top "$container_id" > "$backup_dir/containers/${container_name}_processes.txt" 2>/dev/null || true
            fi
        done <<< "$running_containers"
    fi

    log "SUCCESS" "Container configurations backed up to $backup_dir/containers"
}