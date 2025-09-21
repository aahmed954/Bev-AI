#!/bin/bash

# Infrastructure Readiness Validation Module
# Gate 1: System infrastructure and hardware requirements

# Node configuration
declare -A NODES=(
    ["THANOS"]="192.168.1.100"
    ["ORACLE1"]="192.168.1.101"
    ["STARLORD"]="192.168.1.102"
)

# Minimum requirements
MIN_MEMORY_GB=16
MIN_DISK_GB=500
MIN_DOCKER_VERSION="20.10"
MIN_COMPOSE_VERSION="2.0"

infrastructure_readiness() {
    log "INFO" "Validating infrastructure readiness"
    local validation_passed=true

    # Test 1: Node connectivity
    if ! validate_node_connectivity; then
        validation_passed=false
    fi

    # Test 2: Hardware requirements
    if ! validate_hardware_requirements; then
        validation_passed=false
    fi

    # Test 3: Software requirements
    if ! validate_software_requirements; then
        validation_passed=false
    fi

    # Test 4: Network connectivity
    if ! validate_network_connectivity; then
        validation_passed=false
    fi

    # Test 5: GPU and driver validation
    if ! validate_gpu_requirements; then
        validation_passed=false
    fi

    return $(if [[ "$validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_node_connectivity() {
    log "INFO" "Validating node connectivity"
    local all_nodes_accessible=true

    for node_name in "${!NODES[@]}"; do
        local node_ip="${NODES[$node_name]}"

        # Ping test
        if ping -c 1 -W 5 "$node_ip" >/dev/null 2>&1; then
            log "SUCCESS" "Node $node_name ($node_ip) is reachable"
        else
            log "ERROR" "Node $node_name ($node_ip) is unreachable"
            all_nodes_accessible=false
        fi

        # SSH connectivity test
        if ssh -o ConnectTimeout=10 -o BatchMode=yes "$node_ip" 'echo "SSH test successful"' >/dev/null 2>&1; then
            log "SUCCESS" "SSH access to $node_name is working"
        else
            log "ERROR" "SSH access to $node_name failed"
            all_nodes_accessible=false
        fi
    done

    # Test Tailscale connectivity if available
    if command -v tailscale >/dev/null 2>&1; then
        if tailscale status >/dev/null 2>&1; then
            log "SUCCESS" "Tailscale VPN is active"
        else
            log "WARN" "Tailscale VPN is not active"
        fi
    else
        log "WARN" "Tailscale not installed"
    fi

    return $(if [[ "$all_nodes_accessible" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_hardware_requirements() {
    log "INFO" "Validating hardware requirements"
    local hardware_ok=true

    # Memory validation
    local memory_gb=$(free -g | awk 'NR==2{printf "%.0f", $2}')
    if [[ $memory_gb -ge $MIN_MEMORY_GB ]]; then
        log "SUCCESS" "Memory: ${memory_gb}GB (required: ${MIN_MEMORY_GB}GB)"
    else
        log "ERROR" "Insufficient memory: ${memory_gb}GB (required: ${MIN_MEMORY_GB}GB)"
        hardware_ok=false
    fi

    # Disk space validation
    local disk_gb=$(df / | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
    if [[ $disk_gb -ge $MIN_DISK_GB ]]; then
        log "SUCCESS" "Available disk space: ${disk_gb}GB (required: ${MIN_DISK_GB}GB)"
    else
        log "ERROR" "Insufficient disk space: ${disk_gb}GB (required: ${MIN_DISK_GB}GB)"
        hardware_ok=false
    fi

    # CPU cores validation
    local cpu_cores=$(nproc)
    if [[ $cpu_cores -ge 4 ]]; then
        log "SUCCESS" "CPU cores: $cpu_cores (recommended: 4+)"
    else
        log "WARN" "Limited CPU cores: $cpu_cores (recommended: 4+)"
    fi

    return $(if [[ "$hardware_ok" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_software_requirements() {
    log "INFO" "Validating software requirements"
    local software_ok=true

    # Docker validation
    if command -v docker >/dev/null 2>&1; then
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        if version_compare "$docker_version" "$MIN_DOCKER_VERSION"; then
            log "SUCCESS" "Docker version: $docker_version (required: $MIN_DOCKER_VERSION+)"
        else
            log "ERROR" "Docker version too old: $docker_version (required: $MIN_DOCKER_VERSION+)"
            software_ok=false
        fi

        # Docker daemon status
        if docker info >/dev/null 2>&1; then
            log "SUCCESS" "Docker daemon is running"
        else
            log "ERROR" "Docker daemon is not running or accessible"
            software_ok=false
        fi
    else
        log "ERROR" "Docker not installed"
        software_ok=false
    fi

    # Docker Compose validation
    if command -v docker-compose >/dev/null 2>&1; then
        local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        if version_compare "$compose_version" "$MIN_COMPOSE_VERSION"; then
            log "SUCCESS" "Docker Compose version: $compose_version (required: $MIN_COMPOSE_VERSION+)"
        else
            log "ERROR" "Docker Compose version too old: $compose_version (required: $MIN_COMPOSE_VERSION+)"
            software_ok=false
        fi
    else
        log "ERROR" "Docker Compose not installed"
        software_ok=false
    fi

    # Git validation
    if command -v git >/dev/null 2>&1; then
        log "SUCCESS" "Git is available"
    else
        log "ERROR" "Git not installed"
        software_ok=false
    fi

    return $(if [[ "$software_ok" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_network_connectivity() {
    log "INFO" "Validating network connectivity"
    local network_ok=true

    # Internet connectivity
    if ping -c 1 -W 5 8.8.8.8 >/dev/null 2>&1; then
        log "SUCCESS" "Internet connectivity available"
    else
        log "ERROR" "No internet connectivity"
        network_ok=false
    fi

    # DNS resolution
    if nslookup google.com >/dev/null 2>&1; then
        log "SUCCESS" "DNS resolution working"
    else
        log "ERROR" "DNS resolution failed"
        network_ok=false
    fi

    # Docker Hub connectivity
    if curl -sSf https://hub.docker.com >/dev/null 2>&1; then
        log "SUCCESS" "Docker Hub is accessible"
    else
        log "WARN" "Docker Hub connectivity issues (may affect image pulls)"
    fi

    return $(if [[ "$network_ok" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_gpu_requirements() {
    log "INFO" "Validating GPU requirements"
    local gpu_ok=true

    # NVIDIA GPU validation
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            local gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n1)
            log "SUCCESS" "NVIDIA GPU detected: $gpu_info"

            # Check CUDA availability
            if command -v nvcc >/dev/null 2>&1; then
                local cuda_version=$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+')
                log "SUCCESS" "CUDA version: $cuda_version"
            else
                log "WARN" "CUDA toolkit not found"
            fi

            # Check Docker GPU support
            if docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
                log "SUCCESS" "Docker GPU support is working"
            else
                log "ERROR" "Docker GPU support not working"
                gpu_ok=false
            fi
        else
            log "ERROR" "nvidia-smi failed (driver issues?)"
            gpu_ok=false
        fi
    else
        log "WARN" "NVIDIA drivers not installed (GPU acceleration disabled)"
    fi

    return $(if [[ "$gpu_ok" == "true" ]]; then echo 0; else echo 1; fi)
}

# Version comparison helper
version_compare() {
    local version1="$1"
    local version2="$2"

    if [[ "$version1" == "$version2" ]]; then
        return 0
    fi

    local IFS=.
    local i ver1=($version1) ver2=($version2)

    # Fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done

    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 0
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 1
        fi
    done

    return 0
}