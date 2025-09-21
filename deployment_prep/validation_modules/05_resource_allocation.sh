#!/bin/bash

# Resource Allocation Validation Module
# Gate 5: System resources and capacity validation

# Resource requirements per service
declare -A MEMORY_REQUIREMENTS=(
    ["postgres"]="512"      # MB
    ["redis"]="256"         # MB
    ["elasticsearch"]="2048" # MB
    ["neo4j"]="1024"        # MB
    ["rabbitmq"]="512"      # MB
    ["vault"]="128"         # MB
    ["tor"]="64"            # MB
    ["intelowl_uwsgi"]="1024" # MB
    ["intelowl_nginx"]="128"  # MB
    ["intelowl_celery_beat"]="256" # MB
    ["intelowl_celery_worker"]="2048" # MB
    ["grafana"]="256"       # MB
    ["prometheus"]="1024"   # MB
    ["mcp_server"]="512"    # MB
)

declare -A CPU_REQUIREMENTS=(
    ["postgres"]="1.0"      # cores
    ["redis"]="0.5"         # cores
    ["elasticsearch"]="2.0" # cores
    ["neo4j"]="1.0"         # cores
    ["rabbitmq"]="0.5"      # cores
    ["vault"]="0.2"         # cores
    ["tor"]="0.1"           # cores
    ["intelowl_uwsgi"]="2.0" # cores
    ["intelowl_nginx"]="0.5"  # cores
    ["intelowl_celery_beat"]="0.2" # cores
    ["intelowl_celery_worker"]="4.0" # cores
    ["grafana"]="0.5"       # cores
    ["prometheus"]="1.0"    # cores
    ["mcp_server"]="1.0"    # cores
)

declare -A STORAGE_REQUIREMENTS=(
    ["postgres"]="10"       # GB
    ["redis"]="2"           # GB
    ["elasticsearch"]="20"  # GB
    ["neo4j"]="5"           # GB
    ["rabbitmq"]="1"        # GB
    ["vault"]="1"           # GB
    ["tor"]="0.1"           # GB
    ["intelowl_data"]="50"  # GB
    ["grafana"]="2"         # GB
    ["prometheus"]="30"     # GB
    ["logs"]="20"           # GB
    ["backups"]="100"       # GB
)

# GPU requirements
declare -A GPU_REQUIREMENTS=(
    ["intelowl_celery_worker"]="1"  # GPU count
    ["mcp_server"]="0.5"            # GPU memory fraction
)

# Network bandwidth requirements
declare -A BANDWIDTH_REQUIREMENTS=(
    ["intelowl_nginx"]="100"        # Mbps
    ["elasticsearch"]="50"          # Mbps
    ["prometheus"]="25"             # Mbps
    ["external_apis"]="10"          # Mbps
)

resource_allocation() {
    log "INFO" "Validating resource allocation and capacity"
    local validation_passed=true

    # Test 1: Memory allocation validation
    if ! validate_memory_allocation; then
        validation_passed=false
    fi

    # Test 2: CPU allocation validation
    if ! validate_cpu_allocation; then
        validation_passed=false
    fi

    # Test 3: Storage allocation validation
    if ! validate_storage_allocation; then
        validation_passed=false
    fi

    # Test 4: GPU allocation validation
    if ! validate_gpu_allocation; then
        validation_passed=false
    fi

    # Test 5: Network bandwidth validation
    if ! validate_network_bandwidth; then
        validation_passed=false
    fi

    return $(if [[ "$validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_memory_allocation() {
    log "INFO" "Validating memory allocation"
    local memory_validation_passed=true

    # Calculate total memory requirements
    local total_required=0
    for service in "${!MEMORY_REQUIREMENTS[@]}"; do
        local requirement="${MEMORY_REQUIREMENTS[$service]}"
        total_required=$((total_required + requirement))
    done

    # Get available system memory
    local total_memory_mb=$(free -m | awk 'NR==2{print $2}')
    local available_memory_mb=$(free -m | awk 'NR==2{print $7}')

    log "INFO" "Total memory required: ${total_required}MB"
    log "INFO" "Total system memory: ${total_memory_mb}MB"
    log "INFO" "Available memory: ${available_memory_mb}MB"

    # Check if we have enough memory
    local memory_overhead=$((total_required / 10))  # 10% overhead
    local total_with_overhead=$((total_required + memory_overhead))

    if [[ $available_memory_mb -ge $total_with_overhead ]]; then
        log "SUCCESS" "Sufficient memory available: ${available_memory_mb}MB (required: ${total_with_overhead}MB)"
    else
        log "ERROR" "Insufficient memory: ${available_memory_mb}MB available, ${total_with_overhead}MB required"
        memory_validation_passed=false
    fi

    # Validate per-service memory limits
    for service in "${!MEMORY_REQUIREMENTS[@]}"; do
        local requirement="${MEMORY_REQUIREMENTS[$service]}"
        local percentage=$((requirement * 100 / total_memory_mb))

        if [[ $percentage -le 25 ]]; then
            log "SUCCESS" "Memory allocation for $service: ${requirement}MB (${percentage}% of total)"
        else
            log "WARN" "High memory allocation for $service: ${requirement}MB (${percentage}% of total)"
        fi
    done

    # Check swap configuration
    local swap_total=$(free -m | awk 'NR==3{print $2}')
    if [[ $swap_total -gt 0 ]]; then
        log "SUCCESS" "Swap space available: ${swap_total}MB"
    else
        log "WARN" "No swap space configured"
    fi

    return $(if [[ "$memory_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_cpu_allocation() {
    log "INFO" "Validating CPU allocation"
    local cpu_validation_passed=true

    # Calculate total CPU requirements
    local total_required_cpu=0
    for service in "${!CPU_REQUIREMENTS[@]}"; do
        local requirement="${CPU_REQUIREMENTS[$service]}"
        total_required_cpu=$(echo "$total_required_cpu + $requirement" | bc)
    done

    # Get available CPU cores
    local total_cores=$(nproc)
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')

    log "INFO" "Total CPU cores required: $total_required_cpu"
    log "INFO" "Available CPU cores: $total_cores"
    log "INFO" "Current CPU usage: ${cpu_usage}%"

    # Check if we have enough CPU capacity
    local cpu_overhead=$(echo "$total_required_cpu * 0.1" | bc)
    local total_with_overhead=$(echo "$total_required_cpu + $cpu_overhead" | bc)

    if (( $(echo "$total_cores >= $total_with_overhead" | bc -l) )); then
        log "SUCCESS" "Sufficient CPU capacity: $total_cores cores (required: $total_with_overhead)"
    else
        log "ERROR" "Insufficient CPU capacity: $total_cores cores available, $total_with_overhead required"
        cpu_validation_passed=false
    fi

    # Validate CPU governor settings
    local cpu_governor=""
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]]; then
        cpu_governor=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)
        log "INFO" "CPU governor: $cpu_governor"

        if [[ "$cpu_governor" == "performance" ]]; then
            log "SUCCESS" "CPU governor set to performance mode"
        else
            log "WARN" "CPU governor not set to performance mode (current: $cpu_governor)"
        fi
    fi

    return $(if [[ "$cpu_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_storage_allocation() {
    log "INFO" "Validating storage allocation"
    local storage_validation_passed=true

    # Calculate total storage requirements
    local total_required_gb=0
    for service in "${!STORAGE_REQUIREMENTS[@]}"; do
        local requirement="${STORAGE_REQUIREMENTS[$service]}"
        total_required_gb=$((total_required_gb + requirement))
    done

    # Get available disk space
    local available_gb=$(df / | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
    local total_disk_gb=$(df / | awk 'NR==2 {printf "%.0f", $2/1024/1024}')

    log "INFO" "Total storage required: ${total_required_gb}GB"
    log "INFO" "Available disk space: ${available_gb}GB"
    log "INFO" "Total disk space: ${total_disk_gb}GB"

    # Check if we have enough storage
    local storage_overhead=$((total_required_gb / 10))  # 10% overhead
    local total_with_overhead=$((total_required_gb + storage_overhead))

    if [[ $available_gb -ge $total_with_overhead ]]; then
        log "SUCCESS" "Sufficient storage available: ${available_gb}GB (required: ${total_with_overhead}GB)"
    else
        log "ERROR" "Insufficient storage: ${available_gb}GB available, ${total_with_overhead}GB required"
        storage_validation_passed=false
    fi

    # Check disk performance
    if command -v dd >/dev/null 2>&1; then
        log "INFO" "Testing disk performance..."
        local write_speed=$(dd if=/dev/zero of=/tmp/disk_test bs=1M count=100 2>&1 | grep -o '[0-9.]* MB/s' | tail -n1)
        rm -f /tmp/disk_test

        if [[ -n "$write_speed" ]]; then
            log "INFO" "Disk write speed: $write_speed"
            local speed_num=$(echo "$write_speed" | grep -o '[0-9.]*')
            if (( $(echo "$speed_num > 50" | bc -l) )); then
                log "SUCCESS" "Disk write performance adequate: $write_speed"
            else
                log "WARN" "Slow disk write performance: $write_speed (recommended: >50 MB/s)"
            fi
        fi
    fi

    # Validate Docker storage driver
    if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
        local storage_driver=$(docker info 2>/dev/null | grep "Storage Driver" | awk '{print $3}')
        log "INFO" "Docker storage driver: $storage_driver"

        case "$storage_driver" in
            "overlay2")
                log "SUCCESS" "Docker using optimal storage driver: $storage_driver"
                ;;
            "devicemapper")
                log "WARN" "Docker using older storage driver: $storage_driver (consider overlay2)"
                ;;
            *)
                log "INFO" "Docker storage driver: $storage_driver"
                ;;
        esac
    fi

    return $(if [[ "$storage_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_gpu_allocation() {
    log "INFO" "Validating GPU allocation"
    local gpu_validation_passed=true

    # Check if GPU is required
    if [[ ${#GPU_REQUIREMENTS[@]} -eq 0 ]]; then
        log "INFO" "No GPU requirements specified"
        return 0
    fi

    # Check GPU availability
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi --list-gpus | wc -l)
            local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)

            log "SUCCESS" "GPU detected: $gpu_count GPU(s) with ${gpu_memory}MB memory each"

            # Calculate GPU memory requirements
            local total_gpu_memory_required=0
            for service in "${!GPU_REQUIREMENTS[@]}"; do
                local requirement="${GPU_REQUIREMENTS[$service]}"

                if [[ "$requirement" == *"."* ]]; then
                    # Fractional GPU memory
                    local fraction=$(echo "$requirement" | bc -l)
                    local memory_required=$(echo "$gpu_memory * $fraction" | bc)
                    total_gpu_memory_required=$(echo "$total_gpu_memory_required + $memory_required" | bc)
                    log "INFO" "Service $service requires ${memory_required}MB GPU memory"
                else
                    # Whole GPU count
                    log "INFO" "Service $service requires $requirement GPU(s)"
                fi
            done

            if (( $(echo "$total_gpu_memory_required <= $gpu_memory" | bc -l) )); then
                log "SUCCESS" "Sufficient GPU memory: ${gpu_memory}MB (required: ${total_gpu_memory_required}MB)"
            else
                log "ERROR" "Insufficient GPU memory: ${gpu_memory}MB available, ${total_gpu_memory_required}MB required"
                gpu_validation_passed=false
            fi

            # Check CUDA compatibility
            local cuda_version=$(nvidia-smi | grep "CUDA Version" | grep -oE '[0-9]+\.[0-9]+')
            if [[ -n "$cuda_version" ]]; then
                log "SUCCESS" "CUDA version: $cuda_version"

                # Check minimum CUDA version (11.0+)
                if (( $(echo "$cuda_version >= 11.0" | bc -l) )); then
                    log "SUCCESS" "CUDA version meets requirements"
                else
                    log "WARN" "CUDA version may be too old: $cuda_version (recommended: 11.0+)"
                fi
            fi
        else
            log "ERROR" "NVIDIA drivers installed but GPU not accessible"
            gpu_validation_passed=false
        fi
    else
        log "WARN" "No NVIDIA GPU detected (GPU acceleration disabled)"
    fi

    return $(if [[ "$gpu_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_network_bandwidth() {
    log "INFO" "Validating network bandwidth"
    local bandwidth_validation_passed=true

    # Test internet connectivity speed
    if command -v curl >/dev/null 2>&1; then
        log "INFO" "Testing internet connectivity speed..."

        # Download speed test (simple)
        local start_time=$(date +%s.%N)
        curl -s -o /dev/null "http://speedtest.ftp.otenet.gr/files/test1Mb.db"
        local end_time=$(date +%s.%N)

        local duration=$(echo "$end_time - $start_time" | bc)
        local speed_mbps=$(echo "scale=2; 8 / $duration" | bc)  # 1MB file = 8Mb

        if (( $(echo "$speed_mbps > 10" | bc -l) )); then
            log "SUCCESS" "Internet bandwidth adequate: ${speed_mbps}Mbps"
        else
            log "WARN" "Internet bandwidth may be limited: ${speed_mbps}Mbps"
        fi
    fi

    # Check network interface configuration
    local interfaces=$(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | awk -F': ' '{print $2}')

    for interface in $interfaces; do
        if [[ "$interface" =~ ^(eth|enp|ens|wlan) ]]; then
            local interface_speed=""
            local speed_file="/sys/class/net/$interface/speed"

            if [[ -f "$speed_file" ]]; then
                interface_speed=$(cat "$speed_file" 2>/dev/null)
                if [[ -n "$interface_speed" ]] && [[ "$interface_speed" != "-1" ]]; then
                    log "SUCCESS" "Network interface $interface: ${interface_speed}Mbps"

                    if [[ $interface_speed -ge 1000 ]]; then
                        log "SUCCESS" "Gigabit ethernet available on $interface"
                    elif [[ $interface_speed -ge 100 ]]; then
                        log "SUCCESS" "Fast ethernet available on $interface"
                    else
                        log "WARN" "Limited bandwidth on $interface: ${interface_speed}Mbps"
                    fi
                else
                    log "INFO" "Interface $interface speed unknown"
                fi
            fi
        fi
    done

    # Calculate total bandwidth requirements
    local total_bandwidth=0
    for service in "${!BANDWIDTH_REQUIREMENTS[@]}"; do
        local requirement="${BANDWIDTH_REQUIREMENTS[$service]}"
        total_bandwidth=$((total_bandwidth + requirement))
    done

    log "INFO" "Total estimated bandwidth requirements: ${total_bandwidth}Mbps"

    return $(if [[ "$bandwidth_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

# Resource monitoring functions
generate_resource_summary() {
    cat << EOF
### System Resources

**Memory:**
- Total: $(free -h | awk 'NR==2{print $2}')
- Available: $(free -h | awk 'NR==2{print $7}')
- Used: $(free -h | awk 'NR==2{print $3}')

**CPU:**
- Cores: $(nproc)
- Load Average: $(uptime | awk -F'load average:' '{print $2}')
- Governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")

**Storage:**
- Available: $(df -h / | awk 'NR==2{print $4}')
- Used: $(df -h / | awk 'NR==2{print $3}')
- Total: $(df -h / | awk 'NR==2{print $2}')

**GPU:**
$(if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "- GPUs: $(nvidia-smi --list-gpus | wc -l)"
    echo "- Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)MB per GPU"
    echo "- CUDA: $(nvidia-smi | grep "CUDA Version" | grep -oE '[0-9]+\.[0-9]+' || echo "unknown")"
else
    echo "- No NVIDIA GPU detected"
fi)

**Network:**
$(ip link show | grep -E "^[0-9]+:" | grep -v "lo:" | while read line; do
    interface=$(echo "$line" | awk -F': ' '{print $2}')
    speed=$(cat "/sys/class/net/$interface/speed" 2>/dev/null || echo "unknown")
    echo "- $interface: ${speed}Mbps"
done)
EOF
}