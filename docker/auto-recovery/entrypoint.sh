#!/bin/bash
# Auto-Recovery Service Entrypoint Script
# =======================================

set -euo pipefail

# Configuration
: "${CONFIG_PATH:=/app/config/auto_recovery.yaml}"
: "${LOG_LEVEL:=INFO}"
: "${WORKERS:=4}"
: "${PYTHONPATH:=/app}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [ENTRYPOINT]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

# Function to wait for service dependency
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-60}
    local count=0

    log "Waiting for $host:$port to be available..."

    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            error "Timeout waiting for $host:$port"
            return 1
        fi
        sleep 1
        ((count++))
    done

    success "$host:$port is available"
    return 0
}

# Function to check Docker socket access
check_docker_access() {
    log "Checking Docker socket access..."

    if [ ! -S /var/run/docker.sock ]; then
        error "Docker socket not found at /var/run/docker.sock"
        return 1
    fi

    if ! docker version >/dev/null 2>&1; then
        error "Cannot access Docker daemon"
        return 1
    fi

    success "Docker access confirmed"
    return 0
}

# Function to validate configuration
validate_config() {
    log "Validating configuration..."

    if [ ! -f "$CONFIG_PATH" ]; then
        error "Configuration file not found: $CONFIG_PATH"
        return 1
    fi

    # Validate YAML syntax
    if ! python -c "import yaml; yaml.safe_load(open('$CONFIG_PATH'))" 2>/dev/null; then
        error "Invalid YAML configuration file"
        return 1
    fi

    success "Configuration validated"
    return 0
}

# Function to check required environment variables
check_environment() {
    log "Checking environment variables..."

    local required_vars=(
        "POSTGRES_URI"
        "REDIS_PASSWORD"
    )

    local missing_vars=()

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -gt 0 ]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi

    success "Environment variables validated"
    return 0
}

# Function to setup directories
setup_directories() {
    log "Setting up directories..."

    local dirs=(
        "/app/logs"
        "/app/data"
        "/app/backups"
        "/app/snapshots"
        "/app/metrics"
    )

    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log "Created directory: $dir"
        fi
    done

    success "Directories setup complete"
}

# Function to initialize database
init_database() {
    log "Initializing database connection..."

    python -c "
import asyncio
import sys
sys.path.append('/app')
from infrastructure.auto_recovery import AutoRecoverySystem

async def test_db():
    try:
        system = AutoRecoverySystem()
        if system.postgres_engine:
            with system.postgres_engine.connect() as conn:
                conn.execute('SELECT 1')
            print('Database connection successful')
        else:
            print('No database configured')
        return True
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False

result = asyncio.run(test_db())
sys.exit(0 if result else 1)
"

    if [ $? -eq 0 ]; then
        success "Database connection verified"
    else
        error "Database connection failed"
        return 1
    fi
}

# Function to check Redis connection
check_redis() {
    log "Checking Redis connection..."

    python -c "
import redis
import os

try:
    redis_url = f\"redis://:{os.environ['REDIS_PASSWORD']}@redis:6379/11\"
    client = redis.from_url(redis_url)
    client.ping()
    print('Redis connection successful')
except Exception as e:
    print(f'Redis connection failed: {e}')
    exit(1)
"

    if [ $? -eq 0 ]; then
        success "Redis connection verified"
    else
        error "Redis connection failed"
        return 1
    fi
}

# Function to start health monitoring
start_health_monitoring() {
    log "Starting health monitoring in background..."

    python -c "
import asyncio
import signal
import sys
import logging
sys.path.append('/app')
from infrastructure.auto_recovery import AutoRecoverySystem

# Configure logging
logging.basicConfig(
    level=logging.${LOG_LEVEL},
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def main():
    system = AutoRecoverySystem(
        config_path='${CONFIG_PATH}',
        redis_url='redis://:\${REDIS_PASSWORD}@redis:6379/11'
    )

    # Setup signal handlers
    def signal_handler(signum, frame):
        asyncio.create_task(system.stop_monitoring())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await system.start_monitoring()
        await system.shutdown_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        await system.stop_monitoring()

if __name__ == '__main__':
    asyncio.run(main())
" &

    MONITORING_PID=$!
    echo $MONITORING_PID > /app/monitoring.pid
    success "Health monitoring started (PID: $MONITORING_PID)"
}

# Function to start metrics server
start_metrics_server() {
    log "Starting metrics server..."

    python -c "
import asyncio
from aiohttp import web, web_runner
import json
import sys
sys.path.append('/app')
from infrastructure.auto_recovery import AutoRecoverySystem

async def health_handler(request):
    return web.json_response({'status': 'healthy', 'service': 'auto-recovery'})

async def metrics_handler(request):
    try:
        system = AutoRecoverySystem(config_path='${CONFIG_PATH}')
        status = system.get_system_status()
        return web.json_response(status)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def create_app():
    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_get('/metrics', metrics_handler)
    app.router.add_get('/status', metrics_handler)
    return app

async def main():
    app = await create_app()
    runner = web_runner.AppRunner(app)
    await runner.setup()

    site = web_runner.TCPSite(runner, '0.0.0.0', 8000)
    await site.start()

    print('Metrics server started on port 8000')

    # Keep running
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())
" &

    METRICS_PID=$!
    echo $METRICS_PID > /app/metrics.pid
    success "Metrics server started (PID: $METRICS_PID)"
}

# Function to handle shutdown
shutdown() {
    log "Shutting down auto-recovery system..."

    # Stop monitoring process
    if [ -f /app/monitoring.pid ]; then
        MONITORING_PID=$(cat /app/monitoring.pid)
        if kill -0 "$MONITORING_PID" 2>/dev/null; then
            log "Stopping monitoring process (PID: $MONITORING_PID)"
            kill -TERM "$MONITORING_PID"
            wait "$MONITORING_PID" 2>/dev/null || true
        fi
        rm -f /app/monitoring.pid
    fi

    # Stop metrics server
    if [ -f /app/metrics.pid ]; then
        METRICS_PID=$(cat /app/metrics.pid)
        if kill -0 "$METRICS_PID" 2>/dev/null; then
            log "Stopping metrics server (PID: $METRICS_PID)"
            kill -TERM "$METRICS_PID"
            wait "$METRICS_PID" 2>/dev/null || true
        fi
        rm -f /app/metrics.pid
    fi

    success "Shutdown complete"
    exit 0
}

# Set up signal handlers
trap shutdown SIGINT SIGTERM

# Main execution
main() {
    log "Starting BEV Auto-Recovery System..."
    log "Configuration: $CONFIG_PATH"
    log "Log Level: $LOG_LEVEL"
    log "Workers: $WORKERS"

    # Pre-flight checks
    check_environment || exit 1
    validate_config || exit 1
    setup_directories || exit 1
    check_docker_access || exit 1

    # Wait for dependencies
    log "Waiting for dependencies..."
    wait_for_service postgres 5432 60 || exit 1
    wait_for_service redis 6379 60 || exit 1

    # Initialize connections
    init_database || exit 1
    check_redis || exit 1

    # Start services
    start_metrics_server
    start_health_monitoring

    success "Auto-Recovery System started successfully"
    log "Health endpoint: http://localhost:8000/health"
    log "Metrics endpoint: http://localhost:8000/metrics"

    # Keep the main process running
    while true; do
        # Check if monitoring process is still running
        if [ -f /app/monitoring.pid ]; then
            MONITORING_PID=$(cat /app/monitoring.pid)
            if ! kill -0 "$MONITORING_PID" 2>/dev/null; then
                error "Monitoring process died, restarting..."
                start_health_monitoring
            fi
        fi

        # Check if metrics server is still running
        if [ -f /app/metrics.pid ]; then
            METRICS_PID=$(cat /app/metrics.pid)
            if ! kill -0 "$METRICS_PID" 2>/dev/null; then
                error "Metrics server died, restarting..."
                start_metrics_server
            fi
        fi

        sleep 30
    done
}

# Execute main function
main "$@"