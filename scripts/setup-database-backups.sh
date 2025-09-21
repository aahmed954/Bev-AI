#!/bin/bash

# BEV OSINT Framework - Automated Database Backup System
# Comprehensive backup solution for all databases in the BEV platform

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_ROOT="${BACKUP_ROOT:-/opt/bev/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION_LEVEL="${COMPRESSION_LEVEL:-6}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

echo "💾 Setting up BEV Database Backup System..."

# Create backup directory structure
setup_backup_directories() {
    log "Creating backup directory structure..."

    sudo mkdir -p "$BACKUP_ROOT"/{postgresql,neo4j,redis,elasticsearch,influxdb,mongodb,vector-dbs}
    sudo mkdir -p "$BACKUP_ROOT"/logs
    sudo mkdir -p "$BACKUP_ROOT"/scripts
    sudo mkdir -p "$BACKUP_ROOT"/configs
    sudo chmod -R 755 "$BACKUP_ROOT"

    success "Backup directories created"
}

# Create PostgreSQL backup script
create_postgresql_backup() {
    log "Creating PostgreSQL backup script..."

    cat > "$BACKUP_ROOT/scripts/backup-postgresql.sh" << 'EOF'
#!/bin/bash

# PostgreSQL Backup Script for BEV OSINT Platform

set -euo pipefail

BACKUP_DIR="/opt/bev/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Database connection settings
POSTGRES_HOST="${POSTGRES_HOST:-thanos}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-researcher}"
POSTGRES_DB="${POSTGRES_DB:-osint}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/../logs/postgresql-backup.log"
}

backup_database() {
    local db_name="$1"
    local backup_file="$BACKUP_DIR/${db_name}_${DATE}.sql.gz"

    log "Starting backup of database: $db_name"

    # Create backup with pg_dump
    if docker exec bev_postgres pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$db_name" \
        --verbose \
        --no-password \
        --format=custom \
        --compress=6 | gzip > "$backup_file"; then

        log "✅ PostgreSQL backup completed: $backup_file"

        # Verify backup
        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        else
            log "❌ WARNING: Backup file is empty"
            return 1
        fi
    else
        log "❌ PostgreSQL backup failed for database: $db_name"
        return 1
    fi
}

# Main PostgreSQL databases
databases=("osint" "intelowl" "bev_vectors" "bev_analytics")

for db in "${databases[@]}"; do
    backup_database "$db"
done

# Cleanup old backups
find "$BACKUP_DIR" -name "*.sql.gz" -mtime +$RETENTION_DAYS -delete
log "Old backups cleaned up (older than $RETENTION_DAYS days)"

log "PostgreSQL backup process completed"
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-postgresql.sh"
    success "PostgreSQL backup script created"
}

# Create Neo4j backup script
create_neo4j_backup() {
    log "Creating Neo4j backup script..."

    cat > "$BACKUP_ROOT/scripts/backup-neo4j.sh" << 'EOF'
#!/bin/bash

# Neo4j Backup Script for BEV OSINT Platform

set -euo pipefail

BACKUP_DIR="/opt/bev/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Neo4j connection settings
NEO4J_HOST="${NEO4J_HOST:-thanos}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-BevGraphMaster2024}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/../logs/neo4j-backup.log"
}

backup_neo4j() {
    local backup_file="$BACKUP_DIR/neo4j_${DATE}.tar.gz"

    log "Starting Neo4j backup..."

    # Stop Neo4j temporarily for consistent backup
    docker exec bev_neo4j neo4j stop

    # Create data backup
    if docker exec bev_neo4j tar -czf "/backups/neo4j_${DATE}.tar.gz" /data; then
        # Copy backup out of container
        docker cp "bev_neo4j:/backups/neo4j_${DATE}.tar.gz" "$backup_file"

        log "✅ Neo4j backup completed: $backup_file"

        # Verify backup
        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        else
            log "❌ WARNING: Backup file is empty"
            return 1
        fi
    else
        log "❌ Neo4j backup failed"
        return 1
    fi

    # Restart Neo4j
    docker exec bev_neo4j neo4j start
    log "Neo4j restarted"
}

backup_neo4j

# Cleanup old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
log "Old backups cleaned up (older than $RETENTION_DAYS days)"

log "Neo4j backup process completed"
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-neo4j.sh"
    success "Neo4j backup script created"
}

# Create Redis backup script
create_redis_backup() {
    log "Creating Redis backup script..."

    cat > "$BACKUP_ROOT/scripts/backup-redis.sh" << 'EOF'
#!/bin/bash

# Redis Backup Script for BEV OSINT Platform

set -euo pipefail

BACKUP_DIR="/opt/bev/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/../logs/redis-backup.log"
}

backup_redis() {
    local instance="$1"
    local backup_file="$BACKUP_DIR/redis_${instance}_${DATE}.rdb"

    log "Starting Redis backup for instance: $instance"

    # Force Redis to save current state
    docker exec "bev_redis_${instance}" redis-cli BGSAVE

    # Wait for background save to complete
    while [[ $(docker exec "bev_redis_${instance}" redis-cli LASTSAVE) -eq $(docker exec "bev_redis_${instance}" redis-cli LASTSAVE) ]]; do
        sleep 1
    done

    # Copy RDB file
    if docker cp "bev_redis_${instance}:/data/dump.rdb" "$backup_file"; then
        log "✅ Redis backup completed for $instance: $backup_file"

        # Verify backup
        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        else
            log "❌ WARNING: Backup file is empty"
            return 1
        fi
    else
        log "❌ Redis backup failed for instance: $instance"
        return 1
    fi
}

# Backup all Redis instances
redis_instances=("main" "cache" "sessions")

for instance in "${redis_instances[@]}"; do
    if docker ps | grep -q "bev_redis_${instance}"; then
        backup_redis "$instance"
    else
        log "⚠️  Redis instance not running: $instance"
    fi
done

# Cleanup old backups
find "$BACKUP_DIR" -name "*.rdb" -mtime +$RETENTION_DAYS -delete
log "Old backups cleaned up (older than $RETENTION_DAYS days)"

log "Redis backup process completed"
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-redis.sh"
    success "Redis backup script created"
}

# Create Elasticsearch backup script
create_elasticsearch_backup() {
    log "Creating Elasticsearch backup script..."

    cat > "$BACKUP_ROOT/scripts/backup-elasticsearch.sh" << 'EOF'
#!/bin/bash

# Elasticsearch Backup Script for BEV OSINT Platform

set -euo pipefail

BACKUP_DIR="/opt/bev/backups/elasticsearch"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# Elasticsearch connection settings
ES_HOST="${ES_HOST:-thanos}"
ES_PORT="${ES_PORT:-9200}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/../logs/elasticsearch-backup.log"
}

backup_elasticsearch() {
    local snapshot_name="bev_snapshot_${DATE}"
    local backup_file="$BACKUP_DIR/elasticsearch_${DATE}.tar.gz"

    log "Starting Elasticsearch backup..."

    # Create snapshot repository if not exists
    curl -X PUT "http://$ES_HOST:$ES_PORT/_snapshot/bev_backup_repo" \
        -H 'Content-Type: application/json' \
        -d '{
            "type": "fs",
            "settings": {
                "location": "/opt/elasticsearch/backups",
                "compress": true
            }
        }' || log "Repository may already exist"

    # Create snapshot
    if curl -X PUT "http://$ES_HOST:$ES_PORT/_snapshot/bev_backup_repo/$snapshot_name" \
        -H 'Content-Type: application/json' \
        -d '{
            "indices": "bev-*,osint-*,intel-*",
            "ignore_unavailable": true,
            "include_global_state": false
        }'; then

        log "Snapshot created: $snapshot_name"

        # Wait for snapshot completion
        while true; do
            local status=$(curl -s "http://$ES_HOST:$ES_PORT/_snapshot/bev_backup_repo/$snapshot_name" | \
                jq -r '.snapshots[0].state // "IN_PROGRESS"')

            if [[ "$status" == "SUCCESS" ]]; then
                log "✅ Snapshot completed successfully"
                break
            elif [[ "$status" == "FAILED" ]]; then
                log "❌ Snapshot failed"
                return 1
            else
                log "Snapshot in progress... ($status)"
                sleep 10
            fi
        done

        # Export snapshot data
        docker exec bev_elasticsearch tar -czf "/opt/elasticsearch/backups/${snapshot_name}.tar.gz" \
            "/opt/elasticsearch/backups"

        # Copy backup out of container
        docker cp "bev_elasticsearch:/opt/elasticsearch/backups/${snapshot_name}.tar.gz" "$backup_file"

        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        else
            log "❌ WARNING: Backup file is empty"
            return 1
        fi
    else
        log "❌ Failed to create Elasticsearch snapshot"
        return 1
    fi
}

backup_elasticsearch

# Cleanup old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
log "Old backups cleaned up (older than $RETENTION_DAYS days)"

log "Elasticsearch backup process completed"
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-elasticsearch.sh"
    success "Elasticsearch backup script created"
}

# Create vector databases backup script
create_vector_backup() {
    log "Creating vector databases backup script..."

    cat > "$BACKUP_ROOT/scripts/backup-vector-dbs.sh" << 'EOF'
#!/bin/bash

# Vector Databases Backup Script for BEV OSINT Platform

set -euo pipefail

BACKUP_DIR="/opt/bev/backups/vector-dbs"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BACKUP_DIR/../logs/vector-backup.log"
}

backup_qdrant() {
    local backup_file="$BACKUP_DIR/qdrant_${DATE}.tar.gz"

    log "Starting Qdrant backup..."

    if docker exec bev_qdrant tar -czf "/qdrant/storage/backups/qdrant_${DATE}.tar.gz" \
        /qdrant/storage; then

        docker cp "bev_qdrant:/qdrant/storage/backups/qdrant_${DATE}.tar.gz" "$backup_file"

        log "✅ Qdrant backup completed: $backup_file"

        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        fi
    else
        log "❌ Qdrant backup failed"
        return 1
    fi
}

backup_weaviate() {
    local backup_file="$BACKUP_DIR/weaviate_${DATE}.tar.gz"

    log "Starting Weaviate backup..."

    if docker exec bev_weaviate tar -czf "/var/lib/weaviate/backups/weaviate_${DATE}.tar.gz" \
        /var/lib/weaviate; then

        docker cp "bev_weaviate:/var/lib/weaviate/backups/weaviate_${DATE}.tar.gz" "$backup_file"

        log "✅ Weaviate backup completed: $backup_file"

        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        fi
    else
        log "❌ Weaviate backup failed"
        return 1
    fi
}

backup_milvus() {
    local backup_file="$BACKUP_DIR/milvus_${DATE}.tar.gz"

    log "Starting Milvus backup..."

    if docker exec bev_milvus tar -czf "/var/lib/milvus/backups/milvus_${DATE}.tar.gz" \
        /var/lib/milvus; then

        docker cp "bev_milvus:/var/lib/milvus/backups/milvus_${DATE}.tar.gz" "$backup_file"

        log "✅ Milvus backup completed: $backup_file"

        if [[ -s "$backup_file" ]]; then
            local size=$(du -h "$backup_file" | cut -f1)
            log "Backup size: $size"
        fi
    else
        log "❌ Milvus backup failed"
        return 1
    fi
}

# Backup all vector databases
vector_dbs=("qdrant" "weaviate" "milvus")

for db in "${vector_dbs[@]}"; do
    if docker ps | grep -q "bev_${db}"; then
        "backup_${db}"
    else
        log "⚠️  Vector database not running: $db"
    fi
done

# Cleanup old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
log "Old backups cleaned up (older than $RETENTION_DAYS days)"

log "Vector databases backup process completed"
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-vector-dbs.sh"
    success "Vector databases backup script created"
}

# Create master backup orchestrator
create_master_backup_script() {
    log "Creating master backup orchestrator..."

    cat > "$BACKUP_ROOT/scripts/backup-all-databases.sh" << 'EOF'
#!/bin/bash

# Master Database Backup Orchestrator for BEV OSINT Platform

set -euo pipefail

BACKUP_ROOT="/opt/bev/backups"
DATE=$(date +%Y%m%d_%H%M%S)
PARALLEL_JOBS=3

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$BACKUP_ROOT/logs/master-backup.log"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$BACKUP_ROOT/logs/master-backup.log"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$BACKUP_ROOT/logs/master-backup.log"
}

# Backup functions
run_backup() {
    local name="$1"
    local script="$2"

    log "Starting backup: $name"

    if bash "$script" > "$BACKUP_ROOT/logs/${name}-${DATE}.log" 2>&1; then
        success "$name backup completed"
        return 0
    else
        error "$name backup failed"
        return 1
    fi
}

# Main backup orchestration
main() {
    log "🚀 Starting BEV Database Backup Orchestration"
    log "============================================="

    local start_time=$(date +%s)
    local failed_backups=()

    # Create backup session directory
    local session_dir="$BACKUP_ROOT/sessions/backup_$DATE"
    mkdir -p "$session_dir"

    # Run backups in parallel groups
    log "Group 1: Primary databases (PostgreSQL, Neo4j)"
    (
        run_backup "postgresql" "$BACKUP_ROOT/scripts/backup-postgresql.sh" &
        run_backup "neo4j" "$BACKUP_ROOT/scripts/backup-neo4j.sh" &
        wait
    ) || failed_backups+=("primary-databases")

    log "Group 2: Cache and search (Redis, Elasticsearch)"
    (
        run_backup "redis" "$BACKUP_ROOT/scripts/backup-redis.sh" &
        run_backup "elasticsearch" "$BACKUP_ROOT/scripts/backup-elasticsearch.sh" &
        wait
    ) || failed_backups+=("cache-search")

    log "Group 3: Vector databases"
    run_backup "vector-dbs" "$BACKUP_ROOT/scripts/backup-vector-dbs.sh" || failed_backups+=("vector-dbs")

    # Generate backup report
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    cat > "$session_dir/backup-report.json" << EOL
{
  "session_id": "backup_$DATE",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "duration_seconds": $duration,
  "failed_backups": [$(printf '"%s",' "${failed_backups[@]}" | sed 's/,$//')]
}
EOL

    # Summary
    log "============================================="
    log "Backup session completed in ${duration} seconds"

    if [[ ${#failed_backups[@]} -eq 0 ]]; then
        success "🎉 All database backups completed successfully"
    else
        error "⚠️  ${#failed_backups[@]} backup(s) failed: ${failed_backups[*]}"
    fi

    # Calculate total backup size
    local total_size=$(du -sh "$BACKUP_ROOT" | cut -f1)
    log "Total backup storage used: $total_size"
}

# Cleanup function
cleanup_old_sessions() {
    log "Cleaning up old backup sessions..."
    find "$BACKUP_ROOT/sessions" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
    find "$BACKUP_ROOT/logs" -name "*.log" -mtime +14 -delete 2>/dev/null || true
}

# Pre-backup checks
pre_backup_checks() {
    log "Running pre-backup checks..."

    # Check disk space
    local backup_disk_usage=$(df "$BACKUP_ROOT" | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $backup_disk_usage -gt 80 ]]; then
        error "Backup disk usage is ${backup_disk_usage}% - backup may fail"
        return 1
    fi

    # Check Docker containers
    local running_containers=$(docker ps --format "{{.Names}}" | grep "bev_" | wc -l)
    if [[ $running_containers -lt 5 ]]; then
        error "Only $running_containers BEV containers running - backup may be incomplete"
        return 1
    fi

    success "Pre-backup checks passed"
    return 0
}

# Signal handlers
trap 'log "Backup interrupted"; exit 1' INT TERM

# Execute backup workflow
if pre_backup_checks; then
    main
    cleanup_old_sessions
else
    error "Pre-backup checks failed - aborting backup"
    exit 1
fi
EOF

    chmod +x "$BACKUP_ROOT/scripts/backup-all-databases.sh"
    success "Master backup orchestrator created"
}

# Create backup monitoring and alerts
create_backup_monitoring() {
    log "Creating backup monitoring system..."

    # Backup status checker
    cat > "$BACKUP_ROOT/scripts/check-backup-status.sh" << 'EOF'
#!/bin/bash

# Backup Status Monitoring for BEV OSINT Platform

set -euo pipefail

BACKUP_ROOT="/opt/bev/backups"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_recent_backups() {
    echo "📊 BEV Database Backup Status Report"
    echo "===================================="
    echo "$(date)"
    echo ""

    local databases=("postgresql" "neo4j" "redis" "elasticsearch" "vector-dbs")
    local issues=0

    for db in "${databases[@]}"; do
        local backup_dir="$BACKUP_ROOT/$db"

        if [[ -d "$backup_dir" ]]; then
            local recent_backup=$(find "$backup_dir" -name "*.gz" -o -name "*.rdb" -mtime -1 | head -1)

            if [[ -n "$recent_backup" ]]; then
                local age=$(find "$recent_backup" -printf '%TY-%Tm-%Td %TH:%TM')
                local size=$(du -h "$recent_backup" | cut -f1)
                echo -e "${GREEN}✅ $db${NC}: $age ($size)"
            else
                echo -e "${RED}❌ $db${NC}: No recent backup found"
                ((issues++))
            fi
        else
            echo -e "${YELLOW}⚠️  $db${NC}: Backup directory not found"
            ((issues++))
        fi
    done

    echo ""
    echo "Summary: $issues issue(s) found"

    # Check backup storage usage
    local storage_usage=$(du -sh "$BACKUP_ROOT" | cut -f1)
    echo "Total backup storage: $storage_usage"

    return $issues
}

# Check backup integrity
verify_backup_integrity() {
    echo ""
    echo "🔍 Backup Integrity Check"
    echo "========================"

    local corrupted=0

    # Check PostgreSQL backups
    local latest_pg=$(find "$BACKUP_ROOT/postgresql" -name "*.sql.gz" -mtime -1 | head -1)
    if [[ -n "$latest_pg" ]]; then
        if gzip -t "$latest_pg" 2>/dev/null; then
            echo -e "${GREEN}✅ PostgreSQL backup integrity: OK${NC}"
        else
            echo -e "${RED}❌ PostgreSQL backup integrity: CORRUPTED${NC}"
            ((corrupted++))
        fi
    fi

    # Check compressed backups
    for backup in $(find "$BACKUP_ROOT" -name "*.tar.gz" -mtime -1); do
        local db_name=$(basename "$(dirname "$backup")")
        if tar -tzf "$backup" >/dev/null 2>&1; then
            echo -e "${GREEN}✅ $db_name backup integrity: OK${NC}"
        else
            echo -e "${RED}❌ $db_name backup integrity: CORRUPTED${NC}"
            ((corrupted++))
        fi
    done

    echo ""
    echo "Integrity check complete: $corrupted corruption(s) found"
    return $corrupted
}

# Main function
main() {
    if check_recent_backups && verify_backup_integrity; then
        echo -e "${GREEN}🎉 All backups are healthy${NC}"
        exit 0
    else
        echo -e "${RED}⚠️  Backup issues detected${NC}"
        exit 1
    fi
}

main "$@"
EOF

    chmod +x "$BACKUP_ROOT/scripts/check-backup-status.sh"

    # Create systemd service for automated backups
    cat > config/bev-backup.service << EOF
[Unit]
Description=BEV Database Backup Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
User=root
Group=root
ExecStart=$BACKUP_ROOT/scripts/backup-all-databases.sh
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bev-backup

[Install]
WantedBy=multi-user.target
EOF

    # Create systemd timer for daily backups
    cat > config/bev-backup.timer << EOF
[Unit]
Description=Run BEV Database Backup Daily
Requires=bev-backup.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=1800

[Install]
WantedBy=timers.target
EOF

    success "Backup monitoring and scheduling created"
}

# Create backup restoration scripts
create_restore_scripts() {
    log "Creating database restoration scripts..."

    cat > "$BACKUP_ROOT/scripts/restore-database.sh" << 'EOF'
#!/bin/bash

# Database Restoration Script for BEV OSINT Platform

set -euo pipefail

BACKUP_ROOT="/opt/bev/backups"

usage() {
    echo "Usage: $0 <database_type> <backup_file>"
    echo ""
    echo "Database types: postgresql, neo4j, redis, elasticsearch, vector-dbs"
    echo ""
    echo "Examples:"
    echo "  $0 postgresql /opt/bev/backups/postgresql/osint_20231201_120000.sql.gz"
    echo "  $0 neo4j /opt/bev/backups/neo4j/neo4j_20231201_120000.tar.gz"
    exit 1
}

restore_postgresql() {
    local backup_file="$1"
    local db_name="${2:-osint}"

    echo "🔄 Restoring PostgreSQL database: $db_name"

    if [[ ! -f "$backup_file" ]]; then
        echo "❌ Backup file not found: $backup_file"
        return 1
    fi

    # Stop application services
    docker-compose -f docker-compose.complete.yml stop intelowl_uwsgi intelowl_nginx

    # Restore database
    zcat "$backup_file" | docker exec -i bev_postgres pg_restore \
        -h localhost -p 5432 -U researcher -d "$db_name" \
        --verbose --clean --if-exists

    # Restart services
    docker-compose -f docker-compose.complete.yml start intelowl_uwsgi intelowl_nginx

    echo "✅ PostgreSQL restoration completed"
}

restore_neo4j() {
    local backup_file="$1"

    echo "🔄 Restoring Neo4j database"

    if [[ ! -f "$backup_file" ]]; then
        echo "❌ Backup file not found: $backup_file"
        return 1
    fi

    # Stop Neo4j
    docker exec bev_neo4j neo4j stop

    # Clear existing data
    docker exec bev_neo4j rm -rf /data/databases
    docker exec bev_neo4j rm -rf /data/transactions

    # Restore from backup
    docker exec bev_neo4j tar -xzf "/backups/$(basename "$backup_file")" -C /

    # Start Neo4j
    docker exec bev_neo4j neo4j start

    echo "✅ Neo4j restoration completed"
}

# Main restoration logic
if [[ $# -lt 2 ]]; then
    usage
fi

database_type="$1"
backup_file="$2"

case "$database_type" in
    postgresql)
        restore_postgresql "$backup_file" "${3:-osint}"
        ;;
    neo4j)
        restore_neo4j "$backup_file"
        ;;
    *)
        echo "❌ Unsupported database type: $database_type"
        usage
        ;;
esac
EOF

    chmod +x "$BACKUP_ROOT/scripts/restore-database.sh"
    success "Database restoration scripts created"
}

# Main setup function
main() {
    echo "💾 BEV Database Backup System Setup"
    echo "==================================="
    echo ""

    setup_backup_directories
    echo ""

    create_postgresql_backup
    echo ""

    create_neo4j_backup
    echo ""

    create_redis_backup
    echo ""

    create_elasticsearch_backup
    echo ""

    create_vector_backup
    echo ""

    create_master_backup_script
    echo ""

    create_backup_monitoring
    echo ""

    create_restore_scripts
    echo ""

    success "✅ BEV Database Backup System setup complete!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Install systemd services: sudo cp config/bev-backup.* /etc/systemd/system/"
    echo "2. Enable daily backups: sudo systemctl enable bev-backup.timer"
    echo "3. Start backup timer: sudo systemctl start bev-backup.timer"
    echo "4. Test backup: $BACKUP_ROOT/scripts/backup-all-databases.sh"
    echo "5. Monitor status: $BACKUP_ROOT/scripts/check-backup-status.sh"
    echo ""
    echo "🔧 Manual commands:"
    echo "- Full backup: $BACKUP_ROOT/scripts/backup-all-databases.sh"
    echo "- Status check: $BACKUP_ROOT/scripts/check-backup-status.sh"
    echo "- Restore DB: $BACKUP_ROOT/scripts/restore-database.sh <type> <file>"
}

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi