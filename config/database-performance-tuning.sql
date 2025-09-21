-- ================================================================================
-- BEV OSINT Framework - PostgreSQL Performance Tuning for Enterprise Scale
-- Target: 2000+ concurrent connections, <10ms query latency
-- ================================================================================

-- ================================================================================
-- MEMORY CONFIGURATION (for 64GB RAM server)
-- ================================================================================

-- Shared memory for caching (25% of RAM)
ALTER SYSTEM SET shared_buffers = '16GB';

-- Memory for query workspace (per connection)
ALTER SYSTEM SET work_mem = '256MB';

-- Memory for maintenance operations
ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- Effective cache size hint (75% of RAM)
ALTER SYSTEM SET effective_cache_size = '48GB';

-- Memory for hash tables
ALTER SYSTEM SET hash_mem_multiplier = 2;

-- Huge pages configuration
ALTER SYSTEM SET huge_pages = 'try';

-- ================================================================================
-- CONNECTION POOLING & MANAGEMENT
-- ================================================================================

-- Maximum connections
ALTER SYSTEM SET max_connections = 500;

-- Reserved superuser connections
ALTER SYSTEM SET superuser_reserved_connections = 5;

-- Maximum prepared transactions
ALTER SYSTEM SET max_prepared_transactions = 100;

-- Connection timeout
ALTER SYSTEM SET authentication_timeout = '30s';

-- Statement timeout (prevent long-running queries)
ALTER SYSTEM SET statement_timeout = '30s';

-- Idle session timeout
ALTER SYSTEM SET idle_in_transaction_session_timeout = '5min';

-- ================================================================================
-- PARALLEL QUERY EXECUTION
-- ================================================================================

-- Enable parallel queries
ALTER SYSTEM SET max_parallel_workers_per_gather = 8;
ALTER SYSTEM SET max_parallel_workers = 16;
ALTER SYSTEM SET max_parallel_maintenance_workers = 8;
ALTER SYSTEM SET parallel_leader_participation = on;

-- Parallel query cost thresholds
ALTER SYSTEM SET parallel_setup_cost = 100;
ALTER SYSTEM SET parallel_tuple_cost = 0.01;
ALTER SYSTEM SET min_parallel_table_scan_size = '8MB';
ALTER SYSTEM SET min_parallel_index_scan_size = '512kB';

-- Force parallel execution for large tables
ALTER SYSTEM SET force_parallel_mode = off; -- Use 'on' for testing only

-- ================================================================================
-- QUERY PLANNER OPTIMIZATION
-- ================================================================================

-- Cost model tuning for SSDs
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET cpu_tuple_cost = 0.01;
ALTER SYSTEM SET cpu_index_tuple_cost = 0.005;
ALTER SYSTEM SET cpu_operator_cost = 0.0025;

-- Planner method configuration
ALTER SYSTEM SET enable_seqscan = on;
ALTER SYSTEM SET enable_indexscan = on;
ALTER SYSTEM SET enable_indexonlyscan = on;
ALTER SYSTEM SET enable_bitmapscan = on;
ALTER SYSTEM SET enable_tidscan = on;
ALTER SYSTEM SET enable_sort = on;
ALTER SYSTEM SET enable_hashagg = on;
ALTER SYSTEM SET enable_hashjoin = on;
ALTER SYSTEM SET enable_nestloop = on;
ALTER SYSTEM SET enable_mergejoin = on;
ALTER SYSTEM SET enable_material = on;
ALTER SYSTEM SET enable_partitionwise_join = on;
ALTER SYSTEM SET enable_partitionwise_aggregate = on;

-- JIT compilation
ALTER SYSTEM SET jit = on;
ALTER SYSTEM SET jit_above_cost = 100000;
ALTER SYSTEM SET jit_inline_above_cost = 500000;
ALTER SYSTEM SET jit_optimize_above_cost = 500000;

-- ================================================================================
-- WRITE PERFORMANCE OPTIMIZATION
-- ================================================================================

-- WAL configuration
ALTER SYSTEM SET wal_level = 'replica';
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET wal_compression = on;
ALTER SYSTEM SET wal_log_hints = on;
ALTER SYSTEM SET wal_writer_delay = '200ms';
ALTER SYSTEM SET wal_writer_flush_after = '1MB';

-- Checkpointing
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_wal_size = '16GB';
ALTER SYSTEM SET min_wal_size = '2GB';

-- Background writer
ALTER SYSTEM SET bgwriter_delay = '200ms';
ALTER SYSTEM SET bgwriter_lru_maxpages = 1000;
ALTER SYSTEM SET bgwriter_lru_multiplier = 4.0;
ALTER SYSTEM SET bgwriter_flush_after = '512kB';

-- Commit behavior
ALTER SYSTEM SET synchronous_commit = 'local';
ALTER SYSTEM SET commit_delay = 0;
ALTER SYSTEM SET commit_siblings = 5;

-- ================================================================================
-- VACUUM & MAINTENANCE
-- ================================================================================

-- Autovacuum configuration
ALTER SYSTEM SET autovacuum = on;
ALTER SYSTEM SET autovacuum_max_workers = 8;
ALTER SYSTEM SET autovacuum_naptime = '30s';
ALTER SYSTEM SET autovacuum_vacuum_threshold = 50;
ALTER SYSTEM SET autovacuum_vacuum_scale_factor = 0.05;
ALTER SYSTEM SET autovacuum_analyze_threshold = 50;
ALTER SYSTEM SET autovacuum_analyze_scale_factor = 0.05;
ALTER SYSTEM SET autovacuum_vacuum_cost_delay = '2ms';
ALTER SYSTEM SET autovacuum_vacuum_cost_limit = 10000;

-- Vacuum memory
ALTER SYSTEM SET vacuum_cost_delay = 0;
ALTER SYSTEM SET vacuum_cost_limit = 10000;
ALTER SYSTEM SET autovacuum_work_mem = '1GB';

-- ================================================================================
-- STATISTICS & MONITORING
-- ================================================================================

-- Statistics collection
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;
ALTER SYSTEM SET track_functions = 'all';
ALTER SYSTEM SET track_activity_query_size = 4096;

-- Statistics target for better plans
ALTER SYSTEM SET default_statistics_target = 500;

-- Query performance insights
ALTER SYSTEM SET log_min_duration_statement = '100ms';
ALTER SYSTEM SET log_checkpoints = on;
ALTER SYSTEM SET log_connections = off;
ALTER SYSTEM SET log_disconnections = off;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_temp_files = 0;
ALTER SYSTEM SET log_autovacuum_min_duration = '100ms';

-- pg_stat_statements extension
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET pg_stat_statements.max = 10000;
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.save = on;

-- ================================================================================
-- RELOAD CONFIGURATION
-- ================================================================================

-- Apply configuration changes
SELECT pg_reload_conf();

-- ================================================================================
-- OSINT-SPECIFIC OPTIMIZATIONS
-- ================================================================================

-- Create specialized indexes for OSINT queries
CREATE INDEX CONCURRENTLY idx_osint_data_timestamp
    ON osint_data(created_at DESC, updated_at DESC)
    WHERE deleted_at IS NULL;

CREATE INDEX CONCURRENTLY idx_osint_data_user_query
    ON osint_data(user_id, query_type, created_at DESC)
    WHERE status = 'completed';

CREATE INDEX CONCURRENTLY idx_osint_data_source
    ON osint_data(source_type, source_id)
    INCLUDE (data_hash, metadata);

-- Partial indexes for common queries
CREATE INDEX CONCURRENTLY idx_osint_active_investigations
    ON investigations(user_id, status)
    WHERE status IN ('active', 'pending');

CREATE INDEX CONCURRENTLY idx_osint_recent_alerts
    ON alerts(severity, created_at DESC)
    WHERE acknowledged = false AND created_at > NOW() - INTERVAL '7 days';

-- GIN indexes for JSONB columns
CREATE INDEX CONCURRENTLY idx_osint_metadata_gin
    ON osint_data USING gin(metadata);

CREATE INDEX CONCURRENTLY idx_osint_analysis_gin
    ON analysis_results USING gin(findings);

-- BRIN indexes for time-series data
CREATE INDEX CONCURRENTLY idx_osint_timeseries_brin
    ON osint_timeseries USING brin(timestamp)
    WITH (pages_per_range = 128);

-- ================================================================================
-- TABLE PARTITIONING FOR SCALE
-- ================================================================================

-- Partition main OSINT data table by date
CREATE TABLE osint_data_partitioned (
    LIKE osint_data INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE osint_data_2024_01 PARTITION OF osint_data_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE osint_data_2024_02 PARTITION OF osint_data_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add more partitions as needed...

-- Automatic partition creation function
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    FOR i IN 0..11 LOOP
        start_date := date_trunc('month', CURRENT_DATE) + (i || ' months')::interval;
        end_date := start_date + '1 month'::interval;
        partition_name := 'osint_data_' || to_char(start_date, 'YYYY_MM');

        IF NOT EXISTS (
            SELECT 1 FROM pg_class
            WHERE relname = partition_name
        ) THEN
            EXECUTE format('CREATE TABLE %I PARTITION OF osint_data_partitioned FOR VALUES FROM (%L) TO (%L)',
                partition_name, start_date, end_date);
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule partition creation
SELECT cron.schedule('create-partitions', '0 0 1 * *', 'SELECT create_monthly_partitions()');

-- ================================================================================
-- MATERIALIZED VIEWS FOR COMMON AGGREGATIONS
-- ================================================================================

-- User activity summary
CREATE MATERIALIZED VIEW mv_user_activity AS
SELECT
    user_id,
    DATE(created_at) as activity_date,
    COUNT(*) as query_count,
    COUNT(DISTINCT query_type) as unique_query_types,
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time
FROM osint_data
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY user_id, DATE(created_at)
WITH DATA;

CREATE UNIQUE INDEX ON mv_user_activity(user_id, activity_date);

-- Query performance metrics
CREATE MATERIALIZED VIEW mv_query_performance AS
SELECT
    query_type,
    source_type,
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as query_count,
    AVG(response_time_ms) as avg_response,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY response_time_ms) as median_response,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) as p99_response
FROM osint_data
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY query_type, source_type, DATE_TRUNC('hour', created_at)
WITH DATA;

CREATE UNIQUE INDEX ON mv_query_performance(query_type, source_type, hour);

-- Refresh materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_query_performance;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh
SELECT cron.schedule('refresh-views', '*/15 * * * *', 'SELECT refresh_materialized_views()');

-- ================================================================================
-- QUERY OPTIMIZATION FUNCTIONS
-- ================================================================================

-- Function to analyze and optimize slow queries
CREATE OR REPLACE FUNCTION analyze_slow_queries()
RETURNS TABLE(
    query_text text,
    calls bigint,
    total_time double precision,
    mean_time double precision,
    optimization_suggestion text
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.query,
        s.calls,
        s.total_exec_time,
        s.mean_exec_time,
        CASE
            WHEN s.mean_exec_time > 1000 THEN 'Consider adding indexes or partitioning'
            WHEN s.calls > 10000 THEN 'Consider caching or materialized view'
            WHEN s.rows / NULLIF(s.calls, 0) > 10000 THEN 'Consider pagination or limiting results'
            ELSE 'Query performance acceptable'
        END as suggestion
    FROM pg_stat_statements s
    WHERE s.mean_exec_time > 100  -- Focus on queries over 100ms
    ORDER BY s.total_exec_time DESC
    LIMIT 20;
END;
$$ LANGUAGE plpgsql;

-- ================================================================================
-- CONNECTION POOLING WITH PGBOUNCER CONFIGURATION
-- ================================================================================

-- Note: Add this to pgbouncer.ini

/*
[databases]
bev_osint = host=127.0.0.1 port=5432 dbname=osint pool_size=200 reserve_pool_size=50

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 2000
default_pool_size = 200
reserve_pool_size = 50
reserve_pool_timeout = 5
max_db_connections = 500
max_user_connections = 500
server_lifetime = 3600
server_idle_timeout = 600
server_connect_timeout = 15
server_login_retry = 15
query_timeout = 30
query_wait_timeout = 10
client_idle_timeout = 0
client_login_timeout = 60
autodb_idle_timeout = 3600
stats_period = 60
*/

-- ================================================================================
-- MONITORING QUERIES
-- ================================================================================

-- Check current connections and activity
CREATE VIEW v_connection_stats AS
SELECT
    datname,
    count(*) as connections,
    count(*) FILTER (WHERE state = 'active') as active,
    count(*) FILTER (WHERE state = 'idle') as idle,
    count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_transaction,
    count(*) FILTER (WHERE wait_event IS NOT NULL) as waiting
FROM pg_stat_activity
GROUP BY datname;

-- Check cache hit ratio
CREATE VIEW v_cache_stats AS
SELECT
    datname,
    round(100.0 * sum(blks_hit) / NULLIF(sum(blks_hit + blks_read), 0), 2) as cache_hit_ratio
FROM pg_stat_database
GROUP BY datname;

-- ================================================================================
-- APPLY ALL CHANGES
-- ================================================================================

-- Restart PostgreSQL to apply all changes
-- sudo systemctl restart postgresql

-- Verify settings
SELECT name, setting, unit FROM pg_settings
WHERE name IN ('shared_buffers', 'work_mem', 'max_connections', 'effective_cache_size')
ORDER BY name;