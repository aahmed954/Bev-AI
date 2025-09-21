-- Index Optimization Strategies for BEV OSINT Database
-- Comprehensive indexing strategy for high-performance OSINT operations
-- Version: 1.0.0
-- Date: 2024-09-19

-- ========================================
-- INDEX OPTIMIZATION FRAMEWORK
-- ========================================

-- Create performance monitoring schema
CREATE SCHEMA IF NOT EXISTS performance_monitoring;

-- Index usage tracking table
CREATE TABLE IF NOT EXISTS performance_monitoring.index_usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,
    index_name TEXT NOT NULL,
    index_type TEXT,

    -- Usage statistics
    total_scans BIGINT DEFAULT 0,
    index_scans BIGINT DEFAULT 0,
    sequential_scans BIGINT DEFAULT 0,
    index_hit_ratio DECIMAL(5,2),

    -- Performance metrics
    avg_scan_time DECIMAL(10,3),
    max_scan_time DECIMAL(10,3),
    total_scan_time DECIMAL(15,3),
    bloat_percentage DECIMAL(5,2),

    -- Size metrics
    index_size_mb DECIMAL(10,2),
    table_size_mb DECIMAL(10,2),
    size_ratio DECIMAL(5,2),

    -- Quality indicators
    effectiveness_score DECIMAL(3,2),
    maintenance_overhead DECIMAL(3,2),
    fragmentation_level DECIMAL(3,2),

    -- Temporal tracking
    measurement_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    measurement_period INTERVAL DEFAULT '1 hour',

    -- Recommendations
    optimization_suggestions TEXT[],
    action_required BOOLEAN DEFAULT FALSE,

    -- Metadata
    collected_by TEXT DEFAULT 'automated_monitor',
    notes TEXT
);

-- Create indexes for the monitoring table itself
CREATE INDEX IF NOT EXISTS index_usage_stats_table_idx ON performance_monitoring.index_usage_stats (schema_name, table_name);
CREATE INDEX IF NOT EXISTS index_usage_stats_timestamp_idx ON performance_monitoring.index_usage_stats (measurement_timestamp DESC);
CREATE INDEX IF NOT EXISTS index_usage_stats_effectiveness_idx ON performance_monitoring.index_usage_stats (effectiveness_score DESC, index_hit_ratio DESC);

-- ========================================
-- PERFORMANCE MONITORING FUNCTIONS
-- ========================================

-- Function to analyze index effectiveness
CREATE OR REPLACE FUNCTION performance_monitoring.analyze_index_effectiveness()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    index_name TEXT,
    effectiveness_score DECIMAL(3,2),
    recommendations TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH index_stats AS (
        SELECT
            schemaname::TEXT as schema_name,
            tablename::TEXT as table_name,
            indexname::TEXT as index_name,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
            pg_relation_size(indexrelid) as index_size_bytes
        FROM pg_stat_user_indexes
    ),
    table_stats AS (
        SELECT
            schemaname::TEXT as schema_name,
            tablename::TEXT as table_name,
            seq_scan,
            seq_tup_read,
            n_tup_ins + n_tup_upd + n_tup_del as modification_count,
            pg_relation_size(relid) as table_size_bytes
        FROM pg_stat_user_tables
    )
    SELECT
        i.schema_name,
        i.table_name,
        i.index_name,
        CASE
            WHEN i.idx_scan = 0 THEN 0.0
            WHEN t.seq_scan = 0 THEN 1.0
            ELSE LEAST(1.0, (i.idx_scan::DECIMAL / GREATEST(1, t.seq_scan + i.idx_scan)))
        END::DECIMAL(3,2) as effectiveness_score,
        CASE
            WHEN i.idx_scan = 0 THEN ARRAY['Consider dropping - never used']
            WHEN i.idx_scan < 10 THEN ARRAY['Low usage - review necessity']
            WHEN i.index_size_bytes > t.table_size_bytes * 0.5 THEN ARRAY['Large index - consider optimization']
            WHEN t.modification_count > i.idx_scan * 10 THEN ARRAY['High maintenance overhead']
            ELSE ARRAY['Performing well']
        END as recommendations
    FROM index_stats i
    JOIN table_stats t ON i.schema_name = t.schema_name AND i.table_name = t.table_name
    ORDER BY effectiveness_score ASC, i.index_size_bytes DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to identify missing indexes
CREATE OR REPLACE FUNCTION performance_monitoring.identify_missing_indexes()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    suggested_columns TEXT[],
    reason TEXT,
    priority INTEGER
) AS $$
BEGIN
    RETURN QUERY
    -- Analyze slow queries to suggest missing indexes
    WITH slow_queries AS (
        SELECT
            query,
            calls,
            total_time,
            mean_time,
            rows
        FROM pg_stat_statements
        WHERE mean_time > 100 -- queries taking more than 100ms on average
        ORDER BY mean_time DESC
        LIMIT 50
    )
    SELECT
        'threat_intel'::TEXT as schema_name,
        'threat_indicators'::TEXT as table_name,
        ARRAY['indicator_value', 'indicator_type', 'last_seen']::TEXT[] as suggested_columns,
        'Frequently queried columns in slow queries'::TEXT as reason,
        1 as priority
    WHERE EXISTS (SELECT 1 FROM slow_queries WHERE query LIKE '%threat_indicators%');
END;
$$ LANGUAGE plpgsql;

-- Function to monitor index bloat
CREATE OR REPLACE FUNCTION performance_monitoring.monitor_index_bloat()
RETURNS TABLE (
    schema_name TEXT,
    table_name TEXT,
    index_name TEXT,
    bloat_percentage DECIMAL(5,2),
    wasted_space_mb DECIMAL(10,2),
    action_needed TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH index_bloat AS (
        SELECT
            schemaname::TEXT as schema_name,
            tablename::TEXT as table_name,
            indexname::TEXT as index_name,
            pg_relation_size(indexrelid) as current_size,
            -- Simplified bloat calculation
            CASE
                WHEN pg_relation_size(indexrelid) > 0 THEN
                    ((pg_relation_size(indexrelid)::DECIMAL / GREATEST(1, (idx_scan + 1))) * 100)
                ELSE 0
            END as estimated_bloat_pct
        FROM pg_stat_user_indexes
    )
    SELECT
        ib.schema_name,
        ib.table_name,
        ib.index_name,
        LEAST(100.0, ib.estimated_bloat_pct)::DECIMAL(5,2) as bloat_percentage,
        (ib.current_size * (ib.estimated_bloat_pct / 100) / 1024 / 1024)::DECIMAL(10,2) as wasted_space_mb,
        CASE
            WHEN ib.estimated_bloat_pct > 80 THEN 'REINDEX URGENTLY'
            WHEN ib.estimated_bloat_pct > 50 THEN 'Schedule REINDEX'
            WHEN ib.estimated_bloat_pct > 20 THEN 'Monitor closely'
            ELSE 'Healthy'
        END as action_needed
    FROM index_bloat ib
    WHERE ib.current_size > 1024 * 1024 -- Only indexes larger than 1MB
    ORDER BY ib.estimated_bloat_pct DESC;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- SPECIALIZED INDEX STRATEGIES
-- ========================================

-- High-performance composite indexes for OSINT workloads
DO $$
BEGIN
    -- Marketplace intelligence optimized indexes

    -- Multi-column index for vendor risk analysis
    CREATE INDEX CONCURRENTLY IF NOT EXISTS vendor_profiles_risk_analysis_idx
    ON marketplace_intel.vendor_profiles (risk_score DESC, last_seen DESC, marketplace_name, status)
    WHERE risk_score > 50;

    -- Partial index for active high-risk vendors
    CREATE INDEX CONCURRENTLY IF NOT EXISTS vendor_profiles_active_high_risk_idx
    ON marketplace_intel.vendor_profiles (marketplace_name, vendor_name, feedback_score)
    WHERE status = 'active' AND risk_score > 70;

    -- Product listings price optimization index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS product_listings_price_optimization_idx
    ON marketplace_intel.product_listings (marketplace_name, product_category, currency, price DESC, last_seen DESC)
    WHERE listing_status = 'active';

    -- Transaction analysis index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS transaction_records_analysis_idx
    ON marketplace_intel.transaction_records (marketplace_name, status, transaction_date DESC, transaction_amount DESC)
    WHERE status IN ('completed', 'disputed');

    -- Crypto intelligence optimized indexes

    -- Blockchain flow analysis index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS blockchain_flows_analysis_idx
    ON crypto_intel.blockchain_flows (blockchain_network, flow_type, risk_level, total_amount DESC, flow_start_time DESC);

    -- Wallet transaction risk index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS wallet_transactions_risk_analysis_idx
    ON crypto_intel.wallet_transactions (blockchain_network, is_suspicious, risk_score DESC, amount DESC, timestamp DESC)
    WHERE risk_score > 30;

    -- Exchange movement compliance index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS exchange_movements_compliance_idx
    ON crypto_intel.exchange_movements (exchange_name, movement_type, risk_score DESC, movement_timestamp DESC)
    WHERE sanctions_list_match = TRUE OR ctr_reportable = TRUE;

    -- Threat intelligence optimized indexes

    -- Threat indicator active lookup index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS threat_indicators_active_lookup_idx
    ON threat_intel.threat_indicators (indicator_type, indicator_value, severity, confidence)
    WHERE is_active = TRUE AND expiry_date > NOW();

    -- Security event correlation index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS security_events_correlation_idx
    ON threat_intel.security_events (event_category, severity, source_ip, destination_ip, event_timestamp DESC)
    WHERE response_status NOT IN ('resolved', 'false_positive');

    -- Threat actor attribution index
    CREATE INDEX CONCURRENTLY IF NOT EXISTS threat_actors_attribution_idx
    ON threat_intel.threat_actors (actor_type, suspected_nationality, sophistication_level, activity_level, last_observed DESC);

    RAISE NOTICE 'Created specialized high-performance indexes';
END $$;

-- ========================================
-- VECTOR INDEX OPTIMIZATION
-- ========================================

-- Optimize vector indexes for similarity search performance
DO $$
BEGIN
    -- Marketplace vendor profile similarity
    CREATE INDEX CONCURRENTLY IF NOT EXISTS vendor_profiles_embedding_optimized_idx
    ON marketplace_intel.vendor_profiles
    USING ivfflat (profile_embedding vector_cosine_ops)
    WITH (lists = 500);  -- Increased lists for better performance

    -- Product listing similarity
    CREATE INDEX CONCURRENTLY IF NOT EXISTS product_listings_embedding_optimized_idx
    ON marketplace_intel.product_listings
    USING ivfflat (listing_embedding vector_cosine_ops)
    WITH (lists = 300);

    -- Threat indicator similarity
    CREATE INDEX CONCURRENTLY IF NOT EXISTS threat_indicators_embedding_optimized_idx
    ON threat_intel.threat_indicators
    USING ivfflat (indicator_embedding vector_cosine_ops)
    WITH (lists = 200);

    -- Security event similarity
    CREATE INDEX CONCURRENTLY IF NOT EXISTS security_events_embedding_optimized_idx
    ON threat_intel.security_events
    USING ivfflat (event_embedding vector_cosine_ops)
    WITH (lists = 400);

    RAISE NOTICE 'Optimized vector indexes for similarity search';
END $$;

-- ========================================
-- TIME-SERIES OPTIMIZATION
-- ========================================

-- Optimize time-series queries with BRIN indexes for large tables
DO $$
BEGIN
    -- BRIN indexes for time-series data (very space efficient for ordered data)
    CREATE INDEX CONCURRENTLY IF NOT EXISTS wallet_transactions_timestamp_brin_idx
    ON crypto_intel.wallet_transactions USING BRIN (timestamp, created_at);

    CREATE INDEX CONCURRENTLY IF NOT EXISTS security_events_timestamp_brin_idx
    ON threat_intel.security_events USING BRIN (event_timestamp, detection_timestamp);

    CREATE INDEX CONCURRENTLY IF NOT EXISTS transaction_records_timestamp_brin_idx
    ON marketplace_intel.transaction_records USING BRIN (transaction_date, created_at);

    RAISE NOTICE 'Created BRIN indexes for time-series optimization';
END $$;

-- ========================================
-- FULL-TEXT SEARCH OPTIMIZATION
-- ========================================

-- Optimize full-text search indexes
DO $$
BEGIN
    -- Enhanced full-text search for threat descriptions
    CREATE INDEX CONCURRENTLY IF NOT EXISTS security_events_fulltext_idx
    ON threat_intel.security_events
    USING gin(to_tsvector('english', event_description || ' ' || COALESCE(analyst_notes, '')));

    -- Product listing search optimization
    CREATE INDEX CONCURRENTLY IF NOT EXISTS product_listings_fulltext_idx
    ON marketplace_intel.product_listings
    USING gin(to_tsvector('english', product_title || ' ' || COALESCE(description, '')));

    -- Vendor profile search
    CREATE INDEX CONCURRENTLY IF NOT EXISTS vendor_profiles_fulltext_idx
    ON marketplace_intel.vendor_profiles
    USING gin(to_tsvector('english', vendor_name || ' ' || COALESCE(profile_description, '')));

    RAISE NOTICE 'Created optimized full-text search indexes';
END $$;

-- ========================================
-- AUTOMATED INDEX MAINTENANCE
-- ========================================

-- Function to automatically maintain indexes
CREATE OR REPLACE FUNCTION performance_monitoring.auto_maintain_indexes()
RETURNS TEXT AS $$
DECLARE
    maintenance_report TEXT := '';
    rec RECORD;
BEGIN
    -- Analyze index bloat and perform maintenance
    FOR rec IN
        SELECT schema_name, table_name, index_name, bloat_percentage, action_needed
        FROM performance_monitoring.monitor_index_bloat()
        WHERE action_needed IN ('REINDEX URGENTLY', 'Schedule REINDEX')
    LOOP
        IF rec.action_needed = 'REINDEX URGENTLY' THEN
            BEGIN
                EXECUTE format('REINDEX INDEX CONCURRENTLY %I.%I', rec.schema_name, rec.index_name);
                maintenance_report := maintenance_report || format('REINDEXED: %s.%s (bloat: %s%%)\n',
                    rec.schema_name, rec.index_name, rec.bloat_percentage);
            EXCEPTION
                WHEN OTHERS THEN
                    maintenance_report := maintenance_report || format('FAILED to reindex %s.%s: %s\n',
                        rec.schema_name, rec.index_name, SQLERRM);
            END;
        END IF;
    END LOOP;

    -- Update statistics for all OSINT tables
    ANALYZE marketplace_intel.vendor_profiles;
    ANALYZE marketplace_intel.product_listings;
    ANALYZE crypto_intel.wallet_transactions;
    ANALYZE threat_intel.threat_indicators;
    ANALYZE threat_intel.security_events;

    maintenance_report := maintenance_report || 'Updated table statistics\n';

    RETURN maintenance_report;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- PERFORMANCE MONITORING VIEWS
-- ========================================

-- View for index performance overview
CREATE OR REPLACE VIEW performance_monitoring.index_performance_summary AS
SELECT
    schemaname as schema_name,
    tablename as table_name,
    indexname as index_name,
    idx_scan as scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    CASE
        WHEN idx_scan = 0 THEN 'UNUSED'
        WHEN idx_scan < 100 THEN 'LOW_USAGE'
        WHEN idx_scan < 1000 THEN 'MODERATE_USAGE'
        ELSE 'HIGH_USAGE'
    END as usage_category,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    CASE
        WHEN idx_tup_read > 0 THEN
            ROUND((idx_tup_fetch::DECIMAL / idx_tup_read) * 100, 2)
        ELSE 0
    END as hit_ratio_percent
FROM pg_stat_user_indexes
WHERE schemaname IN ('marketplace_intel', 'crypto_intel', 'threat_intel', 'security_ops', 'autonomous')
ORDER BY idx_scan DESC;

-- View for table performance metrics
CREATE OR REPLACE VIEW performance_monitoring.table_performance_summary AS
SELECT
    schemaname as schema_name,
    tablename as table_name,
    seq_scan as sequential_scans,
    seq_tup_read as seq_tuples_read,
    idx_scan as index_scans,
    idx_tup_fetch as idx_tuples_fetched,
    n_tup_ins as inserts,
    n_tup_upd as updates,
    n_tup_del as deletes,
    CASE
        WHEN seq_scan + idx_scan = 0 THEN 0
        ELSE ROUND((idx_scan::DECIMAL / (seq_scan + idx_scan)) * 100, 2)
    END as index_usage_percent,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size,
    pg_size_pretty(pg_relation_size(relid)) as table_size
FROM pg_stat_user_tables
WHERE schemaname IN ('marketplace_intel', 'crypto_intel', 'threat_intel', 'security_ops', 'autonomous')
ORDER BY pg_total_relation_size(relid) DESC;

-- Create automated monitoring job (requires pg_cron extension)
-- SELECT cron.schedule('index-maintenance', '0 2 * * *', 'SELECT performance_monitoring.auto_maintain_indexes();');

RAISE NOTICE 'Index optimization strategies and performance monitoring framework completed';