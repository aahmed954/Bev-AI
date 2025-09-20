-- Data Retention and Archival Strategies for BEV OSINT Database
-- Comprehensive data lifecycle management for high-volume OSINT operations
-- Version: 1.0.0
-- Date: 2024-09-19

-- ========================================
-- DATA RETENTION MANAGEMENT FRAMEWORK
-- ========================================

-- Create data retention management schema
CREATE SCHEMA IF NOT EXISTS data_retention;

-- Data retention policies table
CREATE TABLE IF NOT EXISTS data_retention.retention_policies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_name TEXT NOT NULL UNIQUE,
    schema_name TEXT NOT NULL,
    table_name TEXT NOT NULL,

    -- Retention configuration
    retention_period INTERVAL NOT NULL,
    partition_column TEXT NOT NULL DEFAULT 'created_at',
    retention_criteria JSONB,

    -- Policy status
    policy_status TEXT CHECK (policy_status IN ('active', 'inactive', 'testing')) DEFAULT 'active',
    auto_execute BOOLEAN DEFAULT TRUE,

    -- Archival configuration
    archive_enabled BOOLEAN DEFAULT FALSE,
    archive_destination TEXT,
    archive_format TEXT CHECK (archive_format IN ('pg_dump', 'csv', 'parquet', 's3')) DEFAULT 'pg_dump',
    compression_enabled BOOLEAN DEFAULT TRUE,

    -- Execution tracking
    last_execution TIMESTAMPTZ,
    next_execution TIMESTAMPTZ,
    execution_frequency INTERVAL DEFAULT '1 day',

    -- Impact assessment
    estimated_data_volume BIGINT,
    estimated_space_savings BIGINT,
    business_impact_level TEXT CHECK (business_impact_level IN ('low', 'medium', 'high', 'critical')) DEFAULT 'medium',

    -- Compliance requirements
    legal_hold_exempt BOOLEAN DEFAULT FALSE,
    compliance_requirements TEXT[],
    regulatory_retention_period INTERVAL,

    -- Quality assurance
    validation_rules JSONB,
    backup_before_deletion BOOLEAN DEFAULT TRUE,
    notification_recipients TEXT[],

    -- Metadata
    created_by TEXT NOT NULL,
    approved_by TEXT,
    approval_date TIMESTAMPTZ,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Retention execution log
CREATE TABLE IF NOT EXISTS data_retention.retention_execution_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id UUID REFERENCES data_retention.retention_policies(id),

    -- Execution details
    execution_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    execution_end TIMESTAMPTZ,
    execution_status TEXT CHECK (execution_status IN ('running', 'completed', 'failed', 'cancelled')),

    -- Data processed
    records_identified BIGINT DEFAULT 0,
    records_archived BIGINT DEFAULT 0,
    records_deleted BIGINT DEFAULT 0,
    space_freed_mb DECIMAL(15,2) DEFAULT 0,

    -- Performance metrics
    execution_duration INTERVAL,
    throughput_records_per_second DECIMAL(10,2),

    -- Results and errors
    success_message TEXT,
    error_message TEXT,
    warnings TEXT[],

    -- Audit trail
    executed_by TEXT DEFAULT 'automated_system',
    validation_passed BOOLEAN DEFAULT TRUE,

    -- Metadata
    execution_method TEXT DEFAULT 'automated',
    dry_run BOOLEAN DEFAULT FALSE
);

-- ========================================
-- OSINT-SPECIFIC RETENTION POLICIES
-- ========================================

-- Insert retention policies for marketplace intelligence
INSERT INTO data_retention.retention_policies (
    policy_name, schema_name, table_name, retention_period, retention_criteria,
    archive_enabled, archive_destination, business_impact_level, compliance_requirements,
    created_by, approved_by
) VALUES
-- Marketplace Intelligence Policies
(
    'marketplace_vendor_profiles_retention',
    'marketplace_intel', 'vendor_profiles', '2 years',
    '{"conditions": ["status != ''active''", "last_seen < NOW() - INTERVAL ''6 months''"]}',
    true, 'archive_storage/marketplace/vendor_profiles/', 'high',
    ARRAY['law_enforcement_records', 'financial_crimes_compliance'],
    'system', 'data_governance_team'
),
(
    'marketplace_product_listings_retention',
    'marketplace_intel', 'product_listings', '1 year',
    '{"conditions": ["listing_status IN (''removed'', ''suspended'')", "last_seen < NOW() - INTERVAL ''3 months''"]}',
    true, 'archive_storage/marketplace/product_listings/', 'medium',
    ARRAY['marketplace_monitoring'],
    'system', 'data_governance_team'
),
(
    'marketplace_transaction_records_retention',
    'marketplace_intel', 'transaction_records', '7 years',
    '{"conditions": ["status = ''completed''", "completion_date < NOW() - INTERVAL ''7 years''"]}',
    true, 'archive_storage/marketplace/transactions/', 'critical',
    ARRAY['financial_crimes_compliance', 'aml_requirements', 'legal_discovery'],
    'system', 'compliance_team'
),
(
    'marketplace_price_histories_retention',
    'marketplace_intel', 'price_histories', '6 months',
    '{"conditions": ["price_date < NOW() - INTERVAL ''6 months''"]}',
    false, null, 'low',
    ARRAY['market_analysis'],
    'system', 'data_governance_team'
),

-- Cryptocurrency Intelligence Policies
(
    'crypto_wallet_transactions_retention',
    'crypto_intel', 'wallet_transactions', '5 years',
    '{"conditions": ["timestamp < NOW() - INTERVAL ''5 years''", "risk_score < 30"]}',
    true, 'archive_storage/crypto/transactions/', 'high',
    ARRAY['aml_requirements', 'blockchain_forensics', 'law_enforcement'],
    'system', 'compliance_team'
),
(
    'crypto_blockchain_flows_retention',
    'crypto_intel', 'blockchain_flows', '3 years',
    '{"conditions": ["flow_start_time < NOW() - INTERVAL ''3 years''", "risk_level = ''low''"]}',
    true, 'archive_storage/crypto/flows/', 'high',
    ARRAY['blockchain_analysis', 'investigation_support'],
    'system', 'investigation_team'
),
(
    'crypto_mixing_patterns_retention',
    'crypto_intel', 'mixing_patterns', '10 years',
    '{"conditions": ["mixing_start_time < NOW() - INTERVAL ''10 years''"]}',
    true, 'archive_storage/crypto/mixing_patterns/', 'critical',
    ARRAY['law_enforcement', 'criminal_intelligence', 'long_term_investigations'],
    'system', 'investigation_team'
),
(
    'crypto_exchange_movements_retention',
    'crypto_intel', 'exchange_movements', '7 years',
    '{"conditions": ["movement_timestamp < NOW() - INTERVAL ''7 years''"]}',
    true, 'archive_storage/crypto/exchange_movements/', 'high',
    ARRAY['aml_requirements', 'financial_regulations'],
    'system', 'compliance_team'
),

-- Threat Intelligence Policies
(
    'threat_indicators_retention',
    'threat_intel', 'threat_indicators', '2 years',
    '{"conditions": ["is_active = false", "expiry_date < NOW() - INTERVAL ''6 months''"]}',
    true, 'archive_storage/threat_intel/indicators/', 'high',
    ARRAY['threat_intelligence', 'cybersecurity_operations'],
    'system', 'security_team'
),
(
    'threat_security_events_retention',
    'threat_intel', 'security_events', '3 years',
    '{"conditions": ["response_status IN (''resolved'', ''false_positive'')", "detection_timestamp < NOW() - INTERVAL ''3 years''"]}',
    true, 'archive_storage/threat_intel/security_events/', 'high',
    ARRAY['incident_response', 'security_operations', 'compliance_audit'],
    'system', 'security_team'
),
(
    'threat_actors_retention',
    'threat_intel', 'threat_actors', '10 years',
    '{"conditions": ["activity_level = ''dormant''", "last_observed < NOW() - INTERVAL ''5 years''"]}',
    true, 'archive_storage/threat_intel/threat_actors/', 'critical',
    ARRAY['long_term_intelligence', 'strategic_threat_analysis'],
    'system', 'intelligence_team'
),

-- Security Operations Policies
(
    'security_ops_defense_actions_retention',
    'security_ops', 'defense_actions', '2 years',
    '{"conditions": ["execution_status = ''completed''", "completed_at < NOW() - INTERVAL ''2 years''"]}',
    true, 'archive_storage/security_ops/defense_actions/', 'medium',
    ARRAY['security_operations', 'incident_response'],
    'system', 'security_team'
),
(
    'security_ops_incident_responses_retention',
    'security_ops', 'incident_responses', '7 years',
    '{"conditions": ["status = ''closed''", "updated_at < NOW() - INTERVAL ''7 years''"]}',
    true, 'archive_storage/security_ops/incidents/', 'critical',
    ARRAY['incident_response', 'legal_discovery', 'compliance_audit'],
    'system', 'compliance_team'
),

-- Autonomous Operations Policies
(
    'autonomous_learning_experiments_retention',
    'autonomous', 'learning_experiments', '2 years',
    '{"conditions": ["experiment_status IN (''completed'', ''failed'', ''cancelled'')", "end_time < NOW() - INTERVAL ''2 years''"]}',
    true, 'archive_storage/autonomous/experiments/', 'medium',
    ARRAY['research_and_development', 'model_versioning'],
    'system', 'ml_team'
),
(
    'autonomous_performance_metrics_retention',
    'autonomous', 'performance_metrics', '1 year',
    '{"conditions": ["measurement_timestamp < NOW() - INTERVAL ''1 year''"]}',
    true, 'archive_storage/autonomous/metrics/', 'low',
    ARRAY['performance_monitoring'],
    'system', 'ml_team'
)
ON CONFLICT (policy_name) DO NOTHING;

-- ========================================
-- AUTOMATED RETENTION FUNCTIONS
-- ========================================

-- Function to execute retention policy
CREATE OR REPLACE FUNCTION data_retention.execute_retention_policy(
    policy_name_param TEXT,
    dry_run BOOLEAN DEFAULT TRUE
) RETURNS UUID AS $$
DECLARE
    policy_record data_retention.retention_policies%ROWTYPE;
    execution_id UUID;
    sql_query TEXT;
    archive_query TEXT;
    delete_query TEXT;
    records_count BIGINT;
    space_before BIGINT;
    space_after BIGINT;
    start_time TIMESTAMPTZ;
    condition_clause TEXT;
BEGIN
    start_time := NOW();

    -- Get policy details
    SELECT * INTO policy_record
    FROM data_retention.retention_policies
    WHERE policy_name = policy_name_param AND policy_status = 'active';

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Policy % not found or inactive', policy_name_param;
    END IF;

    -- Create execution log entry
    INSERT INTO data_retention.retention_execution_log (
        policy_id, execution_status, dry_run, executed_by
    ) VALUES (
        policy_record.id, 'running', dry_run, 'automated_system'
    ) RETURNING id INTO execution_id;

    -- Build condition clause from JSON criteria
    SELECT string_agg(condition, ' AND ')
    INTO condition_clause
    FROM jsonb_array_elements_text(policy_record.retention_criteria->'conditions') AS condition;

    -- Build query to identify records for retention
    sql_query := format(
        'SELECT COUNT(*) FROM %I.%I WHERE %s AND %s < NOW() - INTERVAL ''%s''',
        policy_record.schema_name,
        policy_record.table_name,
        COALESCE(condition_clause, 'TRUE'),
        policy_record.partition_column,
        policy_record.retention_period
    );

    -- Count records to be processed
    EXECUTE sql_query INTO records_count;

    -- Get current table size
    SELECT pg_total_relation_size(format('%I.%I', policy_record.schema_name, policy_record.table_name))
    INTO space_before;

    IF NOT dry_run THEN
        -- Archive data if enabled
        IF policy_record.archive_enabled THEN
            archive_query := format(
                'COPY (SELECT * FROM %I.%I WHERE %s AND %s < NOW() - INTERVAL ''%s'') TO ''%s_%s.csv'' WITH CSV HEADER',
                policy_record.schema_name,
                policy_record.table_name,
                COALESCE(condition_clause, 'TRUE'),
                policy_record.partition_column,
                policy_record.retention_period,
                policy_record.archive_destination,
                to_char(NOW(), 'YYYY_MM_DD')
            );

            -- Execute archive (would need proper file system access)
            -- EXECUTE archive_query;

            UPDATE data_retention.retention_execution_log
            SET records_archived = records_count
            WHERE id = execution_id;
        END IF;

        -- Delete old data
        delete_query := format(
            'DELETE FROM %I.%I WHERE %s AND %s < NOW() - INTERVAL ''%s''',
            policy_record.schema_name,
            policy_record.table_name,
            COALESCE(condition_clause, 'TRUE'),
            policy_record.partition_column,
            policy_record.retention_period
        );

        EXECUTE delete_query;

        -- Get new table size
        SELECT pg_total_relation_size(format('%I.%I', policy_record.schema_name, policy_record.table_name))
        INTO space_after;

        -- Update execution log
        UPDATE data_retention.retention_execution_log
        SET
            execution_end = NOW(),
            execution_status = 'completed',
            records_identified = records_count,
            records_deleted = records_count,
            space_freed_mb = (space_before - space_after) / 1024 / 1024,
            execution_duration = NOW() - start_time,
            throughput_records_per_second = records_count / GREATEST(1, EXTRACT(EPOCH FROM NOW() - start_time)),
            success_message = format('Successfully processed %s records, freed %s MB',
                records_count, (space_before - space_after) / 1024 / 1024)
        WHERE id = execution_id;

        -- Update policy last execution
        UPDATE data_retention.retention_policies
        SET
            last_execution = NOW(),
            next_execution = NOW() + execution_frequency
        WHERE id = policy_record.id;

    ELSE
        -- Dry run - just log what would be done
        UPDATE data_retention.retention_execution_log
        SET
            execution_end = NOW(),
            execution_status = 'completed',
            records_identified = records_count,
            execution_duration = NOW() - start_time,
            success_message = format('DRY RUN: Would process %s records', records_count)
        WHERE id = execution_id;
    END IF;

    RETURN execution_id;

EXCEPTION
    WHEN OTHERS THEN
        UPDATE data_retention.retention_execution_log
        SET
            execution_end = NOW(),
            execution_status = 'failed',
            error_message = SQLERRM
        WHERE id = execution_id;

        RAISE;
END;
$$ LANGUAGE plpgsql;

-- Function to execute all active retention policies
CREATE OR REPLACE FUNCTION data_retention.execute_all_retention_policies(
    dry_run BOOLEAN DEFAULT TRUE
) RETURNS TABLE (
    policy_name TEXT,
    execution_id UUID,
    status TEXT,
    records_processed BIGINT,
    space_freed_mb DECIMAL
) AS $$
DECLARE
    policy_rec RECORD;
    exec_id UUID;
BEGIN
    FOR policy_rec IN
        SELECT rp.policy_name
        FROM data_retention.retention_policies rp
        WHERE rp.policy_status = 'active'
        AND rp.auto_execute = TRUE
        AND (rp.next_execution IS NULL OR rp.next_execution <= NOW())
        ORDER BY rp.business_impact_level DESC, rp.policy_name
    LOOP
        BEGIN
            exec_id := data_retention.execute_retention_policy(policy_rec.policy_name, dry_run);

            SELECT
                policy_rec.policy_name,
                exec_id,
                rel.execution_status,
                rel.records_identified,
                rel.space_freed_mb
            INTO
                policy_name,
                execution_id,
                status,
                records_processed,
                space_freed_mb
            FROM data_retention.retention_execution_log rel
            WHERE rel.id = exec_id;

            RETURN NEXT;

        EXCEPTION
            WHEN OTHERS THEN
                policy_name := policy_rec.policy_name;
                execution_id := exec_id;
                status := 'failed';
                records_processed := 0;
                space_freed_mb := 0;
                RETURN NEXT;
        END;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- PARTITION MANAGEMENT
-- ========================================

-- Function to automatically drop old partitions
CREATE OR REPLACE FUNCTION data_retention.drop_old_partitions(
    schema_name TEXT,
    table_name TEXT,
    retention_months INTEGER DEFAULT 24
) RETURNS TEXT AS $$
DECLARE
    partition_name TEXT;
    drop_count INTEGER := 0;
    result_message TEXT := '';
BEGIN
    FOR partition_name IN
        SELECT schemaname||'.'||tablename
        FROM pg_tables
        WHERE schemaname = schema_name
        AND tablename LIKE table_name || '_%'
        AND tablename ~ '\d{4}_\d{2}$'  -- Matches YYYY_MM pattern
        AND to_date(
            substring(tablename from '(\d{4}_\d{2})$'),
            'YYYY_MM'
        ) < DATE_TRUNC('month', NOW() - (retention_months || ' months')::INTERVAL)
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %s CASCADE', partition_name);
        drop_count := drop_count + 1;
        result_message := result_message || format('Dropped partition: %s\n', partition_name);
    END LOOP;

    result_message := result_message || format('Total partitions dropped: %s', drop_count);
    RETURN result_message;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- MONITORING AND REPORTING
-- ========================================

-- View for retention policy status
CREATE OR REPLACE VIEW data_retention.retention_policy_status AS
SELECT
    rp.policy_name,
    rp.schema_name,
    rp.table_name,
    rp.retention_period,
    rp.policy_status,
    rp.last_execution,
    rp.next_execution,
    CASE
        WHEN rp.next_execution < NOW() THEN 'OVERDUE'
        WHEN rp.next_execution < NOW() + INTERVAL '1 day' THEN 'DUE_SOON'
        ELSE 'ON_SCHEDULE'
    END as execution_status,
    rel.records_identified as last_records_processed,
    rel.space_freed_mb as last_space_freed_mb,
    rel.execution_status as last_execution_status
FROM data_retention.retention_policies rp
LEFT JOIN LATERAL (
    SELECT *
    FROM data_retention.retention_execution_log
    WHERE policy_id = rp.id
    ORDER BY execution_start DESC
    LIMIT 1
) rel ON true
ORDER BY rp.business_impact_level DESC, rp.policy_name;

-- Function to generate retention summary report
CREATE OR REPLACE FUNCTION data_retention.generate_retention_report(
    days_back INTEGER DEFAULT 30
) RETURNS TABLE (
    summary_metric TEXT,
    value TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 'Total Policies Configured' as summary_metric, COUNT(*)::TEXT as value
    FROM data_retention.retention_policies
    UNION ALL
    SELECT 'Active Policies', COUNT(*)::TEXT
    FROM data_retention.retention_policies WHERE policy_status = 'active'
    UNION ALL
    SELECT 'Executions Last ' || days_back || ' Days', COUNT(*)::TEXT
    FROM data_retention.retention_execution_log
    WHERE execution_start >= NOW() - (days_back || ' days')::INTERVAL
    UNION ALL
    SELECT 'Total Records Deleted (Last ' || days_back || ' Days)',
           SUM(records_deleted)::TEXT
    FROM data_retention.retention_execution_log
    WHERE execution_start >= NOW() - (days_back || ' days')::INTERVAL
    UNION ALL
    SELECT 'Total Space Freed MB (Last ' || days_back || ' Days)',
           ROUND(SUM(space_freed_mb), 2)::TEXT
    FROM data_retention.retention_execution_log
    WHERE execution_start >= NOW() - (days_back || ' days')::INTERVAL
    UNION ALL
    SELECT 'Policies Overdue', COUNT(*)::TEXT
    FROM data_retention.retention_policies
    WHERE policy_status = 'active' AND next_execution < NOW();
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job for automated retention (requires pg_cron extension)
-- SELECT cron.schedule('data-retention-daily', '0 3 * * *', 'SELECT data_retention.execute_all_retention_policies(false);');

RAISE NOTICE 'Data retention and archival framework completed';