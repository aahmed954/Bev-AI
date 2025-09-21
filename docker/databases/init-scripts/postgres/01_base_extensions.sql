-- Base PostgreSQL Extensions and Configurations
-- This script should run first to establish core extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";
CREATE EXTENSION IF NOT EXISTS "hstore";
CREATE EXTENSION IF NOT EXISTS "citext";
CREATE EXTENSION IF NOT EXISTS "ltree";

-- Create custom types
DO $$ BEGIN
    CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE confidence_level AS ENUM ('low', 'medium', 'high', 'verified');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE data_source_type AS ENUM ('darkweb', 'clearnet', 'telegram', 'discord', 'social', 'api', 'manual', 'automated');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE threat_category AS ENUM ('malware', 'fraud', 'data_breach', 'ransomware', 'phishing', 'scam', 'drug_trade', 'weapons', 'other');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Optimized configurations for high-performance OSINT workloads
ALTER SYSTEM SET shared_preload_libraries = 'vector,pg_stat_statements';
ALTER SYSTEM SET max_connections = 1000;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET default_statistics_target = 1000;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Enable statement tracking
ALTER SYSTEM SET pg_stat_statements.track = 'all';
ALTER SYSTEM SET pg_stat_statements.max = 10000;

-- Create audit function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        NEW.created_at = COALESCE(NEW.created_at, NOW());
        NEW.updated_at = NOW();
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        NEW.updated_at = NOW();
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Create partitioning helper function
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name TEXT,
    start_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_month TEXT;
    end_month TEXT;
BEGIN
    start_month := to_char(start_date, 'YYYY_MM');
    end_month := to_char(start_date + INTERVAL '1 month', 'YYYY_MM');
    partition_name := table_name || '_' || start_month;

    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
                    FOR VALUES FROM (%L) TO (%L)',
                   partition_name, table_name, start_date, start_date + INTERVAL '1 month');

    -- Create indexes on the partition
    EXECUTE format('CREATE INDEX IF NOT EXISTS %I ON %I (created_at)',
                   partition_name || '_created_at_idx', partition_name);
END;
$$ LANGUAGE plpgsql;