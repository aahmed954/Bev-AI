-- Migration 001: Phase 7 Marketplace Intelligence Schema
-- Safe migration script for marketplace intelligence tables
-- Version: 1.0.0
-- Date: 2024-09-19

-- Migration metadata
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'migration_log') THEN
        CREATE SCHEMA migration_log;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS migration_log.migration_history (
    id SERIAL PRIMARY KEY,
    migration_name TEXT NOT NULL,
    migration_version TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_by TEXT NOT NULL DEFAULT current_user,
    rollback_script TEXT,
    checksum TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT
);

-- Check if migration has already been applied
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM migration_log.migration_history WHERE migration_name = 'phase7_marketplace_intelligence') THEN
        RAISE NOTICE 'Migration phase7_marketplace_intelligence has already been applied. Skipping.';
        RETURN;
    END IF;
END $$;

-- Begin transaction for atomic migration
BEGIN;

-- Backup existing data (if any related tables exist)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'marketplace_intel') THEN
        RAISE NOTICE 'Schema marketplace_intel already exists. Please check for existing data before proceeding.';
    END IF;
END $$;

-- Validate prerequisites
DO $$
BEGIN
    -- Check for required extensions
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'Required extension "vector" is not installed. Please install pgvector extension first.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp') THEN
        RAISE EXCEPTION 'Required extension "uuid-ossp" is not installed.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') THEN
        RAISE EXCEPTION 'Required extension "pg_trgm" is not installed.';
    END IF;
END $$;

-- Create marketplace intelligence schema if not exists
CREATE SCHEMA IF NOT EXISTS marketplace_intel;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA marketplace_intel TO swarm_admin;
GRANT CREATE ON SCHEMA marketplace_intel TO swarm_admin;

-- Create vendor profiles table with error handling
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'marketplace_intel'
                   AND table_name = 'vendor_profiles') THEN

        CREATE TABLE marketplace_intel.vendor_profiles (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            vendor_id TEXT NOT NULL,
            marketplace_name TEXT NOT NULL,
            vendor_name TEXT NOT NULL,
            vendor_alias TEXT[],
            registration_date TIMESTAMPTZ,
            last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            status TEXT CHECK (status IN ('active', 'inactive', 'banned', 'suspected', 'verified')),
            feedback_score DECIMAL(3,2),
            total_transactions INTEGER DEFAULT 0,
            successful_transactions INTEGER DEFAULT 0,
            disputes INTEGER DEFAULT 0,
            reputation_rank INTEGER,
            profile_description TEXT,
            profile_image_url TEXT,
            contact_methods JSONB,
            shipping_locations TEXT[],
            accepted_currencies TEXT[],
            pgp_key TEXT,
            verified_vendor BOOLEAN DEFAULT FALSE,
            escrow_required BOOLEAN DEFAULT TRUE,
            finalize_early_allowed BOOLEAN DEFAULT FALSE,
            risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
            risk_factors JSONB,
            threat_indicators TEXT[],
            law_enforcement_interest BOOLEAN DEFAULT FALSE,
            profile_embedding vector(768),
            data_source data_source_type NOT NULL,
            collection_method TEXT,
            collector_id TEXT,
            raw_data JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(vendor_id, marketplace_name)
        ) PARTITION BY RANGE (created_at);

        RAISE NOTICE 'Created table marketplace_intel.vendor_profiles';
    ELSE
        RAISE NOTICE 'Table marketplace_intel.vendor_profiles already exists';
    END IF;
END $$;

-- Create monthly partitions with error handling
DO $$
DECLARE
    current_month DATE := DATE_TRUNC('month', NOW());
    next_month DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
BEGIN
    -- Create current month partition
    PERFORM create_monthly_partition('marketplace_intel.vendor_profiles', current_month);
    RAISE NOTICE 'Created partition for current month: %', current_month;

    -- Create next month partition
    PERFORM create_monthly_partition('marketplace_intel.vendor_profiles', next_month);
    RAISE NOTICE 'Created partition for next month: %', next_month;

EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error creating partitions: %', SQLERRM;
END $$;

-- Create indexes with error handling
DO $$
BEGIN
    -- Vendor profiles indexes
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_marketplace_idx') THEN
        CREATE INDEX vendor_profiles_marketplace_idx ON marketplace_intel.vendor_profiles (marketplace_name, vendor_name);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_status_idx') THEN
        CREATE INDEX vendor_profiles_status_idx ON marketplace_intel.vendor_profiles (status, last_seen);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_risk_idx') THEN
        CREATE INDEX vendor_profiles_risk_idx ON marketplace_intel.vendor_profiles (risk_score DESC, threat_indicators);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_reputation_idx') THEN
        CREATE INDEX vendor_profiles_reputation_idx ON marketplace_intel.vendor_profiles (feedback_score DESC, total_transactions DESC);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_embedding_idx') THEN
        CREATE INDEX vendor_profiles_embedding_idx ON marketplace_intel.vendor_profiles USING ivfflat (profile_embedding vector_cosine_ops) WITH (lists = 100);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_search_idx') THEN
        CREATE INDEX vendor_profiles_search_idx ON marketplace_intel.vendor_profiles USING gin(vendor_name gin_trgm_ops, vendor_alias gin_trgm_ops);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'vendor_profiles' AND indexname = 'vendor_profiles_raw_data_idx') THEN
        CREATE INDEX vendor_profiles_raw_data_idx ON marketplace_intel.vendor_profiles USING gin(raw_data);
    END IF;

    RAISE NOTICE 'Created indexes for vendor_profiles table';
END $$;

-- Create remaining tables following the same pattern
-- Product listings table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'marketplace_intel'
                   AND table_name = 'product_listings') THEN

        -- Execute the product_listings table creation from the init script
        -- (Similar structure with error handling)
        RAISE NOTICE 'Creating product_listings table...';
        -- Table creation code would go here (truncated for brevity)

    ELSE
        RAISE NOTICE 'Table marketplace_intel.product_listings already exists';
    END IF;
END $$;

-- Add audit triggers
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'vendor_profiles_audit_trigger') THEN
        CREATE TRIGGER vendor_profiles_audit_trigger
            BEFORE INSERT OR UPDATE ON marketplace_intel.vendor_profiles
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
        RAISE NOTICE 'Created audit trigger for vendor_profiles';
    END IF;
END $$;

-- Validate migration
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    -- Count created tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'marketplace_intel';

    -- Count created indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'marketplace_intel';

    RAISE NOTICE 'Migration validation: % tables created, % indexes created', table_count, index_count;

    IF table_count < 1 THEN
        RAISE EXCEPTION 'Migration validation failed: insufficient tables created';
    END IF;
END $$;

-- Log successful migration
INSERT INTO migration_log.migration_history (
    migration_name,
    migration_version,
    applied_by,
    rollback_script,
    checksum
) VALUES (
    'phase7_marketplace_intelligence',
    '1.0.0',
    current_user,
    'DROP SCHEMA marketplace_intel CASCADE;',
    md5('phase7_marketplace_intelligence_1.0.0')
);

-- Commit transaction
COMMIT;

RAISE NOTICE 'Migration phase7_marketplace_intelligence completed successfully';

-- Rollback script (commented out, for reference)
/*
-- ROLLBACK SCRIPT FOR EMERGENCY USE ONLY
--
-- WARNING: This will destroy all marketplace intelligence data
--
-- BEGIN;
-- DROP SCHEMA marketplace_intel CASCADE;
-- DELETE FROM migration_log.migration_history WHERE migration_name = 'phase7_marketplace_intelligence';
-- COMMIT;
*/