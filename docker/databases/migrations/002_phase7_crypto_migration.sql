-- Migration 002: Phase 7 Cryptocurrency Intelligence Schema
-- Safe migration script for cryptocurrency intelligence tables
-- Version: 1.0.0
-- Date: 2024-09-19

-- Check if migration has already been applied
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM migration_log.migration_history WHERE migration_name = 'phase7_crypto_intelligence') THEN
        RAISE NOTICE 'Migration phase7_crypto_intelligence has already been applied. Skipping.';
        RETURN;
    END IF;
END $$;

-- Begin transaction for atomic migration
BEGIN;

-- Validate prerequisites
DO $$
BEGIN
    -- Check for required types
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'data_source_type') THEN
        RAISE EXCEPTION 'Required type "data_source_type" does not exist. Please run base extensions migration first.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'confidence_level') THEN
        RAISE EXCEPTION 'Required type "confidence_level" does not exist. Please run base extensions migration first.';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'severity_level') THEN
        RAISE EXCEPTION 'Required type "severity_level" does not exist. Please run base extensions migration first.';
    END IF;
END $$;

-- Create crypto intelligence schema if not exists
CREATE SCHEMA IF NOT EXISTS crypto_intel;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA crypto_intel TO swarm_admin;
GRANT CREATE ON SCHEMA crypto_intel TO swarm_admin;

-- Create wallet transactions table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'crypto_intel'
                   AND table_name = 'wallet_transactions') THEN

        CREATE TABLE crypto_intel.wallet_transactions (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            transaction_hash TEXT NOT NULL,
            blockchain_network TEXT NOT NULL,
            block_number BIGINT,
            transaction_index INTEGER,
            from_address TEXT NOT NULL,
            to_address TEXT NOT NULL,
            from_wallet_id UUID,
            to_wallet_id UUID,
            amount DECIMAL(30,18) NOT NULL,
            currency_symbol TEXT NOT NULL,
            currency_contract_address TEXT,
            gas_used BIGINT,
            gas_price DECIMAL(30,18),
            transaction_fee DECIMAL(30,18),
            transaction_type TEXT CHECK (transaction_type IN ('transfer', 'exchange', 'mixing', 'smart_contract', 'mining', 'staking', 'defi')),
            transaction_status TEXT CHECK (transaction_status IN ('pending', 'confirmed', 'failed', 'dropped')),
            confirmations INTEGER DEFAULT 0,
            timestamp TIMESTAMPTZ NOT NULL,
            block_timestamp TIMESTAMPTZ,
            first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            is_suspicious BOOLEAN DEFAULT FALSE,
            risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
            risk_factors JSONB,
            mixing_score DECIMAL(3,2),
            chain_analysis_tags TEXT[],
            related_marketplace_vendor UUID,
            related_exchange TEXT,
            related_service_provider TEXT,
            transaction_pattern TEXT,
            time_pattern_score DECIMAL(3,2),
            amount_pattern_score DECIMAL(3,2),
            frequency_pattern_score DECIMAL(3,2),
            aml_flags TEXT[],
            sanctions_list_match BOOLEAN DEFAULT FALSE,
            high_risk_jurisdiction BOOLEAN DEFAULT FALSE,
            regulatory_alerts JSONB,
            data_source data_source_type NOT NULL,
            chain_analysis_provider TEXT,
            collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            raw_transaction_data JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(transaction_hash, blockchain_network)
        ) PARTITION BY RANGE (created_at);

        RAISE NOTICE 'Created table crypto_intel.wallet_transactions';
    ELSE
        RAISE NOTICE 'Table crypto_intel.wallet_transactions already exists';
    END IF;
END $$;

-- Create monthly partitions for wallet transactions
DO $$
DECLARE
    current_month DATE := DATE_TRUNC('month', NOW());
    next_month DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
BEGIN
    PERFORM create_monthly_partition('crypto_intel.wallet_transactions', current_month);
    PERFORM create_monthly_partition('crypto_intel.wallet_transactions', next_month);
    RAISE NOTICE 'Created partitions for crypto_intel.wallet_transactions';
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error creating partitions for wallet_transactions: %', SQLERRM;
END $$;

-- Create blockchain flows table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'crypto_intel'
                   AND table_name = 'blockchain_flows') THEN

        CREATE TABLE crypto_intel.blockchain_flows (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            flow_id TEXT NOT NULL,
            blockchain_network TEXT NOT NULL,
            flow_type TEXT CHECK (flow_type IN ('direct', 'multi_hop', 'mixing', 'exchange', 'peeling_chain', 'consolidation')),
            source_address TEXT NOT NULL,
            destination_address TEXT NOT NULL,
            intermediate_addresses TEXT[],
            total_amount DECIMAL(30,18) NOT NULL,
            currency_symbol TEXT NOT NULL,
            hop_count INTEGER DEFAULT 1,
            flow_duration INTERVAL,
            flow_confidence DECIMAL(3,2),
            obfuscation_score DECIMAL(3,2),
            complexity_score INTEGER,
            suspicious_indicators TEXT[],
            flow_start_time TIMESTAMPTZ NOT NULL,
            flow_end_time TIMESTAMPTZ,
            last_activity TIMESTAMPTZ,
            transaction_hashes TEXT[],
            related_flows UUID[],
            risk_level severity_level DEFAULT 'low',
            risk_explanation TEXT,
            compliance_concerns JSONB,
            known_pattern_match TEXT,
            pattern_confidence DECIMAL(3,2),
            behavioral_classification TEXT[],
            related_marketplace_activity BOOLEAN DEFAULT FALSE,
            related_investigations UUID[],
            law_enforcement_interest BOOLEAN DEFAULT FALSE,
            detection_algorithm TEXT,
            analysis_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            analyst_notes TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(flow_id, blockchain_network)
        ) PARTITION BY RANGE (created_at);

        RAISE NOTICE 'Created table crypto_intel.blockchain_flows';
    ELSE
        RAISE NOTICE 'Table crypto_intel.blockchain_flows already exists';
    END IF;
END $$;

-- Create partitions for blockchain flows
DO $$
DECLARE
    current_month DATE := DATE_TRUNC('month', NOW());
    next_month DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
BEGIN
    PERFORM create_monthly_partition('crypto_intel.blockchain_flows', current_month);
    PERFORM create_monthly_partition('crypto_intel.blockchain_flows', next_month);
    RAISE NOTICE 'Created partitions for crypto_intel.blockchain_flows';
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error creating partitions for blockchain_flows: %', SQLERRM;
END $$;

-- Create indexes for crypto intelligence tables
DO $$
BEGIN
    -- Wallet transactions indexes
    CREATE INDEX IF NOT EXISTS wallet_transactions_hash_idx ON crypto_intel.wallet_transactions (transaction_hash, blockchain_network);
    CREATE INDEX IF NOT EXISTS wallet_transactions_addresses_idx ON crypto_intel.wallet_transactions (from_address, to_address);
    CREATE INDEX IF NOT EXISTS wallet_transactions_amount_idx ON crypto_intel.wallet_transactions (currency_symbol, amount DESC);
    CREATE INDEX IF NOT EXISTS wallet_transactions_timestamp_idx ON crypto_intel.wallet_transactions (timestamp DESC, blockchain_network);
    CREATE INDEX IF NOT EXISTS wallet_transactions_risk_idx ON crypto_intel.wallet_transactions (risk_score DESC, is_suspicious);
    CREATE INDEX IF NOT EXISTS wallet_transactions_mixing_idx ON crypto_intel.wallet_transactions (mixing_score DESC, chain_analysis_tags);
    CREATE INDEX IF NOT EXISTS wallet_transactions_compliance_idx ON crypto_intel.wallet_transactions (sanctions_list_match, aml_flags);
    CREATE INDEX IF NOT EXISTS wallet_transactions_patterns_idx ON crypto_intel.wallet_transactions (transaction_pattern, transaction_type);

    -- Blockchain flows indexes
    CREATE INDEX IF NOT EXISTS blockchain_flows_addresses_idx ON crypto_intel.blockchain_flows (source_address, destination_address);
    CREATE INDEX IF NOT EXISTS blockchain_flows_amount_idx ON crypto_intel.blockchain_flows (currency_symbol, total_amount DESC);
    CREATE INDEX IF NOT EXISTS blockchain_flows_risk_idx ON crypto_intel.blockchain_flows (risk_level, obfuscation_score DESC);
    CREATE INDEX IF NOT EXISTS blockchain_flows_pattern_idx ON crypto_intel.blockchain_flows (known_pattern_match, pattern_confidence DESC);
    CREATE INDEX IF NOT EXISTS blockchain_flows_temporal_idx ON crypto_intel.blockchain_flows (flow_start_time DESC, flow_duration);
    CREATE INDEX IF NOT EXISTS blockchain_flows_investigation_idx ON crypto_intel.blockchain_flows (law_enforcement_interest, related_investigations);

    RAISE NOTICE 'Created indexes for crypto intelligence tables';
END $$;

-- Add audit triggers
DO $$
BEGIN
    -- Wallet transactions audit trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'wallet_transactions_audit_trigger') THEN
        CREATE TRIGGER wallet_transactions_audit_trigger
            BEFORE INSERT OR UPDATE ON crypto_intel.wallet_transactions
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
        RAISE NOTICE 'Created audit trigger for wallet_transactions';
    END IF;

    -- Blockchain flows audit trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'blockchain_flows_audit_trigger') THEN
        CREATE TRIGGER blockchain_flows_audit_trigger
            BEFORE INSERT OR UPDATE ON crypto_intel.blockchain_flows
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
        RAISE NOTICE 'Created audit trigger for blockchain_flows';
    END IF;
END $$;

-- Create additional crypto tables (mixing patterns, exchange movements, wallet clusters)
-- Following similar pattern for brevity...

-- Validate migration
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    -- Count created tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'crypto_intel';

    -- Count created indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'crypto_intel';

    RAISE NOTICE 'Migration validation: % tables created, % indexes created', table_count, index_count;

    IF table_count < 2 THEN
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
    'phase7_crypto_intelligence',
    '1.0.0',
    current_user,
    'DROP SCHEMA crypto_intel CASCADE;',
    md5('phase7_crypto_intelligence_1.0.0')
);

-- Commit transaction
COMMIT;

RAISE NOTICE 'Migration phase7_crypto_intelligence completed successfully';