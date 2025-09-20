-- Phase 7: Alternative Market Intelligence Database Schema
-- Comprehensive marketplace and cryptocurrency intelligence tracking

-- Create marketplace intelligence schema
CREATE SCHEMA IF NOT EXISTS marketplace_intel;

-- ========================================
-- MARKETPLACE VENDOR PROFILES
-- ========================================
CREATE TABLE marketplace_intel.vendor_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vendor_id TEXT NOT NULL,
    marketplace_name TEXT NOT NULL,
    vendor_name TEXT NOT NULL,
    vendor_alias TEXT[],

    -- Profile information
    registration_date TIMESTAMPTZ,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status TEXT CHECK (status IN ('active', 'inactive', 'banned', 'suspected', 'verified')),

    -- Reputation metrics
    feedback_score DECIMAL(3,2),
    total_transactions INTEGER DEFAULT 0,
    successful_transactions INTEGER DEFAULT 0,
    disputes INTEGER DEFAULT 0,
    reputation_rank INTEGER,

    -- Profile details
    profile_description TEXT,
    profile_image_url TEXT,
    contact_methods JSONB,
    shipping_locations TEXT[],
    accepted_currencies TEXT[],

    -- Security indicators
    pgp_key TEXT,
    verified_vendor BOOLEAN DEFAULT FALSE,
    escrow_required BOOLEAN DEFAULT TRUE,
    finalize_early_allowed BOOLEAN DEFAULT FALSE,

    -- Risk assessment
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    risk_factors JSONB,
    threat_indicators TEXT[],
    law_enforcement_interest BOOLEAN DEFAULT FALSE,

    -- Embedding for similarity analysis
    profile_embedding vector(768),

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_method TEXT,
    collector_id TEXT,
    raw_data JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(vendor_id, marketplace_name)
) PARTITION BY RANGE (created_at);

-- Create partitions for current and next month
SELECT create_monthly_partition('marketplace_intel.vendor_profiles', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('marketplace_intel.vendor_profiles', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for vendor profiles
CREATE INDEX vendor_profiles_marketplace_idx ON marketplace_intel.vendor_profiles (marketplace_name, vendor_name);
CREATE INDEX vendor_profiles_status_idx ON marketplace_intel.vendor_profiles (status, last_seen);
CREATE INDEX vendor_profiles_risk_idx ON marketplace_intel.vendor_profiles (risk_score DESC, threat_indicators);
CREATE INDEX vendor_profiles_reputation_idx ON marketplace_intel.vendor_profiles (feedback_score DESC, total_transactions DESC);
CREATE INDEX vendor_profiles_embedding_idx ON marketplace_intel.vendor_profiles USING ivfflat (profile_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX vendor_profiles_search_idx ON marketplace_intel.vendor_profiles USING gin(vendor_name gin_trgm_ops, vendor_alias gin_trgm_ops);
CREATE INDEX vendor_profiles_raw_data_idx ON marketplace_intel.vendor_profiles USING gin(raw_data);

-- ========================================
-- PRODUCT LISTINGS
-- ========================================
CREATE TABLE marketplace_intel.product_listings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vendor_profile_id UUID REFERENCES marketplace_intel.vendor_profiles(id) ON DELETE CASCADE,

    -- Product identification
    listing_id TEXT NOT NULL,
    marketplace_name TEXT NOT NULL,
    product_title TEXT NOT NULL,
    product_category TEXT,
    product_subcategory TEXT,

    -- Product details
    description TEXT,
    price DECIMAL(15,8),
    currency TEXT NOT NULL DEFAULT 'USD',
    quantity_available INTEGER,
    minimum_order INTEGER DEFAULT 1,

    -- Product attributes
    product_images TEXT[],
    product_attributes JSONB,
    shipping_info JSONB,
    delivery_time TEXT,

    -- Classification
    product_type TEXT,
    illegal_content BOOLEAN DEFAULT FALSE,
    controlled_substance BOOLEAN DEFAULT FALSE,
    content_warnings TEXT[],

    -- Listing metadata
    listing_status TEXT CHECK (listing_status IN ('active', 'inactive', 'sold_out', 'removed', 'suspended')),
    views_count INTEGER DEFAULT 0,
    favorites_count INTEGER DEFAULT 0,

    -- Temporal tracking
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    estimated_listing_date TIMESTAMPTZ,

    -- Analysis fields
    price_history JSONB,
    availability_pattern JSONB,
    listing_embedding vector(768),

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_listing_data JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(listing_id, marketplace_name)
) PARTITION BY RANGE (created_at);

-- Create partitions for product listings
SELECT create_monthly_partition('marketplace_intel.product_listings', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('marketplace_intel.product_listings', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for product listings
CREATE INDEX product_listings_vendor_idx ON marketplace_intel.product_listings (vendor_profile_id, listing_status);
CREATE INDEX product_listings_marketplace_idx ON marketplace_intel.product_listings (marketplace_name, product_category);
CREATE INDEX product_listings_price_idx ON marketplace_intel.product_listings (currency, price, last_seen);
CREATE INDEX product_listings_illegal_idx ON marketplace_intel.product_listings (illegal_content, controlled_substance);
CREATE INDEX product_listings_search_idx ON marketplace_intel.product_listings USING gin(to_tsvector('english', product_title || ' ' || COALESCE(description, '')));
CREATE INDEX product_listings_embedding_idx ON marketplace_intel.product_listings USING ivfflat (listing_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX product_listings_attributes_idx ON marketplace_intel.product_listings USING gin(product_attributes);

-- ========================================
-- TRANSACTION RECORDS
-- ========================================
CREATE TABLE marketplace_intel.transaction_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vendor_profile_id UUID REFERENCES marketplace_intel.vendor_profiles(id) ON DELETE CASCADE,
    listing_id UUID REFERENCES marketplace_intel.product_listings(id) ON DELETE SET NULL,

    -- Transaction identification
    transaction_id TEXT,
    marketplace_name TEXT NOT NULL,
    buyer_id TEXT,

    -- Transaction details
    transaction_amount DECIMAL(15,8),
    currency TEXT NOT NULL,
    quantity INTEGER,

    -- Transaction status
    status TEXT CHECK (status IN ('pending', 'paid', 'shipped', 'delivered', 'disputed', 'cancelled', 'completed')),
    payment_method TEXT,
    escrow_used BOOLEAN,

    -- Temporal data
    transaction_date TIMESTAMPTZ,
    payment_date TIMESTAMPTZ,
    shipping_date TIMESTAMPTZ,
    delivery_date TIMESTAMPTZ,
    completion_date TIMESTAMPTZ,

    -- Feedback and ratings
    buyer_rating INTEGER CHECK (buyer_rating BETWEEN 1 AND 5),
    vendor_rating INTEGER CHECK (vendor_rating BETWEEN 1 AND 5),
    buyer_feedback TEXT,
    vendor_feedback TEXT,

    -- Risk indicators
    suspicious_activity BOOLEAN DEFAULT FALSE,
    dispute_reason TEXT,
    automated_flags TEXT[],

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_method TEXT,
    raw_transaction_data JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for transaction records
SELECT create_monthly_partition('marketplace_intel.transaction_records', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('marketplace_intel.transaction_records', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for transaction records
CREATE INDEX transaction_records_vendor_idx ON marketplace_intel.transaction_records (vendor_profile_id, status);
CREATE INDEX transaction_records_marketplace_idx ON marketplace_intel.transaction_records (marketplace_name, transaction_date);
CREATE INDEX transaction_records_amount_idx ON marketplace_intel.transaction_records (currency, transaction_amount DESC);
CREATE INDEX transaction_records_status_idx ON marketplace_intel.transaction_records (status, payment_method);
CREATE INDEX transaction_records_suspicious_idx ON marketplace_intel.transaction_records (suspicious_activity, automated_flags);

-- ========================================
-- REPUTATION SCORES
-- ========================================
CREATE TABLE marketplace_intel.reputation_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vendor_profile_id UUID REFERENCES marketplace_intel.vendor_profiles(id) ON DELETE CASCADE,

    -- Score components
    overall_score DECIMAL(5,2),
    communication_score DECIMAL(5,2),
    shipping_score DECIMAL(5,2),
    product_quality_score DECIMAL(5,2),
    reliability_score DECIMAL(5,2),

    -- Score metadata
    total_reviews INTEGER DEFAULT 0,
    score_calculation_method TEXT,
    confidence confidence_level,

    -- Temporal tracking
    score_period_start TIMESTAMPTZ,
    score_period_end TIMESTAMPTZ,
    calculation_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Historical tracking
    previous_score DECIMAL(5,2),
    score_trend TEXT CHECK (score_trend IN ('increasing', 'decreasing', 'stable', 'volatile')),
    score_change_rate DECIMAL(5,4),

    -- Risk assessment integration
    risk_adjusted_score DECIMAL(5,2),
    risk_factors_impact JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(vendor_profile_id, calculation_date)
);

-- Indexes for reputation scores
CREATE INDEX reputation_scores_vendor_idx ON marketplace_intel.reputation_scores (vendor_profile_id, calculation_date DESC);
CREATE INDEX reputation_scores_overall_idx ON marketplace_intel.reputation_scores (overall_score DESC, confidence);
CREATE INDEX reputation_scores_trend_idx ON marketplace_intel.reputation_scores (score_trend, score_change_rate);

-- ========================================
-- PRICE HISTORIES
-- ========================================
CREATE TABLE marketplace_intel.price_histories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    listing_id UUID REFERENCES marketplace_intel.product_listings(id) ON DELETE CASCADE,
    vendor_profile_id UUID REFERENCES marketplace_intel.vendor_profiles(id) ON DELETE CASCADE,

    -- Price information
    price DECIMAL(15,8) NOT NULL,
    currency TEXT NOT NULL,
    quantity_available INTEGER,

    -- Price context
    price_type TEXT CHECK (price_type IN ('regular', 'bulk_discount', 'limited_time', 'negotiated')),
    discount_percentage DECIMAL(5,2),
    minimum_quantity INTEGER,

    -- Market context
    marketplace_name TEXT NOT NULL,
    competitor_prices JSONB,
    market_average_price DECIMAL(15,8),
    price_position TEXT CHECK (price_position IN ('below_market', 'at_market', 'above_market', 'premium')),

    -- Temporal data
    price_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    price_valid_until TIMESTAMPTZ,

    -- Analysis fields
    price_volatility DECIMAL(5,4),
    demand_indicators JSONB,
    supply_indicators JSONB,

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Create partitions for price histories
SELECT create_monthly_partition('marketplace_intel.price_histories', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('marketplace_intel.price_histories', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for price histories
CREATE INDEX price_histories_listing_idx ON marketplace_intel.price_histories (listing_id, price_date DESC);
CREATE INDEX price_histories_vendor_idx ON marketplace_intel.price_histories (vendor_profile_id, marketplace_name);
CREATE INDEX price_histories_price_idx ON marketplace_intel.price_histories (currency, price, price_date);
CREATE INDEX price_histories_market_idx ON marketplace_intel.price_histories (marketplace_name, price_position);

-- Add audit triggers
CREATE TRIGGER vendor_profiles_audit_trigger
    BEFORE INSERT OR UPDATE ON marketplace_intel.vendor_profiles
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER product_listings_audit_trigger
    BEFORE INSERT OR UPDATE ON marketplace_intel.product_listings
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER transaction_records_audit_trigger
    BEFORE INSERT OR UPDATE ON marketplace_intel.transaction_records
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER reputation_scores_audit_trigger
    BEFORE INSERT OR UPDATE ON marketplace_intel.reputation_scores
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER price_histories_audit_trigger
    BEFORE INSERT OR UPDATE ON marketplace_intel.price_histories
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();