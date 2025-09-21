-- Database initialization script for BEV OSINT MCP Server
-- Creates tables for OSINT results, threat intelligence, and audit logs

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create enum types
CREATE TYPE security_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE target_type AS ENUM ('email', 'domain', 'ip', 'phone', 'username', 'hash', 'wallet');
CREATE TYPE ioc_type AS ENUM ('ip', 'domain', 'hash', 'url', 'email');

-- OSINT Results table
CREATE TABLE IF NOT EXISTS osint_results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target_type target_type NOT NULL,
    target_value VARCHAR(2048) NOT NULL,
    tool_name VARCHAR(100) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    confidence_score DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    risk_score DECIMAL(3,2) DEFAULT 0.0 CHECK (risk_score >= 0.0 AND risk_score <= 1.0),
    sources TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Threat Intelligence table
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ioc_type ioc_type NOT NULL,
    value VARCHAR(2048) NOT NULL,
    threat_types TEXT[] DEFAULT '{}',
    confidence DECIMAL(3,2) DEFAULT 0.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
    severity security_level DEFAULT 'medium',
    sources TEXT[] DEFAULT '{}',
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint on IOC type and value
    UNIQUE(ioc_type, value)
);

-- Audit Logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    client_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    resource VARCHAR(255) NOT NULL,
    success BOOLEAN NOT NULL,
    ip_address INET NOT NULL,
    user_agent TEXT,
    request_data JSONB DEFAULT '{}',
    response_data JSONB DEFAULT '{}',
    security_level security_level DEFAULT 'medium',
    execution_time_ms INTEGER DEFAULT 0
);

-- Security Scan Results table
CREATE TABLE IF NOT EXISTS security_scan_results (
    scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target VARCHAR(2048) NOT NULL,
    scan_type VARCHAR(100) NOT NULL,
    vulnerabilities JSONB DEFAULT '[]',
    security_score DECIMAL(3,2) DEFAULT 0.0 CHECK (security_score >= 0.0 AND security_score <= 1.0),
    risk_level security_level DEFAULT 'medium',
    scan_duration DECIMAL(10,3) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Crypto Transactions table
CREATE TABLE IF NOT EXISTS crypto_transactions (
    transaction_id VARCHAR(255) PRIMARY KEY,
    blockchain VARCHAR(50) NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255) NOT NULL,
    value DECIMAL(20,8) NOT NULL,
    currency VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    block_height BIGINT,
    confirmations INTEGER DEFAULT 0,
    fee DECIMAL(20,8),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Graph Entities table (for Neo4j integration)
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    neo4j_node_id VARCHAR(255),
    entity_type VARCHAR(100) NOT NULL,
    entity_value VARCHAR(2048) NOT NULL,
    properties JSONB DEFAULT '{}',
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint on entity type and value
    UNIQUE(entity_type, entity_value)
);

-- Rate Limiting table (Redis backup)
CREATE TABLE IF NOT EXISTS rate_limits (
    client_id VARCHAR(255) NOT NULL,
    resource VARCHAR(255) NOT NULL,
    requests_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    window_duration INTEGER DEFAULT 60, -- seconds
    
    PRIMARY KEY (client_id, resource, window_start)
);

-- Client Sessions table
CREATE TABLE IF NOT EXISTS client_sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(255) NOT NULL,
    client_name VARCHAR(255),
    client_version VARCHAR(50),
    jwt_token_hash VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    capabilities JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '24 hours',
    is_active BOOLEAN DEFAULT TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_osint_results_target ON osint_results(target_type, target_value);
CREATE INDEX IF NOT EXISTS idx_osint_results_tool ON osint_results(tool_name);
CREATE INDEX IF NOT EXISTS idx_osint_results_created ON osint_results(created_at);
CREATE INDEX IF NOT EXISTS idx_osint_results_confidence ON osint_results(confidence_score);

CREATE INDEX IF NOT EXISTS idx_threat_intel_ioc ON threat_intelligence(ioc_type, value);
CREATE INDEX IF NOT EXISTS idx_threat_intel_severity ON threat_intelligence(severity);
CREATE INDEX IF NOT EXISTS idx_threat_intel_confidence ON threat_intelligence(confidence);
CREATE INDEX IF NOT EXISTS idx_threat_intel_first_seen ON threat_intelligence(first_seen);

CREATE INDEX IF NOT EXISTS idx_audit_logs_client ON audit_logs(client_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_success ON audit_logs(success);
CREATE INDEX IF NOT EXISTS idx_audit_logs_security_level ON audit_logs(security_level);

CREATE INDEX IF NOT EXISTS idx_security_scans_target ON security_scan_results(target);
CREATE INDEX IF NOT EXISTS idx_security_scans_type ON security_scan_results(scan_type);
CREATE INDEX IF NOT EXISTS idx_security_scans_created ON security_scan_results(created_at);

CREATE INDEX IF NOT EXISTS idx_crypto_transactions_blockchain ON crypto_transactions(blockchain);
CREATE INDEX IF NOT EXISTS idx_crypto_transactions_from ON crypto_transactions(from_address);
CREATE INDEX IF NOT EXISTS idx_crypto_transactions_to ON crypto_transactions(to_address);
CREATE INDEX IF NOT EXISTS idx_crypto_transactions_timestamp ON crypto_transactions(timestamp);

CREATE INDEX IF NOT EXISTS idx_graph_entities_type ON graph_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_graph_entities_value ON graph_entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_graph_entities_confidence ON graph_entities(confidence_score);

CREATE INDEX IF NOT EXISTS idx_rate_limits_client ON rate_limits(client_id);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON rate_limits(window_start);

CREATE INDEX IF NOT EXISTS idx_client_sessions_client ON client_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_client_sessions_active ON client_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_client_sessions_expires ON client_sessions(expires_at);

-- Create functions and triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_osint_results_updated_at 
    BEFORE UPDATE ON osint_results 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_threat_intelligence_updated_at 
    BEFORE UPDATE ON threat_intelligence 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_graph_entities_updated_at 
    BEFORE UPDATE ON graph_entities 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create data retention function
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Clean up old audit logs (keep 90 days by default)
    DELETE FROM audit_logs 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    -- Clean up expired sessions
    DELETE FROM client_sessions 
    WHERE expires_at < NOW() OR (NOT is_active AND last_activity < NOW() - INTERVAL '7 days');
    
    -- Clean up old rate limit entries
    DELETE FROM rate_limits 
    WHERE window_start < NOW() - INTERVAL '1 hour';
    
    -- Log cleanup activity
    INSERT INTO audit_logs (client_id, action, resource, success, ip_address, user_agent)
    VALUES ('system', 'data_cleanup', 'database', TRUE, '127.0.0.1', 'postgres_cleanup_function');
END;
$$ LANGUAGE plpgsql;

-- Create initial admin user and API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    key_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key_name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    permissions JSONB DEFAULT '{}',
    rate_limit INTEGER DEFAULT 1000,
    is_active BOOLEAN DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used TIMESTAMP WITH TIME ZONE
);

-- Insert system API key for internal operations
INSERT INTO api_keys (key_name, key_hash, permissions, rate_limit, created_by)
VALUES (
    'system_internal',
    crypt('bev_system_key_2024', gen_salt('bf')),
    '{"tools": ["*"], "resources": ["*"], "admin": true}',
    10000,
    'system_init'
) ON CONFLICT (key_hash) DO NOTHING;

-- Grant permissions to the application user
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO bev_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bev_admin;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO bev_admin;

-- Create a read-only user for monitoring/reporting
CREATE USER IF NOT EXISTS bev_readonly WITH PASSWORD 'BevReadOnly2024!';
GRANT CONNECT ON DATABASE osint TO bev_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO bev_readonly;

-- Security: Enable row level security for sensitive tables
ALTER TABLE threat_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Create policies for row level security (basic example)
CREATE POLICY threat_intel_policy ON threat_intelligence
    FOR ALL TO bev_admin
    USING (true);  -- Admin can see all

CREATE POLICY audit_logs_policy ON audit_logs
    FOR ALL TO bev_admin
    USING (true);  -- Admin can see all

-- Create materialized view for performance metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS performance_metrics AS
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    tool_name,
    COUNT(*) as executions,
    AVG(confidence_score) as avg_confidence,
    AVG(risk_score) as avg_risk
FROM osint_results
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), tool_name
ORDER BY hour DESC, tool_name;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_performance_metrics_hour ON performance_metrics(hour);

-- Refresh materialized view function
CREATE OR REPLACE FUNCTION refresh_performance_metrics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW performance_metrics;
END;
$$ LANGUAGE plpgsql;

-- Final message
DO $$
BEGIN
    RAISE NOTICE 'BEV OSINT MCP Server database initialization completed successfully';
    RAISE NOTICE 'Tables created: osint_results, threat_intelligence, audit_logs, security_scan_results, crypto_transactions, graph_entities, rate_limits, client_sessions, api_keys';
    RAISE NOTICE 'Indexes, triggers, and security policies applied';
    RAISE NOTICE 'System API key created for internal operations';
END
$$;