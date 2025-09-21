-- PostgreSQL Initialization Script for BEV OSINT Framework
-- Creates databases and enables extensions

-- Create databases if they don't exist
SELECT 'CREATE DATABASE osint' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'osint');
SELECT 'CREATE DATABASE intelowl' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'intelowl');
SELECT 'CREATE DATABASE breach_data' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'breach_data');
SELECT 'CREATE DATABASE crypto_analysis' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'crypto_analysis');

-- Enable pgvector extension on osint database
\c osint;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Enable extensions on other databases
\c intelowl;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

\c breach_data;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

\c crypto_analysis;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;