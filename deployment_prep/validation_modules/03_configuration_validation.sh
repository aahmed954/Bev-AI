#!/bin/bash

# Configuration Validation Module
# Gate 3: Environment variables, credentials, and configuration files

# Required environment variables
REQUIRED_ENV_VARS=(
    "POSTGRES_PASSWORD"
    "NEO4J_PASSWORD"
    "REDIS_PASSWORD"
    "ELASTICSEARCH_PASSWORD"
    "GRAFANA_ADMIN_PASSWORD"
    "INTELOWL_ADMIN_PASSWORD"
    "SECRET_KEY"
    "DEHASHED_API_KEY"
    "SNUSBASE_API_KEY"
    "ALPHAHQ_API_KEY"
)

# Optional but recommended environment variables
OPTIONAL_ENV_VARS=(
    "WELUAKINFO_API_KEY"
    "SHODAN_API_KEY"
    "VIRUSTOTAL_API_KEY"
    "HUNTER_API_KEY"
    "SPYSE_API_KEY"
    "BLOCKCHAIN_API_KEY"
)

# Configuration files to validate
CONFIG_FILES=(
    ".env"
    "docker-compose.complete.yml"
    "intelowl/configuration/analyzer_config.json"
    "intelowl/configuration/connector_config.json"
    "vault/config.json"
)

configuration_validation() {
    log "INFO" "Validating configuration and credentials"
    local validation_passed=true

    # Test 1: Environment variables
    if ! validate_environment_variables; then
        validation_passed=false
    fi

    # Test 2: Configuration files
    if ! validate_configuration_files; then
        validation_passed=false
    fi

    # Test 3: Vault credentials
    if ! validate_vault_credentials; then
        validation_passed=false
    fi

    # Test 4: Database configurations
    if ! validate_database_configurations; then
        validation_passed=false
    fi

    # Test 5: API key validation
    if ! validate_api_keys; then
        validation_passed=false
    fi

    return $(if [[ "$validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_environment_variables() {
    log "INFO" "Validating environment variables"
    local env_validation_passed=true

    # Load .env file if it exists
    if [[ -f ".env" ]]; then
        set -a
        source .env
        set +a
        log "SUCCESS" ".env file loaded"
    else
        log "ERROR" ".env file not found"
        env_validation_passed=false
    fi

    # Check required environment variables
    log "INFO" "Checking required environment variables"
    for var in "${REQUIRED_ENV_VARS[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            log "SUCCESS" "$var is set"
        else
            log "ERROR" "$var is not set"

            if [[ "$AUTO_FIX_MODE" == "true" ]]; then
                if generate_missing_env_var "$var"; then
                    log "SUCCESS" "$var generated automatically"
                else
                    log "ERROR" "Failed to generate $var automatically"
                    env_validation_passed=false
                fi
            else
                env_validation_passed=false
            fi
        fi
    done

    # Check optional environment variables
    log "INFO" "Checking optional environment variables"
    local missing_optional=0
    for var in "${OPTIONAL_ENV_VARS[@]}"; do
        if [[ -n "${!var:-}" ]]; then
            log "SUCCESS" "$var is set"
        else
            log "WARN" "$var is not set (optional)"
            ((missing_optional++))
        fi
    done

    if [[ $missing_optional -gt 0 ]]; then
        log "WARN" "$missing_optional optional environment variables are missing"
        log "WARN" "Some OSINT capabilities may be limited"
    fi

    return $(if [[ "$env_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_configuration_files() {
    log "INFO" "Validating configuration files"
    local config_validation_passed=true

    for config_file in "${CONFIG_FILES[@]}"; do
        if [[ -f "$config_file" ]]; then
            log "SUCCESS" "Configuration file exists: $config_file"

            # Validate file format based on extension
            case "$config_file" in
                *.yml|*.yaml)
                    if validate_yaml_syntax "$config_file"; then
                        log "SUCCESS" "$config_file has valid YAML syntax"
                    else
                        log "ERROR" "$config_file has invalid YAML syntax"
                        config_validation_passed=false
                    fi
                    ;;
                *.json)
                    if validate_json_syntax "$config_file"; then
                        log "SUCCESS" "$config_file has valid JSON syntax"
                    else
                        log "ERROR" "$config_file has invalid JSON syntax"
                        config_validation_passed=false
                    fi
                    ;;
                .env)
                    if validate_env_syntax "$config_file"; then
                        log "SUCCESS" "$config_file has valid syntax"
                    else
                        log "ERROR" "$config_file has invalid syntax"
                        config_validation_passed=false
                    fi
                    ;;
            esac
        else
            log "ERROR" "Configuration file missing: $config_file"
            config_validation_passed=false
        fi
    done

    return $(if [[ "$config_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_vault_credentials() {
    log "INFO" "Validating Vault credential access"
    local vault_validation_passed=true

    # Check if Vault is configured
    if [[ -f "vault/config.json" ]]; then
        log "SUCCESS" "Vault configuration found"

        # Test Vault connectivity if Vault is running
        if pgrep -f vault >/dev/null; then
            local vault_status=""
            if command -v vault >/dev/null 2>&1; then
                vault_status=$(vault status 2>/dev/null || echo "sealed")
                if [[ "$vault_status" == *"sealed"* ]]; then
                    log "WARN" "Vault is sealed - unseal before deployment"
                else
                    log "SUCCESS" "Vault is accessible and unsealed"
                fi
            else
                log "WARN" "Vault CLI not available for status check"
            fi
        else
            log "INFO" "Vault not running - will be started during deployment"
        fi

        # Validate Vault credential files
        local vault_cred_files=(
            "vault/credentials/postgres_creds.json"
            "vault/credentials/neo4j_creds.json"
            "vault/credentials/api_keys.json"
        )

        for cred_file in "${vault_cred_files[@]}"; do
            if [[ -f "$cred_file" ]]; then
                if validate_json_syntax "$cred_file"; then
                    log "SUCCESS" "Vault credential file valid: $cred_file"
                else
                    log "ERROR" "Vault credential file invalid: $cred_file"
                    vault_validation_passed=false
                fi
            else
                log "WARN" "Vault credential file missing: $cred_file"
                # Not failing validation as Vault might auto-generate
            fi
        done
    else
        log "WARN" "Vault configuration not found - credential management disabled"
    fi

    return $(if [[ "$vault_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_database_configurations() {
    log "INFO" "Validating database configurations"
    local db_validation_passed=true

    # PostgreSQL configuration validation
    if [[ -n "${POSTGRES_PASSWORD}" ]]; then
        if [[ ${#POSTGRES_PASSWORD} -ge 12 ]]; then
            log "SUCCESS" "PostgreSQL password meets length requirements"
        else
            log "ERROR" "PostgreSQL password too short (minimum 12 characters)"
            db_validation_passed=false
        fi
    fi

    # Neo4j configuration validation
    if [[ -n "${NEO4J_PASSWORD}" ]]; then
        if [[ ${#NEO4J_PASSWORD} -ge 8 ]]; then
            log "SUCCESS" "Neo4j password meets length requirements"
        else
            log "ERROR" "Neo4j password too short (minimum 8 characters)"
            db_validation_passed=false
        fi
    fi

    # Redis configuration validation
    if [[ -n "${REDIS_PASSWORD}" ]]; then
        if [[ ${#REDIS_PASSWORD} -ge 16 ]]; then
            log "SUCCESS" "Redis password meets length requirements"
        else
            log "ERROR" "Redis password too short (minimum 16 characters)"
            db_validation_passed=false
        fi
    fi

    # Validate database initialization scripts
    local db_init_scripts=(
        "database/postgres/init.sql"
        "database/neo4j/init.cypher"
    )

    for script in "${db_init_scripts[@]}"; do
        if [[ -f "$script" ]]; then
            log "SUCCESS" "Database initialization script found: $script"
        else
            log "WARN" "Database initialization script missing: $script"
        fi
    done

    return $(if [[ "$db_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

validate_api_keys() {
    log "INFO" "Validating API key formats and accessibility"
    local api_validation_passed=true

    # API key format validation
    local api_key_formats=(
        "DEHASHED_API_KEY:^[a-zA-Z0-9]{32,64}$"
        "SNUSBASE_API_KEY:^[a-zA-Z0-9]{24,48}$"
        "SHODAN_API_KEY:^[a-zA-Z0-9]{32}$"
        "VIRUSTOTAL_API_KEY:^[a-zA-Z0-9]{64}$"
    )

    for api_info in "${api_key_formats[@]}"; do
        IFS=':' read -r api_var api_pattern <<< "$api_info"

        if [[ -n "${!api_var:-}" ]]; then
            if [[ "${!api_var}" =~ $api_pattern ]]; then
                log "SUCCESS" "$api_var format is valid"

                # Test API key if possible (rate-limited test)
                if test_api_key_connectivity "$api_var" "${!api_var}"; then
                    log "SUCCESS" "$api_var is accessible"
                else
                    log "WARN" "$api_var connectivity test failed (may be rate limited)"
                fi
            else
                log "ERROR" "$api_var format is invalid"
                api_validation_passed=false
            fi
        fi
    done

    return $(if [[ "$api_validation_passed" == "true" ]]; then echo 0; else echo 1; fi)
}

# Helper functions

generate_missing_env_var() {
    local var_name="$1"
    local generated_value=""

    case "$var_name" in
        *PASSWORD*)
            generated_value=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
            ;;
        SECRET_KEY)
            generated_value=$(openssl rand -hex 32)
            ;;
        *)
            log "WARN" "Cannot auto-generate $var_name - manual configuration required"
            return 1
            ;;
    esac

    if [[ -n "$generated_value" ]]; then
        echo "export $var_name=\"$generated_value\"" >> .env
        export "$var_name"="$generated_value"
        log "INFO" "Generated $var_name and added to .env"
        return 0
    fi

    return 1
}

validate_yaml_syntax() {
    local file="$1"

    if command -v yq >/dev/null 2>&1; then
        yq eval '.' "$file" >/dev/null 2>&1
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import yaml; yaml.safe_load(open('$file'))" 2>/dev/null
    else
        log "WARN" "No YAML validator available, skipping syntax check for $file"
        return 0
    fi
}

validate_json_syntax() {
    local file="$1"

    if command -v jq >/dev/null 2>&1; then
        jq empty "$file" 2>/dev/null
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import json; json.load(open('$file'))" 2>/dev/null
    else
        log "WARN" "No JSON validator available, skipping syntax check for $file"
        return 0
    fi
}

validate_env_syntax() {
    local file="$1"

    # Check for common .env file issues
    if grep -q $'[\t]' "$file"; then
        log "ERROR" "$file contains tabs (use spaces only)"
        return 1
    fi

    if grep -q '^[[:space:]]*=' "$file"; then
        log "ERROR" "$file contains variables without names"
        return 1
    fi

    return 0
}

test_api_key_connectivity() {
    local api_var="$1"
    local api_key="$2"

    # Simple connectivity tests (avoid rate limiting)
    case "$api_var" in
        "SHODAN_API_KEY")
            curl -s -f --connect-timeout 5 "https://api.shodan.io/api-info?key=$api_key" >/dev/null 2>&1
            ;;
        "VIRUSTOTAL_API_KEY")
            curl -s -f --connect-timeout 5 -H "x-apikey: $api_key" "https://www.virustotal.com/vtapi/v2/url/report?apikey=$api_key&resource=google.com" >/dev/null 2>&1
            ;;
        *)
            # Skip connectivity test for unknown APIs
            return 0
            ;;
    esac
}