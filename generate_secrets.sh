#!/bin/bash
# Secure Environment Variables Generator

echo "🔐 Generating secure passwords and keys..."
echo "========================================="

# Create secure environment file
cat > .env.secure << 'EOF'
# Secure Environment Variables - NEVER COMMIT THIS FILE
# Generated on $(date)

# Database Passwords
DB_PASSWORD=$(openssl rand -base64 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
NEO4J_PASSWORD=$(openssl rand -base64 32)

# Service Passwords
REDIS_PASSWORD=$(openssl rand -base64 32)
RABBITMQ_PASSWORD=$(openssl rand -base64 32)
KAFKA_PASSWORD=$(openssl rand -base64 32)
SWARM_PASSWORD=$(openssl rand -base64 32)

# Encryption Keys
ENCRYPTION_KEY=$(openssl rand -base64 64)
JWT_SECRET=$(openssl rand -base64 64)
SESSION_SECRET=$(openssl rand -base64 32)

# API Keys (replace with actual values)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
SHODAN_API_KEY=your-shodan-key-here
VIRUSTOTAL_API_KEY=your-virustotal-key-here

# External Service APIs
ETHERSCAN_API_KEY=your-etherscan-key-here
BLOCKCYPHER_API_KEY=your-blockcypher-key-here
ALCHEMY_API_KEY=your-alchemy-key-here

# Security Configuration
SECRETS_BACKEND=env
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token-here

# Application Security
DISABLE_AUTH=true
DEPLOYMENT_MODE=SINGLE_USER
BIND_ADDRESS=127.0.0.1
EXTERNAL_ACCESS=false

# Tor Configuration
TOR_CONTROL_PASSWORD=$(openssl rand -base64 24)

EOF

# Replace placeholders with actual generated passwords
DB_PASS=$(openssl rand -base64 32)
POSTGRES_PASS=$(openssl rand -base64 32)
NEO4J_PASS=$(openssl rand -base64 32)
REDIS_PASS=$(openssl rand -base64 32)
RABBIT_PASS=$(openssl rand -base64 32)
KAFKA_PASS=$(openssl rand -base64 32)
SWARM_PASS=$(openssl rand -base64 32)
ENCRYPT_KEY=$(openssl rand -base64 64)
JWT_KEY=$(openssl rand -base64 64)
SESSION_KEY=$(openssl rand -base64 32)
TOR_PASS=$(openssl rand -base64 24)

# Create actual secure environment file
cat > .env.secure << EOF
# Secure Environment Variables - NEVER COMMIT THIS FILE
# Generated on $(date)

# Database Passwords
DB_PASSWORD=${DB_PASS}
POSTGRES_PASSWORD=${POSTGRES_PASS}
NEO4J_PASSWORD=${NEO4J_PASS}

# Service Passwords
REDIS_PASSWORD=${REDIS_PASS}
RABBITMQ_PASSWORD=${RABBIT_PASS}
KAFKA_PASSWORD=${KAFKA_PASS}
SWARM_PASSWORD=${SWARM_PASS}

# Encryption Keys
ENCRYPTION_KEY=${ENCRYPT_KEY}
JWT_SECRET=${JWT_KEY}
SESSION_SECRET=${SESSION_KEY}

# API Keys (replace with actual values)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here
SHODAN_API_KEY=your-shodan-key-here
VIRUSTOTAL_API_KEY=your-virustotal-key-here

# External Service APIs
ETHERSCAN_API_KEY=your-etherscan-key-here
BLOCKCYPHER_API_KEY=your-blockcypher-key-here
ALCHEMY_API_KEY=your-alchemy-key-here

# Security Configuration
SECRETS_BACKEND=env
VAULT_URL=http://localhost:8200
VAULT_TOKEN=your-vault-token-here

# Application Security
DISABLE_AUTH=true
DEPLOYMENT_MODE=SINGLE_USER
BIND_ADDRESS=127.0.0.1
EXTERNAL_ACCESS=false

# Tor Configuration
TOR_CONTROL_PASSWORD=${TOR_PASS}
EOF

# Set secure permissions
chmod 600 .env.secure

echo "✅ Secure passwords generated in .env.secure"
echo "⚠️  IMPORTANT:"
echo "   - Replace placeholder API keys with real values"
echo "   - File permissions set to 600 (owner read/write only)"
echo "   - NEVER commit .env.secure to version control"
echo ""
echo "📋 Generated passwords:"
echo "   - Database: ${DB_PASS:0:8}..."
echo "   - Redis: ${REDIS_PASS:0:8}..."
echo "   - RabbitMQ: ${RABBIT_PASS:0:8}..."
echo "   - Full values in .env.secure"