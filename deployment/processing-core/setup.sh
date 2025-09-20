#!/bin/bash

# BEV OSINT Framework - Processing Core Node Setup Script
# This script prepares the directory structure and configuration for deployment

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Function to check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check if user is in docker group
    if ! groups | grep -q docker; then
        log_warning "User is not in the docker group. You may need to run Docker commands with sudo."
        log_info "To add your user to the docker group, run: sudo usermod -aG docker \$USER"
        log_info "Then log out and log back in."
    fi

    log_success "Dependencies check completed"
}

# Function to create directory structure
create_directories() {
    log_info "Creating directory structure..."

    # Log directories
    mkdir -p logs/{intelowl,nginx,cytoscape,mcp}
    log_success "Created log directories"

    # SSL directory for certificates
    mkdir -p ssl
    log_success "Created SSL directory"

    # MCP server configuration
    mkdir -p mcp_server/config
    log_success "Created MCP server config directory"

    # IntelOwl custom directories (if they don't exist)
    mkdir -p intelowl/custom_analyzers
    mkdir -p intelowl/custom_connectors
    log_success "Created IntelOwl custom directories"

    # Cytoscape directory (copy from main repo if exists)
    if [[ -d "../../cytoscape" ]]; then
        if [[ ! -d "cytoscape" ]]; then
            cp -r ../../cytoscape ./
            log_success "Copied Cytoscape application from main repository"
        else
            log_info "Cytoscape directory already exists, skipping copy"
        fi
    else
        mkdir -p cytoscape
        log_warning "Cytoscape directory created but source not found in main repo"
        log_info "You may need to manually copy the Cytoscape application"
    fi

    log_success "Directory structure created"
}

# Function to set permissions
set_permissions() {
    log_info "Setting appropriate permissions..."

    # Set permissions for log directories
    chmod -R 755 logs/

    # Set permissions for config directories
    chmod -R 755 mcp_server/

    # Set permissions for SSL directory (restrictive)
    chmod 700 ssl/

    # Set permissions for IntelOwl directories
    chmod -R 755 intelowl/

    log_success "Permissions set"
}

# Function to copy and validate environment file
setup_environment() {
    log_info "Setting up environment configuration..."

    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.template" ]]; then
            cp .env.template .env
            log_success "Created .env file from template"
            log_warning "IMPORTANT: Edit .env file with your actual configuration values"
            log_info "Key values to configure:"
            log_info "  - DATA_CORE_POSTGRES_HOST"
            log_info "  - DATA_CORE_REDIS_HOST"
            log_info "  - DATA_CORE_NEO4J_URI"
            log_info "  - MSG_RABBITMQ_HOST"
            log_info "  - DJANGO_SECRET_KEY"
            log_info "  - JWT_SECRET"
            log_info "  - API keys for OSINT services"
        else
            log_error ".env.template not found. Cannot create .env file."
            exit 1
        fi
    else
        log_info ".env file already exists, skipping creation"
    fi
}

# Function to validate configuration
validate_configuration() {
    log_info "Validating configuration..."

    if [[ ! -f ".env" ]]; then
        log_error ".env file not found. Run setup first."
        return 1
    fi

    # Source the .env file
    set -a
    source .env
    set +a

    # Check critical variables
    local missing_vars=()

    [[ -z "$DATA_CORE_POSTGRES_HOST" ]] && missing_vars+=("DATA_CORE_POSTGRES_HOST")
    [[ -z "$DATA_CORE_REDIS_HOST" ]] && missing_vars+=("DATA_CORE_REDIS_HOST")
    [[ -z "$DATA_CORE_NEO4J_URI" ]] && missing_vars+=("DATA_CORE_NEO4J_URI")
    [[ -z "$MSG_RABBITMQ_HOST" ]] && missing_vars+=("MSG_RABBITMQ_HOST")
    [[ -z "$DJANGO_SECRET_KEY" ]] && missing_vars+=("DJANGO_SECRET_KEY")

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            log_error "  - $var"
        done
        return 1
    fi

    # Test connectivity to external services
    log_info "Testing connectivity to external services..."

    # Test PostgreSQL
    if command -v nc &> /dev/null; then
        if ! nc -z "$DATA_CORE_POSTGRES_HOST" "${DATA_CORE_POSTGRES_PORT:-5432}" 2>/dev/null; then
            log_warning "Cannot connect to PostgreSQL at $DATA_CORE_POSTGRES_HOST:${DATA_CORE_POSTGRES_PORT:-5432}"
        else
            log_success "PostgreSQL connectivity OK"
        fi

        # Test Redis
        if ! nc -z "$DATA_CORE_REDIS_HOST" "${DATA_CORE_REDIS_PORT:-6379}" 2>/dev/null; then
            log_warning "Cannot connect to Redis at $DATA_CORE_REDIS_HOST:${DATA_CORE_REDIS_PORT:-6379}"
        else
            log_success "Redis connectivity OK"
        fi

        # Test RabbitMQ
        if ! nc -z "$MSG_RABBITMQ_HOST" "${MSG_RABBITMQ_PORT:-5672}" 2>/dev/null; then
            log_warning "Cannot connect to RabbitMQ at $MSG_RABBITMQ_HOST:${MSG_RABBITMQ_PORT:-5672}"
        else
            log_success "RabbitMQ connectivity OK"
        fi
    else
        log_warning "netcat (nc) not available, skipping connectivity tests"
    fi

    log_success "Configuration validation completed"
}

# Function to create a sample dark theme CSS
create_sample_theme() {
    log_info "Creating sample dark theme..."

    if [[ ! -f "intelowl/dark_theme.css" ]]; then
        cat > intelowl/dark_theme.css << 'EOF'
/* BEV OSINT Framework - Dark Theme for IntelOwl */

:root {
    --bev-primary: #2c3e50;
    --bev-secondary: #34495e;
    --bev-accent: #3498db;
    --bev-text: #ecf0f1;
    --bev-bg: #1a1a1a;
}

body {
    background-color: var(--bev-bg) !important;
    color: var(--bev-text) !important;
}

.navbar {
    background-color: var(--bev-primary) !important;
}

.card {
    background-color: var(--bev-secondary) !important;
    border-color: var(--bev-accent) !important;
}

.btn-primary {
    background-color: var(--bev-accent) !important;
    border-color: var(--bev-accent) !important;
}

/* Add more custom styles as needed */
EOF
        log_success "Created sample dark theme CSS"
    else
        log_info "Dark theme CSS already exists, skipping creation"
    fi
}

# Function to pull Docker images
pull_images() {
    log_info "Pulling Docker images..."

    if command -v docker-compose &> /dev/null; then
        docker-compose pull
    else
        docker compose pull
    fi

    log_success "Docker images pulled"
}

# Function to display next steps
show_next_steps() {
    log_success "Setup completed successfully!"
    echo
    log_info "Next steps:"
    echo "1. Edit the .env file with your actual configuration values"
    echo "2. Ensure external services (Data Core, Message Infrastructure) are running"
    echo "3. Test connectivity to external services"
    echo "4. Start the processing core services:"
    echo "   docker-compose up -d"
    echo "5. Check service status:"
    echo "   docker-compose ps"
    echo "6. View logs:"
    echo "   docker-compose logs -f"
    echo
    log_info "Access points after deployment:"
    echo "- IntelOwl Web Interface: http://localhost:80"
    echo "- IntelOwl API: http://localhost:8000"
    echo "- Cytoscape Visualization: http://localhost:3000"
    echo "- MCP Server API: http://localhost:3010"
}

# Main function
main() {
    echo "========================================"
    echo "BEV OSINT Framework - Processing Core Setup"
    echo "========================================"
    echo

    case "${1:-setup}" in
        "setup")
            check_root
            check_dependencies
            create_directories
            set_permissions
            setup_environment
            create_sample_theme
            show_next_steps
            ;;
        "validate")
            validate_configuration
            ;;
        "pull")
            pull_images
            ;;
        "clean")
            log_info "Cleaning up..."
            rm -rf logs/
            rm -rf ssl/
            rm -rf mcp_server/config/
            log_success "Cleanup completed"
            ;;
        "help"|"--help"|"-h")
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  setup     - Set up directory structure and configuration (default)"
            echo "  validate  - Validate configuration and test connectivity"
            echo "  pull      - Pull Docker images"
            echo "  clean     - Clean up created directories"
            echo "  help      - Show this help message"
            echo
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"