# BEV OSINT Framework - Cytoscape Visualization Server
# Node.js application for graph visualization and network analysis

FROM node:18-alpine AS base

# Install system dependencies
RUN apk add --no-cache \
    dumb-init \
    curl \
    && rm -rf /var/cache/apk/*

# Create app directory
WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Development stage
FROM base AS development
ENV NODE_ENV=development

# Install all dependencies including dev
RUN npm ci --include=dev

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start development server
CMD ["npm", "run", "dev"]

# Production stage
FROM base AS production
ENV NODE_ENV=production

# Install only production dependencies
RUN npm ci --only=production && npm cache clean --force

# Copy source code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodeapp -u 1001 -G nodejs

# Create directories for logs and data
RUN mkdir -p logs data && \
    chown -R nodeapp:nodejs /app

# Switch to non-root user
USER nodeapp

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start production server
CMD ["npm", "start"]