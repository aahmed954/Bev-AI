# Live2D Avatar Frontend
FROM node:18-alpine

LABEL maintainer="BEV OSINT Team"
LABEL description="Live2D Avatar Frontend Service"

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Expose port
EXPOSE 3000

# Start application
CMD ["npm", "start"]
