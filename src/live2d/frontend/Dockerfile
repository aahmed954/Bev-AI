FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Copy application files
COPY index.html .
COPY avatar.js .
COPY controls.js .
COPY styles.css .
COPY assets/ ./assets/

# Install serve for production
RUN npm install -g serve

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Run the app
CMD ["serve", "-s", ".", "-l", "3000"]