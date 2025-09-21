#!/usr/bin/env node

/**
 * BEV OSINT Framework - Cytoscape Visualization Server
 *
 * This server provides graph visualization capabilities for the BEV OSINT Framework
 * using Cytoscape.js for interactive network analysis and visualization.
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const winston = require('winston');
const neo4j = require('neo4j-driver');
const { Pool } = require('pg');
const path = require('path');
require('dotenv').config();

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Configure Winston logger
const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'cytoscape-server' },
    transports: [
        new winston.transports.File({ filename: './logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: './logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Database connections
let neo4jDriver = null;
let pgPool = null;

// Initialize database connections
async function initializeDatabases() {
    try {
        // Initialize Neo4j connection
        if (process.env.NEO4J_URI) {
            neo4jDriver = neo4j.driver(
                process.env.NEO4J_URI,
                neo4j.auth.basic(
                    process.env.NEO4J_USER || 'neo4j',
                    process.env.NEO4J_PASSWORD || 'password'
                )
            );
            await neo4jDriver.verifyConnectivity();
            logger.info('Connected to Neo4j database');
        }

        // Initialize PostgreSQL connection
        if (process.env.POSTGRES_URI) {
            pgPool = new Pool({
                connectionString: process.env.POSTGRES_URI,
                max: 10,
                idleTimeoutMillis: 30000,
                connectionTimeoutMillis: 5000
            });
            await pgPool.query('SELECT NOW()');
            logger.info('Connected to PostgreSQL database');
        }
    } catch (error) {
        logger.error('Database initialization failed:', error);
        throw error;
    }
}

// Middleware configuration
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            scriptSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"]
        }
    }
}));

app.use(compression());
app.use(cors({
    origin: process.env.CORS_ORIGIN || '*',
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.'
});
app.use('/api/', limiter);

// Logging middleware
app.use(morgan('combined', {
    stream: { write: message => logger.info(message.trim()) }
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Health check endpoint
app.get('/health', (req, res) => {
    const health = {
        status: 'OK',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: process.env.NODE_ENV || 'development',
        version: require('./package.json').version,
        databases: {
            neo4j: neo4jDriver ? 'connected' : 'disconnected',
            postgresql: pgPool ? 'connected' : 'disconnected'
        }
    };
    res.json(health);
});

// API Routes

// Get graph data from Neo4j
app.get('/api/graph/:type', async (req, res) => {
    try {
        const { type } = req.params;
        const { limit = 100, offset = 0 } = req.query;

        if (!neo4jDriver) {
            return res.status(503).json({ error: 'Neo4j not connected' });
        }

        const session = neo4jDriver.session();

        let query;
        switch (type) {
            case 'network':
                query = `
                    MATCH (n)-[r]->(m)
                    RETURN n, r, m
                    LIMIT $limit
                    SKIP $offset
                `;
                break;
            case 'entities':
                query = `
                    MATCH (n)
                    WHERE n.type IS NOT NULL
                    RETURN n
                    LIMIT $limit
                    SKIP $offset
                `;
                break;
            default:
                return res.status(400).json({ error: 'Invalid graph type' });
        }

        const result = await session.run(query, {
            limit: neo4j.int(limit),
            offset: neo4j.int(offset)
        });

        const nodes = new Map();
        const edges = [];

        result.records.forEach(record => {
            record.keys.forEach(key => {
                const item = record.get(key);

                if (item.labels) { // Node
                    const nodeId = item.identity.toString();
                    if (!nodes.has(nodeId)) {
                        nodes.set(nodeId, {
                            data: {
                                id: nodeId,
                                label: item.properties.name || item.properties.id || nodeId,
                                type: item.labels[0],
                                ...item.properties
                            }
                        });
                    }
                } else if (item.type) { // Relationship
                    edges.push({
                        data: {
                            id: item.identity.toString(),
                            source: item.start.toString(),
                            target: item.end.toString(),
                            type: item.type,
                            ...item.properties
                        }
                    });
                }
            });
        });

        await session.close();

        const graphData = {
            nodes: Array.from(nodes.values()),
            edges: edges
        };

        res.json(graphData);
    } catch (error) {
        logger.error('Error fetching graph data:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Search nodes
app.get('/api/search', async (req, res) => {
    try {
        const { q, type } = req.query;

        if (!q) {
            return res.status(400).json({ error: 'Query parameter required' });
        }

        if (!neo4jDriver) {
            return res.status(503).json({ error: 'Neo4j not connected' });
        }

        const session = neo4jDriver.session();

        let query = `
            MATCH (n)
            WHERE toLower(n.name) CONTAINS toLower($query)
        `;

        if (type) {
            query += ` AND n.type = $type`;
        }

        query += `
            RETURN n
            LIMIT 50
        `;

        const result = await session.run(query, { query: q, type });

        const nodes = result.records.map(record => {
            const node = record.get('n');
            return {
                data: {
                    id: node.identity.toString(),
                    label: node.properties.name || node.properties.id,
                    type: node.labels[0],
                    ...node.properties
                }
            };
        });

        await session.close();

        res.json(nodes);
    } catch (error) {
        logger.error('Error searching nodes:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Get analysis results from PostgreSQL
app.get('/api/analysis/:id', async (req, res) => {
    try {
        const { id } = req.params;

        if (!pgPool) {
            return res.status(503).json({ error: 'PostgreSQL not connected' });
        }

        const query = `
            SELECT * FROM analysis_results
            WHERE entity_id = $1
            ORDER BY created_at DESC
        `;

        const result = await pgPool.query(query, [id]);

        res.json(result.rows);
    } catch (error) {
        logger.error('Error fetching analysis data:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Export graph data
app.post('/api/export', async (req, res) => {
    try {
        const { format = 'json', nodes, edges } = req.body;

        if (!nodes || !edges) {
            return res.status(400).json({ error: 'Nodes and edges required' });
        }

        let exportData;
        let contentType;
        let filename;

        switch (format) {
            case 'json':
                exportData = JSON.stringify({ nodes, edges }, null, 2);
                contentType = 'application/json';
                filename = 'graph-export.json';
                break;
            case 'csv':
                // Convert to CSV format
                const csvNodes = nodes.map(n =>
                    `"${n.data.id}","${n.data.label}","${n.data.type}"`
                ).join('\n');
                const csvEdges = edges.map(e =>
                    `"${e.data.source}","${e.data.target}","${e.data.type}"`
                ).join('\n');

                exportData = `Nodes:\nId,Label,Type\n${csvNodes}\n\nEdges:\nSource,Target,Type\n${csvEdges}`;
                contentType = 'text/csv';
                filename = 'graph-export.csv';
                break;
            default:
                return res.status(400).json({ error: 'Unsupported format' });
        }

        res.setHeader('Content-Type', contentType);
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
        res.send(exportData);
    } catch (error) {
        logger.error('Error exporting data:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Serve main application
app.get('/', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>BEV OSINT Framework - Graph Visualization</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { color: #2c3e50; margin-bottom: 20px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
                .status.ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
                .api-link {
                    display: inline-block;
                    margin: 5px 10px 5px 0;
                    padding: 8px 16px;
                    background: #3498db;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                }
                .api-link:hover { background: #2980b9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🕸️ BEV OSINT Framework</h1>
                <h2>Cytoscape Visualization Server</h2>

                <div class="status ok">
                    ✅ Server is running on port ${PORT}
                </div>

                <h3>Available Endpoints:</h3>
                <a href="/health" class="api-link">Health Check</a>
                <a href="/api/graph/network" class="api-link">Network Graph</a>
                <a href="/api/graph/entities" class="api-link">Entity Graph</a>
                <a href="/api/search?q=example" class="api-link">Search API</a>

                <h3>Integration:</h3>
                <p>This server provides REST API endpoints for graph visualization data that can be consumed by the main BEV OSINT Framework interface.</p>

                <h3>Documentation:</h3>
                <p>For detailed API documentation and integration examples, see the main BEV OSINT Framework repository.</p>
            </div>
        </body>
        </html>
    `);
});

// Error handling middleware
app.use((error, req, res, next) => {
    logger.error('Unhandled error:', error);
    res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ error: 'Endpoint not found' });
});

// Graceful shutdown handling
process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down gracefully');

    if (neo4jDriver) {
        await neo4jDriver.close();
        logger.info('Neo4j connection closed');
    }

    if (pgPool) {
        await pgPool.end();
        logger.info('PostgreSQL connection pool closed');
    }

    process.exit(0);
});

process.on('SIGINT', async () => {
    logger.info('SIGINT received, shutting down gracefully');

    if (neo4jDriver) {
        await neo4jDriver.close();
        logger.info('Neo4j connection closed');
    }

    if (pgPool) {
        await pgPool.end();
        logger.info('PostgreSQL connection pool closed');
    }

    process.exit(0);
});

// Start server
async function startServer() {
    try {
        await initializeDatabases();

        app.listen(PORT, '0.0.0.0', () => {
            logger.info(`🚀 Cytoscape server running on port ${PORT}`);
            logger.info(`📊 Environment: ${process.env.NODE_ENV || 'development'}`);
            logger.info(`🔗 Neo4j: ${neo4jDriver ? 'connected' : 'disconnected'}`);
            logger.info(`🗄️  PostgreSQL: ${pgPool ? 'connected' : 'disconnected'}`);
        });
    } catch (error) {
        logger.error('Failed to start server:', error);
        process.exit(1);
    }
}

// Start the server
startServer();