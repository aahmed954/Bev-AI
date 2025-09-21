const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Basic health check
app.get('/health', (req, res) => {
    res.json({ status: 'ok', service: 'cytoscape-server' });
});

// Basic cytoscape endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'BEV Cytoscape Server',
        endpoints: {
            health: '/health',
            graph: '/graph'
        }
    });
});

// Graph rendering endpoint
app.post('/graph', (req, res) => {
    res.json({
        message: 'Graph endpoint ready',
        data: req.body
    });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Cytoscape server running on port ${PORT}`);
});