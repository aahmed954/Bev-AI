<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import cytoscape from 'cytoscape';
    import fcose from 'cytoscape-fcose';
    import * as echarts from 'echarts';
    import DOMPurify from 'dompurify';
    import { invoke } from '@tauri-apps/api/core';
    
    // Register Cytoscape layout
    cytoscape.use(fcose);
    
    interface Market {
        id: string;
        name: string;
        vendors: number;
        products: number;
        volume: number;
        status: 'active' | 'seized' | 'scam' | 'maintenance';
        lastSeen: Date;
        escrowBalance: number;
        categories: string[];
        trustScore: number;
    }
    
    interface Vendor {
        id: string;
        name: string;
        markets: string[];
        rating: number;
        sales: number;
        pgpVerified: boolean;
        specialties: string[];
        riskLevel: 'low' | 'medium' | 'high';
    }
    
    interface Product {
        id: string;
        title: string;
        vendor: string;
        market: string;
        price: number;
        currency: string;
        category: string;
        listingDate: Date;
        escrow: boolean;
    }
    
    interface TrendData {
        timestamp: Date;
        metric: string;
        value: number;
    }
    
    let markets: Market[] = [];
    let vendors: Vendor[] = [];
    let products: Product[] = [];
    let trends: TrendData[] = [];
    let selectedMarket: Market | null = null;
    let searchQuery = '';
    let filterCategory = 'all';
    let riskAlerts: any[] = [];
    
    let graphContainer: HTMLElement;
    let trendChart: HTMLElement;
    let volumeChart: HTMLElement;
    let cy: any;
    let trendChartInstance: any;
    let volumeChartInstance: any;
    let websocket: WebSocket | null = null;
    
    // Real-time monitoring state
    let isMonitoring = false;
    let lastUpdate = new Date();
    let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
    
    // Categories for filtering
    const categories = [
        'all', 'drugs', 'fraud', 'digital_goods', 'counterfeit',
        'hacking', 'weapons', 'documents', 'other'
    ];
    
    onMount(() => {
        initializeGraphs();
        initializeCharts();
        loadInitialData();
        connectWebSocket();
    });
    
    onDestroy(() => {
        if (websocket) {
            websocket.close();
        }
        if (cy) {
            cy.destroy();
        }
        if (trendChartInstance) {
            trendChartInstance.dispose();
        }
        if (volumeChartInstance) {
            volumeChartInstance.dispose();
        }
    });
    
    function initializeGraphs() {
        cy = cytoscape({
            container: graphContainer,
            style: [
                {
                    selector: 'node[type="market"]',
                    style: {
                        'background-color': '#00ff41',
                        'label': 'data(label)',
                        'color': '#00ff41',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': 'mapData(size, 0, 100, 30, 80)',
                        'height': 'mapData(size, 0, 100, 30, 80)',
                        'border-width': 2,
                        'border-color': '#0a0a0a'
                    }
                },
                {
                    selector: 'node[type="vendor"]',
                    style: {
                        'background-color': '#ff9500',
                        'label': 'data(label)',
                        'color': '#ff9500',
                        'text-valign': 'bottom',
                        'text-margin-y': -5,
                        'width': 20,
                        'height': 20
                    }
                },
                {
                    selector: 'node[status="seized"]',
                    style: {
                        'background-color': '#ff0000',
                        'opacity': 0.5
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 'mapData(weight, 0, 10, 1, 5)',
                        'line-color': '#00ff4144',
                        'target-arrow-color': '#00ff41',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.7
                    }
                },
                {
                    selector: '.highlighted',
                    style: {
                        'background-color': '#ffff00',
                        'line-color': '#ffff00',
                        'target-arrow-color': '#ffff00',
                        'z-index': 999
                    }
                }
            ],
            layout: {
                name: 'fcose',
                quality: 'proof',
                randomize: true,
                animate: true,
                animationDuration: 1000,
                nodeDimensionsIncludeLabels: true
            },
            wheelSensitivity: 0.2
        });
        
        // Graph interaction handlers
        cy.on('tap', 'node', (evt: any) => {
            const node = evt.target;
            highlightConnections(node);
            if (node.data('type') === 'market') {
                selectMarket(node.data('marketData'));
            }
        });
        
        cy.on('tap', (evt: any) => {
            if (evt.target === cy) {
                cy.elements().removeClass('highlighted');
                selectedMarket = null;
            }
        });
    }
    
    function initializeCharts() {
        // Trend Chart
        trendChartInstance = echarts.init(trendChart, 'dark');
        const trendOptions = {
            title: {
                text: 'Market Activity Trends',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            legend: {
                data: ['Listings', 'Vendors', 'Transactions'],
                textStyle: { color: '#00ff41' }
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { lineStyle: { color: '#00ff4122' } }
            },
            series: [
                {
                    name: 'Listings',
                    type: 'line',
                    smooth: true,
                    data: [],
                    lineStyle: { color: '#00ff41' },
                    areaStyle: { color: 'rgba(0, 255, 65, 0.2)' }
                },
                {
                    name: 'Vendors',
                    type: 'line',
                    smooth: true,
                    data: [],
                    lineStyle: { color: '#ff9500' }
                },
                {
                    name: 'Transactions',
                    type: 'line',
                    smooth: true,
                    data: [],
                    lineStyle: { color: '#00ccff' }
                }
            ],
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }
        };
        trendChartInstance.setOption(trendOptions);
        
        // Volume Chart
        volumeChartInstance = echarts.init(volumeChart, 'dark');
        const volumeOptions = {
            title: {
                text: 'Transaction Volume by Market',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41',
                formatter: '{b}: {c} BTC'
            },
            xAxis: {
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: '#00ff41' } },
                axisLabel: { rotate: 45 }
            },
            yAxis: {
                type: 'value',
                name: 'Volume (BTC)',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { lineStyle: { color: '#00ff4122' } }
            },
            series: [{
                type: 'bar',
                data: [],
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#00ff41' },
                        { offset: 1, color: '#00ff4166' }
                    ])
                }
            }],
            grid: {
                left: '3%',
                right: '4%',
                bottom: '15%',
                containLabel: true
            }
        };
        volumeChartInstance.setOption(volumeOptions);
        
        // Responsive resize
        window.addEventListener('resize', () => {
            trendChartInstance.resize();
            volumeChartInstance.resize();
        });
    }
    
    async function loadInitialData() {
        try {
            // Fetch initial data through Tauri IPC
            const response = await invoke('get_darknet_data', {
                filter: filterCategory,
                search: searchQuery
            });
            
            // Process and sanitize the response
            const data = JSON.parse(DOMPurify.sanitize(JSON.stringify(response)));
            
            markets = data.markets || [];
            vendors = data.vendors || [];
            products = data.products || [];
            trends = data.trends || [];
            riskAlerts = data.alerts || [];
            
            updateGraph();
            updateCharts();
            
        } catch (error) {
            console.error('Failed to load darknet data:', error);
            // Use mock data for development
            loadMockData();
        }
    }
    
    function loadMockData() {
        // Generate realistic mock data
        markets = [
            {
                id: 'alpha3',
                name: 'AlphaBay3',
                vendors: 2341,
                products: 45123,
                volume: 12.5,
                status: 'active',
                lastSeen: new Date(),
                escrowBalance: 523.7,
                categories: ['drugs', 'fraud', 'digital_goods'],
                trustScore: 8.5
            },
            {
                id: 'darkmarket',
                name: 'DarkMarket',
                vendors: 1876,
                products: 32456,
                volume: 8.3,
                status: 'active',
                lastSeen: new Date(),
                escrowBalance: 312.4,
                categories: ['drugs', 'counterfeit', 'documents'],
                trustScore: 7.2
            },
            {
                id: 'hydra',
                name: 'Hydra',
                vendors: 3245,
                products: 67891,
                volume: 23.7,
                status: 'seized',
                lastSeen: new Date(Date.now() - 86400000 * 30),
                escrowBalance: 0,
                categories: ['drugs'],
                trustScore: 0
            },
            {
                id: 'tormarket',
                name: 'TorMarket',
                vendors: 987,
                products: 18234,
                volume: 5.2,
                status: 'active',
                lastSeen: new Date(),
                escrowBalance: 187.3,
                categories: ['digital_goods', 'hacking', 'fraud'],
                trustScore: 6.8
            }
        ];
        
        vendors = generateMockVendors(50);
        products = generateMockProducts(200);
        trends = generateMockTrends(30);
        
        updateGraph();
        updateCharts();
    }
    
    function generateMockVendors(count: number): Vendor[] {
        const vendorNames = ['CryptoKing', 'ShadowDealer', 'PharmLord', 'DigitalGhost', 'SecureVendor'];
        const result = [];
        
        for (let i = 0; i < count; i++) {
            result.push({
                id: `vendor_${i}`,
                name: vendorNames[i % vendorNames.length] + i,
                markets: markets.slice(0, Math.floor(Math.random() * 3) + 1).map(m => m.id),
                rating: 3 + Math.random() * 2,
                sales: Math.floor(Math.random() * 5000),
                pgpVerified: Math.random() > 0.3,
                specialties: categories.slice(1, Math.floor(Math.random() * 3) + 2),
                riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as any
            });
        }
        
        return result;
    }
    
    function generateMockProducts(count: number): Product[] {
        const result = [];
        
        for (let i = 0; i < count; i++) {
            const vendor = vendors[Math.floor(Math.random() * vendors.length)];
            const market = markets.filter(m => m.status === 'active')[
                Math.floor(Math.random() * markets.filter(m => m.status === 'active').length)
            ];
            
            result.push({
                id: `product_${i}`,
                title: DOMPurify.sanitize(`Product ${i}`),
                vendor: vendor.id,
                market: market.id,
                price: Math.random() * 5,
                currency: 'BTC',
                category: categories[Math.floor(Math.random() * (categories.length - 1)) + 1],
                listingDate: new Date(Date.now() - Math.random() * 86400000 * 30),
                escrow: Math.random() > 0.2
            });
        }
        
        return result;
    }
    
    function generateMockTrends(days: number): TrendData[] {
        const result = [];
        const metrics = ['listings', 'vendors', 'transactions'];
        
        for (let d = days; d >= 0; d--) {
            const date = new Date(Date.now() - d * 86400000);
            
            metrics.forEach(metric => {
                result.push({
                    timestamp: date,
                    metric,
                    value: Math.floor(Math.random() * 1000 + 500 + (30 - d) * 10)
                });
            });
        }
        
        return result;
    }
    
    function connectWebSocket() {
        try {
            connectionStatus = 'connecting';
            
            // Connect through secure WebSocket (wss://)
            websocket = new WebSocket('ws://localhost:3010/darknet-stream');
            
            websocket.onopen = () => {
                connectionStatus = 'connected';
                isMonitoring = true;
                console.log('Connected to darknet monitoring stream');
            };
            
            websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            };
            
            websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                connectionStatus = 'disconnected';
            };
            
            websocket.onclose = () => {
                connectionStatus = 'disconnected';
                isMonitoring = false;
                // Attempt reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            connectionStatus = 'disconnected';
        }
    }
    
    function handleRealtimeUpdate(data: any) {
        lastUpdate = new Date();
        
        switch (data.type) {
            case 'market_update':
                updateMarket(data.payload);
                break;
            case 'new_vendor':
                addVendor(data.payload);
                break;
            case 'new_listing':
                addProduct(data.payload);
                break;
            case 'alert':
                addAlert(data.payload);
                break;
            case 'trend_update':
                updateTrend(data.payload);
                break;
        }
        
        updateGraph();
        updateCharts();
    }
    
    function updateMarket(marketData: Partial<Market>) {
        const index = markets.findIndex(m => m.id === marketData.id);
        if (index !== -1) {
            markets[index] = { ...markets[index], ...marketData };
        } else if (marketData.id && marketData.name) {
            markets.push(marketData as Market);
        }
        markets = markets;
    }
    
    function addVendor(vendor: Vendor) {
        vendors = [...vendors, vendor];
    }
    
    function addProduct(product: Product) {
        products = [...products, product];
    }
    
    function addAlert(alert: any) {
        riskAlerts = [alert, ...riskAlerts].slice(0, 10);
    }
    
    function updateTrend(trendData: TrendData) {
        trends = [...trends, trendData].slice(-100);
    }
    
    function updateGraph() {
        if (!cy) return;
        
        const elements = [];
        
        // Add market nodes
        markets.forEach(market => {
            elements.push({
                data: {
                    id: market.id,
                    label: market.name,
                    type: 'market',
                    status: market.status,
                    size: market.products / 1000,
                    marketData: market
                }
            });
        });
        
        // Add vendor nodes and edges (limited for performance)
        const visibleVendors = vendors.slice(0, 20);
        visibleVendors.forEach(vendor => {
            elements.push({
                data: {
                    id: vendor.id,
                    label: vendor.name,
                    type: 'vendor',
                    vendorData: vendor
                }
            });
            
            vendor.markets.forEach(marketId => {
                elements.push({
                    data: {
                        id: `${vendor.id}-${marketId}`,
                        source: vendor.id,
                        target: marketId,
                        weight: vendor.sales / 1000
                    }
                });
            });
        });
        
        cy.elements().remove();
        cy.add(elements);
        cy.layout({ name: 'fcose', animate: true }).run();
    }
    
    function updateCharts() {
        if (!trendChartInstance || !volumeChartInstance) return;
        
        // Update trend chart
        const listingTrends = trends.filter(t => t.metric === 'listings')
            .map(t => [t.timestamp, t.value]);
        const vendorTrends = trends.filter(t => t.metric === 'vendors')
            .map(t => [t.timestamp, t.value]);
        const transactionTrends = trends.filter(t => t.metric === 'transactions')
            .map(t => [t.timestamp, t.value]);
        
        trendChartInstance.setOption({
            series: [
                { data: listingTrends },
                { data: vendorTrends },
                { data: transactionTrends }
            ]
        });
        
        // Update volume chart
        const volumeData = markets
            .filter(m => m.status === 'active')
            .map(m => ({ name: m.name, value: m.volume }));
        
        volumeChartInstance.setOption({
            xAxis: { data: volumeData.map(d => d.name) },
            series: [{ data: volumeData.map(d => d.value) }]
        });
    }
    
    function highlightConnections(node: any) {
        cy.elements().removeClass('highlighted');
        
        node.addClass('highlighted');
        node.connectedEdges().addClass('highlighted');
        node.connectedEdges().connectedNodes().addClass('highlighted');
    }
    
    function selectMarket(market: Market) {
        selectedMarket = market;
    }
    
    function filterData() {
        loadInitialData();
    }
    
    function exportData() {
        const exportData = {
            markets,
            vendors: vendors.slice(0, 100),
            products: products.slice(0, 500),
            trends,
            exportDate: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `darknet-export-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
</script>

<div class="darknet-monitor">
    <header class="monitor-header">
        <div class="header-left">
            <h2>Darknet Market Intelligence</h2>
            <div class="connection-status" class:connected={connectionStatus === 'connected'}>
                <span class="status-indicator"></span>
                {connectionStatus === 'connected' ? 'LIVE' : connectionStatus.toUpperCase()}
            </div>
        </div>
        
        <div class="header-controls">
            <input
                type="text"
                placeholder="Search markets, vendors, products..."
                bind:value={searchQuery}
                on:input={filterData}
                class="search-input"
            />
            
            <select bind:value={filterCategory} on:change={filterData} class="category-filter">
                {#each categories as category}
                    <option value={category}>
                        {category === 'all' ? 'All Categories' : category.replace('_', ' ').toUpperCase()}
                    </option>
                {/each}
            </select>
            
            <button on:click={exportData} class="export-btn">Export Data</button>
        </div>
    </header>
    
    <div class="monitor-grid">
        <!-- Market Overview -->
        <div class="panel market-overview">
            <h3>Active Markets</h3>
            <div class="market-list">
                {#each markets.filter(m => m.status === 'active') as market}
                    <div 
                        class="market-item" 
                        class:selected={selectedMarket?.id === market.id}
                        on:click={() => selectMarket(market)}
                    >
                        <div class="market-header">
                            <span class="market-name">{market.name}</span>
                            <span class="trust-score">Trust: {market.trustScore}/10</span>
                        </div>
                        <div class="market-metrics">
                            <div class="metric">
                                <span class="label">Vendors</span>
                                <span class="value">{market.vendors.toLocaleString()}</span>
                            </div>
                            <div class="metric">
                                <span class="label">Products</span>
                                <span class="value">{market.products.toLocaleString()}</span>
                            </div>
                            <div class="metric">
                                <span class="label">Volume</span>
                                <span class="value">{market.volume} BTC/day</span>
                            </div>
                            <div class="metric">
                                <span class="label">Escrow</span>
                                <span class="value">{market.escrowBalance} BTC</span>
                            </div>
                        </div>
                        <div class="market-categories">
                            {#each market.categories as cat}
                                <span class="category-tag">{cat}</span>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Graph Visualization -->
        <div class="panel graph-panel">
            <h3>Market Network Graph</h3>
            <div class="graph-container" bind:this={graphContainer}></div>
        </div>
        
        <!-- Risk Alerts -->
        <div class="panel alerts-panel">
            <h3>Risk Alerts</h3>
            <div class="alert-list">
                {#each riskAlerts as alert}
                    <div class="alert-item {alert.severity}">
                        <div class="alert-time">{new Date(alert.timestamp).toLocaleTimeString()}</div>
                        <div class="alert-message">{DOMPurify.sanitize(alert.message)}</div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Selected Market Details -->
        {#if selectedMarket}
        <div class="panel market-details">
            <h3>{selectedMarket.name} Details</h3>
            <div class="details-content">
                <div class="detail-row">
                    <span>Status:</span>
                    <span class="status {selectedMarket.status}">{selectedMarket.status.toUpperCase()}</span>
                </div>
                <div class="detail-row">
                    <span>Last Seen:</span>
                    <span>{selectedMarket.lastSeen.toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span>Trust Score:</span>
                    <span>{selectedMarket.trustScore}/10</span>
                </div>
                <div class="detail-row">
                    <span>Total Vendors:</span>
                    <span>{selectedMarket.vendors.toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span>Total Products:</span>
                    <span>{selectedMarket.products.toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span>Daily Volume:</span>
                    <span>{selectedMarket.volume} BTC</span>
                </div>
                <div class="detail-row">
                    <span>Escrow Balance:</span>
                    <span>{selectedMarket.escrowBalance} BTC</span>
                </div>
            </div>
        </div>
        {/if}
        
        <!-- Trend Charts -->
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={trendChart}></div>
        </div>
        
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={volumeChart}></div>
        </div>
        
        <!-- Top Vendors -->
        <div class="panel vendors-panel">
            <h3>Top Vendors</h3>
            <div class="vendor-list">
                {#each vendors.slice(0, 10) as vendor}
                    <div class="vendor-item">
                        <div class="vendor-header">
                            <span class="vendor-name">{vendor.name}</span>
                            {#if vendor.pgpVerified}
                                <span class="pgp-badge">PGP</span>
                            {/if}
                            <span class="risk-level {vendor.riskLevel}">{vendor.riskLevel.toUpperCase()}</span>
                        </div>
                        <div class="vendor-stats">
                            <span>Rating: {vendor.rating.toFixed(1)}/5</span>
                            <span>Sales: {vendor.sales.toLocaleString()}</span>
                            <span>Markets: {vendor.markets.length}</span>
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Recent Listings -->
        <div class="panel listings-panel">
            <h3>Recent Listings</h3>
            <div class="listing-list">
                {#each products.slice(0, 15) as product}
                    <div class="listing-item">
                        <div class="listing-header">
                            <span class="listing-title">{product.title}</span>
                            <span class="listing-price">{product.price} {product.currency}</span>
                        </div>
                        <div class="listing-meta">
                            <span class="listing-vendor">by {vendors.find(v => v.id === product.vendor)?.name || 'Unknown'}</span>
                            <span class="listing-market">on {markets.find(m => m.id === product.market)?.name || 'Unknown'}</span>
                            {#if product.escrow}
                                <span class="escrow-badge">ESCROW</span>
                            {/if}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar">
        <div class="status-item">
            <span class="label">Last Update:</span>
            <span class="value">{lastUpdate.toLocaleTimeString()}</span>
        </div>
        <div class="status-item">
            <span class="label">Active Markets:</span>
            <span class="value">{markets.filter(m => m.status === 'active').length}</span>
        </div>
        <div class="status-item">
            <span class="label">Total Vendors:</span>
            <span class="value">{vendors.length}</span>
        </div>
        <div class="status-item">
            <span class="label">Total Listings:</span>
            <span class="value">{products.length}</span>
        </div>
        <div class="status-item">
            <span class="label">Monitoring:</span>
            <span class="value" class:active={isMonitoring}>{isMonitoring ? 'ACTIVE' : 'PAUSED'}</span>
        </div>
    </div>
</div>

<style>
    .darknet-monitor {
        padding: 1rem;
        color: var(--text-primary, #00ff41);
        background: var(--bg-primary, #0a0a0a);
        min-height: 100vh;
    }
    
    .monitor-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border, #00ff4133);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    h2 {
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1.5rem;
    }
    
    h3 {
        margin: 0 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.875rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .connection-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0.75rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #ff0000;
        animation: pulse 2s infinite;
    }
    
    .connection-status.connected .status-indicator {
        background: #00ff41;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .header-controls {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .search-input {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        width: 300px;
    }
    
    .search-input::placeholder {
        color: var(--text-secondary, #00ff4166);
    }
    
    .category-filter {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        cursor: pointer;
    }
    
    .export-btn {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        cursor: pointer;
        text-transform: uppercase;
        font-size: 0.75rem;
        transition: all 0.2s;
    }
    
    .export-btn:hover {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .monitor-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .panel {
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        padding: 1rem;
    }
    
    .graph-panel {
        grid-column: span 2;
    }
    
    .chart-panel {
        min-height: 300px;
    }
    
    .graph-container {
        width: 100%;
        height: 500px;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
    }
    
    .chart-container {
        width: 100%;
        height: 300px;
    }
    
    .market-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .market-item {
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        padding: 0.75rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .market-item:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .market-item.selected {
        border-color: var(--text-primary, #00ff41);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .market-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .market-name {
        font-weight: bold;
        font-size: 1rem;
    }
    
    .trust-score {
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .market-metrics {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }
    
    .metric .label {
        color: var(--text-secondary, #00ff4199);
    }
    
    .metric .value {
        font-weight: bold;
    }
    
    .market-categories {
        display: flex;
        gap: 0.25rem;
        flex-wrap: wrap;
    }
    
    .category-tag {
        padding: 0.125rem 0.5rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        font-size: 0.625rem;
        text-transform: uppercase;
    }
    
    .alert-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 300px;
        overflow-y: auto;
    }
    
    .alert-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border-left: 3px solid var(--border, #00ff4133);
        border-radius: 2px;
    }
    
    .alert-item.high {
        border-left-color: #ff0000;
        background: rgba(255, 0, 0, 0.1);
    }
    
    .alert-item.medium {
        border-left-color: #ff9500;
        background: rgba(255, 149, 0, 0.1);
    }
    
    .alert-time {
        font-size: 0.625rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .alert-message {
        font-size: 0.75rem;
    }
    
    .details-content {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .detail-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.875rem;
        padding: 0.25rem 0;
        border-bottom: 1px solid var(--border, #00ff4111);
    }
    
    .status {
        text-transform: uppercase;
        font-weight: bold;
        font-size: 0.75rem;
    }
    
    .status.active { color: #00ff41; }
    .status.seized { color: #ff0000; }
    .status.scam { color: #ff9500; }
    .status.maintenance { color: #00ccff; }
    
    .vendor-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .vendor-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
    }
    
    .vendor-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    
    .vendor-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .pgp-badge {
        padding: 0.125rem 0.25rem;
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
    }
    
    .risk-level {
        margin-left: auto;
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
    }
    
    .risk-level.low {
        background: rgba(0, 255, 65, 0.2);
        color: #00ff41;
    }
    
    .risk-level.medium {
        background: rgba(255, 149, 0, 0.2);
        color: #ff9500;
    }
    
    .risk-level.high {
        background: rgba(255, 0, 0, 0.2);
        color: #ff0000;
    }
    
    .vendor-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .listing-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .listing-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
    }
    
    .listing-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    
    .listing-title {
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .listing-price {
        font-size: 0.875rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .listing-meta {
        display: flex;
        gap: 0.5rem;
        font-size: 0.625rem;
        color: var(--text-secondary, #00ff4166);
    }
    
    .escrow-badge {
        padding: 0.125rem 0.25rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--text-primary, #00ff41);
        border-radius: 2px;
        color: var(--text-primary, #00ff41);
    }
    
    .status-bar {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    .status-item {
        display: flex;
        gap: 0.5rem;
        font-size: 0.75rem;
    }
    
    .status-item .label {
        color: var(--text-secondary, #00ff4199);
    }
    
    .status-item .value {
        font-weight: bold;
    }
    
    .status-item .value.active {
        color: var(--text-primary, #00ff41);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary, #0f0f0f);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border, #00ff4133);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-primary, #00ff4144);
    }
</style>