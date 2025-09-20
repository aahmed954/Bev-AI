<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import cytoscape from 'cytoscape';
    import fcose from 'cytoscape-fcose';
    import * as echarts from 'echarts';
    import DOMPurify from 'dompurify';
    import { invoke } from '@tauri-apps/api/core';
    
    cytoscape.use(fcose);
    
    interface Wallet {
        address: string;
        label?: string;
        balance: number;
        currency: string;
        transactions: number;
        firstSeen: Date;
        lastActive: Date;
        riskScore: number;
        tags: string[];
        cluster?: string;
    }
    
    interface Transaction {
        txId: string;
        from: string;
        to: string;
        amount: number;
        currency: string;
        timestamp: Date;
        fee: number;
        confirmations: number;
        mixer?: boolean;
        suspicious?: boolean;
    }
    
    interface AddressCluster {
        id: string;
        name: string;
        addresses: string[];
        totalBalance: number;
        entity?: string;
        riskLevel: 'low' | 'medium' | 'high' | 'critical';
    }
    
    interface PriceData {
        currency: string;
        price: number;
        change24h: number;
        volume24h: number;
        marketCap: number;
    }
    
    interface MixerAnalysis {
        address: string;
        mixerScore: number;
        obfuscationLayers: number;
        relatedAddresses: string[];
        confidence: number;
    }
    
    // Component state
    let wallets: Wallet[] = [];
    let transactions: Transaction[] = [];
    let clusters: AddressCluster[] = [];
    let priceData: Map<string, PriceData> = new Map();
    let selectedWallet: Wallet | null = null;
    let searchAddress = '';
    let selectedCurrency = 'BTC';
    let timeRange = '24h';
    let showOnlyFlagged = false;
    let mixerAnalysis: MixerAnalysis | null = null;
    
    // Visualization elements
    let graphContainer: HTMLElement;
    let flowChart: HTMLElement;
    let volumeChart: HTMLElement;
    let clusterChart: HTMLElement;
    let cy: any;
    let flowChartInstance: any;
    let volumeChartInstance: any;
    let clusterChartInstance: any;
    
    // WebSocket connection
    let ws: WebSocket | null = null;
    let isTracking = false;
    let lastBlockHeight = 0;
    let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
    
    // Supported currencies
    const currencies = ['BTC', 'ETH', 'XMR', 'LTC', 'USDT', 'USDC'];
    const timeRanges = ['1h', '24h', '7d', '30d', '1y'];
    
    onMount(() => {
        initializeGraph();
        initializeCharts();
        loadInitialData();
        connectBlockchainStream();
    });
    
    onDestroy(() => {
        if (ws) ws.close();
        if (cy) cy.destroy();
        if (flowChartInstance) flowChartInstance.dispose();
        if (volumeChartInstance) volumeChartInstance.dispose();
        if (clusterChartInstance) clusterChartInstance.dispose();
    });
    
    function initializeGraph() {
        cy = cytoscape({
            container: graphContainer,
            style: [
                {
                    selector: 'node[type="wallet"]',
                    style: {
                        'background-color': 'mapData(riskScore, 0, 100, #00ff41, #ff0000)',
                        'label': 'data(label)',
                        'color': '#00ff41',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': 'mapData(balance, 0, 100, 20, 60)',
                        'height': 'mapData(balance, 0, 100, 20, 60)',
                        'border-width': 2,
                        'border-color': '#0a0a0a',
                        'font-size': '10px',
                        'text-wrap': 'ellipsis',
                        'text-max-width': '80px'
                    }
                },
                {
                    selector: 'node[type="exchange"]',
                    style: {
                        'background-color': '#00ccff',
                        'shape': 'diamond',
                        'label': 'data(label)',
                        'width': 40,
                        'height': 40
                    }
                },
                {
                    selector: 'node[type="mixer"]',
                    style: {
                        'background-color': '#ff9500',
                        'shape': 'star',
                        'label': 'data(label)',
                        'width': 35,
                        'height': 35
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 'mapData(amount, 0, 10, 1, 6)',
                        'line-color': 'mapData(suspicious, 0, 1, #00ff4144, #ff000066)',
                        'target-arrow-color': 'mapData(suspicious, 0, 1, #00ff41, #ff0000)',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.8,
                        'label': 'data(label)',
                        'font-size': '8px',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -10
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
                },
                {
                    selector: '.flagged',
                    style: {
                        'background-color': '#ff0000',
                        'border-width': 3,
                        'border-color': '#ff0000'
                    }
                }
            ],
            layout: {
                name: 'fcose',
                quality: 'proof',
                randomize: false,
                animate: true,
                animationDuration: 1000,
                nodeDimensionsIncludeLabels: true,
                nodeRepulsion: 4500,
                idealEdgeLength: 50,
                edgeElasticity: 0.45,
                nestingFactor: 0.1
            },
            wheelSensitivity: 0.2,
            maxZoom: 3,
            minZoom: 0.1
        });
        
        // Graph interactions
        cy.on('tap', 'node[type="wallet"]', (evt: any) => {
            const node = evt.target;
            selectWallet(node.data('walletData'));
            highlightTransactionPath(node);
        });
        
        cy.on('tap', (evt: any) => {
            if (evt.target === cy) {
                cy.elements().removeClass('highlighted');
                selectedWallet = null;
            }
        });
        
        // Right-click for mixer analysis
        cy.on('cxttap', 'node[type="wallet"]', (evt: any) => {
            const address = evt.target.data('id');
            analyzeMixer(address);
        });
    }
    
    function initializeCharts() {
        // Transaction Flow Chart
        flowChartInstance = echarts.init(flowChart, 'dark');
        const flowOptions = {
            title: {
                text: 'Transaction Flow Analysis',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                name: 'Volume (BTC)',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { lineStyle: { color: '#00ff4122' } }
            },
            series: [
                {
                    name: 'Inflow',
                    type: 'line',
                    smooth: true,
                    data: [],
                    lineStyle: { color: '#00ff41' },
                    areaStyle: { color: 'rgba(0, 255, 65, 0.2)' }
                },
                {
                    name: 'Outflow',
                    type: 'line',
                    smooth: true,
                    data: [],
                    lineStyle: { color: '#ff0000' },
                    areaStyle: { color: 'rgba(255, 0, 0, 0.2)' }
                }
            ],
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }
        };
        flowChartInstance.setOption(flowOptions);
        
        // Volume Distribution Chart
        volumeChartInstance = echarts.init(volumeChart, 'dark');
        const volumeOptions = {
            title: {
                text: 'Volume Distribution',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            series: [{
                type: 'pie',
                radius: ['40%', '70%'],
                data: [],
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowOffsetX: 0,
                        shadowColor: 'rgba(0, 255, 65, 0.5)'
                    }
                },
                label: {
                    color: '#00ff41'
                },
                labelLine: {
                    lineStyle: { color: '#00ff41' }
                }
            }]
        };
        volumeChartInstance.setOption(volumeOptions);
        
        // Cluster Analysis Chart
        clusterChartInstance = echarts.init(clusterChart, 'dark');
        const clusterOptions = {
            title: {
                text: 'Address Clustering',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            xAxis: {
                type: 'category',
                data: [],
                axisLine: { lineStyle: { color: '#00ff41' } },
                axisLabel: { rotate: 45 }
            },
            yAxis: {
                type: 'value',
                name: 'Total Balance',
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
        clusterChartInstance.setOption(clusterOptions);
        
        window.addEventListener('resize', () => {
            flowChartInstance.resize();
            volumeChartInstance.resize();
            clusterChartInstance.resize();
        });
    }
    
    async function loadInitialData() {
        try {
            const response = await invoke('get_crypto_data', {
                currency: selectedCurrency,
                timeRange: timeRange
            });
            
            const data = JSON.parse(DOMPurify.sanitize(JSON.stringify(response)));
            
            wallets = data.wallets || [];
            transactions = data.transactions || [];
            clusters = data.clusters || [];
            
            // Update price data
            if (data.prices) {
                currencies.forEach(curr => {
                    if (data.prices[curr]) {
                        priceData.set(curr, data.prices[curr]);
                    }
                });
                priceData = priceData;
            }
            
            updateVisualization();
            
        } catch (error) {
            console.error('Failed to load crypto data:', error);
            loadMockData();
        }
    }
    
    function loadMockData() {
        // Generate realistic mock blockchain data
        const mockAddresses = [
            '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', // Genesis
            'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh',
            '3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy',
            '1dice8EMZmqKvrGE4Qc9bUFf9PX3xaYDp',
            'bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97'
        ];
        
        // Generate mock wallets
        wallets = mockAddresses.map((addr, i) => ({
            address: addr,
            label: i === 0 ? 'Genesis Block' : i === 3 ? 'SatoshiDice' : undefined,
            balance: Math.random() * 100,
            currency: 'BTC',
            transactions: Math.floor(Math.random() * 1000),
            firstSeen: new Date(Date.now() - Math.random() * 365 * 86400000),
            lastActive: new Date(Date.now() - Math.random() * 7 * 86400000),
            riskScore: Math.random() * 100,
            tags: i === 0 ? ['genesis'] : i === 3 ? ['gambling'] : ['unknown'],
            cluster: i < 3 ? 'cluster_1' : undefined
        }));
        
        // Add exchange wallets
        wallets.push(
            {
                address: 'binance_hot_wallet',
                label: 'Binance Hot Wallet',
                balance: 5432.1,
                currency: 'BTC',
                transactions: 50000,
                firstSeen: new Date('2020-01-01'),
                lastActive: new Date(),
                riskScore: 10,
                tags: ['exchange', 'verified'],
                cluster: 'binance'
            },
            {
                address: 'coinbase_cold',
                label: 'Coinbase Cold Storage',
                balance: 10234.5,
                currency: 'BTC',
                transactions: 10000,
                firstSeen: new Date('2019-01-01'),
                lastActive: new Date(),
                riskScore: 5,
                tags: ['exchange', 'cold_storage'],
                cluster: 'coinbase'
            }
        );
        
        // Add mixer wallets
        wallets.push(
            {
                address: 'tornado_cash_1',
                label: 'Tornado Cash Pool',
                balance: 234.5,
                currency: 'ETH',
                transactions: 5000,
                firstSeen: new Date('2021-01-01'),
                lastActive: new Date(),
                riskScore: 90,
                tags: ['mixer', 'high_risk'],
                cluster: 'tornado'
            }
        );
        
        // Generate mock transactions
        transactions = [];
        for (let i = 0; i < 100; i++) {
            const fromWallet = wallets[Math.floor(Math.random() * wallets.length)];
            const toWallet = wallets[Math.floor(Math.random() * wallets.length)];
            
            if (fromWallet.address !== toWallet.address) {
                transactions.push({
                    txId: `tx_${Math.random().toString(36).substr(2, 9)}`,
                    from: fromWallet.address,
                    to: toWallet.address,
                    amount: Math.random() * 10,
                    currency: 'BTC',
                    timestamp: new Date(Date.now() - Math.random() * 86400000),
                    fee: Math.random() * 0.001,
                    confirmations: Math.floor(Math.random() * 100),
                    mixer: fromWallet.tags.includes('mixer') || toWallet.tags.includes('mixer'),
                    suspicious: fromWallet.riskScore > 70 || toWallet.riskScore > 70
                });
            }
        }
        
        // Generate clusters
        clusters = [
            {
                id: 'cluster_1',
                name: 'Early Bitcoin',
                addresses: wallets.slice(0, 3).map(w => w.address),
                totalBalance: wallets.slice(0, 3).reduce((sum, w) => sum + w.balance, 0),
                entity: 'Unknown Early Adopter',
                riskLevel: 'low'
            },
            {
                id: 'binance',
                name: 'Binance Exchange',
                addresses: ['binance_hot_wallet'],
                totalBalance: 5432.1,
                entity: 'Binance Holdings Ltd.',
                riskLevel: 'low'
            },
            {
                id: 'tornado',
                name: 'Tornado Cash',
                addresses: ['tornado_cash_1'],
                totalBalance: 234.5,
                entity: 'Tornado Cash Protocol',
                riskLevel: 'critical'
            }
        ];
        
        // Mock price data
        currencies.forEach(curr => {
            priceData.set(curr, {
                currency: curr,
                price: curr === 'BTC' ? 45000 : curr === 'ETH' ? 3000 : Math.random() * 1000,
                change24h: (Math.random() - 0.5) * 20,
                volume24h: Math.random() * 1000000000,
                marketCap: Math.random() * 100000000000
            });
        });
        
        updateVisualization();
    }
    
    function connectBlockchainStream() {
        try {
            connectionStatus = 'connecting';
            ws = new WebSocket('ws://localhost:3010/crypto-stream');
            
            ws.onopen = () => {
                connectionStatus = 'connected';
                isTracking = true;
                console.log('Connected to blockchain stream');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleBlockchainUpdate(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                connectionStatus = 'disconnected';
            };
            
            ws.onclose = () => {
                connectionStatus = 'disconnected';
                isTracking = false;
                setTimeout(connectBlockchainStream, 5000);
            };
            
        } catch (error) {
            console.error('Failed to connect blockchain stream:', error);
        }
    }
    
    function handleBlockchainUpdate(data: any) {
        switch (data.type) {
            case 'new_block':
                lastBlockHeight = data.height;
                break;
            case 'new_transaction':
                addTransaction(data.transaction);
                break;
            case 'wallet_update':
                updateWallet(data.wallet);
                break;
            case 'price_update':
                if (data.currency && data.price) {
                    priceData.set(data.currency, data.price);
                    priceData = priceData;
                }
                break;
        }
    }
    
    function addTransaction(tx: Transaction) {
        transactions = [tx, ...transactions].slice(0, 1000);
        updateVisualization();
    }
    
    function updateWallet(walletData: Partial<Wallet>) {
        const index = wallets.findIndex(w => w.address === walletData.address);
        if (index !== -1) {
            wallets[index] = { ...wallets[index], ...walletData };
        } else if (walletData.address) {
            wallets.push(walletData as Wallet);
        }
        wallets = wallets;
        updateVisualization();
    }
    
    function updateVisualization() {
        updateGraph();
        updateCharts();
    }
    
    function updateGraph() {
        if (!cy) return;
        
        const elements = [];
        const visibleWallets = showOnlyFlagged 
            ? wallets.filter(w => w.riskScore > 70)
            : wallets;
        
        // Add wallet nodes
        visibleWallets.slice(0, 50).forEach(wallet => {
            elements.push({
                data: {
                    id: wallet.address,
                    label: wallet.label || wallet.address.substring(0, 8) + '...',
                    type: wallet.tags.includes('exchange') ? 'exchange' : 
                          wallet.tags.includes('mixer') ? 'mixer' : 'wallet',
                    balance: wallet.balance,
                    riskScore: wallet.riskScore,
                    walletData: wallet
                },
                classes: wallet.riskScore > 70 ? 'flagged' : ''
            });
        });
        
        // Add transaction edges
        const recentTxs = transactions.slice(0, 100);
        recentTxs.forEach(tx => {
            if (visibleWallets.some(w => w.address === tx.from) &&
                visibleWallets.some(w => w.address === tx.to)) {
                elements.push({
                    data: {
                        id: tx.txId,
                        source: tx.from,
                        target: tx.to,
                        amount: tx.amount,
                        label: `${tx.amount.toFixed(4)} ${tx.currency}`,
                        suspicious: tx.suspicious ? 1 : 0
                    }
                });
            }
        });
        
        cy.elements().remove();
        cy.add(elements);
        cy.layout({ name: 'fcose', animate: true }).run();
    }
    
    function updateCharts() {
        if (!flowChartInstance || !volumeChartInstance || !clusterChartInstance) return;
        
        // Update flow chart
        const flowData = processFlowData();
        flowChartInstance.setOption({
            series: [
                { data: flowData.inflow },
                { data: flowData.outflow }
            ]
        });
        
        // Update volume chart
        const volumeData = processVolumeData();
        volumeChartInstance.setOption({
            series: [{ data: volumeData }]
        });
        
        // Update cluster chart
        const clusterData = clusters.map(c => ({
            name: c.name,
            value: c.totalBalance
        }));
        clusterChartInstance.setOption({
            xAxis: { data: clusterData.map(d => d.name) },
            series: [{ data: clusterData.map(d => d.value) }]
        });
    }
    
    function processFlowData() {
        const inflow: any[] = [];
        const outflow: any[] = [];
        
        // Group transactions by time
        const hourlyData = new Map();
        
        transactions.forEach(tx => {
            const hour = new Date(tx.timestamp);
            hour.setMinutes(0, 0, 0);
            const key = hour.getTime();
            
            if (!hourlyData.has(key)) {
                hourlyData.set(key, { inflow: 0, outflow: 0 });
            }
            
            const data = hourlyData.get(key);
            if (selectedWallet) {
                if (tx.to === selectedWallet.address) {
                    data.inflow += tx.amount;
                } else if (tx.from === selectedWallet.address) {
                    data.outflow += tx.amount;
                }
            }
        });
        
        hourlyData.forEach((data, timestamp) => {
            inflow.push([timestamp, data.inflow]);
            outflow.push([timestamp, data.outflow]);
        });
        
        return { inflow, outflow };
    }
    
    function processVolumeData() {
        const volumeMap = new Map();
        
        transactions.forEach(tx => {
            if (!volumeMap.has(tx.currency)) {
                volumeMap.set(tx.currency, 0);
            }
            volumeMap.set(tx.currency, volumeMap.get(tx.currency) + tx.amount);
        });
        
        return Array.from(volumeMap.entries()).map(([name, value]) => ({
            name,
            value
        }));
    }
    
    function highlightTransactionPath(node: any) {
        cy.elements().removeClass('highlighted');
        
        // Highlight node and its connections
        node.addClass('highlighted');
        
        // Trace transaction path
        const visited = new Set();
        const queue = [node];
        
        while (queue.length > 0) {
            const current = queue.shift();
            if (visited.has(current.id())) continue;
            visited.add(current.id());
            
            current.connectedEdges().forEach((edge: any) => {
                edge.addClass('highlighted');
                const connected = edge.source().id() === current.id() 
                    ? edge.target() 
                    : edge.source();
                
                if (!visited.has(connected.id())) {
                    connected.addClass('highlighted');
                    queue.push(connected);
                }
            });
        }
    }
    
    function selectWallet(wallet: Wallet) {
        selectedWallet = wallet;
        updateCharts();
    }
    
    async function searchAddress() {
        if (!searchAddress) return;
        
        try {
            const result = await invoke('search_address', { 
                address: searchAddress 
            });
            
            const wallet = JSON.parse(DOMPurify.sanitize(JSON.stringify(result)));
            
            if (wallet) {
                wallets = [wallet, ...wallets.filter(w => w.address !== wallet.address)];
                selectWallet(wallet);
                updateVisualization();
                
                // Center graph on searched address
                const node = cy.$(`#${wallet.address}`);
                if (node.length > 0) {
                    cy.animate({
                        center: { eles: node },
                        zoom: 2
                    });
                }
            }
        } catch (error) {
            console.error('Address search failed:', error);
        }
    }
    
    async function analyzeMixer(address: string) {
        try {
            const result = await invoke('analyze_mixer', { address });
            mixerAnalysis = JSON.parse(DOMPurify.sanitize(JSON.stringify(result)));
        } catch (error) {
            console.error('Mixer analysis failed:', error);
            
            // Mock mixer analysis
            mixerAnalysis = {
                address,
                mixerScore: Math.random() * 100,
                obfuscationLayers: Math.floor(Math.random() * 5) + 1,
                relatedAddresses: wallets.slice(0, 3).map(w => w.address),
                confidence: 50 + Math.random() * 50
            };
        }
    }
    
    function exportData() {
        const exportData = {
            wallets: wallets.slice(0, 100),
            transactions: transactions.slice(0, 500),
            clusters,
            priceData: Array.from(priceData.entries()),
            exportDate: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `crypto-analysis-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    function toggleFlagged() {
        showOnlyFlagged = !showOnlyFlagged;
        updateVisualization();
    }
</script>

<div class="crypto-tracker">
    <header class="tracker-header">
        <div class="header-left">
            <h2>Cryptocurrency Tracking</h2>
            <div class="connection-status" class:connected={connectionStatus === 'connected'}>
                <span class="status-indicator"></span>
                {connectionStatus === 'connected' ? 'LIVE' : connectionStatus.toUpperCase()}
            </div>
            {#if lastBlockHeight > 0}
                <span class="block-height">Block: {lastBlockHeight.toLocaleString()}</span>
            {/if}
        </div>
        
        <div class="header-controls">
            <div class="search-group">
                <input
                    type="text"
                    placeholder="Search wallet address..."
                    bind:value={searchAddress}
                    on:keydown={(e) => e.key === 'Enter' && searchAddress()}
                    class="address-input"
                />
                <button on:click={searchAddress} class="search-btn">Track</button>
            </div>
            
            <select bind:value={selectedCurrency} on:change={loadInitialData} class="currency-select">
                {#each currencies as curr}
                    <option value={curr}>{curr}</option>
                {/each}
            </select>
            
            <select bind:value={timeRange} on:change={loadInitialData} class="time-select">
                {#each timeRanges as range}
                    <option value={range}>{range}</option>
                {/each}
            </select>
            
            <button on:click={toggleFlagged} class="filter-btn" class:active={showOnlyFlagged}>
                {showOnlyFlagged ? 'Show All' : 'Flagged Only'}
            </button>
            
            <button on:click={exportData} class="export-btn">Export</button>
        </div>
    </header>
    
    <!-- Price Ticker -->
    <div class="price-ticker">
        {#each Array.from(priceData.values()) as price}
            <div class="price-item">
                <span class="currency-name">{price.currency}</span>
                <span class="price-value">${price.price.toFixed(2)}</span>
                <span class="price-change" class:positive={price.change24h > 0}>
                    {price.change24h > 0 ? '+' : ''}{price.change24h.toFixed(2)}%
                </span>
            </div>
        {/each}
    </div>
    
    <div class="tracker-grid">
        <!-- Wallet List -->
        <div class="panel wallet-panel">
            <h3>Tracked Wallets</h3>
            <div class="wallet-list">
                {#each wallets.slice(0, 20) as wallet}
                    <div 
                        class="wallet-item"
                        class:selected={selectedWallet?.address === wallet.address}
                        class:high-risk={wallet.riskScore > 70}
                        on:click={() => selectWallet(wallet)}
                    >
                        <div class="wallet-header">
                            <span class="wallet-address">
                                {wallet.label || wallet.address.substring(0, 12) + '...'}
                            </span>
                            <span class="risk-score" style="color: hsl({120 - wallet.riskScore * 1.2}, 100%, 50%)">
                                Risk: {wallet.riskScore.toFixed(0)}
                            </span>
                        </div>
                        <div class="wallet-stats">
                            <span>{wallet.balance.toFixed(4)} {wallet.currency}</span>
                            <span>{wallet.transactions} txs</span>
                            <span>Active: {new Date(wallet.lastActive).toLocaleDateString()}</span>
                        </div>
                        <div class="wallet-tags">
                            {#each wallet.tags as tag}
                                <span class="tag">{tag}</span>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Transaction Graph -->
        <div class="panel graph-panel">
            <h3>Transaction Network</h3>
            <div class="graph-container" bind:this={graphContainer}></div>
            <div class="graph-legend">
                <span><span class="legend-dot wallet"></span>Wallet</span>
                <span><span class="legend-dot exchange"></span>Exchange</span>
                <span><span class="legend-dot mixer"></span>Mixer</span>
                <span><span class="legend-dot flagged"></span>Flagged</span>
            </div>
        </div>
        
        <!-- Selected Wallet Details -->
        {#if selectedWallet}
        <div class="panel details-panel">
            <h3>Wallet Details</h3>
            <div class="details-content">
                <div class="detail-row">
                    <span>Address:</span>
                    <span class="mono">{selectedWallet.address}</span>
                </div>
                {#if selectedWallet.label}
                <div class="detail-row">
                    <span>Label:</span>
                    <span>{selectedWallet.label}</span>
                </div>
                {/if}
                <div class="detail-row">
                    <span>Balance:</span>
                    <span>{selectedWallet.balance.toFixed(8)} {selectedWallet.currency}</span>
                </div>
                <div class="detail-row">
                    <span>USD Value:</span>
                    <span>${(selectedWallet.balance * (priceData.get(selectedWallet.currency)?.price || 0)).toFixed(2)}</span>
                </div>
                <div class="detail-row">
                    <span>Transactions:</span>
                    <span>{selectedWallet.transactions.toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span>First Seen:</span>
                    <span>{new Date(selectedWallet.firstSeen).toLocaleDateString()}</span>
                </div>
                <div class="detail-row">
                    <span>Last Active:</span>
                    <span>{new Date(selectedWallet.lastActive).toLocaleString()}</span>
                </div>
                <div class="detail-row">
                    <span>Risk Score:</span>
                    <span style="color: hsl({120 - selectedWallet.riskScore * 1.2}, 100%, 50%)">
                        {selectedWallet.riskScore.toFixed(0)}/100
                    </span>
                </div>
                {#if selectedWallet.cluster}
                <div class="detail-row">
                    <span>Cluster:</span>
                    <span>{selectedWallet.cluster}</span>
                </div>
                {/if}
            </div>
        </div>
        {/if}
        
        <!-- Mixer Analysis -->
        {#if mixerAnalysis}
        <div class="panel mixer-panel">
            <h3>Mixer Analysis</h3>
            <div class="mixer-content">
                <div class="mixer-score">
                    <div class="score-label">Mixer Probability</div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {mixerAnalysis.mixerScore}%"></div>
                    </div>
                    <div class="score-value">{mixerAnalysis.mixerScore.toFixed(1)}%</div>
                </div>
                <div class="mixer-details">
                    <div class="detail-row">
                        <span>Obfuscation Layers:</span>
                        <span>{mixerAnalysis.obfuscationLayers}</span>
                    </div>
                    <div class="detail-row">
                        <span>Confidence:</span>
                        <span>{mixerAnalysis.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="related-addresses">
                        <h4>Related Addresses</h4>
                        {#each mixerAnalysis.relatedAddresses as addr}
                            <div class="related-addr">{addr.substring(0, 16)}...</div>
                        {/each}
                    </div>
                </div>
            </div>
        </div>
        {/if}
        
        <!-- Flow Chart -->
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={flowChart}></div>
        </div>
        
        <!-- Volume Chart -->
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={volumeChart}></div>
        </div>
        
        <!-- Cluster Chart -->
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={clusterChart}></div>
        </div>
        
        <!-- Recent Transactions -->
        <div class="panel transactions-panel">
            <h3>Recent Transactions</h3>
            <div class="transaction-list">
                {#each transactions.slice(0, 20) as tx}
                    <div class="transaction-item" class:suspicious={tx.suspicious}>
                        <div class="tx-header">
                            <span class="tx-id">{tx.txId.substring(0, 8)}...</span>
                            <span class="tx-time">{new Date(tx.timestamp).toLocaleTimeString()}</span>
                        </div>
                        <div class="tx-flow">
                            <span class="tx-from">{tx.from.substring(0, 8)}...</span>
                            <span class="tx-arrow">→</span>
                            <span class="tx-to">{tx.to.substring(0, 8)}...</span>
                        </div>
                        <div class="tx-details">
                            <span class="tx-amount">{tx.amount.toFixed(6)} {tx.currency}</span>
                            <span class="tx-confirmations">{tx.confirmations} conf</span>
                            {#if tx.mixer}
                                <span class="mixer-badge">MIXER</span>
                            {/if}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Address Clusters -->
        <div class="panel clusters-panel">
            <h3>Address Clusters</h3>
            <div class="cluster-list">
                {#each clusters as cluster}
                    <div class="cluster-item" class:critical={cluster.riskLevel === 'critical'}>
                        <div class="cluster-header">
                            <span class="cluster-name">{cluster.name}</span>
                            <span class="risk-level {cluster.riskLevel}">{cluster.riskLevel.toUpperCase()}</span>
                        </div>
                        <div class="cluster-stats">
                            <span>Addresses: {cluster.addresses.length}</span>
                            <span>Balance: {cluster.totalBalance.toFixed(2)} BTC</span>
                        </div>
                        {#if cluster.entity}
                            <div class="cluster-entity">Entity: {cluster.entity}</div>
                        {/if}
                    </div>
                {/each}
            </div>
        </div>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar">
        <div class="status-item">
            <span class="label">Tracking:</span>
            <span class="value" class:active={isTracking}>{isTracking ? 'ACTIVE' : 'PAUSED'}</span>
        </div>
        <div class="status-item">
            <span class="label">Wallets:</span>
            <span class="value">{wallets.length}</span>
        </div>
        <div class="status-item">
            <span class="label">Transactions:</span>
            <span class="value">{transactions.length}</span>
        </div>
        <div class="status-item">
            <span class="label">Flagged:</span>
            <span class="value">{wallets.filter(w => w.riskScore > 70).length}</span>
        </div>
        <div class="status-item">
            <span class="label">Total Volume:</span>
            <span class="value">{transactions.reduce((sum, tx) => sum + tx.amount, 0).toFixed(2)} BTC</span>
        </div>
    </div>
</div>

<style>
    .crypto-tracker {
        padding: 1rem;
        color: var(--text-primary, #00ff41);
        background: var(--bg-primary, #0a0a0a);
        min-height: 100vh;
    }
    
    .tracker-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
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
    
    h4 {
        margin: 0.5rem 0;
        font-size: 0.75rem;
        text-transform: uppercase;
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
    
    .block-height {
        padding: 0.25rem 0.75rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    .header-controls {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    
    .search-group {
        display: flex;
        gap: 0.5rem;
    }
    
    .address-input {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        width: 300px;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
    }
    
    .address-input::placeholder {
        color: var(--text-secondary, #00ff4166);
    }
    
    .search-btn, .filter-btn, .export-btn {
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
    
    .search-btn:hover, .filter-btn:hover, .export-btn:hover {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .filter-btn.active {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .currency-select, .time-select {
        padding: 0.5rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        cursor: pointer;
    }
    
    .price-ticker {
        display: flex;
        gap: 2rem;
        padding: 0.75rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        margin-bottom: 1rem;
        overflow-x: auto;
    }
    
    .price-item {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        white-space: nowrap;
    }
    
    .currency-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .price-value {
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
    }
    
    .price-change {
        font-size: 0.75rem;
        color: #ff0000;
    }
    
    .price-change.positive {
        color: #00ff41;
    }
    
    .tracker-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
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
        position: relative;
    }
    
    .graph-legend {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
        font-size: 0.75rem;
    }
    
    .legend-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 0.25rem;
    }
    
    .legend-dot.wallet {
        background: #00ff41;
    }
    
    .legend-dot.exchange {
        background: #00ccff;
    }
    
    .legend-dot.mixer {
        background: #ff9500;
    }
    
    .legend-dot.flagged {
        background: #ff0000;
    }
    
    .chart-container {
        width: 100%;
        height: 300px;
    }
    
    .wallet-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .wallet-item {
        padding: 0.75rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .wallet-item:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .wallet-item.selected {
        border-color: var(--text-primary, #00ff41);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .wallet-item.high-risk {
        border-color: #ff000044;
        background: rgba(255, 0, 0, 0.05);
    }
    
    .wallet-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .wallet-address {
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    .risk-score {
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .wallet-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .wallet-tags {
        display: flex;
        gap: 0.25rem;
        flex-wrap: wrap;
    }
    
    .tag {
        padding: 0.125rem 0.5rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        font-size: 0.625rem;
        text-transform: uppercase;
    }
    
    .details-content, .mixer-content {
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
    
    .mono {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        word-break: break-all;
    }
    
    .mixer-score {
        margin-bottom: 1rem;
    }
    
    .score-label {
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .score-bar {
        width: 100%;
        height: 20px;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #00ff41, #ff9500, #ff0000);
        transition: width 0.3s;
    }
    
    .score-value {
        text-align: center;
        margin-top: 0.25rem;
        font-weight: bold;
    }
    
    .related-addresses {
        margin-top: 0.5rem;
    }
    
    .related-addr {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        padding: 0.25rem 0;
        color: var(--text-secondary, #00ff4199);
    }
    
    .transaction-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .transaction-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
    }
    
    .transaction-item.suspicious {
        border-color: #ff000044;
        background: rgba(255, 0, 0, 0.05);
    }
    
    .tx-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
        font-size: 0.75rem;
    }
    
    .tx-id {
        font-family: 'Courier New', monospace;
        color: var(--text-secondary, #00ff4199);
    }
    
    .tx-time {
        color: var(--text-secondary, #00ff4199);
    }
    
    .tx-flow {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
    }
    
    .tx-arrow {
        color: var(--text-primary, #00ff41);
    }
    
    .tx-details {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .tx-amount {
        font-weight: bold;
        color: var(--text-primary, #00ff41);
    }
    
    .mixer-badge {
        padding: 0.125rem 0.25rem;
        background: #ff9500;
        color: var(--bg-primary, #0a0a0a);
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
    }
    
    .cluster-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .cluster-item {
        padding: 0.75rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
    }
    
    .cluster-item.critical {
        border-color: #ff000044;
        background: rgba(255, 0, 0, 0.05);
    }
    
    .cluster-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .cluster-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .risk-level {
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
        text-transform: uppercase;
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
    
    .risk-level.critical {
        background: rgba(255, 0, 0, 0.4);
        color: #ff0000;
        animation: pulse 2s infinite;
    }
    
    .cluster-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .cluster-entity {
        margin-top: 0.25rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
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