<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import * as echarts from 'echarts';
    import cytoscape from 'cytoscape';
    import fcose from 'cytoscape-fcose';
    import DOMPurify from 'dompurify';
    import { invoke } from '@tauri-apps/api/core';
    
    cytoscape.use(fcose);
    
    interface ThreatActor {
        id: string;
        name: string;
        aliases: string[];
        origin: string;
        active: boolean;
        sophistication: 'low' | 'medium' | 'high' | 'advanced';
        targets: string[];
        ttps: string[];
        campaigns: string[];
        lastSeen: Date;
        description: string;
    }
    
    interface IOC {
        id: string;
        type: 'ip' | 'domain' | 'hash' | 'email' | 'url' | 'cve' | 'yara';
        value: string;
        threatLevel: number;
        firstSeen: Date;
        lastSeen: Date;
        campaigns: string[];
        tags: string[];
        confidence: number;
        sources: string[];
    }
    
    interface Campaign {
        id: string;
        name: string;
        actor: string;
        status: 'active' | 'dormant' | 'completed';
        startDate: Date;
        endDate?: Date;
        targets: string[];
        industries: string[];
        techniques: string[];
        iocs: string[];
        severity: 'low' | 'medium' | 'high' | 'critical';
    }
    
    interface MitreAttack {
        tactic: string;
        techniques: {
            id: string;
            name: string;
            used: boolean;
            count: number;
        }[];
    }
    
    interface ThreatFeed {
        id: string;
        name: string;
        enabled: boolean;
        lastUpdate: Date;
        iocCount: number;
        reliability: number;
    }
    
    // Component state
    let threatActors: ThreatActor[] = [];
    let iocs: IOC[] = [];
    let campaigns: Campaign[] = [];
    let mitreMatrix: MitreAttack[] = [];
    let threatFeeds: ThreatFeed[] = [];
    let selectedActor: ThreatActor | null = null;
    let selectedCampaign: Campaign | null = null;
    let selectedIOC: IOC | null = null;
    
    // Filters
    let searchQuery = '';
    let iocTypeFilter = 'all';
    let severityFilter = 'all';
    let timeRange = '24h';
    let showActiveOnly = false;
    
    // Visualization elements
    let threatGraphContainer: HTMLElement;
    let threatMapContainer: HTMLElement;
    let iocTimelineContainer: HTMLElement;
    let severityChartContainer: HTMLElement;
    let cy: any;
    let threatMapInstance: any;
    let timelineInstance: any;
    let severityInstance: any;
    
    // Real-time monitoring
    let ws: WebSocket | null = null;
    let isMonitoring = false;
    let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
    let alertQueue: any[] = [];
    let lastUpdate = new Date();
    
    // Constants
    const iocTypes = ['all', 'ip', 'domain', 'hash', 'email', 'url', 'cve', 'yara'];
    const severityLevels = ['all', 'low', 'medium', 'high', 'critical'];
    const timeRanges = ['1h', '24h', '7d', '30d', '90d'];
    
    onMount(() => {
        initializeThreatGraph();
        initializeCharts();
        loadThreatData();
        connectThreatStream();
    });
    
    onDestroy(() => {
        if (ws) ws.close();
        if (cy) cy.destroy();
        if (threatMapInstance) threatMapInstance.dispose();
        if (timelineInstance) timelineInstance.dispose();
        if (severityInstance) severityInstance.dispose();
    });
    
    function initializeThreatGraph() {
        cy = cytoscape({
            container: threatGraphContainer,
            style: [
                {
                    selector: 'node[type="actor"]',
                    style: {
                        'background-color': 'mapData(sophistication, 0, 3, #00ff41, #ff0000)',
                        'label': 'data(label)',
                        'color': '#00ff41',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': 50,
                        'height': 50,
                        'shape': 'hexagon',
                        'border-width': 2,
                        'border-color': '#0a0a0a',
                        'font-size': '10px'
                    }
                },
                {
                    selector: 'node[type="campaign"]',
                    style: {
                        'background-color': '#ff9500',
                        'label': 'data(label)',
                        'shape': 'roundrectangle',
                        'width': 40,
                        'height': 30,
                        'font-size': '8px'
                    }
                },
                {
                    selector: 'node[type="target"]',
                    style: {
                        'background-color': '#00ccff',
                        'label': 'data(label)',
                        'shape': 'ellipse',
                        'width': 35,
                        'height': 35,
                        'font-size': '8px'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#00ff4144',
                        'target-arrow-color': '#00ff41',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.7,
                        'label': 'data(label)',
                        'font-size': '8px',
                        'text-rotation': 'autorotate'
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
                    selector: '.active-threat',
                    style: {
                        'border-width': 3,
                        'border-color': '#ff0000',
                        'background-color': '#ff000033'
                    }
                }
            ],
            layout: {
                name: 'fcose',
                quality: 'proof',
                randomize: false,
                animate: true,
                animationDuration: 1000,
                nodeDimensionsIncludeLabels: true
            },
            wheelSensitivity: 0.2
        });
        
        cy.on('tap', 'node', (evt: any) => {
            const node = evt.target;
            const type = node.data('type');
            
            if (type === 'actor') {
                selectActor(node.data('actorData'));
            } else if (type === 'campaign') {
                selectCampaign(node.data('campaignData'));
            }
            
            highlightConnections(node);
        });
        
        cy.on('tap', (evt: any) => {
            if (evt.target === cy) {
                cy.elements().removeClass('highlighted');
                selectedActor = null;
                selectedCampaign = null;
            }
        });
    }
    
    function initializeCharts() {
        // Threat Map
        threatMapInstance = echarts.init(threatMapContainer, 'dark');
        const mapOptions = {
            title: {
                text: 'Global Threat Activity',
                textStyle: { color: '#00ff41' },
                left: 'center'
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41',
                formatter: '{b}: {c} threats'
            },
            visualMap: {
                min: 0,
                max: 100,
                calculable: true,
                inRange: {
                    color: ['#00ff4133', '#ff950066', '#ff0000aa']
                },
                textStyle: { color: '#00ff41' }
            },
            series: [{
                name: 'Threats',
                type: 'map',
                map: 'world',
                roam: true,
                emphasis: {
                    label: {
                        show: true,
                        color: '#00ff41'
                    },
                    itemStyle: {
                        areaColor: '#00ff41'
                    }
                },
                data: []
            }]
        };
        threatMapInstance.setOption(mapOptions);
        
        // IOC Timeline
        timelineInstance = echarts.init(iocTimelineContainer, 'dark');
        const timelineOptions = {
            title: {
                text: 'IOC Timeline',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'axis',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            legend: {
                data: ['IPs', 'Domains', 'Hashes', 'URLs'],
                textStyle: { color: '#00ff41' }
            },
            xAxis: {
                type: 'time',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { show: false }
            },
            yAxis: {
                type: 'value',
                name: 'Count',
                axisLine: { lineStyle: { color: '#00ff41' } },
                splitLine: { lineStyle: { color: '#00ff4122' } }
            },
            series: [
                {
                    name: 'IPs',
                    type: 'line',
                    stack: 'total',
                    data: [],
                    areaStyle: { color: 'rgba(0, 255, 65, 0.3)' }
                },
                {
                    name: 'Domains',
                    type: 'line',
                    stack: 'total',
                    data: [],
                    areaStyle: { color: 'rgba(0, 204, 255, 0.3)' }
                },
                {
                    name: 'Hashes',
                    type: 'line',
                    stack: 'total',
                    data: [],
                    areaStyle: { color: 'rgba(255, 149, 0, 0.3)' }
                },
                {
                    name: 'URLs',
                    type: 'line',
                    stack: 'total',
                    data: [],
                    areaStyle: { color: 'rgba(255, 0, 0, 0.3)' }
                }
            ],
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            }
        };
        timelineInstance.setOption(timelineOptions);
        
        // Severity Distribution
        severityInstance = echarts.init(severityChartContainer, 'dark');
        const severityOptions = {
            title: {
                text: 'Threat Severity Distribution',
                textStyle: { color: '#00ff41' }
            },
            tooltip: {
                trigger: 'item',
                backgroundColor: '#0a0a0a',
                borderColor: '#00ff41'
            },
            series: [{
                type: 'gauge',
                startAngle: 180,
                endAngle: 0,
                min: 0,
                max: 100,
                splitNumber: 5,
                axisLine: {
                    lineStyle: {
                        width: 30,
                        color: [
                            [0.2, '#00ff41'],
                            [0.4, '#00ccff'],
                            [0.6, '#ff9500'],
                            [0.8, '#ff6600'],
                            [1, '#ff0000']
                        ]
                    }
                },
                pointer: {
                    itemStyle: {
                        color: 'auto'
                    }
                },
                axisTick: {
                    distance: -30,
                    length: 8,
                    lineStyle: {
                        color: '#00ff41',
                        width: 2
                    }
                },
                splitLine: {
                    distance: -30,
                    length: 30,
                    lineStyle: {
                        color: '#00ff41',
                        width: 4
                    }
                },
                axisLabel: {
                    color: '#00ff41',
                    distance: 40
                },
                detail: {
                    valueAnimation: true,
                    formatter: '{value}',
                    color: '#00ff41',
                    fontSize: 20
                },
                data: [{ value: 0, name: 'Threat Level' }]
            }]
        };
        severityInstance.setOption(severityOptions);
        
        window.addEventListener('resize', () => {
            threatMapInstance.resize();
            timelineInstance.resize();
            severityInstance.resize();
        });
    }
    
    async function loadThreatData() {
        try {
            const response = await invoke('get_threat_intel_data', {
                timeRange: timeRange,
                filter: {
                    iocType: iocTypeFilter,
                    severity: severityFilter,
                    search: searchQuery
                }
            });
            
            const data = JSON.parse(DOMPurify.sanitize(JSON.stringify(response)));
            
            threatActors = data.actors || [];
            iocs = data.iocs || [];
            campaigns = data.campaigns || [];
            mitreMatrix = data.mitre || [];
            threatFeeds = data.feeds || [];
            
            updateVisualizations();
            
        } catch (error) {
            console.error('Failed to load threat intel:', error);
            loadMockData();
        }
    }
    
    function loadMockData() {
        // Generate realistic threat intelligence mock data
        threatActors = [
            {
                id: 'apt28',
                name: 'APT28',
                aliases: ['Fancy Bear', 'Sofacy', 'Pawn Storm'],
                origin: 'Russia',
                active: true,
                sophistication: 'advanced',
                targets: ['Government', 'Military', 'Defense'],
                ttps: ['T1566', 'T1083', 'T1057', 'T1055'],
                campaigns: ['campaign_1', 'campaign_3'],
                lastSeen: new Date(),
                description: 'Advanced persistent threat group attributed to Russian intelligence'
            },
            {
                id: 'lazarus',
                name: 'Lazarus Group',
                aliases: ['Hidden Cobra', 'Guardians of Peace'],
                origin: 'North Korea',
                active: true,
                sophistication: 'advanced',
                targets: ['Financial', 'Cryptocurrency', 'Entertainment'],
                ttps: ['T1189', 'T1203', 'T1105', 'T1571'],
                campaigns: ['campaign_2', 'campaign_4'],
                lastSeen: new Date(),
                description: 'State-sponsored threat group known for financial theft and destructive attacks'
            },
            {
                id: 'fin7',
                name: 'FIN7',
                aliases: ['Carbanak', 'Carbon Spider'],
                origin: 'Eastern Europe',
                active: true,
                sophistication: 'high',
                targets: ['Retail', 'Hospitality', 'Restaurant'],
                ttps: ['T1566', 'T1055', 'T1003', 'T1486'],
                campaigns: ['campaign_5'],
                lastSeen: new Date(Date.now() - 86400000),
                description: 'Financially motivated threat group targeting POS systems'
            },
            {
                id: 'apt29',
                name: 'APT29',
                aliases: ['Cozy Bear', 'The Dukes'],
                origin: 'Russia',
                active: true,
                sophistication: 'advanced',
                targets: ['Government', 'Think Tanks', 'Healthcare'],
                ttps: ['T1027', 'T1070', 'T1078', 'T1133'],
                campaigns: ['campaign_6', 'campaign_7'],
                lastSeen: new Date(),
                description: 'Sophisticated threat group linked to Russian SVR'
            }
        ];
        
        // Generate IOCs
        iocs = [];
        const iocTemplates = {
            ip: ['192.168.', '10.0.', '172.16.', '185.220.', '91.219.'],
            domain: ['malware-c2.com', 'phishing-site.net', 'evil-payload.org', 'backdoor-server.io'],
            hash: ['d41d8cd98f00b204e9800998ecf8427e', 'e3b0c44298fc1c149afbf4c8996fb924'],
            email: ['threat@evil.com', 'phishing@malicious.net', 'ransomware@darkweb.onion'],
            url: ['http://evil.com/payload', 'https://c2server.net/beacon', 'http://phishing.org/login'],
            cve: ['CVE-2024-0001', 'CVE-2024-0002', 'CVE-2024-0003'],
            yara: ['rule_malware_1', 'rule_backdoor_2', 'rule_ransomware_3']
        };
        
        Object.entries(iocTemplates).forEach(([type, templates]) => {
            templates.forEach((template, index) => {
                for (let i = 0; i < 5; i++) {
                    const value = type === 'ip' 
                        ? template + Math.floor(Math.random() * 255) + '.' + Math.floor(Math.random() * 255)
                        : type === 'hash'
                        ? template + Math.random().toString(16).substr(2, 8)
                        : template.replace(/\d+/, String(i));
                    
                    iocs.push({
                        id: `ioc_${type}_${index}_${i}`,
                        type: type as any,
                        value: value,
                        threatLevel: Math.floor(Math.random() * 100),
                        firstSeen: new Date(Date.now() - Math.random() * 30 * 86400000),
                        lastSeen: new Date(Date.now() - Math.random() * 86400000),
                        campaigns: [`campaign_${Math.floor(Math.random() * 7) + 1}`],
                        tags: ['malware', 'c2', 'phishing'].slice(0, Math.floor(Math.random() * 3) + 1),
                        confidence: 50 + Math.random() * 50,
                        sources: ['OSINT', 'Honeypot', 'Sandbox', 'Partner'].slice(0, Math.floor(Math.random() * 3) + 1)
                    });
                }
            });
        });
        
        // Generate campaigns
        campaigns = [
            {
                id: 'campaign_1',
                name: 'Operation CloudHopper',
                actor: 'apt28',
                status: 'active',
                startDate: new Date('2024-01-01'),
                targets: ['MSPs', 'Cloud Providers'],
                industries: ['Technology', 'Telecommunications'],
                techniques: ['Spear Phishing', 'Supply Chain Compromise'],
                iocs: iocs.slice(0, 20).map(i => i.id),
                severity: 'critical'
            },
            {
                id: 'campaign_2',
                name: 'WannaCry 2.0',
                actor: 'lazarus',
                status: 'dormant',
                startDate: new Date('2024-02-01'),
                endDate: new Date('2024-06-01'),
                targets: ['Healthcare', 'Critical Infrastructure'],
                industries: ['Healthcare', 'Energy'],
                techniques: ['Ransomware', 'EternalBlue Exploit'],
                iocs: iocs.slice(20, 40).map(i => i.id),
                severity: 'critical'
            },
            {
                id: 'campaign_3',
                name: 'SolarWinds Follow-up',
                actor: 'apt28',
                status: 'active',
                startDate: new Date('2024-03-01'),
                targets: ['Government Contractors', 'IT Companies'],
                industries: ['Government', 'Technology'],
                techniques: ['Supply Chain Attack', 'Backdoor'],
                iocs: iocs.slice(40, 60).map(i => i.id),
                severity: 'high'
            },
            {
                id: 'campaign_4',
                name: 'Crypto Heist',
                actor: 'lazarus',
                status: 'active',
                startDate: new Date('2024-04-01'),
                targets: ['Crypto Exchanges', 'DeFi Platforms'],
                industries: ['Financial', 'Cryptocurrency'],
                techniques: ['Social Engineering', 'Zero-day Exploits'],
                iocs: iocs.slice(60, 80).map(i => i.id),
                severity: 'high'
            },
            {
                id: 'campaign_5',
                name: 'POS Malware Wave',
                actor: 'fin7',
                status: 'active',
                startDate: new Date('2024-05-01'),
                targets: ['Retail Chains', 'Restaurants'],
                industries: ['Retail', 'Hospitality'],
                techniques: ['POS Malware', 'Card Skimming'],
                iocs: iocs.slice(80, 100).map(i => i.id),
                severity: 'medium'
            },
            {
                id: 'campaign_6',
                name: 'Embassy Siege',
                actor: 'apt29',
                status: 'dormant',
                startDate: new Date('2023-12-01'),
                endDate: new Date('2024-03-01'),
                targets: ['Embassies', 'Foreign Ministries'],
                industries: ['Government'],
                techniques: ['Watering Hole', 'Credential Harvesting'],
                iocs: iocs.slice(100, 120).map(i => i.id),
                severity: 'high'
            },
            {
                id: 'campaign_7',
                name: 'COVID Research Theft',
                actor: 'apt29',
                status: 'completed',
                startDate: new Date('2023-06-01'),
                endDate: new Date('2023-12-01'),
                targets: ['Research Institutes', 'Pharmaceutical'],
                industries: ['Healthcare', 'Research'],
                techniques: ['Spear Phishing', 'Data Exfiltration'],
                iocs: iocs.slice(120, 140).map(i => i.id),
                severity: 'critical'
            }
        ];
        
        // MITRE ATT&CK Matrix
        mitreMatrix = [
            {
                tactic: 'Initial Access',
                techniques: [
                    { id: 'T1566', name: 'Phishing', used: true, count: 45 },
                    { id: 'T1189', name: 'Drive-by Compromise', used: true, count: 23 },
                    { id: 'T1190', name: 'Exploit Public-Facing App', used: true, count: 31 }
                ]
            },
            {
                tactic: 'Execution',
                techniques: [
                    { id: 'T1203', name: 'Exploitation for Execution', used: true, count: 28 },
                    { id: 'T1059', name: 'Command & Scripting', used: true, count: 52 }
                ]
            },
            {
                tactic: 'Persistence',
                techniques: [
                    { id: 'T1078', name: 'Valid Accounts', used: true, count: 37 },
                    { id: 'T1133', name: 'External Remote Services', used: true, count: 19 }
                ]
            },
            {
                tactic: 'Defense Evasion',
                techniques: [
                    { id: 'T1027', name: 'Obfuscated Files', used: true, count: 41 },
                    { id: 'T1070', name: 'Indicator Removal', used: true, count: 33 }
                ]
            },
            {
                tactic: 'Command & Control',
                techniques: [
                    { id: 'T1071', name: 'Application Layer Protocol', used: true, count: 48 },
                    { id: 'T1571', name: 'Non-Standard Port', used: true, count: 22 }
                ]
            },
            {
                tactic: 'Exfiltration',
                techniques: [
                    { id: 'T1041', name: 'Exfiltration Over C2', used: true, count: 36 },
                    { id: 'T1048', name: 'Exfiltration Over Protocol', used: true, count: 29 }
                ]
            }
        ];
        
        // Threat Feeds
        threatFeeds = [
            { id: 'feed1', name: 'AlienVault OTX', enabled: true, lastUpdate: new Date(), iocCount: 15234, reliability: 85 },
            { id: 'feed2', name: 'MISP Community', enabled: true, lastUpdate: new Date(), iocCount: 8921, reliability: 90 },
            { id: 'feed3', name: 'Tor Exit Nodes', enabled: true, lastUpdate: new Date(), iocCount: 1823, reliability: 100 },
            { id: 'feed4', name: 'Phishing URLs', enabled: false, lastUpdate: new Date(Date.now() - 86400000), iocCount: 5432, reliability: 75 }
        ];
        
        updateVisualizations();
    }
    
    function connectThreatStream() {
        try {
            connectionStatus = 'connecting';
            ws = new WebSocket('ws://localhost:3010/threat-stream');
            
            ws.onopen = () => {
                connectionStatus = 'connected';
                isMonitoring = true;
                console.log('Connected to threat intelligence stream');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleThreatUpdate(data);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                connectionStatus = 'disconnected';
            };
            
            ws.onclose = () => {
                connectionStatus = 'disconnected';
                isMonitoring = false;
                setTimeout(connectThreatStream, 5000);
            };
            
        } catch (error) {
            console.error('Failed to connect threat stream:', error);
        }
    }
    
    function handleThreatUpdate(data: any) {
        lastUpdate = new Date();
        
        switch (data.type) {
            case 'new_ioc':
                addIOC(data.ioc);
                break;
            case 'actor_update':
                updateActor(data.actor);
                break;
            case 'campaign_update':
                updateCampaign(data.campaign);
                break;
            case 'threat_alert':
                addAlert(data.alert);
                break;
        }
        
        updateVisualizations();
    }
    
    function addIOC(ioc: IOC) {
        iocs = [ioc, ...iocs].slice(0, 1000);
    }
    
    function updateActor(actorData: Partial<ThreatActor>) {
        const index = threatActors.findIndex(a => a.id === actorData.id);
        if (index !== -1) {
            threatActors[index] = { ...threatActors[index], ...actorData };
        }
        threatActors = threatActors;
    }
    
    function updateCampaign(campaignData: Partial<Campaign>) {
        const index = campaigns.findIndex(c => c.id === campaignData.id);
        if (index !== -1) {
            campaigns[index] = { ...campaigns[index], ...campaignData };
        }
        campaigns = campaigns;
    }
    
    function addAlert(alert: any) {
        alertQueue = [alert, ...alertQueue].slice(0, 50);
    }
    
    function updateVisualizations() {
        updateThreatGraph();
        updateCharts();
    }
    
    function updateThreatGraph() {
        if (!cy) return;
        
        const elements = [];
        
        // Add threat actor nodes
        threatActors.forEach(actor => {
            elements.push({
                data: {
                    id: actor.id,
                    label: actor.name,
                    type: 'actor',
                    sophistication: ['low', 'medium', 'high', 'advanced'].indexOf(actor.sophistication),
                    actorData: actor
                },
                classes: actor.active ? 'active-threat' : ''
            });
        });
        
        // Add campaign nodes
        campaigns.forEach(campaign => {
            elements.push({
                data: {
                    id: campaign.id,
                    label: campaign.name,
                    type: 'campaign',
                    campaignData: campaign
                }
            });
            
            // Link to actor
            elements.push({
                data: {
                    id: `${campaign.actor}-${campaign.id}`,
                    source: campaign.actor,
                    target: campaign.id,
                    label: 'operates'
                }
            });
        });
        
        // Add target nodes
        const targets = new Set<string>();
        campaigns.forEach(c => c.targets.forEach(t => targets.add(t)));
        
        targets.forEach(target => {
            elements.push({
                data: {
                    id: `target_${target}`,
                    label: target,
                    type: 'target'
                }
            });
            
            // Link campaigns to targets
            campaigns.filter(c => c.targets.includes(target)).forEach(campaign => {
                elements.push({
                    data: {
                        id: `${campaign.id}-target_${target}`,
                        source: campaign.id,
                        target: `target_${target}`,
                        label: 'targets'
                    }
                });
            });
        });
        
        cy.elements().remove();
        cy.add(elements);
        cy.layout({ name: 'fcose', animate: true }).run();
    }
    
    function updateCharts() {
        if (!threatMapInstance || !timelineInstance || !severityInstance) return;
        
        // Update threat map with mock world data
        const mapData = [
            { name: 'United States', value: Math.floor(Math.random() * 100) },
            { name: 'China', value: Math.floor(Math.random() * 100) },
            { name: 'Russia', value: Math.floor(Math.random() * 100) },
            { name: 'United Kingdom', value: Math.floor(Math.random() * 100) },
            { name: 'Germany', value: Math.floor(Math.random() * 100) }
        ];
        
        threatMapInstance.setOption({
            series: [{ data: mapData }]
        });
        
        // Update IOC timeline
        const timelineData = processTimelineData();
        timelineInstance.setOption({
            series: [
                { data: timelineData.ip },
                { data: timelineData.domain },
                { data: timelineData.hash },
                { data: timelineData.url }
            ]
        });
        
        // Update severity gauge
        const avgSeverity = calculateAverageSeverity();
        severityInstance.setOption({
            series: [{
                data: [{ value: avgSeverity, name: 'Threat Level' }]
            }]
        });
    }
    
    function processTimelineData() {
        const data: any = {
            ip: [],
            domain: [],
            hash: [],
            url: []
        };
        
        // Group IOCs by day and type
        const dayMap = new Map();
        
        iocs.forEach(ioc => {
            const day = new Date(ioc.firstSeen);
            day.setHours(0, 0, 0, 0);
            const key = day.getTime();
            
            if (!dayMap.has(key)) {
                dayMap.set(key, { ip: 0, domain: 0, hash: 0, url: 0 });
            }
            
            const dayData = dayMap.get(key);
            if (dayData[ioc.type]) {
                dayData[ioc.type]++;
            }
        });
        
        dayMap.forEach((counts, timestamp) => {
            data.ip.push([timestamp, counts.ip]);
            data.domain.push([timestamp, counts.domain]);
            data.hash.push([timestamp, counts.hash]);
            data.url.push([timestamp, counts.url]);
        });
        
        return data;
    }
    
    function calculateAverageSeverity() {
        const severityValues = {
            low: 25,
            medium: 50,
            high: 75,
            critical: 100
        };
        
        const activeCampaigns = campaigns.filter(c => c.status === 'active');
        if (activeCampaigns.length === 0) return 0;
        
        const total = activeCampaigns.reduce((sum, c) => sum + severityValues[c.severity], 0);
        return Math.round(total / activeCampaigns.length);
    }
    
    function highlightConnections(node: any) {
        cy.elements().removeClass('highlighted');
        node.addClass('highlighted');
        node.connectedEdges().addClass('highlighted');
        node.connectedEdges().connectedNodes().addClass('highlighted');
    }
    
    function selectActor(actor: ThreatActor) {
        selectedActor = actor;
    }
    
    function selectCampaign(campaign: Campaign) {
        selectedCampaign = campaign;
    }
    
    function selectIOC(ioc: IOC) {
        selectedIOC = ioc;
    }
    
    function filterData() {
        loadThreatData();
    }
    
    function toggleActive() {
        showActiveOnly = !showActiveOnly;
        filterData();
    }
    
    function toggleFeed(feed: ThreatFeed) {
        feed.enabled = !feed.enabled;
        threatFeeds = threatFeeds;
        // In production, this would update the backend
    }
    
    function exportIOCs() {
        const exportData = {
            iocs: iocs.map(ioc => ({
                type: ioc.type,
                value: ioc.value,
                confidence: ioc.confidence,
                tags: ioc.tags
            })),
            exportDate: new Date().toISOString(),
            count: iocs.length
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `iocs-export-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    function createYARARule() {
        // Placeholder for YARA rule generation
        alert('YARA rule generation would open in editor');
    }
</script>

<div class="threat-intel">
    <header class="intel-header">
        <div class="header-left">
            <h2>Threat Intelligence</h2>
            <div class="connection-status" class:connected={connectionStatus === 'connected'}>
                <span class="status-indicator"></span>
                {connectionStatus === 'connected' ? 'LIVE' : connectionStatus.toUpperCase()}
            </div>
            <span class="last-update">Updated: {lastUpdate.toLocaleTimeString()}</span>
        </div>
        
        <div class="header-controls">
            <input
                type="text"
                placeholder="Search IOCs, actors, campaigns..."
                bind:value={searchQuery}
                on:input={filterData}
                class="search-input"
            />
            
            <select bind:value={iocTypeFilter} on:change={filterData} class="filter-select">
                {#each iocTypes as type}
                    <option value={type}>{type === 'all' ? 'All IOCs' : type.toUpperCase()}</option>
                {/each}
            </select>
            
            <select bind:value={severityFilter} on:change={filterData} class="filter-select">
                {#each severityLevels as level}
                    <option value={level}>{level === 'all' ? 'All Severities' : level.toUpperCase()}</option>
                {/each}
            </select>
            
            <select bind:value={timeRange} on:change={filterData} class="filter-select">
                {#each timeRanges as range}
                    <option value={range}>{range}</option>
                {/each}
            </select>
            
            <button on:click={toggleActive} class="filter-btn" class:active={showActiveOnly}>
                {showActiveOnly ? 'Show All' : 'Active Only'}
            </button>
            
            <button on:click={exportIOCs} class="export-btn">Export IOCs</button>
            <button on:click={createYARARule} class="yara-btn">YARA Rule</button>
        </div>
    </header>
    
    <!-- Threat Summary -->
    <div class="threat-summary">
        <div class="summary-item">
            <span class="summary-label">Active Actors</span>
            <span class="summary-value">{threatActors.filter(a => a.active).length}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Active Campaigns</span>
            <span class="summary-value">{campaigns.filter(c => c.status === 'active').length}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">IOCs (24h)</span>
            <span class="summary-value">
                {iocs.filter(i => new Date().getTime() - new Date(i.firstSeen).getTime() < 86400000).length}
            </span>
        </div>
        <div class="summary-item critical">
            <span class="summary-label">Critical Threats</span>
            <span class="summary-value">{campaigns.filter(c => c.severity === 'critical').length}</span>
        </div>
    </div>
    
    <div class="intel-grid">
        <!-- Threat Actors -->
        <div class="panel actors-panel">
            <h3>Threat Actors</h3>
            <div class="actors-list">
                {#each threatActors as actor}
                    <div 
                        class="actor-item"
                        class:selected={selectedActor?.id === actor.id}
                        class:active={actor.active}
                        on:click={() => selectActor(actor)}
                    >
                        <div class="actor-header">
                            <span class="actor-name">{actor.name}</span>
                            <span class="sophistication {actor.sophistication}">
                                {actor.sophistication.toUpperCase()}
                            </span>
                        </div>
                        <div class="actor-details">
                            <span>Origin: {actor.origin}</span>
                            <span>Campaigns: {actor.campaigns.length}</span>
                            <span>Last: {new Date(actor.lastSeen).toLocaleDateString()}</span>
                        </div>
                        <div class="actor-aliases">
                            {#each actor.aliases.slice(0, 2) as alias}
                                <span class="alias">{alias}</span>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Threat Graph -->
        <div class="panel graph-panel">
            <h3>Threat Landscape</h3>
            <div class="graph-container" bind:this={threatGraphContainer}></div>
        </div>
        
        <!-- Active Campaigns -->
        <div class="panel campaigns-panel">
            <h3>Active Campaigns</h3>
            <div class="campaigns-list">
                {#each campaigns.filter(c => showActiveOnly ? c.status === 'active' : true) as campaign}
                    <div 
                        class="campaign-item"
                        class:selected={selectedCampaign?.id === campaign.id}
                        on:click={() => selectCampaign(campaign)}
                    >
                        <div class="campaign-header">
                            <span class="campaign-name">{campaign.name}</span>
                            <span class="severity {campaign.severity}">{campaign.severity.toUpperCase()}</span>
                        </div>
                        <div class="campaign-details">
                            <span>Actor: {threatActors.find(a => a.id === campaign.actor)?.name || 'Unknown'}</span>
                            <span>Status: {campaign.status}</span>
                            <span>IOCs: {campaign.iocs.length}</span>
                        </div>
                        <div class="campaign-targets">
                            {#each campaign.industries.slice(0, 3) as industry}
                                <span class="target-tag">{industry}</span>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- IOC Feed -->
        <div class="panel iocs-panel">
            <h3>Recent IOCs</h3>
            <div class="ioc-list">
                {#each iocs.slice(0, 30) as ioc}
                    <div 
                        class="ioc-item"
                        class:high-threat={ioc.threatLevel > 70}
                        on:click={() => selectIOC(ioc)}
                    >
                        <div class="ioc-header">
                            <span class="ioc-type {ioc.type}">{ioc.type.toUpperCase()}</span>
                            <span class="threat-level" style="color: hsl({120 - ioc.threatLevel * 1.2}, 100%, 50%)">
                                {ioc.threatLevel}%
                            </span>
                        </div>
                        <div class="ioc-value">{ioc.value}</div>
                        <div class="ioc-meta">
                            <span>{new Date(ioc.firstSeen).toLocaleDateString()}</span>
                            <span>Conf: {ioc.confidence.toFixed(0)}%</span>
                            {#each ioc.tags.slice(0, 2) as tag}
                                <span class="ioc-tag">{tag}</span>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- MITRE ATT&CK Matrix -->
        <div class="panel mitre-panel">
            <h3>MITRE ATT&CK Coverage</h3>
            <div class="mitre-matrix">
                {#each mitreMatrix as tactic}
                    <div class="mitre-tactic">
                        <h4>{tactic.tactic}</h4>
                        <div class="techniques">
                            {#each tactic.techniques as technique}
                                <div 
                                    class="technique" 
                                    class:used={technique.used}
                                    title={`${technique.name} (${technique.count} uses)`}
                                >
                                    <span class="tech-id">{technique.id}</span>
                                    <span class="tech-count">{technique.count}</span>
                                </div>
                            {/each}
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Selected Details -->
        {#if selectedActor}
        <div class="panel details-panel">
            <h3>{selectedActor.name} Details</h3>
            <div class="details-content">
                <div class="detail-row">
                    <span>Origin:</span>
                    <span>{selectedActor.origin}</span>
                </div>
                <div class="detail-row">
                    <span>Sophistication:</span>
                    <span class="{selectedActor.sophistication}">{selectedActor.sophistication.toUpperCase()}</span>
                </div>
                <div class="detail-row">
                    <span>Status:</span>
                    <span class:active={selectedActor.active}>{selectedActor.active ? 'ACTIVE' : 'DORMANT'}</span>
                </div>
                <div class="detail-row">
                    <span>Last Seen:</span>
                    <span>{new Date(selectedActor.lastSeen).toLocaleString()}</span>
                </div>
                <div class="detail-section">
                    <h4>Aliases</h4>
                    {#each selectedActor.aliases as alias}
                        <span class="alias">{alias}</span>
                    {/each}
                </div>
                <div class="detail-section">
                    <h4>Targets</h4>
                    {#each selectedActor.targets as target}
                        <span class="target-tag">{target}</span>
                    {/each}
                </div>
                <div class="detail-section">
                    <h4>TTPs</h4>
                    {#each selectedActor.ttps as ttp}
                        <span class="ttp-tag">{ttp}</span>
                    {/each}
                </div>
                <p class="description">{selectedActor.description}</p>
            </div>
        </div>
        {/if}
        
        <!-- Charts -->
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={threatMapContainer}></div>
        </div>
        
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={iocTimelineContainer}></div>
        </div>
        
        <div class="panel chart-panel">
            <div class="chart-container" bind:this={severityChartContainer}></div>
        </div>
        
        <!-- Threat Feeds -->
        <div class="panel feeds-panel">
            <h3>Threat Feeds</h3>
            <div class="feeds-list">
                {#each threatFeeds as feed}
                    <div class="feed-item" class:enabled={feed.enabled}>
                        <div class="feed-header">
                            <span class="feed-name">{feed.name}</span>
                            <button 
                                class="feed-toggle" 
                                on:click={() => toggleFeed(feed)}
                            >
                                {feed.enabled ? 'ON' : 'OFF'}
                            </button>
                        </div>
                        <div class="feed-stats">
                            <span>IOCs: {feed.iocCount.toLocaleString()}</span>
                            <span>Reliability: {feed.reliability}%</span>
                            <span>Updated: {new Date(feed.lastUpdate).toLocaleTimeString()}</span>
                        </div>
                    </div>
                {/each}
            </div>
        </div>
        
        <!-- Alert Queue -->
        <div class="panel alerts-panel">
            <h3>Threat Alerts</h3>
            <div class="alert-list">
                {#each alertQueue.slice(0, 10) as alert}
                    <div class="alert-item {alert.severity}">
                        <div class="alert-time">{new Date(alert.timestamp).toLocaleTimeString()}</div>
                        <div class="alert-message">{DOMPurify.sanitize(alert.message)}</div>
                        <div class="alert-source">{alert.source}</div>
                    </div>
                {/each}
            </div>
        </div>
    </div>
    
    <!-- Status Bar -->
    <div class="status-bar">
        <div class="status-item">
            <span class="label">Monitoring:</span>
            <span class="value" class:active={isMonitoring}>{isMonitoring ? 'ACTIVE' : 'PAUSED'}</span>
        </div>
        <div class="status-item">
            <span class="label">Total IOCs:</span>
            <span class="value">{iocs.length.toLocaleString()}</span>
        </div>
        <div class="status-item">
            <span class="label">Active Actors:</span>
            <span class="value">{threatActors.filter(a => a.active).length}</span>
        </div>
        <div class="status-item">
            <span class="label">Active Campaigns:</span>
            <span class="value">{campaigns.filter(c => c.status === 'active').length}</span>
        </div>
        <div class="status-item">
            <span class="label">Feeds Active:</span>
            <span class="value">{threatFeeds.filter(f => f.enabled).length}/{threatFeeds.length}</span>
        </div>
    </div>
</div>

<style>
    .threat-intel {
        padding: 1rem;
        color: var(--text-primary, #00ff41);
        background: var(--bg-primary, #0a0a0a);
        min-height: 100vh;
    }
    
    .intel-header {
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
    
    .last-update {
        padding: 0.25rem 0.75rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        font-size: 0.75rem;
    }
    
    .header-controls {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    .search-input {
        padding: 0.5rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        width: 250px;
    }
    
    .search-input::placeholder {
        color: var(--text-secondary, #00ff4166);
    }
    
    .filter-select {
        padding: 0.5rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 4px;
        cursor: pointer;
    }
    
    .filter-btn, .export-btn, .yara-btn {
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
    
    .filter-btn:hover, .export-btn:hover, .yara-btn:hover {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .filter-btn.active {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .threat-summary {
        display: flex;
        gap: 2rem;
        padding: 0.75rem 1rem;
        background: var(--bg-secondary, #1a1a1a);
        border: 1px solid var(--border, #00ff4133);
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    .summary-item {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .summary-label {
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
        text-transform: uppercase;
    }
    
    .summary-value {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 0.25rem;
    }
    
    .summary-item.critical .summary-value {
        color: #ff0000;
    }
    
    .intel-grid {
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
    }
    
    .chart-container {
        width: 100%;
        height: 300px;
    }
    
    .actors-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .actor-item {
        padding: 0.75rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .actor-item:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .actor-item.selected {
        border-color: var(--text-primary, #00ff41);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .actor-item.active {
        border-left: 3px solid #ff0000;
    }
    
    .actor-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .actor-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .sophistication {
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .sophistication.low {
        background: rgba(0, 255, 65, 0.2);
        color: #00ff41;
    }
    
    .sophistication.medium {
        background: rgba(0, 204, 255, 0.2);
        color: #00ccff;
    }
    
    .sophistication.high {
        background: rgba(255, 149, 0, 0.2);
        color: #ff9500;
    }
    
    .sophistication.advanced {
        background: rgba(255, 0, 0, 0.2);
        color: #ff0000;
    }
    
    .actor-details {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .actor-aliases {
        display: flex;
        gap: 0.25rem;
        flex-wrap: wrap;
    }
    
    .alias {
        padding: 0.125rem 0.5rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        font-size: 0.625rem;
    }
    
    .campaigns-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .campaign-item {
        padding: 0.75rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .campaign-item:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .campaign-item.selected {
        border-color: var(--text-primary, #00ff41);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    .campaign-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .campaign-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .severity {
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    
    .severity.low {
        background: rgba(0, 255, 65, 0.2);
        color: #00ff41;
    }
    
    .severity.medium {
        background: rgba(255, 149, 0, 0.2);
        color: #ff9500;
    }
    
    .severity.high {
        background: rgba(255, 102, 0, 0.2);
        color: #ff6600;
    }
    
    .severity.critical {
        background: rgba(255, 0, 0, 0.2);
        color: #ff0000;
        animation: pulse 2s infinite;
    }
    
    .campaign-details {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .campaign-targets {
        display: flex;
        gap: 0.25rem;
        flex-wrap: wrap;
    }
    
    .target-tag, .ttp-tag {
        padding: 0.125rem 0.5rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        font-size: 0.625rem;
    }
    
    .ioc-list {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .ioc-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .ioc-item:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .ioc-item.high-threat {
        border-left: 3px solid #ff0000;
    }
    
    .ioc-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }
    
    .ioc-type {
        padding: 0.125rem 0.25rem;
        border-radius: 2px;
        font-size: 0.625rem;
        font-weight: bold;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--text-primary, #00ff41);
    }
    
    .threat-level {
        font-weight: bold;
        font-size: 0.75rem;
    }
    
    .ioc-value {
        font-family: 'Courier New', monospace;
        font-size: 0.75rem;
        word-break: break-all;
        margin-bottom: 0.25rem;
    }
    
    .ioc-meta {
        display: flex;
        gap: 0.5rem;
        font-size: 0.625rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .ioc-tag {
        padding: 0.125rem 0.25rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4111);
        border-radius: 2px;
    }
    
    .mitre-matrix {
        display: flex;
        gap: 1rem;
        overflow-x: auto;
    }
    
    .mitre-tactic {
        min-width: 120px;
    }
    
    .techniques {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }
    
    .technique {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 2px;
        font-size: 0.625rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .technique:hover {
        border-color: var(--text-primary, #00ff41);
    }
    
    .technique.used {
        background: rgba(0, 255, 65, 0.1);
        border-color: var(--text-primary, #00ff4133);
    }
    
    .tech-id {
        font-weight: bold;
    }
    
    .tech-count {
        color: var(--text-secondary, #00ff4199);
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
    
    .detail-section {
        margin-top: 0.5rem;
    }
    
    .description {
        margin-top: 0.5rem;
        font-size: 0.75rem;
        line-height: 1.4;
        color: var(--text-secondary, #00ff4199);
    }
    
    .feeds-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .feed-item {
        padding: 0.75rem;
        background: var(--bg-tertiary, #0f0f0f);
        border: 1px solid var(--border, #00ff4122);
        border-radius: 4px;
        opacity: 0.5;
        transition: all 0.2s;
    }
    
    .feed-item.enabled {
        opacity: 1;
        border-color: var(--text-primary, #00ff4133);
    }
    
    .feed-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .feed-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .feed-toggle {
        padding: 0.125rem 0.5rem;
        background: var(--bg-primary, #0a0a0a);
        border: 1px solid var(--border, #00ff4133);
        color: var(--text-primary, #00ff41);
        border-radius: 2px;
        cursor: pointer;
        font-size: 0.625rem;
        font-weight: bold;
    }
    
    .feed-toggle:hover {
        background: var(--text-primary, #00ff41);
        color: var(--bg-primary, #0a0a0a);
    }
    
    .feed-stats {
        display: flex;
        gap: 1rem;
        font-size: 0.75rem;
        color: var(--text-secondary, #00ff4199);
    }
    
    .alert-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .alert-item {
        padding: 0.5rem;
        background: var(--bg-tertiary, #0f0f0f);
        border-left: 3px solid var(--border, #00ff4133);
        border-radius: 2px;
    }
    
    .alert-item.low {
        border-left-color: #00ff41;
    }
    
    .alert-item.medium {
        border-left-color: #ff9500;
    }
    
    .alert-item.high {
        border-left-color: #ff6600;
    }
    
    .alert-item.critical {
        border-left-color: #ff0000;
        background: rgba(255, 0, 0, 0.1);
    }
    
    .alert-time {
        font-size: 0.625rem;
        color: var(--text-secondary, #00ff4199);
        margin-bottom: 0.25rem;
    }
    
    .alert-message {
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
    }
    
    .alert-source {
        font-size: 0.625rem;
        color: var(--text-secondary, #00ff4166);
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