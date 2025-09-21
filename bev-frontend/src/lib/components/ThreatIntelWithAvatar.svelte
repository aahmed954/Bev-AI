<!--
Enhanced Threat Intelligence Component with Avatar Integration
Features: Real-time threat analysis with avatar feedback and emotional responses
Connected to: Avatar WebSocket Client for dynamic threat announcements
-->

<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import * as echarts from 'echarts';
    import cytoscape from 'cytoscape';
    import fcose from 'cytoscape-fcose';
    import DOMPurify from 'dompurify';
    import { invoke } from '@tauri-apps/api/core';
    import { writable } from 'svelte/store';
    
    // Avatar integration
    import { avatarClient, OSINTAvatarIntegration } from '$lib/services/AvatarWebSocketClient';
    
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
        riskScore: number;
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
        avatarReported: boolean;
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
        avatarAlerted: boolean;
    }
    
    interface AvatarThreatState {
        current_threat_level: 'green' | 'yellow' | 'orange' | 'red';
        active_investigations: number;
        recent_alerts: number;
        mood: 'calm' | 'focused' | 'concerned' | 'alert' | 'alarmed';
        last_announcement: string;
    }
    
    // Component state
    let threatActors: ThreatActor[] = [];
    let iocs: IOC[] = [];
    let campaigns: Campaign[] = [];
    let selectedView = 'dashboard';
    let searchQuery = '';
    let threatLevelFilter = 'all';
    let isLoading = false;
    let isConnected = false;
    let wsConnection: WebSocket | null = null;
    
    // Avatar integration state
    const avatarThreatState = writable<AvatarThreatState>({
        current_threat_level: 'green',
        active_investigations: 0,
        recent_alerts: 0,
        mood: 'calm',
        last_announcement: ''
    });
    
    // Enhanced threat tracking for avatar responses
    let threatHistory: Array<{
        timestamp: Date,
        type: 'actor' | 'ioc' | 'campaign',
        severity: 'low' | 'medium' | 'high' | 'critical',
        description: string,
        avatar_response: string
    }> = [];
    
    // Chart containers
    let threatChart: echarts.ECharts;
    let networkChart: cytoscape.Core;
    let threatChartContainer: HTMLElement;
    let networkContainer: HTMLElement;
    
    // Real-time updates
    let updateInterval: number;
    let avatarUpdateInterval: number;
    
    onMount(async () => {
        await initializeThreatIntel();
        await initializeAvatarIntegration();
        await loadThreatData();
        setupCharts();
        startRealTimeUpdates();
    });
    
    onDestroy(() => {
        cleanup();
    });
    
    async function initializeThreatIntel() {
        try {
            // Connect to threat intelligence API
            const response = await fetch('http://localhost:3010/threat-intel/status');
            isConnected = response.ok;
            
            if (isConnected) {
                // Initialize WebSocket for real-time threat updates
                wsConnection = new WebSocket('ws://localhost:3010/threat-intel/ws');
                
                wsConnection.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleRealTimeThreatUpdate(data);
                };
                
                wsConnection.onopen = () => {
                    console.log('Threat intelligence WebSocket connected');
                };
                
                wsConnection.onclose = () => {
                    console.log('Threat intelligence WebSocket disconnected');
                    isConnected = false;
                };
            }
        } catch (error) {
            console.error('Failed to initialize threat intelligence:', error);
            isConnected = false;
        }
    }
    
    async function initializeAvatarIntegration() {
        try {
            // Initialize avatar client if not already connected
            if (avatarClient.getState().interaction_mode === 'idle') {
                await OSINTAvatarIntegration.initialize();
            }
            
            // Setup avatar threat monitoring
            avatarClient.subscribe('osint_event', handleAvatarOSINTEvent);
            
            // Initial avatar announcement
            await avatarClient.setEmotion('focused');
            await avatarClient.speak('Threat intelligence monitoring activated. Scanning for security threats and indicators of compromise.', 'focused');
            
            avatarThreatState.update(state => ({
                ...state,
                mood: 'focused',
                last_announcement: 'Threat monitoring activated'
            }));
            
        } catch (error) {
            console.error('Failed to initialize avatar integration:', error);
        }
    }
    
    async function loadThreatData() {
        isLoading = true;
        
        try {
            // Load threat actors
            const actorsResponse = await fetch('http://localhost:3010/threat-intel/actors');
            if (actorsResponse.ok) {
                threatActors = await actorsResponse.json();
                await announceNewThreats(threatActors);
            }
            
            // Load IOCs
            const iocsResponse = await fetch('http://localhost:3010/threat-intel/iocs');
            if (iocsResponse.ok) {
                iocs = await iocsResponse.json();
                await analyzeIOCsWithAvatar(iocs);
            }
            
            // Load campaigns
            const campaignsResponse = await fetch('http://localhost:3010/threat-intel/campaigns');
            if (campaignsResponse.ok) {
                campaigns = await campaignsResponse.json();
                await evaluateCampaignsWithAvatar(campaigns);
            }
            
            updateThreatLevel();
            
        } catch (error) {
            console.error('Failed to load threat data:', error);
            await avatarClient.expressConcern('Unable to load threat intelligence data. System may be compromised.');
        } finally {
            isLoading = false;
        }
    }
    
    async function announceNewThreats(actors: ThreatActor[]) {
        const highRiskActors = actors.filter(actor => 
            actor.active && 
            (actor.sophistication === 'high' || actor.sophistication === 'advanced') &&
            actor.riskScore > 80
        );
        
        if (highRiskActors.length > 0) {
            await avatarClient.setEmotion('alert');
            await avatarClient.speak(
                `Attention: ${highRiskActors.length} high-risk threat actors detected. Review recommended.`,
                'alert',
                'high'
            );
            
            avatarThreatState.update(state => ({
                ...state,
                mood: 'alert',
                recent_alerts: state.recent_alerts + highRiskActors.length,
                last_announcement: `${highRiskActors.length} high-risk actors detected`
            }));
            
            // Log threat for history
            threatHistory.push({
                timestamp: new Date(),
                type: 'actor',
                severity: 'high',
                description: `${highRiskActors.length} high-risk threat actors detected`,
                avatar_response: 'Alert emotion with high priority announcement'
            });
        }
    }
    
    async function analyzeIOCsWithAvatar(indicators: IOC[]) {
        const criticalIOCs = indicators.filter(ioc => 
            ioc.threatLevel >= 8 && 
            ioc.confidence >= 0.8 &&
            !ioc.avatarReported
        );
        
        if (criticalIOCs.length > 0) {
            await avatarClient.setEmotion('concerned');
            
            for (const ioc of criticalIOCs.slice(0, 3)) { // Limit to first 3 for voice
                await avatarClient.speak(
                    `Critical indicator found: ${ioc.type} with threat level ${ioc.threatLevel}`,
                    'concerned'
                );
                
                ioc.avatarReported = true;
            }
            
            avatarThreatState.update(state => ({
                ...state,
                mood: 'concerned',
                recent_alerts: state.recent_alerts + criticalIOCs.length
            }));
            
            threatHistory.push({
                timestamp: new Date(),
                type: 'ioc',
                severity: 'critical',
                description: `${criticalIOCs.length} critical IOCs identified`,
                avatar_response: 'Concerned emotion with detailed IOC announcements'
            });
        }
    }
    
    async function evaluateCampaignsWithAvatar(campaignData: Campaign[]) {
        const activeCriticalCampaigns = campaignData.filter(campaign => 
            campaign.status === 'active' && 
            campaign.severity === 'critical' &&
            !campaign.avatarAlerted
        );
        
        if (activeCriticalCampaigns.length > 0) {
            await avatarClient.setEmotion('alarmed');
            await avatarClient.speak(
                `URGENT: ${activeCriticalCampaigns.length} critical active threat campaigns detected. Immediate attention required.`,
                'alarmed',
                'high'
            );
            
            avatarThreatState.update(state => ({
                ...state,
                mood: 'alarmed',
                current_threat_level: 'red',
                active_investigations: activeCriticalCampaigns.length,
                last_announcement: 'Critical campaigns detected - urgent attention required'
            }));
            
            // Mark campaigns as alerted
            activeCriticalCampaigns.forEach(campaign => {
                campaign.avatarAlerted = true;
            });
            
            threatHistory.push({
                timestamp: new Date(),
                type: 'campaign',
                severity: 'critical',
                description: `${activeCriticalCampaigns.length} critical active campaigns`,
                avatar_response: 'Alarmed emotion with urgent announcement'
            });
        }
    }
    
    function updateThreatLevel() {
        const criticalThreats = [
            ...threatActors.filter(a => a.active && a.riskScore >= 90),
            ...iocs.filter(i => i.threatLevel >= 9),
            ...campaigns.filter(c => c.status === 'active' && c.severity === 'critical')
        ];
        
        const highThreats = [
            ...threatActors.filter(a => a.active && a.riskScore >= 70),
            ...iocs.filter(i => i.threatLevel >= 7),
            ...campaigns.filter(c => c.status === 'active' && c.severity === 'high')
        ];
        
        let threatLevel: 'green' | 'yellow' | 'orange' | 'red' = 'green';
        let mood: 'calm' | 'focused' | 'concerned' | 'alert' | 'alarmed' = 'calm';
        
        if (criticalThreats.length > 0) {
            threatLevel = 'red';
            mood = 'alarmed';
        } else if (highThreats.length > 3) {
            threatLevel = 'orange';
            mood = 'alert';
        } else if (highThreats.length > 0) {
            threatLevel = 'yellow';
            mood = 'concerned';
        } else {
            threatLevel = 'green';
            mood = 'focused';
        }
        
        avatarThreatState.update(state => ({
            ...state,
            current_threat_level: threatLevel,
            mood
        }));
    }
    
    function handleRealTimeThreatUpdate(data: any) {
        switch (data.type) {
            case 'new_threat_actor':
                threatActors = [...threatActors, data.actor];
                announceNewThreats([data.actor]);
                break;
                
            case 'new_ioc':
                iocs = [...iocs, data.ioc];
                analyzeIOCsWithAvatar([data.ioc]);
                break;
                
            case 'campaign_update':
                const campaignIndex = campaigns.findIndex(c => c.id === data.campaign.id);
                if (campaignIndex >= 0) {
                    campaigns[campaignIndex] = data.campaign;
                } else {
                    campaigns = [...campaigns, data.campaign];
                }
                evaluateCampaignsWithAvatar([data.campaign]);
                break;
                
            case 'threat_level_change':
                handleThreatLevelChange(data.level, data.reason);
                break;
        }
        
        updateThreatLevel();
        updateCharts();
    }
    
    async function handleThreatLevelChange(newLevel: string, reason: string) {
        const emotionMap = {
            'low': 'calm',
            'medium': 'focused',
            'high': 'concerned',
            'critical': 'alarmed'
        };
        
        const emotion = emotionMap[newLevel] || 'focused';
        await avatarClient.setEmotion(emotion);
        
        if (newLevel === 'critical') {
            await avatarClient.speak(
                `CRITICAL THREAT LEVEL ESCALATION: ${reason}. Initiating enhanced monitoring protocols.`,
                'alarmed',
                'high'
            );
        } else if (newLevel === 'high') {
            await avatarClient.speak(
                `Threat level elevated to HIGH due to ${reason}. Increased vigilance recommended.`,
                'concerned'
            );
        }
        
        avatarThreatState.update(state => ({
            ...state,
            mood: emotion as any,
            last_announcement: `Threat level: ${newLevel.toUpperCase()} - ${reason}`
        }));
    }
    
    function handleAvatarOSINTEvent(event: any) {
        if (event.data.type === 'threat_investigation_start') {
            avatarThreatState.update(state => ({
                ...state,
                active_investigations: state.active_investigations + 1
            }));
        } else if (event.data.type === 'threat_investigation_complete') {
            avatarThreatState.update(state => ({
                ...state,
                active_investigations: Math.max(0, state.active_investigations - 1)
            }));
        }
    }
    
    function setupCharts() {
        if (threatChartContainer) {
            threatChart = echarts.init(threatChartContainer);
            updateThreatChart();
        }
        
        if (networkContainer) {
            networkChart = cytoscape({
                container: networkContainer,
                style: [
                    {
                        selector: 'node',
                        style: {
                            'background-color': '#3b82f6',
                            'label': 'data(label)',
                            'font-size': '12px',
                            'color': '#ffffff'
                        }
                    },
                    {
                        selector: 'edge',
                        style: {
                            'line-color': '#6b7280',
                            'width': 2
                        }
                    }
                ],
                layout: {
                    name: 'fcose',
                    quality: 'default',
                    randomize: false,
                    animate: true
                }
            });
            updateNetworkChart();
        }
    }
    
    function updateThreatChart() {
        if (!threatChart) return;
        
        const threatLevels = {
            'Critical': campaigns.filter(c => c.severity === 'critical').length,
            'High': campaigns.filter(c => c.severity === 'high').length,
            'Medium': campaigns.filter(c => c.severity === 'medium').length,
            'Low': campaigns.filter(c => c.severity === 'low').length
        };
        
        const option = {
            title: {
                text: 'Threat Level Distribution',
                textStyle: { color: '#ffffff' }
            },
            backgroundColor: 'transparent',
            series: [{
                type: 'pie',
                data: Object.entries(threatLevels).map(([name, value]) => ({
                    name,
                    value,
                    itemStyle: {
                        color: name === 'Critical' ? '#ef4444' :
                               name === 'High' ? '#f97316' :
                               name === 'Medium' ? '#eab308' : '#22c55e'
                    }
                }))
            }]
        };
        
        threatChart.setOption(option);
    }
    
    function updateNetworkChart() {
        if (!networkChart) return;
        
        const elements = [
            ...threatActors.slice(0, 10).map(actor => ({
                data: {
                    id: actor.id,
                    label: actor.name,
                    type: 'actor'
                }
            })),
            ...campaigns.slice(0, 5).map(campaign => ({
                data: {
                    id: campaign.id,
                    label: campaign.name,
                    type: 'campaign'
                }
            }))
        ];
        
        networkChart.elements().remove();
        networkChart.add(elements);
        networkChart.layout({ name: 'fcose' }).run();
    }
    
    function updateCharts() {
        updateThreatChart();
        updateNetworkChart();
    }
    
    function startRealTimeUpdates() {
        updateInterval = window.setInterval(async () => {
            if (isConnected) {
                await loadThreatData();
            }
        }, 30000); // Update every 30 seconds
        
        avatarUpdateInterval = window.setInterval(async () => {
            // Periodic avatar status updates
            const state = $avatarThreatState;
            if (state.active_investigations > 0) {
                await avatarClient.speak(
                    `Status update: ${state.active_investigations} threat investigations in progress.`,
                    'focused',
                    'low'
                );
            }
        }, 300000); // Update every 5 minutes
    }
    
    async function announceManualThreat(severity: 'low' | 'medium' | 'high' | 'critical', description: string) {
        const emotionMap = {
            'low': 'focused',
            'medium': 'concerned',
            'high': 'alert',
            'critical': 'alarmed'
        };
        
        await avatarClient.setEmotion(emotionMap[severity]);
        await avatarClient.speak(
            `Manual threat reported: ${severity.toUpperCase()} severity. ${description}`,
            emotionMap[severity],
            severity === 'critical' ? 'high' : 'normal'
        );
        
        threatHistory.push({
            timestamp: new Date(),
            type: 'actor',
            severity,
            description: `Manual report: ${description}`,
            avatar_response: `${emotionMap[severity]} emotion with manual threat announcement`
        });
    }
    
    function cleanup() {
        if (updateInterval) {
            clearInterval(updateInterval);
        }
        
        if (avatarUpdateInterval) {
            clearInterval(avatarUpdateInterval);
        }
        
        if (wsConnection) {
            wsConnection.close();
        }
        
        if (threatChart) {
            threatChart.dispose();
        }
        
        if (networkChart) {
            networkChart.destroy();
        }
    }
    
    function getThreatLevelColor(level: string): string {
        switch (level) {
            case 'red': return 'bg-red-500';
            case 'orange': return 'bg-orange-500';
            case 'yellow': return 'bg-yellow-500';
            case 'green': return 'bg-green-500';
            default: return 'bg-gray-500';
        }
    }
    
    function getMoodIcon(mood: string): string {
        switch (mood) {
            case 'calm': return '😌';
            case 'focused': return '🧐';
            case 'concerned': return '😟';
            case 'alert': return '🚨';
            case 'alarmed': return '😰';
            default: return '🤖';
        }
    }
</script>

<!-- Enhanced Threat Intelligence with Avatar Integration -->
<div class="threat-intel-with-avatar bg-gray-900 text-white p-6">
    <!-- Header with Avatar Status -->
    <div class="flex items-center justify-between mb-6">
        <div>
            <h1 class="text-2xl font-bold text-cyan-400">Threat Intelligence</h1>
            <p class="text-gray-400">AI-enhanced threat monitoring with real-time avatar feedback</p>
        </div>
        
        <!-- Avatar Threat State Display -->
        <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div class="flex items-center space-x-4">
                <div class="text-center">
                    <div class="text-2xl mb-1">{getMoodIcon($avatarThreatState.mood)}</div>
                    <div class="text-xs text-gray-400">Avatar Mood</div>
                </div>
                
                <div class="text-center">
                    <div class="w-4 h-4 rounded-full mx-auto mb-1 {getThreatLevelColor($avatarThreatState.current_threat_level)}"></div>
                    <div class="text-xs text-gray-400">Threat Level</div>
                </div>
                
                <div class="text-center">
                    <div class="text-lg font-bold text-cyan-400">{$avatarThreatState.active_investigations}</div>
                    <div class="text-xs text-gray-400">Active</div>
                </div>
                
                <div class="text-center">
                    <div class="text-lg font-bold text-red-400">{$avatarThreatState.recent_alerts}</div>
                    <div class="text-xs text-gray-400">Alerts</div>
                </div>
            </div>
            
            {#if $avatarThreatState.last_announcement}
                <div class="mt-3 text-xs text-gray-300 border-t border-gray-700 pt-2">
                    Latest: {$avatarThreatState.last_announcement}
                </div>
            {/if}
        </div>
    </div>
    
    <!-- Threat Overview Cards -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-gray-400 text-sm">Threat Actors</p>
                    <p class="text-2xl font-bold text-white">{threatActors.length}</p>
                </div>
                <div class="text-red-400 text-xl">👥</div>
            </div>
            <div class="text-xs text-gray-500 mt-2">
                {threatActors.filter(a => a.active).length} active
            </div>
        </div>
        
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-gray-400 text-sm">IOCs</p>
                    <p class="text-2xl font-bold text-white">{iocs.length}</p>
                </div>
                <div class="text-yellow-400 text-xl">⚠️</div>
            </div>
            <div class="text-xs text-gray-500 mt-2">
                {iocs.filter(i => i.threatLevel >= 7).length} high severity
            </div>
        </div>
        
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-gray-400 text-sm">Campaigns</p>
                    <p class="text-2xl font-bold text-white">{campaigns.length}</p>
                </div>
                <div class="text-blue-400 text-xl">🎯</div>
            </div>
            <div class="text-xs text-gray-500 mt-2">
                {campaigns.filter(c => c.status === 'active').length} active
            </div>
        </div>
        
        <div class="bg-gray-800 rounded-lg p-4">
            <div class="flex items-center justify-between">
                <div>
                    <p class="text-gray-400 text-sm">Threat Level</p>
                    <p class="text-2xl font-bold capitalize {
                        $avatarThreatState.current_threat_level === 'red' ? 'text-red-400' :
                        $avatarThreatState.current_threat_level === 'orange' ? 'text-orange-400' :
                        $avatarThreatState.current_threat_level === 'yellow' ? 'text-yellow-400' :
                        'text-green-400'
                    }">{$avatarThreatState.current_threat_level}</p>
                </div>
                <div class="{
                    $avatarThreatState.current_threat_level === 'red' ? 'text-red-400' :
                    $avatarThreatState.current_threat_level === 'orange' ? 'text-orange-400' :
                    $avatarThreatState.current_threat_level === 'yellow' ? 'text-yellow-400' :
                    'text-green-400'
                } text-xl">🛡️</div>
            </div>
        </div>
    </div>
    
    <!-- Main Content -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Threat Chart -->
        <div class="bg-gray-800 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-white mb-4">Threat Distribution</h3>
            <div bind:this={threatChartContainer} class="w-full h-64"></div>
        </div>
        
        <!-- Network Chart -->
        <div class="bg-gray-800 rounded-lg p-4">
            <h3 class="text-lg font-semibold text-white mb-4">Threat Actor Network</h3>
            <div bind:this={networkContainer} class="w-full h-64"></div>
        </div>
    </div>
    
    <!-- Avatar Threat History -->
    <div class="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 class="text-lg font-semibold text-white mb-4">Avatar Threat Announcements</h3>
        {#if threatHistory.length === 0}
            <div class="text-center py-8 text-gray-400">
                <div class="text-4xl mb-2">🤖</div>
                <p>No threat announcements yet</p>
            </div>
        {:else}
            <div class="space-y-3 max-h-64 overflow-y-auto">
                {#each threatHistory.slice(-10).reverse() as threat}
                    <div class="bg-gray-900 rounded p-3 border-l-4 {
                        threat.severity === 'critical' ? 'border-red-500' :
                        threat.severity === 'high' ? 'border-orange-500' :
                        threat.severity === 'medium' ? 'border-yellow-500' :
                        'border-blue-500'
                    }">
                        <div class="flex items-center justify-between mb-1">
                            <span class="text-sm font-medium text-white capitalize">{threat.type} - {threat.severity}</span>
                            <span class="text-xs text-gray-500">{threat.timestamp.toLocaleTimeString()}</span>
                        </div>
                        <p class="text-sm text-gray-300">{threat.description}</p>
                        <p class="text-xs text-purple-400 mt-1">Avatar: {threat.avatar_response}</p>
                    </div>
                {/each}
            </div>
        {/if}
    </div>
    
    <!-- Manual Threat Reporting -->
    <div class="mt-6 bg-gray-800 rounded-lg p-4">
        <h3 class="text-lg font-semibold text-white mb-4">Manual Threat Reporting</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
            <button
                on:click={() => announceManualThreat('low', 'User reported low priority threat')}
                class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
            >
                Report Low
            </button>
            <button
                on:click={() => announceManualThreat('medium', 'User reported medium priority threat')}
                class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded transition-colors"
            >
                Report Medium
            </button>
            <button
                on:click={() => announceManualThreat('high', 'User reported high priority threat')}
                class="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
            >
                Report High
            </button>
            <button
                on:click={() => announceManualThreat('critical', 'User reported critical threat requiring immediate attention')}
                class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
            >
                Report Critical
            </button>
        </div>
    </div>
</div>

<style>
    .threat-intel-with-avatar {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Custom scrollbar */
    :global(.threat-intel-with-avatar *::-webkit-scrollbar) {
        width: 6px;
    }
    
    :global(.threat-intel-with-avatar *::-webkit-scrollbar-track) {
        background: #374151;
    }
    
    :global(.threat-intel-with-avatar *::-webkit-scrollbar-thumb) {
        background: #6b7280;
        border-radius: 3px;
    }
</style>