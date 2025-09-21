<script lang="ts">
  import { onMount } from 'svelte';
  import { io, Socket } from 'socket.io-client';
  import * as echarts from 'echarts';
  import { fade, fly } from 'svelte/transition';
  
  let socket: Socket;
  let socialFeeds = [];
  let selectedPlatforms = {
    twitter: true,
    telegram: true,
    discord: true,
    reddit: true,
    mastodon: false
  };
  
  let sentimentChart: echarts.ECharts;
  let networkGraph: echarts.ECharts;
  let trendsChart: echarts.ECharts;
  
  let activeMonitors = 0;
  let totalPosts = 0;
  let threatScore = 0;
  let keywords = '';
  let userProfiles = [];
  let networkNodes = [];
  let networkLinks = [];
  
  // Real-time feed data
  let feedData = {
    twitter: [],
    telegram: [],
    discord: [],
    reddit: []
  };
  
  onMount(() => {
    initializeWebSocket();
    initializeCharts();
    
    return () => {
      if (socket) socket.disconnect();
      if (sentimentChart) sentimentChart.dispose();
      if (networkGraph) networkGraph.dispose();
      if (trendsChart) trendsChart.dispose();
    };
  });
  
  function initializeWebSocket() {
    socket = io('ws://localhost:3001', {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity
    });
    
    socket.on('social_feed', (data) => {
      handleSocialFeed(data);
    });
    
    socket.on('sentiment_update', (data) => {
      updateSentimentChart(data);
    });
    
    socket.on('network_update', (data) => {
      updateNetworkGraph(data);
    });
    
    socket.on('trend_alert', (data) => {
      handleTrendAlert(data);
    });
    
    socket.on('connect', () => {
      activeMonitors++;
      socket.emit('subscribe_social', { platforms: selectedPlatforms, keywords });
    });
  }
  
  function initializeCharts() {
    // Sentiment Analysis Chart
    const sentimentDom = document.getElementById('sentiment-chart');
    sentimentChart = echarts.init(sentimentDom, 'dark');
    
    const sentimentOption = {
      title: { text: 'Sentiment Analysis', textStyle: { color: '#00ffff' }},
      tooltip: { trigger: 'axis' },
      legend: { data: ['Positive', 'Neutral', 'Negative'], textStyle: { color: '#888' }},
      xAxis: { type: 'time', boundaryGap: false },
      yAxis: { type: 'value' },
      series: [
        {
          name: 'Positive',
          type: 'line',
          smooth: true,
          data: [],
          itemStyle: { color: '#00ff00' }
        },
        {
          name: 'Neutral',
          type: 'line',
          smooth: true,
          data: [],
          itemStyle: { color: '#ffff00' }
        },
        {
          name: 'Negative',
          type: 'line',
          smooth: true,
          data: [],
          itemStyle: { color: '#ff0000' }
        }
      ]
    };
    sentimentChart.setOption(sentimentOption);
    
    // Network Graph
    const networkDom = document.getElementById('network-graph');
    networkGraph = echarts.init(networkDom, 'dark');
    
    const networkOption = {
      title: { text: 'Social Network Graph', textStyle: { color: '#00ffff' }},
      tooltip: {},
      animationDurationUpdate: 1500,
      animationEasingUpdate: 'quinticInOut',
      series: [{
        type: 'graph',
        layout: 'force',
        data: [],
        links: [],
        roam: true,
        label: { show: true, position: 'right', formatter: '{b}' },
        lineStyle: { color: 'source', curveness: 0.3 },
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 10 }
        }
      }]
    };
    networkGraph.setOption(networkOption);
    
    // Trends Chart
    const trendsDom = document.getElementById('trends-chart');
    trendsChart = echarts.init(trendsDom, 'dark');
    
    const trendsOption = {
      title: { text: 'Trending Topics', textStyle: { color: '#00ffff' }},
      tooltip: { trigger: 'item' },
      series: [{
        type: 'wordCloud',
        shape: 'circle',
        gridSize: 8,
        sizeRange: [12, 50],
        rotationRange: [-90, 90],
        data: []
      }]
    };
    trendsChart.setOption(trendsOption);
  }
  
  function handleSocialFeed(data) {
    feedData[data.platform] = [...feedData[data.platform], data].slice(-100);
    totalPosts++;
    
    // Update threat scoring
    if (data.threatLevel) {
      threatScore = Math.min(100, threatScore + data.threatLevel * 0.1);
    }
    
    // Extract user profiles for analysis
    if (data.user && !userProfiles.find(u => u.id === data.user.id)) {
      userProfiles = [...userProfiles, data.user];
      addNetworkNode(data.user);
    }
    
    // Check for network connections
    if (data.mentions) {
      data.mentions.forEach(mention => {
        addNetworkLink(data.user.id, mention.id);
      });
    }
  }
  
  function addNetworkNode(user) {
    networkNodes = [...networkNodes, {
      id: user.id,
      name: user.username,
      value: user.followers || 1,
      category: user.platform,
      symbolSize: Math.log(user.followers + 1) * 5
    }];
    updateNetworkGraph();
  }
  
  function addNetworkLink(source, target) {
    if (!networkLinks.find(l => l.source === source && l.target === target)) {
      networkLinks = [...networkLinks, { source, target }];
      updateNetworkGraph();
    }
  }
  
  function updateNetworkGraph() {
    if (!networkGraph) return;
    networkGraph.setOption({
      series: [{
        data: networkNodes,
        links: networkLinks
      }]
    });
  }
  
  function updateSentimentChart(data) {
    if (!sentimentChart) return;
    const option = sentimentChart.getOption();
    
    const timestamp = Date.now();
    option.series[0].data.push([timestamp, data.positive]);
    option.series[1].data.push([timestamp, data.neutral]);
    option.series[2].data.push([timestamp, data.negative]);
    
    // Keep only last 100 points
    option.series.forEach(s => {
      if (s.data.length > 100) s.data.shift();
    });
    
    sentimentChart.setOption(option);
  }
  
  function handleTrendAlert(data) {
    // Update trends chart with new trending topics
    if (!trendsChart) return;
    
    const trendData = data.trends.map(trend => ({
      name: trend.topic,
      value: trend.count,
      textStyle: {
        color: trend.sentiment === 'negative' ? '#ff0000' : 
               trend.sentiment === 'positive' ? '#00ff00' : '#ffff00'
      }
    }));
    
    trendsChart.setOption({
      series: [{ data: trendData }]
    });
  }
  
  function searchKeywords() {
    if (socket && socket.connected) {
      socket.emit('update_keywords', { keywords, platforms: selectedPlatforms });
    }
  }
  
  function togglePlatform(platform) {
    selectedPlatforms[platform] = !selectedPlatforms[platform];
    if (socket && socket.connected) {
      socket.emit('update_platforms', selectedPlatforms);
    }
  }
  
  function analyzeProfile(userId) {
    if (socket && socket.connected) {
      socket.emit('analyze_profile', { userId });
    }
  }
</script>

<style>
  .social-intel-container {
    @apply p-6 space-y-6 bg-gray-900 text-gray-100;
  }
  
  .platform-selector {
    @apply flex gap-4 p-4 bg-gray-800 rounded-lg;
  }
  
  .platform-toggle {
    @apply px-4 py-2 rounded transition-colors cursor-pointer;
  }
  
  .platform-toggle.active {
    @apply bg-cyan-600 text-white;
  }
  
  .platform-toggle.inactive {
    @apply bg-gray-700 text-gray-400 hover:bg-gray-600;
  }
  
  .feed-container {
    @apply grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4;
  }
  
  .feed-item {
    @apply p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-cyan-500 transition-colors;
  }
  
  .chart-container {
    @apply h-64 md:h-96 bg-gray-800 rounded-lg p-4;
  }
  
  .stats-grid {
    @apply grid grid-cols-4 gap-4 mb-6;
  }
  
  .stat-card {
    @apply p-4 bg-gradient-to-br from-gray-800 to-gray-900 rounded-lg border border-gray-700;
  }
</style>

<div class="social-intel-container">
  <h2 class="text-3xl font-bold text-cyan-400 mb-6">Social Media Intelligence</h2>
  
  <!-- Stats Overview -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="text-sm text-gray-400">Active Monitors</div>
      <div class="text-2xl font-bold text-cyan-400">{activeMonitors}</div>
    </div>
    <div class="stat-card">
      <div class="text-sm text-gray-400">Total Posts</div>
      <div class="text-2xl font-bold text-green-400">{totalPosts}</div>
    </div>
    <div class="stat-card">
      <div class="text-sm text-gray-400">Threat Score</div>
      <div class="text-2xl font-bold" class:text-red-400={threatScore > 70} 
           class:text-yellow-400={threatScore > 40 && threatScore <= 70}
           class:text-green-400={threatScore <= 40}>
        {threatScore.toFixed(1)}%
      </div>
    </div>
    <div class="stat-card">
      <div class="text-sm text-gray-400">Profiles Tracked</div>
      <div class="text-2xl font-bold text-purple-400">{userProfiles.length}</div>
    </div>
  </div>  
  <!-- Platform Selector -->
  <div class="platform-selector">
    <span class="text-gray-400 mr-4">Monitor Platforms:</span>
    {#each Object.keys(selectedPlatforms) as platform}
      <button 
        class="platform-toggle"
        class:active={selectedPlatforms[platform]}
        class:inactive={!selectedPlatforms[platform]}
        on:click={() => togglePlatform(platform)}>
        {platform}
      </button>
    {/each}
  </div>
  
  <!-- Keyword Search -->
  <div class="flex gap-4 mb-6">
    <input 
      type="text" 
      bind:value={keywords}
      placeholder="Enter keywords to monitor (comma-separated)"
      class="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-gray-100 placeholder-gray-500"
    />
    <button 
      on:click={searchKeywords}
      class="px-6 py-2 bg-cyan-600 text-white rounded-lg hover:bg-cyan-700 transition-colors">
      Start Monitoring
    </button>
  </div>
  
  <!-- Visualization Grid -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
    <div class="chart-container">
      <div id="sentiment-chart" class="w-full h-full"></div>
    </div>
    <div class="chart-container">
      <div id="network-graph" class="w-full h-full"></div>
    </div>
  </div>
  
  <div class="chart-container mb-6">
    <div id="trends-chart" class="w-full h-full"></div>
  </div>  
  <!-- Live Feed Grid -->
  <h3 class="text-xl font-semibold text-cyan-400 mb-4">Live Social Feeds</h3>
  <div class="feed-container">
    {#each Object.entries(feedData) as [platform, posts]}
      {#if selectedPlatforms[platform]}
        <div class="space-y-3">
          <h4 class="text-lg font-medium text-gray-300 capitalize">{platform}</h4>
          {#each posts.slice(-5) as post (post.id)}
            <div class="feed-item" in:fly={{ y: 20, duration: 300 }}>
              <div class="flex items-center gap-2 mb-2">
                <img src={post.user?.avatar || '/default-avatar.png'} alt="avatar" class="w-8 h-8 rounded-full" />
                <span class="text-sm font-medium text-gray-300">@{post.user?.username || 'unknown'}</span>
                {#if post.threatLevel > 0.7}
                  <span class="ml-auto text-xs px-2 py-1 bg-red-600 rounded">HIGH RISK</span>
                {/if}
              </div>
              <p class="text-sm text-gray-400 line-clamp-3">{post.content}</p>
              <div class="flex justify-between items-center mt-2 text-xs text-gray-500">
                <span>{new Date(post.timestamp).toLocaleTimeString()}</span>
                <button 
                  on:click={() => analyzeProfile(post.user.id)}
                  class="text-cyan-400 hover:text-cyan-300">
                  Analyze
                </button>
              </div>
            </div>
          {/each}
        </div>
      {/if}
    {/each}
  </div>
</div>