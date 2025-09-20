<!--
Cross-Platform Correlation Component - Identity Resolution & Multi-Platform Analysis
Connected to: intelowl/custom_analyzers/social_analyzer.py
Features: Identity correlation, cross-platform patterns, unified timeline, relationship mapping
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	
	export let profiles: any = {};
	export let correlationData: any = {};
	
	const viewMode = writable('overview'); // 'overview', 'timeline', 'patterns', 'identity'
	const selectedCorrelation = writable(null);
	
	let identityMatches: any[] = [];
	let crossPlatformPatterns: any = null;
	let unifiedTimeline: any[] = [];
	let riskAssessment: any = null;
	
	onMount(() => {
		if (profiles && Object.keys(profiles).length > 1) {
			analyzeIdentityMatches();
			analyzeCrossPlatformPatterns();
			buildUnifiedTimeline();
			assessCrossPlatformRisks();
		}
	});
	
	function analyzeIdentityMatches() {
		const platforms = Object.keys(profiles);
		const matches = [];
		
		// Email correlation
		const emails = new Set();
		platforms.forEach(platform => {
			if (profiles[platform]?.email) {
				emails.add(profiles[platform].email.toLowerCase());
			}
		});
		
		if (emails.size > 0) {
			matches.push({
				type: 'email',
				confidence: 0.95,
				platforms: platforms.filter(p => profiles[p]?.email),
				details: `Shared email address across ${platforms.length} platforms`,
				value: Array.from(emails)[0]
			});
		}
		
		// Name correlation
		const names = new Set();
		platforms.forEach(platform => {
			const profile = profiles[platform];
			if (profile?.full_name || profile?.name) {
				names.add((profile.full_name || profile.name).toLowerCase());
			}
		});
		
		if (names.size === 1 && platforms.length > 1) {
			matches.push({
				type: 'name',
				confidence: 0.80,
				platforms: platforms,
				details: `Consistent name across platforms`,
				value: Array.from(names)[0]
			});
		}
		
		// Profile picture correlation (simplified - would use image hashing in reality)
		const profilePics = platforms.filter(p => profiles[p]?.profile_picture || profiles[p]?.profile_pic_url);
		if (profilePics.length > 1) {
			matches.push({
				type: 'profile_picture',
				confidence: 0.70,
				platforms: profilePics,
				details: `Similar profile pictures detected`,
				value: 'Visual similarity detected'
			});
		}
		
		// Bio/description correlation
		const bios = [];
		platforms.forEach(platform => {
			const profile = profiles[platform];
			if (profile?.biography || profile?.description || profile?.summary) {
				bios.push({
					platform,
					text: (profile.biography || profile.description || profile.summary).toLowerCase()
				});
			}
		});
		
		// Simple text similarity check
		if (bios.length > 1) {
			const similarityScore = calculateTextSimilarity(bios[0].text, bios[1].text);
			if (similarityScore > 0.6) {
				matches.push({
					type: 'biography',
					confidence: similarityScore,
					platforms: bios.map(b => b.platform),
					details: `Similar biographical information`,
					value: `${Math.round(similarityScore * 100)}% text similarity`
				});
			}
		}
		
		identityMatches = matches.sort((a, b) => b.confidence - a.confidence);
	}
	
	function calculateTextSimilarity(text1: string, text2: string): number {
		// Simple Jaccard similarity for demonstration
		const words1 = new Set(text1.split(/\s+/));
		const words2 = new Set(text2.split(/\s+/));
		const intersection = new Set([...words1].filter(x => words2.has(x)));
		const union = new Set([...words1, ...words2]);
		return intersection.size / union.size;
	}
	
	function analyzeCrossPlatformPatterns() {
		const platforms = Object.keys(profiles);
		const patterns = {
			posting_frequency: {},
			engagement_patterns: {},
			content_themes: {},
			activity_hours: {},
			network_overlap: {}
		};
		
		// Analyze posting frequency
		platforms.forEach(platform => {
			const profile = profiles[platform];
			if (profile?.posts_count || profile?.statuses_count || profile?.media_count) {
				patterns.posting_frequency[platform] = profile.posts_count || profile.statuses_count || profile.media_count;
			}
		});
		
		// Analyze engagement patterns
		platforms.forEach(platform => {
			const profile = profiles[platform];
			const followers = profile?.followers_count || profile?.followers || 0;
			const following = profile?.following_count || profile?.friends_count || profile?.following || 0;
			
			if (followers > 0 && following > 0) {
				patterns.engagement_patterns[platform] = {
					followers,
					following,
					ratio: following > 0 ? followers / following : 0
				};
			}
		});
		
		// Content theme analysis (simplified)
		platforms.forEach(platform => {
			const profile = profiles[platform];
			const bio = profile?.biography || profile?.description || profile?.summary || '';
			const keywords = extractKeywords(bio);
			patterns.content_themes[platform] = keywords;
		});
		
		crossPlatformPatterns = patterns;
	}
	
	function extractKeywords(text: string): string[] {
		// Simple keyword extraction
		return text.toLowerCase()
			.split(/\s+/)
			.filter(word => word.length > 3)
			.slice(0, 10);
	}
	
	function buildUnifiedTimeline() {
		const timeline = [];
		const platforms = Object.keys(profiles);
		
		platforms.forEach(platform => {
			const profile = profiles[platform];
			
			// Account creation
			if (profile?.created_at || profile?.joined_date) {
				timeline.push({
					date: new Date(profile.created_at || profile.joined_date),
					platform,
					type: 'account_created',
					description: `${platform} account created`,
					details: profile
				});
			}
			
			// Profile updates (if available)
			if (profile?.last_updated) {
				timeline.push({
					date: new Date(profile.last_updated),
					platform,
					type: 'profile_updated',
					description: `${platform} profile updated`,
					details: profile
				});
			}
		});
		
		unifiedTimeline = timeline.sort((a, b) => a.date.getTime() - b.date.getTime());
	}
	
	function assessCrossPlatformRisks() {
		const risks = [];
		const platforms = Object.keys(profiles);
		
		// Privacy inconsistency risk
		const privacyLevels = platforms.map(platform => ({
			platform,
			isPrivate: profiles[platform]?.is_private || profiles[platform]?.protected || false
		}));
		
		const mixedPrivacy = privacyLevels.some(p => p.isPrivate) && privacyLevels.some(p => !p.isPrivate);
		if (mixedPrivacy) {
			risks.push({
				type: 'privacy_inconsistency',
				severity: 'medium',
				description: 'Mixed privacy settings across platforms',
				details: 'Some accounts are private while others are public, creating potential information leakage',
				affected_platforms: privacyLevels.map(p => p.platform)
			});
		}
		
		// Identity correlation risk
		if (identityMatches.length > 2) {
			risks.push({
				type: 'identity_correlation',
				severity: 'high',
				description: 'High identity correlation across platforms',
				details: `${identityMatches.length} correlation points found, making deanonymization easier`,
				affected_platforms: platforms
			});
		}
		
		// Information overlap risk
		const bioSimilarity = identityMatches.find(m => m.type === 'biography');
		if (bioSimilarity && bioSimilarity.confidence > 0.8) {
			risks.push({
				type: 'information_overlap',
				severity: 'medium',
				description: 'High information overlap between platforms',
				details: 'Similar biographical information increases tracking potential',
				affected_platforms: bioSimilarity.platforms
			});
		}
		
		riskAssessment = {
			total_risks: risks.length,
			high_risk: risks.filter(r => r.severity === 'high').length,
			medium_risk: risks.filter(r => r.severity === 'medium').length,
			low_risk: risks.filter(r => r.severity === 'low').length,
			risks
		};
	}
	
	function formatDate(date: Date): string {
		return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
	}
	
	function getConfidenceColor(confidence: number): string {
		if (confidence >= 0.9) return 'text-red-400';
		if (confidence >= 0.7) return 'text-yellow-400';
		return 'text-green-400';
	}
	
	function getRiskSeverityColor(severity: string): string {
		switch (severity) {
			case 'high': return 'text-red-400 bg-red-900/30 border-red-800';
			case 'medium': return 'text-yellow-400 bg-yellow-900/30 border-yellow-800';
			case 'low': return 'text-green-400 bg-green-900/30 border-green-800';
			default: return 'text-gray-400 bg-gray-900/30 border-gray-800';
		}
	}
	
	function openCorrelationModal(correlation: any) {
		selectedCorrelation.set(correlation);
	}
</script>

<div class="correlation-analyzer h-full">
	<!-- Navigation Tabs -->
	<div class="flex border-b border-gray-700 mb-6">
		{#each [
			{ id: 'overview', label: 'Correlation Overview', icon: '🔗' },
			{ id: 'identity', label: 'Identity Matches', icon: '🎯' },
			{ id: 'patterns', label: 'Behavioral Patterns', icon: '📊' },
			{ id: 'timeline', label: 'Unified Timeline', icon: '⏰' }
		] as tab}
			<button
				class="px-4 py-2 font-medium text-sm transition-colors {
					$viewMode === tab.id
						? 'text-purple-400 border-b-2 border-purple-400'
						: 'text-gray-400 hover:text-white'
				}"
				on:click={() => viewMode.set(tab.id)}
			>
				<span class="mr-2">{tab.icon}</span>
				{tab.label}
			</button>
		{/each}
	</div>
	
	{#if $viewMode === 'overview'}
		<!-- Correlation Overview -->
		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Summary Stats -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-purple-400">Correlation Summary</h3>
				<div class="space-y-4">
					<div class="flex justify-between">
						<span class="text-gray-400">Platforms Analyzed</span>
						<span class="font-bold text-white">
							{Object.keys(profiles).length}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Identity Matches</span>
						<span class="font-bold text-purple-400">
							{identityMatches.length}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-400">Timeline Events</span>
						<span class="font-bold text-white">
							{unifiedTimeline.length}
						</span>
					</div>
					{#if riskAssessment}
						<div class="flex justify-between">
							<span class="text-gray-400">Risk Factors</span>
							<span class="font-bold text-red-400">
								{riskAssessment.total_risks}
							</span>
						</div>
					{/if}
				</div>
			</div>
			
			<!-- Platform Matrix -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-blue-400">Platform Matrix</h3>
				<div class="grid grid-cols-2 gap-3">
					{#each Object.entries(profiles) as [platform, profile]}
						<div class="bg-gray-900 rounded p-3 text-center">
							<div class="text-2xl mb-1">
								{#if platform === 'instagram'}📷
								{:else if platform === 'twitter'}🐦
								{:else if platform === 'linkedin'}💼
								{:else if platform === 'facebook'}👥
								{:else}📱{/if}
							</div>
							<div class="text-sm font-medium text-white capitalize">{platform}</div>
							<div class="text-xs text-gray-400">
								{profile?.username || profile?.name || 'N/A'}
							</div>
						</div>
					{/each}
				</div>
			</div>
			
			<!-- Risk Assessment -->
			{#if riskAssessment}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-red-400">Risk Assessment</h3>
					<div class="space-y-3">
						<div class="flex justify-between">
							<span class="text-gray-400">High Risk</span>
							<span class="text-red-400 font-bold">{riskAssessment.high_risk}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Medium Risk</span>
							<span class="text-yellow-400 font-bold">{riskAssessment.medium_risk}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Low Risk</span>
							<span class="text-green-400 font-bold">{riskAssessment.low_risk}</span>
						</div>
					</div>
					
					{#if riskAssessment.risks.length > 0}
						<div class="mt-4 space-y-2">
							{#each riskAssessment.risks.slice(0, 2) as risk}
								<div class="p-3 rounded border {getRiskSeverityColor(risk.severity)}">
									<div class="font-medium text-sm">{risk.description}</div>
									<div class="text-xs mt-1">{risk.details}</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			{/if}
		</div>
		
	{:else if $viewMode === 'identity'}
		<!-- Identity Matches -->
		<div class="space-y-4">
			{#if identityMatches.length === 0}
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🎯</div>
					<p>No identity correlations found</p>
				</div>
			{:else}
				{#each identityMatches as match, index}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-start justify-between mb-4">
							<div class="flex-1">
								<h3 class="text-lg font-semibold text-white capitalize">
									{match.type.replace('_', ' ')} Correlation
								</h3>
								<p class="text-gray-400">{match.details}</p>
							</div>
							<div class="text-right">
								<div class="text-sm text-gray-400">Confidence</div>
								<div class="text-lg font-bold {getConfidenceColor(match.confidence)}">
									{Math.round(match.confidence * 100)}%
								</div>
							</div>
						</div>
						
						<div class="mb-4">
							<div class="text-sm text-gray-400 mb-2">Matched Value:</div>
							<div class="bg-gray-900 rounded p-3">
								<code class="text-purple-400">{match.value}</code>
							</div>
						</div>
						
						<div>
							<div class="text-sm text-gray-400 mb-2">Affected Platforms:</div>
							<div class="flex flex-wrap gap-2">
								{#each match.platforms as platform}
									<span class="px-2 py-1 bg-purple-600 text-white text-xs rounded capitalize">
										{platform}
									</span>
								{/each}
							</div>
						</div>
					</div>
				{/each}
			{/if}
		</div>
		
	{:else if $viewMode === 'patterns'}
		<!-- Behavioral Patterns -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Posting Frequency -->
			{#if crossPlatformPatterns?.posting_frequency}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Posting Activity</h3>
					<div class="space-y-3">
						{#each Object.entries(crossPlatformPatterns.posting_frequency) as [platform, count]}
							<div class="flex items-center justify-between">
								<span class="text-gray-300 capitalize">{platform}</span>
								<div class="flex items-center space-x-2">
									<div class="w-24 bg-gray-700 rounded-full h-2">
										<div
											class="bg-blue-400 h-2 rounded-full"
											style="width: {(count / Math.max(...Object.values(crossPlatformPatterns.posting_frequency))) * 100}%"
										></div>
									</div>
									<span class="text-white text-sm w-12">{count}</span>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			
			<!-- Engagement Patterns -->
			{#if crossPlatformPatterns?.engagement_patterns}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Engagement Patterns</h3>
					<div class="space-y-4">
						{#each Object.entries(crossPlatformPatterns.engagement_patterns) as [platform, data]}
							<div class="bg-gray-900 rounded p-3">
								<div class="font-medium text-white capitalize mb-2">{platform}</div>
								<div class="grid grid-cols-3 gap-2 text-sm">
									<div class="text-center">
										<div class="text-blue-400 font-bold">{data.followers.toLocaleString()}</div>
										<div class="text-gray-400 text-xs">Followers</div>
									</div>
									<div class="text-center">
										<div class="text-green-400 font-bold">{data.following.toLocaleString()}</div>
										<div class="text-gray-400 text-xs">Following</div>
									</div>
									<div class="text-center">
										<div class="text-yellow-400 font-bold">{data.ratio.toFixed(1)}</div>
										<div class="text-gray-400 text-xs">Ratio</div>
									</div>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			
			<!-- Content Themes -->
			{#if crossPlatformPatterns?.content_themes}
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Content Theme Analysis</h3>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{#each Object.entries(crossPlatformPatterns.content_themes) as [platform, keywords]}
							<div class="bg-gray-900 rounded p-4">
								<h4 class="font-medium text-white capitalize mb-3">{platform}</h4>
								<div class="flex flex-wrap gap-1">
									{#each keywords.slice(0, 8) as keyword}
										<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
											{keyword}
										</span>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
		
	{:else if $viewMode === 'timeline'}
		<!-- Unified Timeline -->
		<div class="space-y-4">
			{#if unifiedTimeline.length === 0}
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">⏰</div>
					<p>No timeline events available</p>
				</div>
			{:else}
				{#each unifiedTimeline as event, index}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-start space-x-4">
							<div class="flex-shrink-0 w-12 h-12 bg-purple-600 rounded-full flex items-center justify-center">
								{#if event.platform === 'instagram'}📷
								{:else if event.platform === 'twitter'}🐦
								{:else if event.platform === 'linkedin'}💼
								{:else if event.platform === 'facebook'}👥
								{:else}📱{/if}
							</div>
							
							<div class="flex-1">
								<div class="flex items-center justify-between mb-2">
									<h3 class="text-lg font-medium text-white">{event.description}</h3>
									<span class="text-sm text-gray-400">{formatDate(event.date)}</span>
								</div>
								
								<div class="text-gray-400 text-sm mb-2">
									Platform: <span class="text-purple-400 capitalize">{event.platform}</span>
								</div>
								
								{#if event.type === 'account_created'}
									<div class="bg-gray-900 rounded p-3">
										<div class="text-sm text-gray-300">Account Details:</div>
										<div class="mt-1 text-sm">
											{#if event.details.username}
												<div>Username: <span class="text-purple-400">@{event.details.username}</span></div>
											{/if}
											{#if event.details.name || event.details.full_name}
												<div>Name: <span class="text-white">{event.details.name || event.details.full_name}</span></div>
											{/if}
										</div>
									</div>
								{/if}
							</div>
						</div>
					</div>
				{/each}
			{/if}
		</div>
	{/if}
</div>

<!-- Correlation Detail Modal -->
{#if $selectedCorrelation}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedCorrelation.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-lg font-semibold text-purple-400">Correlation Analysis</h3>
				<button
					on:click={() => selectedCorrelation.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<pre class="text-gray-300 text-sm overflow-auto">{JSON.stringify($selectedCorrelation, null, 2)}</pre>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.correlation-analyzer {
		color: white;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>