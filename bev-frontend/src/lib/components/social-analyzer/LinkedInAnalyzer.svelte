<!--
LinkedIn Analyzer Component - Professional Network Analysis
Connected to: intelowl/custom_analyzers/social_analyzer.py
Features: Professional profile analysis, connection mapping, job history, skills analysis
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	
	export let profileData: any = {};
	export let connections: any[] = [];
	
	const viewMode = writable('profile'); // 'profile', 'connections', 'skills', 'experience'
	const selectedConnection = writable(null);
	
	let skillsAnalysis: any = null;
	let experienceAnalysis: any = null;
	let networkMetrics: any = null;
	
	onMount(() => {
		if (profileData && Object.keys(profileData).length > 0) {
			analyzeSkills();
			analyzeExperience();
			calculateNetworkMetrics();
		}
	});
	
	function analyzeSkills() {
		if (profileData.skills && profileData.skills.length > 0) {
			const skillCategories = profileData.skills.reduce((acc, skill) => {
				const category = skill.category || 'Other';
				if (!acc[category]) acc[category] = [];
				acc[category].push(skill);
				return acc;
			}, {});
			
			skillsAnalysis = {
				categories: skillCategories,
				totalSkills: profileData.skills.length,
				topSkills: profileData.skills
					.sort((a, b) => (b.endorsements || 0) - (a.endorsements || 0))
					.slice(0, 10)
			};
		}
	}
	
	function analyzeExperience() {
		if (profileData.experience && profileData.experience.length > 0) {
			const companies = new Set(profileData.experience.map(exp => exp.company));
			const industries = new Set(profileData.experience.map(exp => exp.industry).filter(Boolean));
			
			const totalExperience = profileData.experience.reduce((total, exp) => {
				if (exp.start_date && exp.end_date) {
					const start = new Date(exp.start_date);
					const end = exp.end_date === 'Present' ? new Date() : new Date(exp.end_date);
					return total + (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 365);
				}
				return total;
			}, 0);
			
			experienceAnalysis = {
				totalCompanies: companies.size,
				totalIndustries: industries.size,
				totalYears: Math.round(totalExperience * 10) / 10,
				currentRole: profileData.experience.find(exp => exp.end_date === 'Present'),
				recentExperience: profileData.experience.slice(0, 5)
			};
		}
	}
	
	function calculateNetworkMetrics() {
		if (connections && connections.length > 0) {
			const industries = connections.reduce((acc, conn) => {
				if (conn.industry) {
					acc[conn.industry] = (acc[conn.industry] || 0) + 1;
				}
				return acc;
			}, {});
			
			const companies = connections.reduce((acc, conn) => {
				if (conn.company) {
					acc[conn.company] = (acc[conn.company] || 0) + 1;
				}
				return acc;
			}, {});
			
			networkMetrics = {
				totalConnections: connections.length,
				topIndustries: Object.entries(industries)
					.sort(([,a], [,b]) => b - a)
					.slice(0, 5)
					.map(([industry, count]) => ({ industry, count })),
				topCompanies: Object.entries(companies)
					.sort(([,a], [,b]) => b - a)
					.slice(0, 5)
					.map(([company, count]) => ({ company, count })),
				mutualConnections: connections.filter(conn => conn.mutual_connections > 0).length
			};
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatDate(dateString: string): string {
		if (dateString === 'Present') return 'Present';
		return new Date(dateString).toLocaleDateString();
	}
	
	function calculateDuration(startDate: string, endDate: string): string {
		const start = new Date(startDate);
		const end = endDate === 'Present' ? new Date() : new Date(endDate);
		const diffTime = Math.abs(end.getTime() - start.getTime());
		const diffYears = Math.floor(diffTime / (1000 * 60 * 60 * 24 * 365));
		const diffMonths = Math.floor((diffTime % (1000 * 60 * 60 * 24 * 365)) / (1000 * 60 * 60 * 24 * 30));
		
		if (diffYears > 0) {
			return `${diffYears} year${diffYears > 1 ? 's' : ''} ${diffMonths} month${diffMonths > 1 ? 's' : ''}`;
		}
		return `${diffMonths} month${diffMonths > 1 ? 's' : ''}`;
	}
	
	function openConnectionModal(connection: any) {
		selectedConnection.set(connection);
	}
</script>

<div class="linkedin-analyzer h-full">
	<!-- Navigation Tabs -->
	<div class="flex border-b border-gray-700 mb-6">
		{#each [
			{ id: 'profile', label: 'Profile Overview', icon: '👤' },
			{ id: 'experience', label: 'Experience', icon: '💼' },
			{ id: 'skills', label: 'Skills Analysis', icon: '🎯' },
			{ id: 'connections', label: 'Network', icon: '🔗' }
		] as tab}
			<button
				class="px-4 py-2 font-medium text-sm transition-colors {
					$viewMode === tab.id
						? 'text-blue-400 border-b-2 border-blue-400'
						: 'text-gray-400 hover:text-white'
				}"
				on:click={() => viewMode.set(tab.id)}
			>
				<span class="mr-2">{tab.icon}</span>
				{tab.label}
			</button>
		{/each}
	</div>
	
	{#if $viewMode === 'profile'}
		<!-- Profile Overview -->
		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Profile Info -->
			<div class="lg:col-span-2">
				<div class="bg-gray-800 rounded-lg p-6">
					<div class="flex items-start space-x-4">
						{#if profileData.profile_picture}
							<img
								src={profileData.profile_picture}
								alt="Profile"
								class="w-20 h-20 rounded-full border-2 border-blue-400"
							/>
						{:else}
							<div class="w-20 h-20 rounded-full bg-gray-700 flex items-center justify-center">
								<span class="text-2xl">💼</span>
							</div>
						{/if}
						
						<div class="flex-1">
							<h2 class="text-xl font-bold text-white">
								{profileData.full_name || 'Unknown Professional'}
							</h2>
							<p class="text-blue-400 font-medium">{profileData.headline || 'Professional'}</p>
							<p class="text-gray-400 text-sm mt-1">
								📍 {profileData.location || 'Location not specified'}
							</p>
							{#if profileData.industry}
								<p class="text-gray-400 text-sm">
									🏢 {profileData.industry}
								</p>
							{/if}
						</div>
					</div>
					
					{#if profileData.summary}
						<div class="mt-4 p-4 bg-gray-900 rounded">
							<h4 class="font-medium text-gray-300 mb-2">Professional Summary</h4>
							<p class="text-gray-400 whitespace-pre-wrap">{profileData.summary}</p>
						</div>
					{/if}
					
					{#if experienceAnalysis && experienceAnalysis.currentRole}
						<div class="mt-4 p-4 bg-blue-900/20 border border-blue-800 rounded">
							<h4 class="font-medium text-blue-300 mb-2">Current Position</h4>
							<div class="text-white font-medium">{experienceAnalysis.currentRole.title}</div>
							<div class="text-blue-400">{experienceAnalysis.currentRole.company}</div>
							<div class="text-gray-400 text-sm mt-1">
								{formatDate(experienceAnalysis.currentRole.start_date)} - Present
							</div>
						</div>
					{/if}
				</div>
			</div>
			
			<!-- Professional Stats -->
			<div>
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Professional Stats</h3>
					<div class="space-y-4">
						<div class="flex justify-between">
							<span class="text-gray-400">Connections</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.connections_count || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Followers</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.followers_count || 0)}
							</span>
						</div>
						{#if experienceAnalysis}
							<div class="flex justify-between">
								<span class="text-gray-400">Experience</span>
								<span class="font-bold text-green-400">
									{experienceAnalysis.totalYears} years
								</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Companies</span>
								<span class="font-bold text-white">
									{experienceAnalysis.totalCompanies}
								</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Industries</span>
								<span class="font-bold text-white">
									{experienceAnalysis.totalIndustries}
								</span>
							</div>
						{/if}
						{#if skillsAnalysis}
							<div class="flex justify-between">
								<span class="text-gray-400">Skills</span>
								<span class="font-bold text-yellow-400">
									{skillsAnalysis.totalSkills}
								</span>
							</div>
						{/if}
					</div>
				</div>
				
				<!-- Network Insights -->
				{#if networkMetrics}
					<div class="bg-gray-800 rounded-lg p-6 mt-6">
						<h3 class="text-lg font-semibold mb-4 text-green-400">Network Insights</h3>
						<div class="space-y-3">
							<div class="flex justify-between">
								<span class="text-gray-400">Total Network</span>
								<span class="text-white">{networkMetrics.totalConnections}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Mutual Connections</span>
								<span class="text-white">{networkMetrics.mutualConnections}</span>
							</div>
							{#if networkMetrics.topIndustries.length > 0}
								<div>
									<div class="text-gray-400 text-sm mb-2">Top Industry</div>
									<div class="text-white">
										{networkMetrics.topIndustries[0].industry}
										<span class="text-gray-400 text-sm">
											({networkMetrics.topIndustries[0].count} connections)
										</span>
									</div>
								</div>
							{/if}
						</div>
					</div>
				{/if}
			</div>
		</div>
		
	{:else if $viewMode === 'experience'}
		<!-- Experience Analysis -->
		<div class="space-y-4">
			{#if !experienceAnalysis || !experienceAnalysis.recentExperience || experienceAnalysis.recentExperience.length === 0}
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">💼</div>
					<p>No experience data available</p>
				</div>
			{:else}
				{#each experienceAnalysis.recentExperience as experience, index}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-start justify-between mb-4">
							<div class="flex-1">
								<h3 class="text-lg font-semibold text-white">{experience.title}</h3>
								<div class="text-blue-400 font-medium">{experience.company}</div>
								<div class="text-gray-400 text-sm mt-1">
									{formatDate(experience.start_date)} - {formatDate(experience.end_date || 'Present')}
									{#if experience.start_date}
										<span class="ml-2">
											({calculateDuration(experience.start_date, experience.end_date || 'Present')})
										</span>
									{/if}
								</div>
								{#if experience.location}
									<div class="text-gray-400 text-sm">📍 {experience.location}</div>
								{/if}
							</div>
							{#if experience.end_date === 'Present' || !experience.end_date}
								<span class="px-2 py-1 bg-green-600 text-white text-xs rounded-full">
									Current
								</span>
							{/if}
						</div>
						
						{#if experience.description}
							<div class="mb-4">
								<p class="text-gray-300 whitespace-pre-wrap">{experience.description}</p>
							</div>
						{/if}
						
						{#if experience.skills && experience.skills.length > 0}
							<div>
								<h4 class="text-sm font-medium text-gray-300 mb-2">Skills Used:</h4>
								<div class="flex flex-wrap gap-2">
									{#each experience.skills as skill}
										<span class="px-2 py-1 bg-blue-600 text-white text-xs rounded">
											{skill}
										</span>
									{/each}
								</div>
							</div>
						{/if}
					</div>
				{/each}
			{/if}
		</div>
		
	{:else if $viewMode === 'skills'}
		<!-- Skills Analysis -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Top Skills -->
			{#if skillsAnalysis && skillsAnalysis.topSkills}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Top Endorsed Skills</h3>
					<div class="space-y-3">
						{#each skillsAnalysis.topSkills as skill, index}
							<div class="flex items-center justify-between">
								<span class="text-gray-300">{skill.name}</span>
								<div class="flex items-center space-x-2">
									<div class="w-24 bg-gray-700 rounded-full h-2">
										<div
											class="bg-yellow-400 h-2 rounded-full"
											style="width: {skill.endorsements ? (skill.endorsements / skillsAnalysis.topSkills[0].endorsements) * 100 : 0}%"
										></div>
									</div>
									<span class="text-yellow-400 text-sm w-8">
										{skill.endorsements || 0}
									</span>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			
			<!-- Skills by Category -->
			{#if skillsAnalysis && skillsAnalysis.categories}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Skills by Category</h3>
					<div class="space-y-4">
						{#each Object.entries(skillsAnalysis.categories) as [category, skills]}
							<div>
								<h4 class="font-medium text-gray-300 mb-2">{category}</h4>
								<div class="flex flex-wrap gap-2">
									{#each skills as skill}
										<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
											{skill.name}
											{#if skill.endorsements}
												<span class="text-yellow-400 ml-1">({skill.endorsements})</span>
											{/if}
										</span>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
		
	{:else if $viewMode === 'connections'}
		<!-- Network Analysis -->
		<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
			<!-- Network Overview -->
			{#if networkMetrics}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Network Overview</h3>
					<div class="space-y-4">
						<div class="text-center">
							<div class="text-3xl font-bold text-white">{networkMetrics.totalConnections}</div>
							<div class="text-gray-400 text-sm">Total Connections</div>
						</div>
						
						<div class="space-y-2">
							<h4 class="text-sm font-medium text-gray-300">Top Industries</h4>
							{#each networkMetrics.topIndustries as industry}
								<div class="flex justify-between text-sm">
									<span class="text-gray-400">{industry.industry}</span>
									<span class="text-white">{industry.count}</span>
								</div>
							{/each}
						</div>
						
						<div class="space-y-2">
							<h4 class="text-sm font-medium text-gray-300">Top Companies</h4>
							{#each networkMetrics.topCompanies as company}
								<div class="flex justify-between text-sm">
									<span class="text-gray-400">{company.company}</span>
									<span class="text-white">{company.count}</span>
								</div>
							{/each}
						</div>
					</div>
				</div>
			{/if}
			
			<!-- Connection List -->
			<div class="lg:col-span-2">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Recent Connections</h3>
					<div class="space-y-3">
						{#if connections.length === 0}
							<div class="text-center py-8 text-gray-400">
								<div class="text-3xl mb-2">🔗</div>
								<p>No connection data available</p>
							</div>
						{:else}
							{#each connections.slice(0, 10) as connection, index}
								<div
									class="flex items-center space-x-4 p-3 bg-gray-900 rounded cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openConnectionModal(connection)}
								>
									{#if connection.profile_picture}
										<img
											src={connection.profile_picture}
											alt="Connection"
											class="w-10 h-10 rounded-full"
										/>
									{:else}
										<div class="w-10 h-10 rounded-full bg-gray-600 flex items-center justify-center">
											<span class="text-sm">👤</span>
										</div>
									{/if}
									
									<div class="flex-1">
										<div class="font-medium text-white">{connection.name}</div>
										<div class="text-sm text-gray-400">{connection.headline || connection.title}</div>
										{#if connection.company}
											<div class="text-xs text-blue-400">{connection.company}</div>
										{/if}
									</div>
									
									{#if connection.mutual_connections}
										<div class="text-xs text-gray-400">
											{connection.mutual_connections} mutual
										</div>
									{/if}
								</div>
							{/each}
						{/if}
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<!-- Connection Detail Modal -->
{#if $selectedConnection}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedConnection.set(null)}>
		<div class="max-w-2xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-lg font-semibold text-blue-400">Connection Details</h3>
				<button
					on:click={() => selectedConnection.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="flex items-start space-x-4">
				{#if $selectedConnection.profile_picture}
					<img
						src={$selectedConnection.profile_picture}
						alt="Connection"
						class="w-16 h-16 rounded-full"
					/>
				{:else}
					<div class="w-16 h-16 rounded-full bg-gray-700 flex items-center justify-center">
						<span class="text-xl">👤</span>
					</div>
				{/if}
				
				<div class="flex-1">
					<h4 class="text-xl font-bold text-white">{$selectedConnection.name}</h4>
					<p class="text-blue-400 font-medium">{$selectedConnection.headline || $selectedConnection.title}</p>
					{#if $selectedConnection.company}
						<p class="text-gray-400">{$selectedConnection.company}</p>
					{/if}
					{#if $selectedConnection.location}
						<p class="text-gray-400 text-sm">📍 {$selectedConnection.location}</p>
					{/if}
				</div>
			</div>
			
			{#if $selectedConnection.summary}
				<div class="mt-4 p-4 bg-gray-900 rounded">
					<p class="text-gray-300">{$selectedConnection.summary}</p>
				</div>
			{/if}
			
			<div class="mt-4 grid grid-cols-2 gap-4 text-sm">
				{#if $selectedConnection.mutual_connections}
					<div>
						<span class="text-gray-400">Mutual Connections:</span>
						<span class="text-white ml-2">{$selectedConnection.mutual_connections}</span>
					</div>
				{/if}
				{#if $selectedConnection.industry}
					<div>
						<span class="text-gray-400">Industry:</span>
						<span class="text-white ml-2">{$selectedConnection.industry}</span>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.linkedin-analyzer {
		color: white;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>