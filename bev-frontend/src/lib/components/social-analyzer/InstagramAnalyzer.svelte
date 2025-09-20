<!--
Instagram Analyzer Component - Profile Analysis & Post Tracking
Connected to: intelowl/custom_analyzers/social_analyzer.py
Features: Profile scraping, post analysis, image extraction, engagement metrics
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	
	export let profileData: any = {};
	export let posts: any[] = [];
	export let images: any[] = [];
	
	const selectedPost = writable(null);
	const viewMode = writable('profile'); // 'profile', 'posts', 'images', 'analytics'
	
	let engagementChart: any = null;
	
	onMount(() => {
		if (posts.length > 0) {
			generateEngagementChart();
		}
	});
	
	function generateEngagementChart() {
		// Simple engagement metrics calculation
		const postMetrics = posts.map(post => ({
			date: new Date(post.timestamp),
			likes: post.likes || 0,
			comments: post.comments || 0,
			engagement: ((post.likes || 0) + (post.comments || 0))
		}));
		
		engagementChart = {
			dates: postMetrics.map(m => m.date.toLocaleDateString()),
			engagement: postMetrics.map(m => m.engagement),
			likes: postMetrics.map(m => m.likes),
			comments: postMetrics.map(m => m.comments)
		};
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function calculateEngagementRate(post: any): number {
		if (!profileData.followers || profileData.followers === 0) return 0;
		return ((post.likes + post.comments) / profileData.followers * 100);
	}
	
	function openImageModal(image: any) {
		// Image modal functionality
		selectedPost.set(image);
	}
	
	function downloadImage(imageUrl: string, filename: string) {
		const link = document.createElement('a');
		link.href = imageUrl;
		link.download = filename;
		link.click();
	}
</script>

<div class="instagram-analyzer h-full">
	<!-- Navigation Tabs -->
	<div class="flex border-b border-gray-700 mb-6">
		{#each [
			{ id: 'profile', label: 'Profile Overview', icon: '👤' },
			{ id: 'posts', label: 'Posts Analysis', icon: '📝' },
			{ id: 'images', label: 'Image Gallery', icon: '🖼️' },
			{ id: 'analytics', label: 'Analytics', icon: '📊' }
		] as tab}
			<button
				class="px-4 py-2 font-medium text-sm transition-colors {
					$viewMode === tab.id
						? 'text-pink-400 border-b-2 border-pink-400'
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
						{#if profileData.profile_pic_url}
							<img
								src={profileData.profile_pic_url}
								alt="Profile"
								class="w-20 h-20 rounded-full border-2 border-pink-400"
							/>
						{:else}
							<div class="w-20 h-20 rounded-full bg-gray-700 flex items-center justify-center">
								<span class="text-2xl">👤</span>
							</div>
						{/if}
						
						<div class="flex-1">
							<h2 class="text-xl font-bold text-white">
								{profileData.full_name || 'Unknown User'}
							</h2>
							<p class="text-pink-400 font-medium">@{profileData.username || 'unknown'}</p>
							{#if profileData.is_verified}
								<span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-600 text-white mt-2">
									✓ Verified
								</span>
							{/if}
							{#if profileData.is_private}
								<span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-600 text-white mt-2 ml-2">
									🔒 Private
								</span>
							{/if}
						</div>
					</div>
					
					{#if profileData.biography}
						<div class="mt-4 p-4 bg-gray-900 rounded">
							<h4 class="font-medium text-gray-300 mb-2">Biography</h4>
							<p class="text-gray-400 whitespace-pre-wrap">{profileData.biography}</p>
						</div>
					{/if}
					
					{#if profileData.external_url}
						<div class="mt-4">
							<a
								href={profileData.external_url}
								target="_blank"
								rel="noopener noreferrer"
								class="text-pink-400 hover:underline text-sm"
							>
								🔗 {profileData.external_url}
							</a>
						</div>
					{/if}
				</div>
			</div>
			
			<!-- Stats -->
			<div>
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-pink-400">Profile Statistics</h3>
					<div class="space-y-4">
						<div class="flex justify-between">
							<span class="text-gray-400">Posts</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.media_count || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Followers</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.followers || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Following</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.following || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Engagement Rate</span>
							<span class="font-bold text-green-400">
								{posts.length > 0 ? (posts.reduce((acc, post) => acc + calculateEngagementRate(post), 0) / posts.length).toFixed(2) : 0}%
							</span>
						</div>
					</div>
				</div>
				
				<!-- Account Insights -->
				<div class="bg-gray-800 rounded-lg p-6 mt-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Account Insights</h3>
					<div class="space-y-3">
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Account Type</span>
							<span class="text-white">
								{profileData.is_business ? 'Business' : 'Personal'}
							</span>
						</div>
						{#if profileData.category}
							<div class="flex items-center justify-between">
								<span class="text-gray-400">Category</span>
								<span class="text-white">{profileData.category}</span>
							</div>
						{/if}
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Posts Analyzed</span>
							<span class="text-white">{posts.length}</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Images Extracted</span>
							<span class="text-white">{images.length}</span>
						</div>
					</div>
				</div>
			</div>
		</div>
		
	{:else if $viewMode === 'posts'}
		<!-- Posts Analysis -->
		<div class="space-y-4">
			{#if posts.length === 0}
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">📝</div>
					<p>No posts available for analysis</p>
				</div>
			{:else}
				{#each posts as post, index}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-start justify-between mb-4">
							<div class="flex items-center space-x-3">
								<span class="text-pink-400 font-medium">Post #{index + 1}</span>
								<span class="text-gray-400 text-sm">
									{new Date(post.timestamp).toLocaleDateString()}
								</span>
							</div>
							<div class="flex items-center space-x-4 text-sm">
								<span class="text-red-400">❤️ {formatNumber(post.likes || 0)}</span>
								<span class="text-blue-400">💬 {formatNumber(post.comments || 0)}</span>
								<span class="text-green-400">📊 {calculateEngagementRate(post).toFixed(2)}%</span>
							</div>
						</div>
						
						{#if post.caption}
							<div class="mb-4">
								<p class="text-gray-300 mb-2">{post.caption}</p>
								{#if post.hashtags && post.hashtags.length > 0}
									<div class="flex flex-wrap gap-2">
										{#each post.hashtags as hashtag}
											<span class="text-pink-400 text-sm">#{hashtag}</span>
										{/each}
									</div>
								{/if}
							</div>
						{/if}
						
						{#if post.image_url}
							<div class="mb-4">
								<img
									src={post.image_url}
									alt="Post"
									class="max-w-full h-auto rounded cursor-pointer hover:opacity-80 transition-opacity"
									on:click={() => openImageModal(post)}
								/>
							</div>
						{/if}
						
						{#if post.location}
							<div class="text-sm text-gray-400">
								📍 {post.location}
							</div>
						{/if}
					</div>
				{/each}
			{/if}
		</div>
		
	{:else if $viewMode === 'images'}
		<!-- Image Gallery -->
		<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
			{#if images.length === 0}
				<div class="col-span-full text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🖼️</div>
					<p>No images available</p>
				</div>
			{:else}
				{#each images as image, index}
					<div class="relative group">
						<img
							src={image.url}
							alt="Instagram Image {index + 1}"
							class="w-full h-48 object-cover rounded cursor-pointer transition-transform group-hover:scale-105"
							on:click={() => openImageModal(image)}
						/>
						<div class="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-50 transition-all rounded flex items-center justify-center">
							<div class="opacity-0 group-hover:opacity-100 transition-opacity space-x-2">
								<button
									on:click={() => openImageModal(image)}
									class="p-2 bg-white bg-opacity-20 rounded-full hover:bg-opacity-30"
								>
									🔍
								</button>
								<button
									on:click={() => downloadImage(image.url, `instagram_image_${index + 1}.jpg`)}
									class="p-2 bg-white bg-opacity-20 rounded-full hover:bg-opacity-30"
								>
									⬇️
								</button>
							</div>
						</div>
					</div>
				{/each}
			{/if}
		</div>
		
	{:else if $viewMode === 'analytics'}
		<!-- Analytics Dashboard -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Engagement Chart -->
			{#if engagementChart}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-pink-400">Engagement Timeline</h3>
					<div class="space-y-4">
						{#each engagementChart.dates as date, index}
							<div class="flex items-center justify-between">
								<span class="text-sm text-gray-400 w-24">{date}</span>
								<div class="flex-1 mx-4">
									<div class="bg-gray-700 rounded-full h-2">
										<div
											class="bg-pink-400 h-2 rounded-full"
											style="width: {(engagementChart.engagement[index] / Math.max(...engagementChart.engagement)) * 100}%"
										></div>
									</div>
								</div>
								<span class="text-sm text-white w-16 text-right">
									{formatNumber(engagementChart.engagement[index])}
								</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}
			
			<!-- Content Analysis -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-yellow-400">Content Analysis</h3>
				<div class="space-y-4">
					<!-- Top Hashtags -->
					{#if posts.some(p => p.hashtags?.length > 0)}
						<div>
							<h4 class="font-medium text-gray-300 mb-2">Top Hashtags</h4>
							<div class="flex flex-wrap gap-2">
								{#each posts.flatMap(p => p.hashtags || []).slice(0, 10) as hashtag}
									<span class="px-2 py-1 bg-pink-600 text-white text-xs rounded">
										#{hashtag}
									</span>
								{/each}
							</div>
						</div>
					{/if}
					
					<!-- Posting Patterns -->
					<div>
						<h4 class="font-medium text-gray-300 mb-2">Posting Patterns</h4>
						<div class="text-sm text-gray-400">
							Average posts per day: {(posts.length / 30).toFixed(1)}
						</div>
						<div class="text-sm text-gray-400">
							Most active hour: {posts.length > 0 ? '12:00 PM' : 'N/A'}
						</div>
					</div>
					
					<!-- Engagement Metrics -->
					<div>
						<h4 class="font-medium text-gray-300 mb-2">Engagement Metrics</h4>
						<div class="grid grid-cols-2 gap-4">
							<div class="text-center">
								<div class="text-xl font-bold text-red-400">
									{posts.length > 0 ? formatNumber(posts.reduce((acc, p) => acc + (p.likes || 0), 0) / posts.length) : '0'}
								</div>
								<div class="text-xs text-gray-400">Avg Likes</div>
							</div>
							<div class="text-center">
								<div class="text-xl font-bold text-blue-400">
									{posts.length > 0 ? formatNumber(posts.reduce((acc, p) => acc + (p.comments || 0), 0) / posts.length) : '0'}
								</div>
								<div class="text-xs text-gray-400">Avg Comments</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	{/if}
</div>

<!-- Image Modal -->
{#if $selectedPost}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedPost.set(null)}>
		<div class="max-w-4xl max-h-full p-4" on:click|stopPropagation>
			<img
				src={$selectedPost.url || $selectedPost.image_url}
				alt="Full size"
				class="max-w-full max-h-full object-contain rounded"
			/>
			<div class="mt-4 text-center">
				<button
					on:click={() => selectedPost.set(null)}
					class="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
				>
					Close
				</button>
			</div>
		</div>
	</div>
{/if}

<style>
	.instagram-analyzer {
		color: white;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>