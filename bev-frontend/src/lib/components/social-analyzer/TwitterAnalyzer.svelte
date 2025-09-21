<!--
Twitter Analyzer Component - Tweet Analysis & Sentiment Tracking
Connected to: intelowl/custom_analyzers/social_analyzer.py
Features: Tweet analysis, sentiment tracking, hashtag trends, network analysis
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	
	export let profileData: any = {};
	export let tweets: any[] = [];
	
	const selectedTweet = writable(null);
	const viewMode = writable('profile'); // 'profile', 'tweets', 'sentiment', 'network'
	const sentimentFilter = writable('all'); // 'all', 'positive', 'negative', 'neutral'
	
	let tweetMetrics: any = null;
	let sentimentAnalysis: any = null;
	let hashtagTrends: any[] = [];
	
	onMount(() => {
		if (tweets.length > 0) {
			calculateMetrics();
			analyzeSentiment();
			extractHashtagTrends();
		}
	});
	
	function calculateMetrics() {
		const totalTweets = tweets.length;
		const totalLikes = tweets.reduce((acc, tweet) => acc + (tweet.likes || 0), 0);
		const totalRetweets = tweets.reduce((acc, tweet) => acc + (tweet.retweets || 0), 0);
		const totalReplies = tweets.reduce((acc, tweet) => acc + (tweet.replies || 0), 0);
		
		tweetMetrics = {
			totalTweets,
			totalLikes,
			totalRetweets,
			totalReplies,
			avgLikes: totalTweets > 0 ? (totalLikes / totalTweets).toFixed(1) : 0,
			avgRetweets: totalTweets > 0 ? (totalRetweets / totalTweets).toFixed(1) : 0,
			avgReplies: totalTweets > 0 ? (totalReplies / totalTweets).toFixed(1) : 0,
			engagementRate: profileData.followers > 0 ? ((totalLikes + totalRetweets + totalReplies) / (totalTweets * profileData.followers) * 100).toFixed(2) : 0
		};
	}
	
	function analyzeSentiment() {
		const sentiments = tweets.map(tweet => tweet.sentiment || 'neutral');
		const sentimentCounts = sentiments.reduce((acc, sentiment) => {
			acc[sentiment] = (acc[sentiment] || 0) + 1;
			return acc;
		}, {});
		
		sentimentAnalysis = {
			positive: sentimentCounts.positive || 0,
			negative: sentimentCounts.negative || 0,
			neutral: sentimentCounts.neutral || 0,
			total: tweets.length
		};
	}
	
	function extractHashtagTrends() {
		const hashtagMap = new Map();
		
		tweets.forEach(tweet => {
			if (tweet.hashtags) {
				tweet.hashtags.forEach(hashtag => {
					const count = hashtagMap.get(hashtag) || 0;
					hashtagMap.set(hashtag, count + 1);
				});
			}
		});
		
		hashtagTrends = Array.from(hashtagMap.entries())
			.map(([hashtag, count]) => ({ hashtag, count }))
			.sort((a, b) => b.count - a.count)
			.slice(0, 10);
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function getSentimentColor(sentiment: string): string {
		switch (sentiment) {
			case 'positive': return 'text-green-400';
			case 'negative': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function getSentimentEmoji(sentiment: string): string {
		switch (sentiment) {
			case 'positive': return '😊';
			case 'negative': return '😔';
			default: return '😐';
		}
	}
	
	function filteredTweets() {
		if ($sentimentFilter === 'all') return tweets;
		return tweets.filter(tweet => tweet.sentiment === $sentimentFilter);
	}
	
	function openTweetModal(tweet: any) {
		selectedTweet.set(tweet);
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString() + ' ' + new Date(dateString).toLocaleTimeString();
	}
</script>

<div class="twitter-analyzer h-full">
	<!-- Navigation Tabs -->
	<div class="flex border-b border-gray-700 mb-6">
		{#each [
			{ id: 'profile', label: 'Profile Overview', icon: '👤' },
			{ id: 'tweets', label: 'Tweet Analysis', icon: '🐦' },
			{ id: 'sentiment', label: 'Sentiment Tracking', icon: '📊' },
			{ id: 'network', label: 'Network Analysis', icon: '🕸️' }
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
						{#if profileData.profile_image_url}
							<img
								src={profileData.profile_image_url}
								alt="Profile"
								class="w-20 h-20 rounded-full border-2 border-blue-400"
							/>
						{:else}
							<div class="w-20 h-20 rounded-full bg-gray-700 flex items-center justify-center">
								<span class="text-2xl">🐦</span>
							</div>
						{/if}
						
						<div class="flex-1">
							<h2 class="text-xl font-bold text-white">
								{profileData.name || 'Unknown User'}
							</h2>
							<p class="text-blue-400 font-medium">@{profileData.username || 'unknown'}</p>
							<div class="flex items-center space-x-4 mt-2">
								{#if profileData.verified}
									<span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-blue-600 text-white">
										✓ Verified
									</span>
								{/if}
								{#if profileData.protected}
									<span class="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-600 text-white">
										🔒 Protected
									</span>
								{/if}
							</div>
						</div>
					</div>
					
					{#if profileData.description}
						<div class="mt-4 p-4 bg-gray-900 rounded">
							<h4 class="font-medium text-gray-300 mb-2">Bio</h4>
							<p class="text-gray-400 whitespace-pre-wrap">{profileData.description}</p>
						</div>
					{/if}
					
					<div class="mt-4 grid grid-cols-2 gap-4 text-sm">
						{#if profileData.location}
							<div class="flex items-center text-gray-400">
								<span class="mr-2">📍</span>
								{profileData.location}
							</div>
						{/if}
						{#if profileData.url}
							<div class="flex items-center">
								<span class="mr-2">🔗</span>
								<a href={profileData.url} target="_blank" class="text-blue-400 hover:underline">
									{profileData.url}
								</a>
							</div>
						{/if}
						{#if profileData.created_at}
							<div class="flex items-center text-gray-400">
								<span class="mr-2">📅</span>
								Joined {new Date(profileData.created_at).toLocaleDateString()}
							</div>
						{/if}
					</div>
				</div>
			</div>
			
			<!-- Stats -->
			<div>
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Profile Statistics</h3>
					<div class="space-y-4">
						<div class="flex justify-between">
							<span class="text-gray-400">Tweets</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.statuses_count || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Followers</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.followers_count || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Following</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.friends_count || 0)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Listed</span>
							<span class="font-bold text-white">
								{formatNumber(profileData.listed_count || 0)}
							</span>
						</div>
						{#if tweetMetrics}
							<div class="flex justify-between">
								<span class="text-gray-400">Engagement Rate</span>
								<span class="font-bold text-green-400">
									{tweetMetrics.engagementRate}%
								</span>
							</div>
						{/if}
					</div>
				</div>
				
				<!-- Tweet Metrics -->
				{#if tweetMetrics}
					<div class="bg-gray-800 rounded-lg p-6 mt-6">
						<h3 class="text-lg font-semibold mb-4 text-yellow-400">Tweet Metrics</h3>
						<div class="space-y-3">
							<div class="flex justify-between">
								<span class="text-gray-400">Analyzed Tweets</span>
								<span class="text-white">{tweetMetrics.totalTweets}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Avg Likes</span>
								<span class="text-red-400">{tweetMetrics.avgLikes}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Avg Retweets</span>
								<span class="text-green-400">{tweetMetrics.avgRetweets}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Avg Replies</span>
								<span class="text-blue-400">{tweetMetrics.avgReplies}</span>
							</div>
						</div>
					</div>
				{/if}
			</div>
		</div>
		
	{:else if $viewMode === 'tweets'}
		<!-- Tweet Analysis -->
		<div class="mb-4 flex items-center justify-between">
			<h3 class="text-lg font-semibold text-blue-400">Tweet Analysis</h3>
			<div class="flex items-center space-x-4">
				<label class="text-sm text-gray-400">Filter by sentiment:</label>
				<select
					bind:value={$sentimentFilter}
					class="bg-gray-800 border border-gray-700 rounded px-3 py-1 text-white text-sm"
				>
					<option value="all">All Tweets</option>
					<option value="positive">Positive</option>
					<option value="negative">Negative</option>
					<option value="neutral">Neutral</option>
				</select>
			</div>
		</div>
		
		<div class="space-y-4">
			{#if filteredTweets().length === 0}
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🐦</div>
					<p>No tweets available for analysis</p>
				</div>
			{:else}
				{#each filteredTweets() as tweet, index}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-start justify-between mb-4">
							<div class="flex items-center space-x-3">
								<span class="text-blue-400 font-medium">Tweet #{index + 1}</span>
								<span class="text-gray-400 text-sm">
									{formatDate(tweet.created_at)}
								</span>
								{#if tweet.sentiment}
									<span class="flex items-center {getSentimentColor(tweet.sentiment)} text-sm">
										{getSentimentEmoji(tweet.sentiment)} {tweet.sentiment}
									</span>
								{/if}
							</div>
							<div class="flex items-center space-x-4 text-sm">
								<span class="text-red-400">❤️ {formatNumber(tweet.likes || 0)}</span>
								<span class="text-green-400">🔄 {formatNumber(tweet.retweets || 0)}</span>
								<span class="text-blue-400">💬 {formatNumber(tweet.replies || 0)}</span>
							</div>
						</div>
						
						<div class="mb-4">
							<p class="text-gray-300 mb-2 text-base leading-relaxed">{tweet.text}</p>
							{#if tweet.hashtags && tweet.hashtags.length > 0}
								<div class="flex flex-wrap gap-2">
									{#each tweet.hashtags as hashtag}
										<span class="text-blue-400 text-sm">#{hashtag}</span>
									{/each}
								</div>
							{/if}
						</div>
						
						{#if tweet.media && tweet.media.length > 0}
							<div class="mb-4 grid grid-cols-2 gap-2">
								{#each tweet.media as media}
									<img
										src={media.url}
										alt="Tweet media"
										class="rounded cursor-pointer hover:opacity-80 transition-opacity"
										on:click={() => openTweetModal(tweet)}
									/>
								{/each}
							</div>
						{/if}
						
						<div class="flex items-center justify-between text-sm text-gray-400">
							{#if tweet.in_reply_to_screen_name}
								<span>Reply to @{tweet.in_reply_to_screen_name}</span>
							{/if}
							{#if tweet.is_retweet}
								<span class="text-green-400">🔄 Retweet</span>
							{/if}
							{#if tweet.is_quote}
								<span class="text-yellow-400">💬 Quote Tweet</span>
							{/if}
						</div>
					</div>
				{/each}
			{/if}
		</div>
		
	{:else if $viewMode === 'sentiment'}
		<!-- Sentiment Analysis -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
			<!-- Sentiment Overview -->
			{#if sentimentAnalysis}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Sentiment Distribution</h3>
					<div class="space-y-4">
						<div class="flex items-center justify-between">
							<div class="flex items-center">
								<span class="text-green-400 mr-2">😊</span>
								<span class="text-gray-300">Positive</span>
							</div>
							<div class="flex items-center space-x-2">
								<div class="w-32 bg-gray-700 rounded-full h-2">
									<div
										class="bg-green-400 h-2 rounded-full"
										style="width: {(sentimentAnalysis.positive / sentimentAnalysis.total) * 100}%"
									></div>
								</div>
								<span class="text-white text-sm w-16">
									{sentimentAnalysis.positive} ({((sentimentAnalysis.positive / sentimentAnalysis.total) * 100).toFixed(1)}%)
								</span>
							</div>
						</div>
						
						<div class="flex items-center justify-between">
							<div class="flex items-center">
								<span class="text-gray-400 mr-2">😐</span>
								<span class="text-gray-300">Neutral</span>
							</div>
							<div class="flex items-center space-x-2">
								<div class="w-32 bg-gray-700 rounded-full h-2">
									<div
										class="bg-gray-400 h-2 rounded-full"
										style="width: {(sentimentAnalysis.neutral / sentimentAnalysis.total) * 100}%"
									></div>
								</div>
								<span class="text-white text-sm w-16">
									{sentimentAnalysis.neutral} ({((sentimentAnalysis.neutral / sentimentAnalysis.total) * 100).toFixed(1)}%)
								</span>
							</div>
						</div>
						
						<div class="flex items-center justify-between">
							<div class="flex items-center">
								<span class="text-red-400 mr-2">😔</span>
								<span class="text-gray-300">Negative</span>
							</div>
							<div class="flex items-center space-x-2">
								<div class="w-32 bg-gray-700 rounded-full h-2">
									<div
										class="bg-red-400 h-2 rounded-full"
										style="width: {(sentimentAnalysis.negative / sentimentAnalysis.total) * 100}%"
									></div>
								</div>
								<span class="text-white text-sm w-16">
									{sentimentAnalysis.negative} ({((sentimentAnalysis.negative / sentimentAnalysis.total) * 100).toFixed(1)}%)
								</span>
							</div>
						</div>
					</div>
				</div>
			{/if}
			
			<!-- Hashtag Trends -->
			{#if hashtagTrends.length > 0}
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Top Hashtags</h3>
					<div class="space-y-3">
						{#each hashtagTrends as hashtag, index}
							<div class="flex items-center justify-between">
								<span class="text-blue-400">#{hashtag.hashtag}</span>
								<div class="flex items-center space-x-2">
									<div class="w-20 bg-gray-700 rounded-full h-2">
										<div
											class="bg-blue-400 h-2 rounded-full"
											style="width: {(hashtag.count / hashtagTrends[0].count) * 100}%"
										></div>
									</div>
									<span class="text-white text-sm w-8">{hashtag.count}</span>
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		</div>
		
	{:else if $viewMode === 'network'}
		<!-- Network Analysis -->
		<div class="bg-gray-800 rounded-lg p-6">
			<h3 class="text-lg font-semibold mb-4 text-blue-400">Network Analysis</h3>
			<div class="text-center py-12 text-gray-400">
				<div class="text-4xl mb-4">🕸️</div>
				<p>Network visualization will be displayed here</p>
				<p class="text-sm mt-2">Analyzing mentions, replies, and retweet patterns...</p>
			</div>
		</div>
	{/if}
</div>

<!-- Tweet Detail Modal -->
{#if $selectedTweet}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedTweet.set(null)}>
		<div class="max-w-2xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-lg font-semibold text-blue-400">Tweet Details</h3>
				<button
					on:click={() => selectedTweet.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<p class="text-gray-300 text-base leading-relaxed">{$selectedTweet.text}</p>
				
				<div class="flex items-center space-x-6 text-sm">
					<span class="text-red-400">❤️ {formatNumber($selectedTweet.likes || 0)}</span>
					<span class="text-green-400">🔄 {formatNumber($selectedTweet.retweets || 0)}</span>
					<span class="text-blue-400">💬 {formatNumber($selectedTweet.replies || 0)}</span>
				</div>
				
				<div class="text-sm text-gray-400">
					{formatDate($selectedTweet.created_at)}
				</div>
				
				{#if $selectedTweet.media && $selectedTweet.media.length > 0}
					<div class="grid grid-cols-1 gap-2">
						{#each $selectedTweet.media as media}
							<img
								src={media.url}
								alt="Tweet media"
								class="w-full rounded"
							/>
						{/each}
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.twitter-analyzer {
		color: white;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>