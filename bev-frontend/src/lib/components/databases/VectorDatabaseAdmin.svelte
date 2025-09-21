<!--
Vector Database Administration - Qdrant & Weaviate Management
Connected to: src/infrastructure/vector_db_manager.py
Features: Index management, embedding visualization, similarity search, performance metrics
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const vectorDatabases = writable({
		qdrant: {
			status: 'unknown',
			collections: [],
			total_vectors: 0,
			memory_usage: 0,
			disk_usage: 0
		},
		weaviate: {
			status: 'unknown',
			classes: [],
			total_objects: 0,
			memory_usage: 0,
			disk_usage: 0
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'qdrant', 'weaviate', 'search', 'metrics'
	const selectedDatabase = writable('qdrant');
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Search interface
	let searchQuery = {
		text: '',
		vector: [],
		similarity_threshold: 0.8,
		limit: 10,
		collection: '',
		filters: {}
	};
	
	let searchResults = [];
	
	// Index creation forms
	let newQdrantCollection = {
		name: '',
		vector_size: 384,
		distance: 'Cosine',
		shard_number: 1,
		replication_factor: 1
	};
	
	let newWeaviateClass = {
		class: '',
		description: '',
		vectorizer: 'text2vec-transformers',
		properties: []
	};
	
	const distanceMetrics = ['Cosine', 'Euclidean', 'Dot', 'Manhattan'];
	const vectorizers = ['text2vec-transformers', 'text2vec-openai', 'text2vec-cohere', 'none'];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadVectorDatabases();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/vector-db-admin');
			
			ws.onopen = () => {
				console.log('Vector DB Admin WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleVectorDBUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Vector DB WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Vector DB WebSocket connection failed:', error);
		}
	}
	
	function handleVectorDBUpdate(data: any) {
		switch (data.type) {
			case 'qdrant_collection_created':
				addQdrantCollection(data.collection);
				break;
			case 'weaviate_class_created':
				addWeaviateClass(data.class);
				break;
			case 'vector_search_result':
				updateSearchResults(data.results);
				break;
			case 'metrics_update':
				updateMetrics(data.database, data.metrics);
				break;
		}
	}
	
	async function loadVectorDatabases() {
		isLoading.set(true);
		try {
			const [qdrantData, weaviateData] = await Promise.all([
				invoke('get_qdrant_info'),
				invoke('get_weaviate_info')
			]);
			
			vectorDatabases.update(current => ({
				qdrant: { ...current.qdrant, ...qdrantData },
				weaviate: { ...current.weaviate, ...weaviateData }
			}));
		} catch (error) {
			console.error('Failed to load vector databases:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function createQdrantCollection() {
		try {
			const result = await invoke('create_qdrant_collection', { config: newQdrantCollection });
			addQdrantCollection(result);
			
			// Reset form
			newQdrantCollection = {
				name: '',
				vector_size: 384,
				distance: 'Cosine',
				shard_number: 1,
				replication_factor: 1
			};
			
			dispatch('collection_created', result);
		} catch (error) {
			console.error('Failed to create Qdrant collection:', error);
		}
	}
	
	async function createWeaviateClass() {
		try {
			const result = await invoke('create_weaviate_class', { config: newWeaviateClass });
			addWeaviateClass(result);
			
			// Reset form
			newWeaviateClass = {
				class: '',
				description: '',
				vectorizer: 'text2vec-transformers',
				properties: []
			};
			
			dispatch('class_created', result);
		} catch (error) {
			console.error('Failed to create Weaviate class:', error);
		}
	}
	
	async function performVectorSearch() {
		try {
			const result = await invoke('vector_search', {
				database: $selectedDatabase,
				query: searchQuery
			});
			
			searchResults = result;
			dispatch('search_completed', result);
		} catch (error) {
			console.error('Failed to perform vector search:', error);
		}
	}
	
	async function deleteCollection(database: string, name: string) {
		try {
			await invoke('delete_vector_collection', { database, name });
			
			if (database === 'qdrant') {
				vectorDatabases.update(current => ({
					...current,
					qdrant: {
						...current.qdrant,
						collections: current.qdrant.collections.filter(c => c.name !== name)
					}
				}));
			} else if (database === 'weaviate') {
				vectorDatabases.update(current => ({
					...current,
					weaviate: {
						...current.weaviate,
						classes: current.weaviate.classes.filter(c => c.class !== name)
					}
				}));
			}
		} catch (error) {
			console.error('Failed to delete collection:', error);
		}
	}
	
	function addQdrantCollection(collection: any) {
		vectorDatabases.update(current => ({
			...current,
			qdrant: {
				...current.qdrant,
				collections: [...current.qdrant.collections, collection]
			}
		}));
	}
	
	function addWeaviateClass(classObj: any) {
		vectorDatabases.update(current => ({
			...current,
			weaviate: {
				...current.weaviate,
				classes: [...current.weaviate.classes, classObj]
			}
		}));
	}
	
	function updateSearchResults(results: any[]) {
		searchResults = results;
	}
	
	function updateMetrics(database: string, metrics: any) {
		vectorDatabases.update(current => ({
			...current,
			[database]: { ...current[database], ...metrics }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'healthy':
			case 'connected': return 'text-green-400';
			case 'warning': return 'text-yellow-400';
			case 'error':
			case 'disconnected': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
	}
	
	function openCollectionModal(collection: any) {
		// Open collection detail modal
	}
</script>

<!-- Vector Database Admin -->
<div class="vector-db-admin h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-emerald-400">Vector Database Administration</h1>
			<div class="flex items-center space-x-4">
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($vectorDatabases.qdrant.status)}"></div>
					<span class="text-sm text-gray-400">Qdrant</span>
				</div>
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($vectorDatabases.weaviate.status)}"></div>
					<span class="text-sm text-gray-400">Weaviate</span>
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-emerald-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadVectorDatabases}
						class="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Database Overview -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{$vectorDatabases.qdrant.collections.length}</div>
				<div class="text-sm text-gray-400">Qdrant Collections</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-purple-400">{$vectorDatabases.weaviate.classes.length}</div>
				<div class="text-sm text-gray-400">Weaviate Classes</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">
					{formatNumber($vectorDatabases.qdrant.total_vectors + $vectorDatabases.weaviate.total_objects)}
				</div>
				<div class="text-sm text-gray-400">Total Vectors</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">
					{formatBytes($vectorDatabases.qdrant.memory_usage + $vectorDatabases.weaviate.memory_usage)}
				</div>
				<div class="text-sm text-gray-400">Memory Usage</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Database Overview', icon: '📊' },
				{ id: 'qdrant', label: 'Qdrant Management', icon: '🔵' },
				{ id: 'weaviate', label: 'Weaviate Management', icon: '🟣' },
				{ id: 'search', label: 'Vector Search', icon: '🔍' },
				{ id: 'metrics', label: 'Performance', icon: '📈' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-emerald-500 text-emerald-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => selectedTab.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $selectedTab === 'overview'}
			<!-- Database Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Qdrant Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<div class="flex items-center justify-between mb-4">
						<h3 class="text-lg font-semibold text-blue-400">Qdrant Vector Database</h3>
						<span class="px-2 py-1 rounded text-xs {getStatusColor($vectorDatabases.qdrant.status)} bg-gray-700">
							{$vectorDatabases.qdrant.status}
						</span>
					</div>
					
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div>
								<div class="text-xl font-bold text-white">{$vectorDatabases.qdrant.collections.length}</div>
								<div class="text-xs text-gray-400">Collections</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{formatNumber($vectorDatabases.qdrant.total_vectors)}</div>
								<div class="text-xs text-gray-400">Vectors</div>
							</div>
						</div>
						
						<div class="space-y-2">
							<h4 class="font-medium text-gray-300">Recent Collections</h4>
							{#each $vectorDatabases.qdrant.collections.slice(0, 3) as collection}
								<div class="flex justify-between items-center bg-gray-900 rounded p-2">
									<span class="text-white text-sm">{collection.name}</span>
									<div class="text-xs text-gray-400">
										{formatNumber(collection.points_count || 0)} points
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
				
				<!-- Weaviate Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<div class="flex items-center justify-between mb-4">
						<h3 class="text-lg font-semibold text-purple-400">Weaviate Vector Database</h3>
						<span class="px-2 py-1 rounded text-xs {getStatusColor($vectorDatabases.weaviate.status)} bg-gray-700">
							{$vectorDatabases.weaviate.status}
						</span>
					</div>
					
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div>
								<div class="text-xl font-bold text-white">{$vectorDatabases.weaviate.classes.length}</div>
								<div class="text-xs text-gray-400">Classes</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{formatNumber($vectorDatabases.weaviate.total_objects)}</div>
								<div class="text-xs text-gray-400">Objects</div>
							</div>
						</div>
						
						<div class="space-y-2">
							<h4 class="font-medium text-gray-300">Recent Classes</h4>
							{#each $vectorDatabases.weaviate.classes.slice(0, 3) as classObj}
								<div class="flex justify-between items-center bg-gray-900 rounded p-2">
									<span class="text-white text-sm">{classObj.class}</span>
									<div class="text-xs text-gray-400">
										{formatNumber(classObj.objectsCount || 0)} objects
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'qdrant'}
			<!-- Qdrant Management -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Create Collection -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Create Qdrant Collection</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Collection Name</label>
							<input
								type="text"
								bind:value={newQdrantCollection.name}
								placeholder="Enter collection name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Vector Size</label>
							<input
								type="number"
								bind:value={newQdrantCollection.vector_size}
								min="1"
								max="4096"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Distance Metric</label>
							<select
								bind:value={newQdrantCollection.distance}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							>
								{#each distanceMetrics as metric}
									<option value={metric}>{metric}</option>
								{/each}
							</select>
						</div>
						
						<button
							on:click={createQdrantCollection}
							class="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
							disabled={!newQdrantCollection.name}
						>
							Create Collection
						</button>
					</div>
				</div>
				
				<!-- Qdrant Collections -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Qdrant Collections</h3>
					{#if $vectorDatabases.qdrant.collections.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">🔵</div>
							<p>No collections found</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $vectorDatabases.qdrant.collections as collection}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{collection.name}</h4>
											<span class="text-sm text-gray-400">
												{formatNumber(collection.points_count || 0)} points
											</span>
										</div>
										<div class="flex items-center space-x-2">
											<button
												on:click={() => openCollectionModal(collection)}
												class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
											>
												View
											</button>
											<button
												on:click={() => deleteCollection('qdrant', collection.name)}
												class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
											>
												Delete
											</button>
										</div>
									</div>
									
									<div class="grid grid-cols-3 gap-4 text-sm">
										<div>
											<span class="text-gray-400">Vector Size:</span>
											<span class="text-white ml-2">{collection.config?.params?.vectors?.size || 'N/A'}</span>
										</div>
										<div>
											<span class="text-gray-400">Distance:</span>
											<span class="text-white ml-2">{collection.config?.params?.vectors?.distance || 'N/A'}</span>
										</div>
										<div>
											<span class="text-gray-400">Status:</span>
											<span class="text-white ml-2 {getStatusColor(collection.status)}">{collection.status || 'Active'}</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'weaviate'}
			<!-- Weaviate Management -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Create Class -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Create Weaviate Class</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Class Name</label>
							<input
								type="text"
								bind:value={newWeaviateClass.class}
								placeholder="Enter class name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Description</label>
							<textarea
								bind:value={newWeaviateClass.description}
								placeholder="Enter class description"
								rows="3"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							></textarea>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Vectorizer</label>
							<select
								bind:value={newWeaviateClass.vectorizer}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							>
								{#each vectorizers as vectorizer}
									<option value={vectorizer}>{vectorizer}</option>
								{/each}
							</select>
						</div>
						
						<button
							on:click={createWeaviateClass}
							class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium transition-colors"
							disabled={!newWeaviateClass.class}
						>
							Create Class
						</button>
					</div>
				</div>
				
				<!-- Weaviate Classes -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Weaviate Classes</h3>
					{#if $vectorDatabases.weaviate.classes.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">🟣</div>
							<p>No classes found</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $vectorDatabases.weaviate.classes as classObj}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{classObj.class}</h4>
											<span class="text-sm text-gray-400">
												{formatNumber(classObj.objectsCount || 0)} objects
											</span>
										</div>
										<div class="flex items-center space-x-2">
											<button
												class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors"
											>
												View
											</button>
											<button
												on:click={() => deleteCollection('weaviate', classObj.class)}
												class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
											>
												Delete
											</button>
										</div>
									</div>
									
									{#if classObj.description}
										<p class="text-gray-400 text-sm mb-2">{classObj.description}</p>
									{/if}
									
									<div class="text-sm">
										<span class="text-gray-400">Vectorizer:</span>
										<span class="text-white ml-2">{classObj.vectorizer || 'N/A'}</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'search'}
			<!-- Vector Search Interface -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Search Form -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-emerald-400">Vector Similarity Search</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Database</label>
							<select
								bind:value={$selectedDatabase}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
							>
								<option value="qdrant">Qdrant</option>
								<option value="weaviate">Weaviate</option>
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Search Text</label>
							<textarea
								bind:value={searchQuery.text}
								placeholder="Enter text to search for similar vectors"
								rows="4"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
							></textarea>
						</div>
						
						<div class="grid grid-cols-2 gap-4">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">
									Similarity Threshold: {searchQuery.similarity_threshold}
								</label>
								<input
									type="range"
									bind:value={searchQuery.similarity_threshold}
									min="0.1"
									max="1.0"
									step="0.05"
									class="w-full"
								/>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Result Limit</label>
								<input
									type="number"
									bind:value={searchQuery.limit}
									min="1"
									max="100"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-emerald-500"
								/>
							</div>
						</div>
						
						<button
							on:click={performVectorSearch}
							class="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded font-medium transition-colors"
							disabled={!searchQuery.text.trim()}
						>
							Search Vectors
						</button>
					</div>
				</div>
				
				<!-- Search Results -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Search Results</h3>
					{#if searchResults.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">🔍</div>
							<p>No search results</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each searchResults as result}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">Result #{result.id}</span>
										<span class="text-sm text-emerald-400">
											{(result.score * 100).toFixed(1)}% match
										</span>
									</div>
									<div class="text-sm text-gray-400 mb-2">
										{result.payload?.text || result.properties?.content || 'No preview available'}
									</div>
									<div class="text-xs text-gray-500">
										Vector: [{result.vector?.slice(0, 3).map(v => v.toFixed(3)).join(', ')}...]
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'metrics'}
			<!-- Performance Metrics -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Database Performance</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Qdrant Memory:</span>
							<span class="text-white">{formatBytes($vectorDatabases.qdrant.memory_usage)}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Weaviate Memory:</span>
							<span class="text-white">{formatBytes($vectorDatabases.weaviate.memory_usage)}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Total Vectors:</span>
							<span class="text-white">
								{formatNumber($vectorDatabases.qdrant.total_vectors + $vectorDatabases.weaviate.total_objects)}
							</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Search Performance</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">📊</div>
						<p>Search performance metrics</p>
						<p class="text-sm mt-2">Query latency and throughput charts</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Vector Databases | Qdrant + Weaviate
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('backup_databases')}
					class="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded text-sm font-medium transition-colors"
				>
					Backup All
				</button>
				<button
					on:click={() => dispatch('optimize_indices')}
					class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm font-medium transition-colors"
				>
					Optimize Indices
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.vector-db-admin {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.vector-db-admin *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.vector-db-admin *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.vector-db-admin *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.vector-db-admin *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>