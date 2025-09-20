<!-- BEV OSINT Main Layout Component -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import Sidebar from '$lib/components/navigation/Sidebar.svelte';
  import Header from '$lib/components/navigation/Header.svelte';
  import { userPreferences, initializeApp } from '$lib/stores/app';
  
  export let data: { user?: { name: string } } = {};
  
  let sidebarCollapsed = false;
  
  // Subscribe to user preferences
  $: userPreferences.update(prefs => ({ ...prefs, sidebarCollapsed }));
  
  // Get page title from route
  $: pageTitle = getPageTitle($page.url.pathname);
  
  function getPageTitle(path: string): string {
    const titles: Record<string, string> = {
      '/': 'Intelligence Dashboard',
      '/darknet': 'Darknet Markets Analysis',
      '/crypto': 'Cryptocurrency Tracking',
      '/threats': 'Threat Intelligence',
      '/social': 'Social Intelligence',
      '/assistant': 'AI Assistant',
      '/reports': 'Intelligence Reports',
      '/opsec': 'OPSEC Status',
      '/settings': 'System Settings',
    };
    return titles[path] || 'BEV OSINT Framework';
  }
  
  onMount(async () => {
    // Initialize application on mount
    await initializeApp();
  });
</script>

<div class="flex h-screen bg-dark-bg-primary overflow-hidden">
  <!-- Sidebar -->
  <aside class="{sidebarCollapsed ? 'w-16' : 'w-64'} transition-all duration-200 flex-shrink-0">
    <Sidebar bind:collapsed={sidebarCollapsed} />
  </aside>
  
  <!-- Main Content Area -->
  <div class="flex-1 flex flex-col overflow-hidden">
    <!-- Header -->
    <Header userName={data.user?.name || 'Operator'}>
      <svelte:fragment slot="title">{pageTitle}</svelte:fragment>
    </Header>
    
    <!-- Page Content -->
    <main class="flex-1 overflow-auto bg-dark-bg-primary">
      <div class="h-full">
        <slot />
      </div>
    </main>
  </div>
</div>

<style>
  :global(body) {
    @apply bg-dark-bg-primary text-dark-text-primary;
  }
  
  :global(.scrollbar-thin) {
    scrollbar-width: thin;
    scrollbar-color: theme('colors.dark.border.subtle') transparent;
  }
  
  :global(.scrollbar-thin::-webkit-scrollbar) {
    width: 8px;
    height: 8px;
  }
  
  :global(.scrollbar-thin::-webkit-scrollbar-track) {
    background: transparent;
  }
  
  :global(.scrollbar-thin::-webkit-scrollbar-thumb) {
    background-color: theme('colors.dark.border.subtle');
    border-radius: 4px;
  }
  
  :global(.scrollbar-thin::-webkit-scrollbar-thumb:hover) {
    background-color: theme('colors.dark.border.default');
  }
</style>
