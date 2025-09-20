<!-- BEV OSINT Panel Component -->
<script lang="ts">
  import { cn } from '$lib/utils/cn';
  
  export let title: string | undefined = undefined;
  export let subtitle: string | undefined = undefined;
  export let collapsible = false;
  export let collapsed = false;
  
  let className = '';
  export { className as class };
  
  $: panelClasses = cn(
    'bg-dark-bg-secondary rounded-lg border border-dark-border-subtle',
    className
  );
</script>

<div class={panelClasses} {...$$restProps}>
  {#if title}
    <div class="px-4 py-3 border-b border-dark-border-subtle">
      <div class="flex items-center justify-between">
        <div>
          <h3 class="text-lg font-semibold text-dark-text-primary">{title}</h3>
          {#if subtitle}
            <p class="text-sm text-dark-text-secondary mt-0.5">{subtitle}</p>
          {/if}
        </div>
        {#if collapsible}
          <button
            type="button"
            on:click={() => (collapsed = !collapsed)}
            class="p-1.5 rounded hover:bg-dark-bg-tertiary transition-colors"
            aria-expanded={!collapsed}
          >
            <svg
              class="w-5 h-5 text-dark-text-secondary transition-transform {collapsed ? '' : 'rotate-180'}"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        {/if}
      </div>
    </div>
  {/if}
  
  {#if !collapsed}
    <div class="p-4">
      <slot />
    </div>
  {/if}
</div>
