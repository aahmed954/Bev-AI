<!-- BEV OSINT Button Component -->
<script lang="ts">
  import type { ComponentProps } from 'svelte';
  import { cn } from '$lib/utils/cn';
  
  export let variant: 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline' = 'primary';
  export let size: 'xs' | 'sm' | 'md' | 'lg' = 'md';
  export let loading = false;
  export let disabled = false;
  export let fullWidth = false;
  export let type: ComponentProps<'button'>['type'] = 'button';
  
  let className = '';
  export { className as class };
  
  const variants = {
    primary: 'bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-800 focus:ring-primary-500',
    secondary: 'bg-dark-bg-secondary text-dark-text-primary hover:bg-dark-bg-tertiary active:bg-dark-bg-elevated border border-dark-border-default',
    danger: 'bg-red-600 text-white hover:bg-red-700 active:bg-red-800 focus:ring-red-500',
    ghost: 'text-dark-text-primary hover:bg-dark-bg-secondary active:bg-dark-bg-tertiary',
    outline: 'border border-dark-border-default text-dark-text-primary hover:bg-dark-bg-secondary active:bg-dark-bg-tertiary'
  };
  
  const sizes = {
    xs: 'h-7 px-3 text-xs',
    sm: 'h-8 px-4 text-sm',
    md: 'h-10 px-5 text-base',
    lg: 'h-12 px-6 text-lg'
  };
  
  $: buttonClasses = cn(
    'inline-flex items-center justify-center font-medium rounded-md transition-colors duration-150',
    'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-dark-bg-primary',
    'disabled:opacity-50 disabled:cursor-not-allowed',
    variants[variant],
    sizes[size],
    fullWidth && 'w-full',
    loading && 'relative !text-transparent hover:!text-transparent cursor-wait',
    className
  );
</script>

<button
  {type}
  {disabled}
  class={buttonClasses}
  on:click
  on:submit
  {...$$restProps}
>
  {#if loading}
    <div class="absolute inset-0 flex items-center justify-center">
      <svg class="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
    </div>
  {/if}
  <slot />
</button>
