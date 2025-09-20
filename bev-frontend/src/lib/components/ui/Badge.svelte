<!-- BEV OSINT Badge Component -->
<script lang="ts">
  import { cn } from '$lib/utils/cn';
  
  export let variant: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'proxied' = 'default';
  export let size: 'xs' | 'sm' | 'md' = 'sm';
  export let pulse = false;
  
  let className = '';
  export { className as class };
  
  const variants = {
    default: 'bg-dark-bg-tertiary text-dark-text-secondary',
    success: 'bg-green-600/20 text-green-400 border-green-600/30',
    warning: 'bg-yellow-600/20 text-yellow-400 border-yellow-600/30',
    danger: 'bg-red-600/20 text-red-400 border-red-600/30',
    info: 'bg-blue-600/20 text-blue-400 border-blue-600/30',
    proxied: 'bg-purple-600/20 text-purple-400 border-purple-600/30'
  };
  
  const sizes = {
    xs: 'text-xs px-2 py-0.5',
    sm: 'text-sm px-2.5 py-0.5',
    md: 'text-base px-3 py-1'
  };
  
  $: badgeClasses = cn(
    'inline-flex items-center rounded-md font-medium border',
    variants[variant],
    sizes[size],
    pulse && 'animate-pulse',
    className
  );
</script>

<span class={badgeClasses} {...$$restProps}>
  <slot />
</span>
