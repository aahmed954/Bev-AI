<!-- BEV OSINT Card Component -->
<script lang="ts">
  import { cn } from '$lib/utils/cn';
  
  export let variant: 'default' | 'elevated' | 'bordered' | 'threat' = 'default';
  export let padding: 'none' | 'sm' | 'md' | 'lg' = 'md';
  export let interactive = false;
  
  let className = '';
  export { className as class };
  
  const variants = {
    default: 'bg-dark-bg-secondary',
    elevated: 'bg-dark-bg-elevated shadow-lg',
    bordered: 'bg-dark-bg-secondary border border-dark-border-default',
    threat: 'bg-dark-bg-secondary border border-red-600/50 shadow-glow-threat'
  };
  
  const paddings = {
    none: '',
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6'
  };
  
  $: cardClasses = cn(
    'rounded-lg',
    variants[variant],
    paddings[padding],
    interactive && 'transition-all duration-150 hover:bg-dark-bg-tertiary cursor-pointer',
    className
  );
</script>

<div class={cardClasses} on:click on:keydown {...$$restProps}>
  <slot />
</div>
