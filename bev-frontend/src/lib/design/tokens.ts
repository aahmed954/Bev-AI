// BEV OSINT Design Tokens - Intelligence-grade UI specifications

export const colors = {
  // Primary colors for OSINT operations
  primary: {
    50: '#f0f9ff',
    100: '#e0f2fe',
    200: '#bae6fd',
    300: '#7dd3fc',
    400: '#38bdf8',
    500: '#0ea5e9',
    600: '#0284c7',
    700: '#0369a1',
    800: '#075985',
    900: '#0c4a6e',
    950: '#082f49',
  },
  
  // Threat level indicators
  threat: {
    critical: '#dc2626',
    high: '#ea580c',
    medium: '#f59e0b',
    low: '#84cc16',
    info: '#06b6d4',
  },
  
  // Dark theme optimized for long analysis sessions
  dark: {
    bg: {
      primary: '#0a0a0b',
      secondary: '#141416',
      tertiary: '#1c1c1f',
      elevated: '#232328',
    },
    border: {
      subtle: '#2a2a30',
      default: '#3a3a42',
      strong: '#52525b',
    },
    text: {
      primary: '#fafafa',
      secondary: '#a1a1aa',
      tertiary: '#71717a',
      inverse: '#09090b',
    },
  },
  
  // Operational status colors
  status: {
    online: '#10b981',
    offline: '#6b7280',
    error: '#ef4444',
    warning: '#f59e0b',
    proxied: '#8b5cf6', // Tor/proxy active
  },
};

export const typography = {
  fonts: {
    mono: 'JetBrains Mono, Consolas, Monaco, monospace',
    sans: 'Inter, system-ui, -apple-system, sans-serif',
    display: 'Space Grotesk, Inter, sans-serif',
  },
  
  sizes: {
    xs: '0.75rem',    // 12px
    sm: '0.875rem',   // 14px
    base: '1rem',     // 16px
    lg: '1.125rem',   // 18px
    xl: '1.25rem',    // 20px
    '2xl': '1.5rem',  // 24px
    '3xl': '1.875rem', // 30px
    '4xl': '2.25rem', // 36px
  },
  
  weights: {
    light: 300,
    regular: 400,
    medium: 500,
    semibold: 600,
    bold: 700,
  },
  
  lineHeights: {
    tight: 1.25,
    snug: 1.375,
    normal: 1.5,
    relaxed: 1.625,
    loose: 1.75,
  },
};

export const spacing = {
  0: '0',
  px: '1px',
  0.5: '0.125rem',
  1: '0.25rem',
  1.5: '0.375rem',
  2: '0.5rem',
  2.5: '0.625rem',
  3: '0.75rem',
  3.5: '0.875rem',
  4: '1rem',
  5: '1.25rem',
  6: '1.5rem',
  7: '1.75rem',
  8: '2rem',
  9: '2.25rem',
  10: '2.5rem',
  12: '3rem',
  14: '3.5rem',
  16: '4rem',
  20: '5rem',
};

export const borderRadius = {
  none: '0',
  sm: '0.125rem',
  base: '0.25rem',
  md: '0.375rem',
  lg: '0.5rem',
  xl: '0.75rem',
  '2xl': '1rem',
  '3xl': '1.5rem',
  full: '9999px',
};

export const shadows = {
  sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
  base: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
  md: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
  lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
  xl: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
  '2xl': '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
  
  // Intelligence-specific shadows
  glow: {
    primary: '0 0 20px rgba(14, 165, 233, 0.3)',
    threat: '0 0 20px rgba(239, 68, 68, 0.3)',
    success: '0 0 20px rgba(16, 185, 129, 0.3)',
    warning: '0 0 20px rgba(245, 158, 11, 0.3)',
  },
};

export const animations = {
  durations: {
    instant: '0ms',
    fast: '150ms',
    normal: '250ms',
    slow: '350ms',
    slower: '500ms',
  },
  
  easings: {
    linear: 'linear',
    in: 'cubic-bezier(0.4, 0, 1, 1)',
    out: 'cubic-bezier(0, 0, 0.2, 1)',
    inOut: 'cubic-bezier(0.4, 0, 0.2, 1)',
    bounce: 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
  },
};

export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px',
  '3xl': '1920px',
  '4xl': '2560px',
};

export const zIndex = {
  base: 0,
  dropdown: 10,
  overlay: 20,
  modal: 30,
  popover: 40,
  tooltip: 50,
  notification: 60,
  critical: 9999,
};
