import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
    plugins: [sveltekit()],
    server: {
        port: 5173,
        strictPort: true,
        host: 'localhost'
    },
    build: {
        target: 'esnext',
        minify: 'esbuild',
        sourcemap: false
    },
    optimizeDeps: {
        include: ['cytoscape', 'echarts', 'dompurify']
    }
});