import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// only applies to the vite dev server!!

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  optimizeDeps: {
    include: ['react-plotly.js/factory', 'plotly.js-dist-min'],
  },
  server: {
    // vite dev server port
    port:5173,
    strictPort: true,
    proxy: {
      "/api": {
        // NOTE - edit if changing backend port
        target: 'http://localhost:9000',
        changeOrigin: true,
      }
    }
  }
})
