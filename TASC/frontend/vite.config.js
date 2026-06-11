import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['react-plotly.js/factory', 'plotly.js-dist-min'],
  },
  server: {
    port:5173,
    strictPort: true,
    proxy: {
      "/api": {
        // NOTE - edit if changing port
        target: 'http://localhost:9000',
        changeOrigin: true,
      }
    }
  }
})
