import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  assetsInclude: ['**/*.glb'],
  server: {
    port: 3000,
    proxy: {
      '/predict': 'http://localhost:8000',
      '/health':  'http://localhost:8000',
      '/models':  'http://localhost:8000',
    }
  }
})
