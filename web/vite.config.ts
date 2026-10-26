import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  base: process.env.PAGES_BASE || '/financial-research-workbench/',
  plugins: [react()],
  build: {outDir: '../dist', emptyOutDir: true},
});
