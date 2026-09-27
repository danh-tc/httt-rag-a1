import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev (`npm run dev`, port 5173) API calls are proxied to the FastAPI server on :8000.
// In production FastAPI serves the built `dist/` itself, so the paths are the same.
const API = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/search": API, "/dataset": API, "/figures": API },
  },
});
