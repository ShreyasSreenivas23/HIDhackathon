import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api/* -> FastAPI on :8000 (override with VITE_API_TARGET, e.g. the AWS endpoint)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
