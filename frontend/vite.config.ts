/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API (uvicorn app.main:app) runs on :8000 during development.
const api = process.env.API_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  // FastAPI serves the build from app/web at "/".
  build: { outDir: "../app/web", emptyOutDir: true },
  server: { proxy: { "/v1": api, "/health": api } },
  test: { environment: "jsdom" },
});
