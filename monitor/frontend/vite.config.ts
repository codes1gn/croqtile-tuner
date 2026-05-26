import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const isGHPages = process.env.GITHUB_PAGES === "true";
const repoBase = isGHPages ? "/croqtile-tuner/" : "/";

export default defineConfig({
  base: repoBase,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api/events": {
        target: "http://localhost:8642",
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes) => {
            proxyRes.headers["cache-control"] = "no-cache";
            proxyRes.headers["x-accel-buffering"] = "no";
          });
        },
      },
      "/api": {
        target: "http://localhost:8642",
        changeOrigin: true,
      },
    },
  },
});
