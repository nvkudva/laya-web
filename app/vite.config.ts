import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// COOP/COEP are required for wasm threads. credentialless (not require-corp) so a
// cross-origin weight host without a CORP header still loads.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "credentialless",
};

export default defineConfig({
  plugins: [react()],
  server: { headers: isolation },
  preview: { headers: isolation },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: { rollupOptions: { input: { main: "index.html", probe: "probe.html", parity: "parity.html" } } },
});
