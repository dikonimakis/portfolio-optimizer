import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// portfolio-optimizer.jsx lives one directory above this project's
// node_modules (it's the real library source, not a copy), so plain
// resolution can't find react/react-dom/recharts from there. Alias them
// explicitly rather than installing a second copy at the project root,
// which would break that project's own zero-dependency design.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      react: path.resolve(__dirname, "node_modules/react"),
      "react-dom": path.resolve(__dirname, "node_modules/react-dom"),
      recharts: path.resolve(__dirname, "node_modules/recharts"),
    },
  },
  server: {
    fs: { allow: [path.resolve(__dirname, ".."), __dirname] },
  },
});
