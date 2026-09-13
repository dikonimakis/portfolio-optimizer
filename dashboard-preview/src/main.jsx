import React from "react";
import { createRoot } from "react-dom/client";
import PortfolioOptimizer from "../../portfolio-optimizer.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <PortfolioOptimizer />
  </React.StrictMode>
);
