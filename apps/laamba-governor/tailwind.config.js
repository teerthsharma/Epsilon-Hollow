/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        gov: {
          bg: "#0D0D0D",
          panel: "#141414",
          border: "#2A2A2A",
          text: "#E0E0E0",
          dim: "#888888",
          accent: "#00E5FF",
          data: "#FF8C00",
          topo: "#00E5FF",
          task: "#00E676",
          compare: "#FFEA00",
          fusion: "#AA00FF",
          bus: "#FF4081",
          export: "#BDBDBD",
          viz: "#FF1744",
          error: "#FF1744",
          warn: "#FFD600",
          ok: "#00E676",
        },
      },
      fontFamily: {
        sans: ["Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Cascadia Code", "monospace"],
      },
    },
  },
  plugins: [],
};
