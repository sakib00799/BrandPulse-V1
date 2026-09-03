import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        ink: "#15231f",
        canvas: "#f3f1e9",
        moss: "#315c4c",
        lime: "#cde85b",
        coral: "#f0785f",
      },
      boxShadow: {
        panel: "0 18px 50px rgba(21, 35, 31, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
