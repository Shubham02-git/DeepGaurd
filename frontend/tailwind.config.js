/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f4ff",
          100: "#dde7ff",
          200: "#c3d3ff",
          300: "#9ab5ff",
          400: "#6d8aff",
          500: "#4a63f5",
          600: "#3344e8",
          700: "#2a34d4",
          800: "#252cab",
          900: "#262e8a",
        },
        dark: {
          900: "#050810",
          800: "#0a0f1e",
          700: "#0f1629",
          600: "#161f38",
          500: "#1e2a4a",
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "noise": "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E\")",
      }
    },
  },
  plugins: [],
}
