/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        gray: {
          50: "var(--c-text)",
          100: "var(--c-text)",
          200: "var(--c-text)",
          300: "var(--c-textMuted)",
          400: "var(--c-textMuted)",
          500: "var(--c-textFaint)",
          600: "var(--c-textFaint)",
          700: "var(--c-border)",
          800: "var(--c-border)",
          900: "var(--c-bgSurface)",
          950: "var(--c-bg)",
        },
        slate: {
          200: "var(--c-text)",
          300: "var(--c-textMuted)",
          500: "var(--c-textFaint)",
          600: "var(--c-textFaint)",
          700: "var(--c-border)",
          900: "var(--c-bgSurface)",
          950: "var(--c-bg)",
        },
        cyan: {
          300: "var(--c-accent)",
          400: "var(--c-accent)",
          500: "var(--c-accent)",
          600: "var(--c-accentMuted)",
          700: "var(--c-accentMuted)",
          800: "var(--c-accentMuted)",
          950: "var(--c-accentBg)",
        },
        blue: {
          400: "var(--c-accent)",
          500: "var(--c-accent)",
          600: "var(--c-accent)",
        },
        violet: {
          300: "var(--c-accent)",
          500: "var(--c-accent)",
        },
      },
    },
  },
  plugins: [],
};
