import { createContext, useContext, useState, useEffect, type ReactNode } from "react";

export type ThemeId = "midnight" | "amber" | "neon" | "arctic" | "teal" | "crimson" | "green" | "pink";

export interface ThemeColors {
  bg: string;
  bgSurface: string;
  bgElevated: string;
  border: string;
  borderSubtle: string;
  text: string;
  textMuted: string;
  textFaint: string;
  accent: string;
  accentMuted: string;
  accentBg: string;
  headerBg: string;
  headerBorder: string;
  success: string;
  warning: string;
  error: string;
}

export const THEMES: Record<ThemeId, { label: string; colors: ThemeColors }> = {
  midnight: {
    label: "Midnight",
    colors: {
      bg: "#101114",
      bgSurface: "#171a1e",
      bgElevated: "#1e2228",
      border: "#2a3038",
      borderSubtle: "#23282f80",
      text: "#e8ecf0",
      textMuted: "#9ba4b0",
      textFaint: "#556070",
      accent: "#38bdf8",
      accentMuted: "#0e7490",
      accentBg: "#0c2d3f",
      headerBg: "#171a1eee",
      headerBorder: "#2a3038",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  amber: {
    label: "Ember",
    colors: {
      bg: "#141210",
      bgSurface: "#1c1916",
      bgElevated: "#25211c",
      border: "#382f26",
      borderSubtle: "#30271f80",
      text: "#f0ebe4",
      textMuted: "#b5a898",
      textFaint: "#6e5f4f",
      accent: "#f59e0b",
      accentMuted: "#92400e",
      accentBg: "#2d1a04",
      headerBg: "#1c1916ee",
      headerBorder: "#382f26",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  neon: {
    label: "Phantom",
    colors: {
      bg: "#110f16",
      bgSurface: "#19161f",
      bgElevated: "#211d2a",
      border: "#302a3c",
      borderSubtle: "#28233380",
      text: "#ece8f4",
      textMuted: "#a89fba",
      textFaint: "#5f5574",
      accent: "#a78bfa",
      accentMuted: "#6d28d9",
      accentBg: "#1e0f3d",
      headerBg: "#19161fee",
      headerBorder: "#302a3c",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  arctic: {
    label: "Arctic",
    colors: {
      bg: "#0f1318",
      bgSurface: "#161b22",
      bgElevated: "#1d242e",
      border: "#2b3545",
      borderSubtle: "#232d3b80",
      text: "#e6ecf5",
      textMuted: "#8b9bb5",
      textFaint: "#4c5f7a",
      accent: "#60a5fa",
      accentMuted: "#1d4ed8",
      accentBg: "#0a1e3f",
      headerBg: "#161b22ee",
      headerBorder: "#2b3545",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  teal: {
    label: "Deep Sea",
    colors: {
      bg: "#0e1414",
      bgSurface: "#151d1d",
      bgElevated: "#1c2626",
      border: "#283636",
      borderSubtle: "#1f2f2f80",
      text: "#e4f0ef",
      textMuted: "#88aaa8",
      textFaint: "#4a706d",
      accent: "#2dd4bf",
      accentMuted: "#0f766e",
      accentBg: "#052e2a",
      headerBg: "#151d1dee",
      headerBorder: "#283636",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  crimson: {
    label: "Scarlet",
    colors: {
      bg: "#141011",
      bgSurface: "#1d1618",
      bgElevated: "#261d20",
      border: "#3a282d",
      borderSubtle: "#30212580",
      text: "#f2e8ea",
      textMuted: "#b89da5",
      textFaint: "#755760",
      accent: "#fb7185",
      accentMuted: "#be123c",
      accentBg: "#3b0a1a",
      headerBg: "#1d1618ee",
      headerBorder: "#3a282d",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  green: {
    label: "Forest",
    colors: {
      bg: "#0e1310",
      bgSurface: "#151c17",
      bgElevated: "#1c261e",
      border: "#28372b",
      borderSubtle: "#1f2e2280",
      text: "#e4f2e8",
      textMuted: "#88b098",
      textFaint: "#4a7558",
      accent: "#4ade80",
      accentMuted: "#15803d",
      accentBg: "#052e16",
      headerBg: "#151c17ee",
      headerBorder: "#28372b",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
  pink: {
    label: "Dusk",
    colors: {
      bg: "#141015",
      bgSurface: "#1d161f",
      bgElevated: "#271e2a",
      border: "#3a2b3e",
      borderSubtle: "#30233480",
      text: "#f2e8f4",
      textMuted: "#b89dc0",
      textFaint: "#6e5078",
      accent: "#f0abfc",
      accentMuted: "#a21caf",
      accentBg: "#350b3a",
      headerBg: "#1d161fee",
      headerBorder: "#3a2b3e",
      success: "#34d399",
      warning: "#fbbf24",
      error: "#f87171",
    },
  },
};

const STORAGE_KEY = "croqtuner-theme";

interface ThemeContextValue {
  theme: ThemeId;
  setTheme: (id: ThemeId) => void;
  colors: ThemeColors;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "midnight",
  setTheme: () => {},
  colors: THEMES.midnight.colors,
});

function applyThemeVars(colors: ThemeColors) {
  const root = document.documentElement;
  Object.entries(colors).forEach(([key, value]) => {
    root.style.setProperty(`--c-${key}`, value);
  });
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<ThemeId>(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as ThemeId | null;
    return stored && stored in THEMES ? stored : "midnight";
  });

  const setTheme = (id: ThemeId) => {
    setThemeState(id);
    localStorage.setItem(STORAGE_KEY, id);
  };

  useEffect(() => {
    applyThemeVars(THEMES[theme].colors);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, setTheme, colors: THEMES[theme].colors }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  return useContext(ThemeContext);
}
