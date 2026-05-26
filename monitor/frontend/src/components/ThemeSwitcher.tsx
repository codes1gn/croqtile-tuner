import { useState, useRef, useEffect } from "react";
import { useTheme, THEMES, type ThemeId } from "../theme";

const THEME_IDS = Object.keys(THEMES) as ThemeId[];

export function ThemeSwitcher() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const current = THEMES[theme];

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
        style={{
          backgroundColor: "var(--c-bgElevated)",
          border: "1px solid var(--c-border)",
          color: "var(--c-text)",
        }}
      >
        <span
          className="w-3.5 h-3.5 rounded-full shrink-0"
          style={{ background: `linear-gradient(135deg, ${current.colors.bg} 50%, ${current.colors.accent} 50%)` }}
        />
        <span>{current.label}</span>
        <svg className="w-3 h-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div
          className="absolute top-full left-0 mt-1 py-1 rounded-lg shadow-xl z-50 min-w-[180px]"
          style={{ backgroundColor: "var(--c-bgSurface)", border: "1px solid var(--c-border)" }}
        >
          {THEME_IDS.map((id) => {
            const t = THEMES[id];
            const isActive = id === theme;
            return (
              <button
                key={id}
                onClick={() => { setTheme(id); setOpen(false); }}
                className="w-full flex items-center gap-2.5 px-3 py-2 text-xs transition-colors text-left"
                style={{
                  color: isActive ? "var(--c-accent)" : "var(--c-textMuted)",
                  backgroundColor: isActive ? "var(--c-accentBg)" : "transparent",
                }}
                onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.backgroundColor = "var(--c-bgElevated)"; }}
                onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.backgroundColor = "transparent"; }}
              >
                <span
                  className="w-4 h-4 rounded-full shrink-0 border"
                  style={{
                    background: `linear-gradient(135deg, ${t.colors.bg} 50%, ${t.colors.accent} 50%)`,
                    borderColor: isActive ? t.colors.accent : t.colors.border,
                  }}
                />
                <span className="font-medium">{t.label}</span>
                {isActive && (
                  <svg className="w-3 h-3 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
