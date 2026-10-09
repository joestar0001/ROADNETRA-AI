// Shared Tailwind (CDN) theme for every RoadNetra page. Load right after cdn.tailwindcss.com.
tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: "#060A13",   // page background
        panel: "#0B1120",    // sidebars, top bar
        card: "#0F172A",     // cards
        raised: "#16203A",   // hovered / nested surfaces
        line: "rgba(148,163,184,0.14)",
        "line-strong": "rgba(148,163,184,0.26)",
        ink: "#E6EBF5",      // primary text
        muted: "#94A3B8",    // secondary text
        dim: "#64748B",      // tertiary text
        brand: { DEFAULT: "#3B82F6", strong: "#2563EB", soft: "#93C5FD" },
        cyan: { DEFAULT: "#22D3EE", soft: "#A5F3FC" },
        danger: { DEFAULT: "#EF4444", soft: "#FCA5A5" },
        warn: { DEFAULT: "#F59E0B", soft: "#FCD34D" },
        ok: { DEFAULT: "#10B981", soft: "#6EE7B7" },
        info: { DEFAULT: "#0EA5E9", soft: "#7DD3FC" },
      },
      fontFamily: {
        display: ['"Plus Jakarta Sans"', "Inter", "system-ui", "sans-serif"],
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        card: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 10px 30px -12px rgba(0,0,0,0.6)",
        glow: "0 0 0 1px rgba(59,130,246,0.35), 0 12px 40px -10px rgba(59,130,246,0.45)",
        "glow-red": "0 0 0 1px rgba(239,68,68,0.4), 0 12px 40px -10px rgba(239,68,68,0.45)",
      },
      borderRadius: { xl: "0.875rem", "2xl": "1.125rem" },
      keyframes: {
        "fade-up": { "0%": { opacity: 0, transform: "translateY(8px)" }, "100%": { opacity: 1, transform: "none" } },
        scan: { "0%": { transform: "translateY(-100%)" }, "100%": { transform: "translateY(100%)" } },
      },
      animation: { "fade-up": "fade-up .5s ease both", scan: "scan 4s linear infinite" },
    },
  },
};
