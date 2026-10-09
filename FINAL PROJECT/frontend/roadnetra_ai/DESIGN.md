---
name: RoadNetra-AI
colors:
  surface: '#0f131c'
  surface-dim: '#0f131c'
  surface-bright: '#353942'
  surface-container-lowest: '#0a0e16'
  surface-container-low: '#181c24'
  surface-container: '#1c2028'
  surface-container-high: '#262a33'
  surface-container-highest: '#31353e'
  on-surface: '#dfe2ee'
  on-surface-variant: '#c3c6d7'
  inverse-surface: '#dfe2ee'
  inverse-on-surface: '#2c3039'
  outline: '#8d90a0'
  outline-variant: '#434655'
  surface-tint: '#b4c5ff'
  primary: '#b4c5ff'
  on-primary: '#002a78'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#0053db'
  secondary: '#68dba9'
  on-secondary: '#003825'
  secondary-container: '#25a475'
  on-secondary-container: '#00311f'
  tertiary: '#ffb4ab'
  on-tertiary: '#690005'
  tertiary-container: '#d52022'
  on-tertiary-container: '#ffecea'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#85f8c4'
  secondary-fixed-dim: '#68dba9'
  on-secondary-fixed: '#002114'
  on-secondary-fixed-variant: '#005137'
  tertiary-fixed: '#ffdad6'
  tertiary-fixed-dim: '#ffb4ab'
  on-tertiary-fixed: '#410002'
  on-tertiary-fixed-variant: '#93000b'
  background: '#0f131c'
  on-background: '#dfe2ee'
  surface-variant: '#31353e'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
  data-mono:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
    letterSpacing: -0.01em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-compact: 0.5rem
  margin: 1.5rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
  space-2xl: 2rem
  space-3xl: 3rem
---

## Brand & Style

This design system establishes an operational standard for mission-critical urban intelligence and highway surveillance. The brand personality blends institutional authority with the precise, high-density utility of elite engineering platforms. The visual language conveys calm under pressure: operators managing multi-lane expressways, emergency dispatch, and traffic signal networks require high information density without cognitive friction.

The aesthetic fuses modern technical minimalism with situational awareness UI. It rejects decorative excess in favor of crisp structural divisions, unambiguous telemetry legibility, and immediate visual hierarchy. The emotional response is one of total operational control, institutional trust, and zero latency in decision-making. High-visibility situational accents punctuate a disciplined neutral canvas, allowing urgent incidents to break through ambient telemetry immediately.

## Colors

The palette operates on a disciplined 60-30-10 distribution model tailored for intense 24/7 monitoring environments, defaulting to Control Room Dark Mode while supporting an institutional Light Mode.

- **60% Dominant Foundation**: In Dark Mode, deep obsidian and night-slate tones (`#0B0F17` canvas background, `#111827` panel layer, `#1E293B` elevated cards) eliminate eye fatigue during overnight shifts while preserving contrast for GIS layers. In Light Mode, clean architectural slates dominate (`#FFFFFF` base, `#F8FAFC` canvas, `#F1F5F9` panels).
- **30% Structural & Content**: In Dark Mode, hair-line structural boundaries use `#334155`, primary labels and readouts take `#F8FAFC`, and secondary telemetry uses `#94A3B8`. In Light Mode, structural rules use `#E2E8F0` and `#CBD5E1`, with primary body text in `#0F172A` and muted indicators in `#334155`.
- **10% Brand & Telemetry Accents**: Interactive authority resides in Cobalt Blue (`#2563EB` interactive, `#1D4ED8` hover, `#3B82F6` dark-mode focus). Safety alerts follow international emergency management norms:
  - **P1 Critical / Crash / Blockage**: Red (`#DC2626`)
  - **P2 High / Congestion / Stall**: Amber (`#D97706`)
  - **Operational Normal / Free-flow**: Emerald Green (`#059669`)
  - **Info / Sensor / Weather**: Cyan (`#0284C7`)

## Typography

Typography enforces a strict hierarchy between strategic overview headings and dense data scanning.

- **Headings (Plus Jakarta Sans)**: Delivers geometric confidence, clarity, and authority. Used for view titles, panel headers, and critical modal confirmations. Its humanist proportions keep the interface approachable despite dense technical context.
- **Body & Data Readouts (Inter)**: Employs `cv05`, `cv08`, and OpenType tabular figures (`tnum`) across all telemetry readouts, GPS coordinates, timestamp feeds, and speed metrics to eliminate visual jitter during real-time UI updates.
- **Data Labels & Overlays**: Micro-labels (11px, bold, uppercase) maintain rapid legibility on live camera overlays and map pins even on low-resolution field terminals.

## Layout & Spacing

The layout is built around a rigid 8px spatial grid, with a 4px sub-grid for high-density components (data tables, mini telemetry chips, and filter ribbons).

- **Grid Architecture**: Multi-monitor desktop screens (1440px to 4K control walls) use a fluid multi-pane layout: a collapsable 280px left navigation rail, a variable-width primary map or live camera grid, and a 420px contextual right drawer for incident triaging and live feeds.
- **Breakpoints**:
  - `Desktop/Command Center (>=1280px)`: Fixed-width contextual sidebars with fluid map/camera viewports. Split screens support side-by-side stream feeds.
  - `Field Tablet (768px - 1279px)`: Off-canvas sliding drawers for telemetry inspection; persistent alert bar anchored at the top.
  - `Mobile Patrol (<768px)`: Single-column stacked layout with bottom navigation sheet and floating quick-action triage buttons.
- **Internal Density**: Dense data rows utilize `space-xs` (4px) and `space-sm` (8px) gaps. Modals and floating command cards use `space-lg` (16px) and `space-xl` (24px) padding to ensure unambiguous separation between critical controls.

## Elevation & Depth

Visual depth combines surface tint separation with micro-bordered ambient shadows, avoiding high blur radiuses that reduce crispness on video walls.

- **Base Layer (Surface 0)**: Obsidian `#0B0F17` (Dark) / `#F8FAFC` (Light). Provides the infinite foundation for mapping layers and full-screen camera feeds.
- **Structural Panels (Surface 1)**: `#111827` (Dark) / `#FFFFFF` (Light). 1px continuous border of `#1E293B` (Dark) or `#E2E8F0` (Light). Used for static sidebars, toolbars, and top navigation.
- **Interactive Cards & Modals (Surface 2)**: `#1E293B` (Dark) / `#FFFFFF` (Light). Supported by a subtle technical shadow: `0 4px 12px -2px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.06)` in dark mode, and `0 4px 12px -2px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(15, 23, 42, 0.05)` in light mode.
- **Overlays, HUD, & Map Pins (Surface 3)**: Semitransparent slate (`rgba(17, 24, 39, 0.85)` in dark mode) paired with `backdrop-filter: blur(8px)` and high-contrast alert borders. Real-time alert toasts float above all views with a distinct directional amber or red left accent border (3px).

## Shapes

The design system uses a restrained corner radius model (`roundedness: 1`) to preserve an engineered, instrumentation-grade feel.

- **Controls and Inputs (`rounded`, 4px)**: Buttons, text fields, dropdown triggers, and incident log table rows use tight 4px radiuses to maximize layout compactness and alignment with data grids.
- **Cards and Panels (`rounded-md`, 6px to 8px)**: Incident detail modals, camera stream wrappers, and alert drawers use 8px (`rounded-lg`) to soften focal boundaries without feeling consumer-oriented.
- **Telemetry Chips and Badges (`rounded-full`, 9999px)**: Status badges (e.g., `LIVE`, `DISPATCHED`) and filter pills utilize pill shapes to clearly contrast against rectangular data windows and video grids.

## Components

### Buttons
- **Primary**: Solid Cobalt Blue background (`#2563EB`), white text, 4px radius, 36px standard height (`space-md` horizontal padding). Hover transitions to `#1D4ED8`. Focus outline: 2px offset in `#3B82F6`.
- **Secondary**: Surface-2 background (`#1E293B` Dark / `#F1F5F9` Light), 1px slate border (`#334155` / `#CBD5E1`), 4px radius.
- **Outline**: Transparent background with 1px border (`#334155` / `#E2E8F0`). Hover evokes `#1E293B20`.
- **Danger**: Solid `#DC2626` background for urgent actions (e.g., `Trigger Emergency Broadcast`, `Close Expressway Lanes`).
- **Ghost**: Zero border or fill; text color `#94A3B8`, brightening on hover to `#F8FAFC`.

### Severity Badges & Status Chips
- **P1 Critical**: Pill-shaped, high-contrast red badge (`bg-red-500/15`, text `#EF4444`, 1px border `red-500/30`). Paired with a pulsing 6px indicator dot.
- **P2 High**: Amber badge (`bg-amber-500/15`, text `#F59E0B`, 1px border `amber-500/30`).
- **P3 Moderate**: Cyan/Blue badge (`bg-sky-500/15`, text `#0EA5E9`, 1px border `sky-500/30`).
- **P4 Minor**: Neutral slate badge (`bg-slate-500/15`, text `#94A3B8`, 1px border `slate-500/30`).
- **Operational Status Chips**: `LIVE` (Emerald ping indicator dot), `DISPATCHED` (Cobalt Blue fill), `INVESTIGATING` (Amber fill), `RESOLVED` (Muted slate fill).

### High-Density Data Tables
- Horizontal borders only (`#1E293B` Dark / `#E2E8F0` Light). Zero vertical gridlines.
- Row height: Fixed 36px for condensed telemetry views; 44px for incident queues.
- Monospace figures for coordinates, timestamps (ISO 8601 formatted), license plates, and speed estimates.
- Hover state: Full-row background fill tint (`#1E293B50` Dark / `#F8FAFC` Light) with persistent action triggers appearing in the last column.

### Incident Cards & HUD Overlays
- Structurally grouped by: Header (Timestamp + Severity Badge), Body (Camera snapshot thumbnail, Highway Marker / GPS, AI Confidence score), Footer (Assigned Patrol Unit + Quick Action Triage).
- Map coordinates rendered in monospace micro-labels with quick-copy utility.

### Inputs & Filters
- Form controls: 36px height, dark slate fill (`#0F172A`), 1px structural border (`#334155`), clean focus ring (`#2563EB`).
- Multi-select filter pills: Pill-shaped with count indicators (e.g., `Cameras [42]`, `Congestion > 60%`).

### Real-time Alert Toasts & Empty States
- **Alert Toasts**: Fixed bottom-right positioning, high z-index (900), Surface 2 container with solid 3px colored alert spine on the left edge. Dismissible with countdown progress bar.
- **Empty States**: Technical wireframe icon, subdued slate instruction text, primary CTA to reset filter parameters or calibrate sensor zone.