# Ĥolmes: brand kit and implementation notes

Ĥolmes is the project's name (formerly CircuitSleuth). It reconstructs the Hamiltonian of a
quantum system (superconducting circuits and other systems) from spectroscopy data using
machine learning. The name is a pun: in physics notation, the "hat" on Ĥ marks the
Hamiltonian operator, and the logo replaces that hat with a detective's fedora.

**Task for Claude Code:** rebrand the website to Ĥolmes using the files in this folder.
Replace every user-facing "CircuitSleuth" with "Ĥolmes", install the favicon and app icons,
put the logo lockup in the site header, and add the social-preview meta tags. The
checklist is at the bottom.

---

## Name

| Use | Write it as |
|---|---|
| Display / UI text | **Ĥolmes** (capital H with circumflex, U+0124; HTML entity `&Hcirc;`) |
| Plain-ASCII contexts (URLs, repo or package names, file names, env vars, code identifiers) | `holmes` |
| Screen-reader / alt text | "Ĥolmes" (or "Holmes") |
| Tagline (optional) | *Hamiltonians, deduced from spectra.* |

Never put the Ĥ in URLs, slugs or identifiers.

## Files in this folder

| File | What it is | Where it goes |
|---|---|---|
| `favicon.svg` | Rounded navy icon, scalable favicon | site root / `public/` |
| `favicon.ico` | 16, 32 and 48 px fallback favicon | site root / `public/` |
| `favicon-32.png` | 32 px PNG favicon | `public/` |
| `apple-touch-icon.png` | 180 px, full-bleed square (iOS rounds the corners itself) | `public/` |
| `icon-192.png`, `icon-512.png` | Rounded app icons for the web manifest | `public/` |
| `icon-maskable-512.png` | Full-bleed square with safe padding, `"purpose": "maskable"` | `public/` |
| `holmes-icon.svg` | The app icon (navy rounded square, white H, amber fedora) | anywhere an icon/avatar is needed |
| `holmes-mark.svg` | Logo mark only, transparent background, **navy H**: for light backgrounds | header/footer on light themes |
| `holmes-mark-reverse.svg` | Logo mark only, transparent background, **white H**: for dark backgrounds | header/footer on dark themes |
| `og-image.png` | 1200×630 social-preview card | `public/`, referenced by Open Graph / Twitter tags |
| `lockup-preview-light.png`, `lockup-preview-dark.png` | Reference images of the header lockup: **for reference only, don't ship** | n/a |

Copy the shipping files into the framework's static-asset folder (`public/` for
Next.js, Vite and most React setups; the site root for plain HTML).

## Colors

| Token | Hex | Use |
|---|---|---|
| `--holmes-navy` | `#12223D` | Primary brand color, icon background, wordmark on light |
| `--holmes-amber` | `#D99A1E` | The fedora; accents, links on dark, highlights |
| `--holmes-amber-dark` | `#8A5A0C` | Hat band (inside the logo only) |
| `--holmes-white` | `#FFFFFF` | The H on dark, text on navy |

Amber on white fails text contrast, so don't use amber for body text on light backgrounds.
On navy, amber and white both pass.

## Typography

- **Wordmark:** *Playfair Display*, weight 700 (Google Fonts, free).
  `<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap" rel="stylesheet">`
  The font **must** include the Latin Extended subset, because that's where Ĥ lives. The
  Google Fonts link above serves it automatically. If the font is self-hosted, include
  the `latin-ext` files. Verified: Playfair Display has U+0124. **DM Serif Display does
  not**, so don't use it for the wordmark: the browser would render the Ĥ in a different
  fallback font.
- Body and UI text can stay whatever the site already uses.

## Header lockup

Mark on the left, wordmark on the right, vertically centered (see `lockup-preview-*.png`).

```html
<a href="/" class="brand" aria-label="Ĥolmes home">
  <img src="/holmes-mark.svg" alt="" width="40" height="37">
  <span class="brand-name">Ĥolmes</span>
</a>
```

```css
.brand { display: inline-flex; align-items: center; gap: 0.6em; text-decoration: none; color: #12223D; }
.brand img { height: 1.6em; width: auto; }
.brand-name { font-family: "Playfair Display", Georgia, serif; font-weight: 700; font-size: 1.6rem; line-height: 1; }
/* On a dark header use /holmes-mark-reverse.svg and color: #FFFFFF */
```

Keep the mark's proportions: set only height (or only width). Don't recolor, rotate, stretch
or add effects to it.

## `<head>` tags

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<meta name="theme-color" content="#12223D">

<title>Ĥolmes: Hamiltonians, deduced from spectra</title>
<meta name="description" content="Ĥolmes reconstructs the Hamiltonian of a quantum system from its spectroscopy data using machine learning.">

<meta property="og:title" content="Ĥolmes">
<meta property="og:description" content="Hamiltonians, deduced from spectra.">
<meta property="og:image" content="/og-image.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="/og-image.png">
```

Once the site has a deployed domain, change `og:image` and `twitter:image` to absolute URLs
(`https://<domain>/og-image.png`); some link previewers ignore relative paths.
For Next.js App Router, put these in `metadata` in `app/layout.tsx` instead of raw tags.

`site.webmanifest`:

```json
{
  "name": "Ĥolmes",
  "short_name": "Ĥolmes",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "theme_color": "#12223D",
  "background_color": "#12223D",
  "display": "standalone"
}
```

## Inline icon (in case the SVG file is missing)

```svg
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200" role="img" aria-label="Ĥolmes logo">
  <rect width="200" height="200" rx="44" fill="#12223D"/>
  <g transform="translate(16 16) scale(0.84)">
    <g transform="translate(100 100) scale(1.16) translate(-100 -115)">
      <path d="M24 92 C56 80 144 80 176 92 C154 103 46 103 24 92 Z" fill="#D99A1E"/>
      <path d="M64 88 C62 60 70 44 86 46 C94 47 98 52 100 55 C102 52 106 47 114 46 C130 44 138 60 136 88 Z" fill="#D99A1E"/>
      <path d="M63 76 L137 76 L136 88 L64 88 Z" fill="#8A5A0C"/>
      <path d="M100 55 L100 68" stroke="#8A5A0C" stroke-width="3" stroke-linecap="round" fill="none"/>
      <path d="M54 100 H78 V132 H122 V100 H146 V186 H122 V154 H78 V186 H54 Z" fill="#FFFFFF"/>
    </g>
  </g>
</svg>
```

The H is a single path on purpose. Don't rebuild it from separate rectangles, because
anti-aliasing then draws faint seams where the pieces meet.

## Checklist

- [ ] Copy the shipping files into `public/` (or the site root).
- [ ] Replace user-facing "CircuitSleuth" text with "Ĥolmes" (leave code identifiers, or rename them to `holmes` if that's simple and safe).
- [ ] Load Playfair Display 700 (with Latin Extended) and add the header lockup.
- [ ] Add the favicon, apple-touch-icon, manifest and theme-color tags.
- [ ] Add the title, description and Open Graph / Twitter tags.
- [ ] Check in the browser: the favicon shows in the tab, and the Ĥ in the header renders in Playfair Display (the circumflex matches the letter's style, not a fallback font).
- [ ] Commit with a message like "Rebrand to Ĥolmes: logo, favicons, social card".
