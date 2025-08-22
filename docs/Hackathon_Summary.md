# GitHubWatcher – AI‑Assisted GitHub Repository Analyzer (One‑Pager)

## What It Is
Analyze any GitHub repository and instantly see activity, documentation quality, security signals, AI insights, and more. Includes a compare view, exports, and demo presets.

## Why It Matters
- Choose better dependencies faster
- Communicate health/risks clearly to non‑maintainers
- Great demo value: visual, fast, and AI‑enhanced

## Live Demo Flow (60–90s)
1. Open `/results` and click a preset (e.g., `psf/requests`).
2. Show repo overview and animated quality scores.
3. README preview + Gemini summary (purpose, features, stack, status).
4. New metrics: Releases cadence, PR merge time, issue response/close; CI/docs badges.
5. Portia badge + insights (if enabled). Explain fallback if inactive.
6. Export PNG/PDF and copy share link.
7. Optionally use `/compare` to show side‑by‑side repos.
8. `/stack` explains architecture/choices.

## Key Features
- Repository overview, language chart, topics
- README analysis (scraped), AI summary via Gemini
- Portia AI code quality + review (when key + network available)
- Health dashboard with documentation/activity/community/overall scores
- Releases/PR/Issue flow metrics; CI/Docs detection
- Compare two repos; export PNG/PDF; share link
- Optional Redis caching; rate‑limit aware fallbacks

## Architecture
- Backend: FastAPI (`main.py`)
- Templates/UI: Jinja2 + Tailwind (`templates/*.html`) + tiny JS
- GitHub API: PyGithub
- Scraping: httpx + trafilatura
- AI: google‑generativeai (Gemini), Portia API
- Caching: Redis (optional)
- Charts: Chart.js
- Security/Infra: CORS, CSP headers, health check, feature flags

## Notable Endpoints
- Pages: `/`, `/results`, `/compare`, `/stack`, `/about`
- API: `/api/analyze?url=<repo>`, `/api/cache/clear`, `/healthz`

## Configuration (env)
```
GITHUB_API_KEY=<optional>
GEMINI_API_KEY=<optional>
PORTIA_API_KEY=<recommended>
PORTIA_BASE_URL=https://api.portialabs.ai
ENABLE_PORTIA=true|false
ENABLE_GEMINI=true|false
REDIS_HOST, REDIS_PORT, REDIS_DB
CORS_ORIGINS
PORT
```

## Judge Talking Points
- Dual‑AI design: Gemini (docs understanding) + Portia (code quality)
- Graceful fallbacks: works even when AI is unavailable
- Real signals: releases cadence, PR/issue flow, CI/Docs hygiene
- Developer‑friendly UX: presets, compare, exports, share links
- Production‑minded: health checks, security headers, PORT‑aware run, auto‑deploy from GitHub

## Roadmap (if asked)
- Star growth timeline + bus factor visualization
- PDF export with branding
- Batch analysis and historical tracking
- Serverless friendly build (optional)
