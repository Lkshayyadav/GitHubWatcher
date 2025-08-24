import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Also attempt to load .env files relative to this file and the repo root
try:
    possible_env_paths = [Path(__file__).with_name(".env"), Path(__file__).parent.parent / ".env"]
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
except Exception:
    pass

import base64
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Try to import optional dependencies
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Caching will be disabled.")

try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    print("Warning: PyGithub not available. GitHub API integration will be limited.")

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: Google Generative AI not available. AI analysis will be disabled.")

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("Warning: Trafilatura not available. Web scraping will be limited.")

# Create FastAPI app instance with lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        print("Server has started. Visit: http://127.0.0.1:5000")
    except Exception:
        pass
    yield
    # Shutdown
    try:
        print("Server is shutting down...")
    except Exception:
        pass

app = FastAPI(
    title="GitHub Repository Checker", 
    description="A modern tool for checking GitHub repositories",
    lifespan=lifespan
)

# CORS and security headers
try:
    from fastapi.middleware.cors import CORSMiddleware
    cors_origins = os.getenv("CORS_ORIGINS", "*")
    allow_origins = [o.strip() for o in cors_origins.split(",")] if cors_origins else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
except Exception:
    pass

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    try:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' 'unsafe-inline' https: data:;"
        )
    except Exception:
        pass
    return response

# Initialize Redis client if available
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            decode_responses=True,
            socket_connect_timeout=5,  # 5 second timeout
            socket_timeout=5,
        )
        # Test connection
        redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        print("Continuing without Redis - caching will be disabled")
        redis_client = None

# Initialize GitHub client if available
github_client = None
if GITHUB_AVAILABLE:
    github_token = os.getenv("GITHUB_API_KEY")
    if github_token:
        masked = f"****{github_token[-4:]}" if len(github_token) >= 4 else "(set)"
        print(f"GITHUB_API_KEY detected: {masked}")
    if github_token:
        try:
            github_client = Github(github_token)
            print("GitHub client initialized")
        except Exception as e:
            print(f"GitHub client initialization failed: {e}")
    else:
        print("Warning: GITHUB_API_KEY not found in environment")

# Initialize Gemini client if available
if GENAI_AVAILABLE:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            print("Gemini AI configured")
        except Exception as e:
            print(f"Gemini AI configuration failed: {e}")
    else:
        print("Warning: GEMINI_API_KEY not found in environment")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Configuration flags
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"


# Helper functions
def generate_cache_key(url: str) -> str:
    """Generate a cache key for a GitHub URL"""
    return f"repo_analysis:{hashlib.md5(url.encode()).hexdigest()}"


def get_cached_analysis(url: str) -> Optional[Dict[Any, Any]]:
    """Get cached analysis for a GitHub URL"""
    if not redis_client:
        return None

    try:
        cache_key = generate_cache_key(url)
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        print(f"Cache retrieval error: {e}")

    return None


def cache_analysis(url: str, analysis_data: Dict[Any, Any], ttl: int = 3600) -> bool:
    """Cache analysis data for a GitHub URL. Only cache successful results."""
    if not redis_client:
        return False

    # Do not cache failures to avoid sticky errors
    if analysis_data.get("status") != "success":
        return False

    try:
        cache_key = generate_cache_key(url)
        redis_client.setex(cache_key, ttl, json.dumps(analysis_data))
        return True
    except Exception as e:
        print(f"Cache storage error: {e}")
        return False


def extract_repo_info(url: str) -> tuple[str, str]:
    """Extract owner and repo name from GitHub URL"""
    # Clean URL patterns
    patterns = [r"https?://github\.com/([^/]+)/([^/]+)/?.*", r"github\.com/([^/]+)/([^/]+)/?.*"]

    for pattern in patterns:
        match = re.match(pattern, url.strip())
        if match:
            owner, repo_name = match.groups()
            # Remove .git suffix if present
            repo_name = repo_name.rstrip(".git")
            return owner, repo_name

    raise ValueError(f"Invalid GitHub repository URL: {url}")


def get_contributors_count(repo) -> int:
    """Get the number of contributors with proper error handling"""
    try:
        contributors = list(repo.get_contributors())
        return len(contributors)
    except Exception as e:
        print(f"Error fetching contributors: {e}")
        return 0


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if not dt:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_last_commit_date(repo) -> Optional[str]:
    """Get the last commit date with proper error handling"""
    try:
        commits = repo.get_commits()
        if commits.totalCount > 0:
            latest_commit = commits[0]
            dt = _to_utc(latest_commit.commit.author.date)
            return dt.isoformat() if dt else None
    except Exception as e:
        print(f"Error fetching last commit: {e}")
    return None


def get_commit_frequency(repo) -> Dict[str, Any]:
    """Get commit frequency and recent activity patterns"""
    try:
        commits = repo.get_commits()
        if commits.totalCount == 0:
            return {"total_commits": 0, "recent_activity": "none", "avg_commits_per_week": 0}

        # Get recent commits (last 30 days)
        recent_commits = []
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)

        for commit in commits[:100]:  # Check last 100 commits
            commit_dt = _to_utc(commit.commit.author.date)
            if commit_dt and commit_dt > thirty_days_ago:
                recent_commits.append(commit)
            else:
                break

        # Calculate activity level
        if len(recent_commits) >= 20:
            activity_level = "very_active"
        elif len(recent_commits) >= 10:
            activity_level = "active"
        elif len(recent_commits) >= 5:
            activity_level = "moderate"
        elif len(recent_commits) >= 1:
            activity_level = "low"
        else:
            activity_level = "inactive"

        # Calculate weekly average
        created = _to_utc(getattr(repo, "created_at", None))
        now_utc = datetime.now(timezone.utc)
        weeks_since_creation = max(1, (now_utc - created).days // 7) if created else 1
        avg_commits_per_week = commits.totalCount / weeks_since_creation

        return {
            "total_commits": commits.totalCount,
            "recent_commits": len(recent_commits),
            "recent_activity": activity_level,
            "avg_commits_per_week": round(avg_commits_per_week, 1),
            "last_commit_days_ago": (
                (now_utc - _to_utc(commits[0].commit.author.date)).days if commits[0] else None
            ),
        }
    except Exception as e:
        print(f"Error calculating commit frequency: {e}")
        return {"total_commits": 0, "recent_activity": "unknown", "avg_commits_per_week": 0}


def get_issue_metrics(repo) -> Dict[str, Any]:
    """Get issue and PR metrics"""
    try:
        issues = repo.get_issues(state="open")
        pulls = repo.get_pulls(state="open")

        # Calculate response time (average time to first response)
        response_times = []
        for issue in issues[:20]:  # Sample recent issues
            if issue.comments > 0:
                comments = issue.get_comments()
                if comments.totalCount > 0:
                    first_response = _to_utc(comments[0].created_at)
                    created_at = _to_utc(issue.created_at)
                    response_time = (
                        (first_response - created_at).days if first_response and created_at else 0
                    )
                    response_times.append(response_time)

        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        # Median time to close for recent closed issues
        close_times = []
        try:
            closed_issues = repo.get_issues(state="closed")
            for issue in closed_issues[:50]:
                if issue.closed_at and issue.created_at:
                    closed = _to_utc(issue.closed_at)
                    created = _to_utc(issue.created_at)
                    if closed and created:
                        close_times.append((closed - created).days)
        except Exception:
            pass
        median_close_time = None
        if close_times:
            s = sorted(close_times)
            mid = len(s) // 2
            median_close_time = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2

        return {
            "open_issues": issues.totalCount,
            "open_pulls": pulls.totalCount,
            "avg_response_time_days": round(avg_response_time, 1) if avg_response_time else None,
            "median_close_time_days": (
                round(median_close_time, 1) if median_close_time is not None else None
            ),
            "issue_health": (
                "good" if avg_response_time and avg_response_time < 7 else "needs_attention"
            ),
        }
    except Exception as e:
        print(f"Error fetching issue metrics: {e}")
        return {
            "open_issues": 0,
            "open_pulls": 0,
            "avg_response_time_days": None,
            "issue_health": "unknown",
        }


def calculate_quality_score(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall quality scores for different aspects"""
    scores = {"documentation": 0, "activity": 0, "community": 0, "overall": 0}

    # Documentation score (0-100)
    doc_score = 0
    if repo_data.get("code_quality", {}).get("has_readme"):
        doc_score += 30
    if repo_data.get("code_quality", {}).get("readme_size", 0) > 1000:
        doc_score += 20
    if repo_data.get("basic_info", {}).get("description"):
        doc_score += 20
    if repo_data.get("code_quality", {}).get("topics"):
        doc_score += 15
    if repo_data.get("community_health", {}).get("has_wiki"):
        doc_score += 15
    scores["documentation"] = min(100, doc_score)

    # Activity score (0-100)
    activity_score = 0
    commit_freq = repo_data.get("activity_metrics", {}).get("commit_frequency", {})
    if commit_freq.get("recent_activity") == "very_active":
        activity_score += 40
    elif commit_freq.get("recent_activity") == "active":
        activity_score += 30
    elif commit_freq.get("recent_activity") == "moderate":
        activity_score += 20
    elif commit_freq.get("recent_activity") == "low":
        activity_score += 10

    stars = repo_data.get("activity_metrics", {}).get("stars", 0)
    if stars > 1000:
        activity_score += 30
    elif stars > 100:
        activity_score += 20
    elif stars > 10:
        activity_score += 10

    if commit_freq.get("avg_commits_per_week", 0) > 5:
        activity_score += 30
    elif commit_freq.get("avg_commits_per_week", 0) > 1:
        activity_score += 20

    scores["activity"] = min(100, activity_score)

    # Community score (0-100)
    community_score = 0
    contributors = repo_data.get("activity_metrics", {}).get("contributors_count", 0)
    if contributors > 10:
        community_score += 30
    elif contributors > 5:
        community_score += 20
    elif contributors > 1:
        community_score += 10

    forks = repo_data.get("activity_metrics", {}).get("forks", 0)
    if forks > 100:
        community_score += 25
    elif forks > 10:
        community_score += 15
    elif forks > 1:
        community_score += 10

    if repo_data.get("community_health", {}).get("has_issues"):
        community_score += 15
    if repo_data.get("community_health", {}).get("license"):
        community_score += 20

    scores["community"] = min(100, community_score)

    # Overall score (weighted average)
    scores["overall"] = round(
        (scores["documentation"] * 0.3 + scores["activity"] * 0.4 + scores["community"] * 0.3), 1
    )

    return scores


def scrape_readme_content(repo_url: str, owner: str, repo_name: str) -> Dict[str, Any]:
    """Scrape README and About content from repository"""
    scraped_data = {
        "readme_content": None,
        "readme_length": 0,
        "full_readme_content": None,  # Store full content for AI analysis
        "about_content": None,
        "scraping_status": "not_attempted",
    }

    try:
        if TRAFILATURA_AVAILABLE:
            # Use trafilatura to extract content
            readme_url = f"{repo_url}/blob/main/README.md"
            downloaded = trafilatura.fetch_url(readme_url)
            if downloaded:
                readme_content = trafilatura.extract(downloaded)
                if readme_content:
                    scraped_data["full_readme_content"] = readme_content  # Full content for AI
                    scraped_data["readme_content"] = readme_content[:2000]  # Truncated for display
                    scraped_data["readme_length"] = len(readme_content)

            # Try alternative README locations
            if not scraped_data["readme_content"]:
                alt_urls = [
                    f"{repo_url}/blob/master/README.md",
                    f"{repo_url}/blob/main/readme.md",
                    f"{repo_url}/blob/master/readme.md",
                ]
                for alt_url in alt_urls:
                    downloaded = trafilatura.fetch_url(alt_url)
                    if downloaded:
                        readme_content = trafilatura.extract(downloaded)
                        if readme_content:
                            scraped_data["full_readme_content"] = readme_content
                            scraped_data["readme_content"] = readme_content[:2000]
                            scraped_data["readme_length"] = len(readme_content)
                            break

            scraped_data["scraping_status"] = (
                "success" if scraped_data["readme_content"] else "no_content_found"
            )

        else:
            # Fallback using httpx for basic content extraction
            with httpx.Client(timeout=10.0) as client:
                readme_url = f"https://raw.githubusercontent.com/{owner}/{repo_name}/main/README.md"
                try:
                    response = client.get(readme_url)
                    if response.status_code == 200:
                        full_content = response.text
                        scraped_data["full_readme_content"] = full_content
                        scraped_data["readme_content"] = full_content[:2000]
                        scraped_data["readme_length"] = len(full_content)
                        scraped_data["scraping_status"] = "success"
                except:
                    # Try master branch
                    master_url = (
                        f"https://raw.githubusercontent.com/{owner}/{repo_name}/master/README.md"
                    )
                    try:
                        response = client.get(master_url)
                        if response.status_code == 200:
                            full_content = response.text
                            scraped_data["full_readme_content"] = full_content
                            scraped_data["readme_content"] = full_content[:2000]
                            scraped_data["readme_length"] = len(full_content)
                            scraped_data["scraping_status"] = "success"
                    except Exception as e:
                        scraped_data["scraping_status"] = f"error: {str(e)}"

    except Exception as e:
        scraped_data["scraping_status"] = f"error: {str(e)}"
        print(f"Web scraping error: {e}")

    return scraped_data


def analyze_with_gemini(readme_content: str, repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze repository using Google Gemini AI"""
    ai_analysis = {
        "purpose": None,
        "features": [],
        "tech_stack": {},
        "status": "unknown",
        "readme_summary": None,
        "analysis_status": "not_attempted",
    }

    if not GENAI_AVAILABLE or not os.getenv("GEMINI_API_KEY"):
        ai_analysis["analysis_status"] = "gemini_not_available"
        return ai_analysis

    try:
        # Prepare context for AI analysis
        context_info = {
            "repo_name": repo_data.get("basic_info", {}).get("name", ""),
            "description": repo_data.get("basic_info", {}).get("description", ""),
            "language": repo_data.get("basic_info", {}).get("language", ""),
            "topics": repo_data.get("code_quality", {}).get("topics", []),
            "languages": repo_data.get("code_quality", {}).get("languages", {}),
            "stars": repo_data.get("activity_metrics", {}).get("stars", 0),
            "last_updated": repo_data.get("basic_info", {}).get("updated_at", ""),
            "archived": repo_data.get("community_health", {}).get("archived", False),
        }

        # Create comprehensive analysis prompt
        prompt = f"""Analyze this GitHub repository and provide a structured JSON response with exactly these fields:

Repository Context:
- Name: {context_info["repo_name"]}
- Description: {context_info["description"]}
- Primary Language: {context_info["language"]}
- Stars: {context_info["stars"]}
- Topics: {", ".join(context_info["topics"]) if context_info["topics"] else "None"}
- Archived: {context_info["archived"]}

README Content:
{readme_content[:8000]}  

Please analyze the repository and return a JSON object with exactly these fields:
- "purpose": A concise 1-2 sentence summary of what this project does/achieves
- "features": An array of 3-6 key features or capabilities (strings)
- "tech_stack": An object categorizing technologies like {{"Frontend": "React, CSS", "Backend": "Python, FastAPI", "Database": "PostgreSQL"}}
- "status": One of: "Active", "Archived", "Inactive", "Experimental", "Mature"
- "readme_summary": A 2-3 sentence summary explaining what this README tells us about the project

Important: Return ONLY valid JSON, no additional text or explanation."""

        # Configure Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1000,
            ),
        )

        if response.text:
            # Try to parse JSON response (handle markdown code blocks)
            try:
                response_text = response.text.strip()

                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith("```"):
                    response_text = response_text[3:]  # Remove ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove trailing ```

                response_text = response_text.strip()

                ai_data = json.loads(response_text)

                # Validate expected fields
                required_fields = ["purpose", "features", "tech_stack", "status", "readme_summary"]
                if all(field in ai_data for field in required_fields):
                    ai_analysis.update(ai_data)
                    ai_analysis["analysis_status"] = "success"
                else:
                    ai_analysis["analysis_status"] = "invalid_format"
                    ai_analysis["raw_response"] = response.text[:500]

            except json.JSONDecodeError as json_err:
                ai_analysis["analysis_status"] = f"json_parse_error: {str(json_err)}"
                ai_analysis["raw_response"] = response.text[:500]
        else:
            ai_analysis["analysis_status"] = "no_response"

    except Exception as e:
        ai_analysis["analysis_status"] = f"error: {str(e)}"
        print(f"Gemini AI analysis error: {e}")

    return ai_analysis
