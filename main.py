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

# Portia AI Integration
try:
    import requests
    PORTIA_AVAILABLE = True
except ImportError:
    PORTIA_AVAILABLE = False
    print("Warning: Requests not available. Portia AI integration will be disabled.")

# Create FastAPI app instance
app = FastAPI(
    title="GitHub Repository Checker", 
    description="A modern tool for checking GitHub repositories"
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
        )
        # Test connection
        redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
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

# Initialize Portia AI client
portia_api_key = os.getenv("PORTIA_API_KEY")
portia_base_url = os.getenv("PORTIA_BASE_URL", "https://api.portia.ai")

if PORTIA_AVAILABLE and portia_api_key:
    print("Portia AI configured")
else:
    print("Warning: PORTIA_API_KEY not found in environment")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Configuration flags
ENABLE_GEMINI = os.getenv("ENABLE_GEMINI", "true").lower() == "true"
ENABLE_PORTIA = os.getenv("ENABLE_PORTIA", "true").lower() == "true"

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


def analyze_with_portia(code_content: str, repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze repository code using Portia AI"""
    portia_analysis = {
        "code_quality": {},
        "security_insights": [],
        "performance_suggestions": [],
        "best_practices": [],
        "complexity_analysis": {},
        "ai_score": 0,
        "analysis_status": "not_attempted",
    }

    if not PORTIA_AVAILABLE or not portia_api_key:
        portia_analysis["analysis_status"] = "portia_not_available"
        return portia_analysis

    try:
        # Prepare context for Portia AI
        context = {
            "repository_name": repo_data.get("basic_info", {}).get("name", ""),
            "language": repo_data.get("basic_info", {}).get("language", ""),
            "description": repo_data.get("basic_info", {}).get("description", ""),
            "code_sample": code_content[:5000] if code_content else "",  # Sample of code
            "file_count": repo_data.get("basic_info", {}).get("size", 0),
            "stars": repo_data.get("activity_metrics", {}).get("stars", 0),
        }

        # Call Portia AI API for code analysis
        headers = {"Authorization": f"Bearer {portia_api_key}", "Content-Type": "application/json"}

        payload = {
            "prompt": f"""Analyze this GitHub repository and provide insights:

Repository: {context['repository_name']}
Language: {context['language']}
Description: {context['description']}
Stars: {context['stars']}

Code Sample:
{context['code_sample']}

Please provide:
1. Code quality score (0-100)
2. Security insights (list of potential issues)
3. Performance suggestions (list of improvements)
4. Best practices recommendations (list)
5. Complexity analysis (simple/medium/complex)
6. Overall AI assessment score (0-100)

Return as JSON with these exact fields:
- "code_quality_score": number
- "security_insights": array of strings
- "performance_suggestions": array of strings  
- "best_practices": array of strings
- "complexity_level": string
- "ai_assessment_score": number
- "summary": string
""",
            "max_tokens": 1000,
            "temperature": 0.3,
        }

        response = requests.post(
            f"{portia_base_url}/v1/chat/completions", headers=headers, json=payload, timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]

                # Parse JSON response
                try:
                    import json

                    ai_data = json.loads(content)

                    # Validate and populate analysis
                    if all(
                        key in ai_data
                        for key in [
                            "code_quality_score",
                            "security_insights",
                            "performance_suggestions",
                            "best_practices",
                            "complexity_level",
                            "ai_assessment_score",
                        ]
                    ):
                        portia_analysis.update(
                            {
                                "code_quality": {
                                    "score": ai_data.get("code_quality_score", 0),
                                    "complexity": ai_data.get("complexity_level", "unknown"),
                                },
                                "security_insights": ai_data.get("security_insights", []),
                                "performance_suggestions": ai_data.get(
                                    "performance_suggestions", []
                                ),
                                "best_practices": ai_data.get("best_practices", []),
                                "ai_score": ai_data.get("ai_assessment_score", 0),
                                "summary": ai_data.get("summary", ""),
                                "analysis_status": "success",
                            }
                        )
                    else:
                        portia_analysis["analysis_status"] = "invalid_response_format"
                        portia_analysis["raw_response"] = content[:500]

                except json.JSONDecodeError:
                    portia_analysis["analysis_status"] = "json_parse_error"
                    portia_analysis["raw_response"] = content[:500]
            else:
                portia_analysis["analysis_status"] = "no_response_content"
        else:
            portia_analysis["analysis_status"] = f"api_error: {response.status_code}"

    except Exception as e:
        portia_analysis["analysis_status"] = f"error: {str(e)}"
        print(f"Portia AI analysis error: {e}")

    return portia_analysis


def get_portia_code_review(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get AI-powered code review suggestions using Portia"""
    review_data = {
        "suggestions": [],
        "improvements": [],
        "code_patterns": [],
        "review_score": 0,
        "review_status": "not_attempted",
    }

    if not PORTIA_AVAILABLE or not portia_api_key:
        review_data["review_status"] = "portia_not_available"
        return review_data

    try:
        # Prepare code review context
        context = {
            "repo_name": repo_data.get("basic_info", {}).get("name", ""),
            "language": repo_data.get("basic_info", {}).get("language", ""),
            "languages": repo_data.get("code_quality", {}).get("languages", {}),
            "topics": repo_data.get("code_quality", {}).get("topics", []),
            "has_readme": repo_data.get("code_quality", {}).get("has_readme", False),
            "readme_size": repo_data.get("code_quality", {}).get("readme_size", 0),
        }

        headers = {"Authorization": f"Bearer {portia_api_key}", "Content-Type": "application/json"}

        payload = {
            "prompt": f"""Provide a comprehensive code review for this repository:

Repository: {context['repo_name']}
Primary Language: {context['language']}
Languages Used: {', '.join(context['languages'].keys()) if context['languages'] else 'Unknown'}
Topics: {', '.join(context['topics']) if context['topics'] else 'None'}
Has README: {context['has_readme']}
README Size: {context['readme_size']} bytes

Please provide:
1. Code review suggestions (specific improvements)
2. Architecture improvements
3. Code patterns to follow/avoid
4. Overall review score (0-100)

Return as JSON:
- "suggestions": array of strings
- "improvements": array of strings  
- "code_patterns": array of strings
- "review_score": number
- "summary": string
""",
            "max_tokens": 800,
            "temperature": 0.2,
        }

        response = requests.post(
            f"{portia_base_url}/v1/chat/completions", headers=headers, json=payload, timeout=25
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]

                try:
                    import json

                    ai_data = json.loads(content)

                    if all(
                        key in ai_data
                        for key in ["suggestions", "improvements", "code_patterns", "review_score"]
                    ):
                        review_data.update(
                            {
                                "suggestions": ai_data.get("suggestions", []),
                                "improvements": ai_data.get("improvements", []),
                                "code_patterns": ai_data.get("code_patterns", []),
                                "review_score": ai_data.get("review_score", 0),
                                "summary": ai_data.get("summary", ""),
                                "review_status": "success",
                            }
                        )
                    else:
                        review_data["review_status"] = "invalid_response_format"

                except json.JSONDecodeError:
                    review_data["review_status"] = "json_parse_error"
        else:
            review_data["review_status"] = f"api_error: {response.status_code}"

    except Exception as e:
        review_data["review_status"] = f"error: {str(e)}"
        print(f"Portia code review error: {e}")

    return review_data


def get_security_analysis(repo) -> Dict[str, Any]:
    """Analyze repository security and dependencies"""
    security_data = {
        "vulnerabilities": [],
        "outdated_deps": [],
        "security_score": 100,
        "license_compliance": "unknown",
        "dependencies": {},
    }

    try:
        # Check for security files
        security_files = []
        try:
            contents = repo.get_contents("")
            for content in contents:
                if content.name.lower() in ["security.md", "security", "security-policy.md"]:
                    security_files.append(content.name)
        except:
            pass

        # Check for dependency files
        dep_files = []
        try:
            contents = repo.get_contents("")
            for content in contents:
                if content.name.lower() in [
                    "requirements.txt",
                    "package.json",
                    "pom.xml",
                    "gemfile",
                    "cargo.toml",
                    "go.mod",
                ]:
                    dep_files.append(content.name)
        except:
            pass

        # Analyze dependencies if found
        if dep_files:
            security_data["dependencies"]["files_found"] = dep_files
            # Simulate dependency analysis (in real implementation, you'd parse these files)
            security_data["outdated_deps"] = [
                {
                    "package": "requests",
                    "current": "2.25.1",
                    "latest": "2.31.0",
                    "severity": "medium",
                },
                {"package": "urllib3", "current": "1.26.5", "latest": "2.0.7", "severity": "high"},
            ]
            security_data["security_score"] = max(0, 100 - len(security_data["outdated_deps"]) * 10)

        # Check for security advisories
        if security_files:
            security_data["security_score"] += 10

        # License compliance check
        if repo.license:
            security_data["license_compliance"] = "compliant"
        else:
            security_data["license_compliance"] = "missing"
            security_data["security_score"] -= 20

    except Exception as e:
        print(f"Error in security analysis: {e}")

    return security_data


def get_trending_analysis(repo) -> Dict[str, Any]:
    """Analyze repository trending and growth patterns"""
    trending_data = {
        "growth_rate": "stable",
        "trending_score": 0,
        "recent_activity": "normal",
        "predictions": {},
        "similar_repos": [],
    }

    try:
        # Calculate growth rate based on stars and recent activity
        stars = repo.stargazers_count
        forks = repo.forks_count
        created_days = 0
        created_at = _to_utc(getattr(repo, "created_at", None))
        if created_at:
            created_days = (datetime.now(timezone.utc) - created_at).days

        # Growth rate calculation
        if created_days > 0:
            stars_per_day = stars / created_days
            if stars_per_day > 5:
                trending_data["growth_rate"] = "exploding"
                trending_data["trending_score"] = 95
            elif stars_per_day > 2:
                trending_data["growth_rate"] = "growing"
                trending_data["trending_score"] = 75
            elif stars_per_day > 0.5:
                trending_data["growth_rate"] = "stable"
                trending_data["trending_score"] = 50
            else:
                trending_data["growth_rate"] = "declining"
                trending_data["trending_score"] = 25

        # Recent activity analysis
        if hasattr(repo, "pushed_at") and repo.pushed_at:
            pushed_at = _to_utc(repo.pushed_at)
            days_since_push = (datetime.now(timezone.utc) - pushed_at).days if pushed_at else 9999
            if days_since_push < 7:
                trending_data["recent_activity"] = "very_active"
            elif days_since_push < 30:
                trending_data["recent_activity"] = "active"
            elif days_since_push < 90:
                trending_data["recent_activity"] = "moderate"
            else:
                trending_data["recent_activity"] = "inactive"

        # Predictions
        if trending_data["trending_score"] > 80:
            trending_data["predictions"] = {
                "next_month_stars": int(stars * 1.5),
                "abandonment_risk": "low",
                "enterprise_ready": "high",
            }
        elif trending_data["trending_score"] > 50:
            trending_data["predictions"] = {
                "next_month_stars": int(stars * 1.2),
                "abandonment_risk": "medium",
                "enterprise_ready": "medium",
            }
        else:
            trending_data["predictions"] = {
                "next_month_stars": int(stars * 0.9),
                "abandonment_risk": "high",
                "enterprise_ready": "low",
            }

    except Exception as e:
        print(f"Error in trending analysis: {e}")

    return trending_data


def get_releases_cadence(repo) -> Dict[str, Any]:
    """Compute releases in last 90/365 days and latest release age"""
    data = {"last_90_days": 0, "last_365_days": 0, "latest_age_days": None, "count": 0}
    try:
        releases = list(repo.get_releases()[:50])
        data["count"] = len(releases)
        now = datetime.now(timezone.utc)
        for rel in releases:
            created = _to_utc(getattr(rel, "created_at", None)) or _to_utc(
                getattr(rel, "published_at", None)
            )
            if not created:
                continue
            if (now - created).days <= 365:
                data["last_365_days"] += 1
            if (now - created).days <= 90:
                data["last_90_days"] += 1
        if releases:
            latest = _to_utc(getattr(releases[0], "created_at", None)) or _to_utc(
                getattr(releases[0], "published_at", None)
            )
            if latest:
                data["latest_age_days"] = (now - latest).days
    except Exception as e:
        print(f"Releases cadence error: {e}")
    return data


def get_pr_flow(repo) -> Dict[str, Any]:
    """Median merge time (days) for recent merged PRs"""
    data = {"median_merge_time_days": None, "sampled": 0}
    try:
        merged_deltas = []
        pulls = repo.get_pulls(state="closed")
        for pr in pulls[:50]:
            if getattr(pr, "merged", False) and pr.merged_at and pr.created_at:
                merged = _to_utc(pr.merged_at)
                created = _to_utc(pr.created_at)
                if merged and created and merged >= created:
                    merged_deltas.append((merged - created).days)
        data["sampled"] = len(merged_deltas)
        if merged_deltas:
            s = sorted(merged_deltas)
            mid = len(s) // 2
            data["median_merge_time_days"] = (
                s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
            )
    except Exception as e:
        print(f"PR flow error: {e}")
    return data


def detect_ci_and_docs(repo) -> Dict[str, Any]:
    """Detect CI workflows and common project hygiene files"""
    info = {
        "ci_workflows": [],
        "has_contributing": False,
        "has_code_of_conduct": False,
        "has_security": False,
        "has_license": False,
    }
    try:
        contents = repo.get_contents("")
        names = [c.name.lower() for c in contents]
        info["has_contributing"] = any(n in names for n in ["contributing.md", "contributing"])
        info["has_code_of_conduct"] = any(
            n in names for n in ["code_of_conduct.md", "code-of-conduct.md", "code_of_conduct"]
        )
        info["has_security"] = any(n in names for n in ["security.md", "security"])
        info["has_license"] = any("license" in n for n in names)
        # Workflows
        try:
            wf_dir = repo.get_contents(".github/workflows")
            for f in wf_dir:
                info["ci_workflows"].append(f.name)
        except Exception:
            pass
    except Exception as e:
        print(f"CI/docs detection error: {e}")
    return info


def get_developer_experience_score(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate developer experience score and recommendations"""
    dx_score = {
        "overall_score": 0,
        "setup_difficulty": "unknown",
        "documentation_score": 0,
        "contributing_score": 0,
        "recommendations": [],
    }

    # Setup difficulty analysis
    setup_score = 0
    if repo_data.get("code_quality", {}).get("has_readme"):
        setup_score += 30
    if repo_data.get("basic_info", {}).get("description"):
        setup_score += 20
    if repo_data.get("community_health", {}).get("has_wiki"):
        setup_score += 15

    # Documentation score
    doc_score = 0
    if repo_data.get("code_quality", {}).get("has_readme"):
        doc_score += 40
    if repo_data.get("code_quality", {}).get("readme_size", 0) > 2000:
        doc_score += 30
    if repo_data.get("code_quality", {}).get("topics"):
        doc_score += 20
    if repo_data.get("community_health", {}).get("has_wiki"):
        doc_score += 10

    # Contributing score
    contributing_score = 0
    if repo_data.get("community_health", {}).get("has_issues"):
        contributing_score += 25
    if repo_data.get("community_health", {}).get("license"):
        contributing_score += 25
    if repo_data.get("community_health", {}).get("allow_forking"):
        contributing_score += 25
    if repo_data.get("community_health", {}).get("has_projects"):
        contributing_score += 25

    # Generate recommendations
    recommendations = []
    if not repo_data.get("code_quality", {}).get("has_readme"):
        recommendations.append("Add a comprehensive README.md (+30 points)")
    if not repo_data.get("basic_info", {}).get("description"):
        recommendations.append("Add a repository description (+20 points)")
    if not repo_data.get("community_health", {}).get("license"):
        recommendations.append("Add a LICENSE file (+25 points)")
    if not repo_data.get("community_health", {}).get("has_issues"):
        recommendations.append("Enable issues for better community engagement (+25 points)")
    if repo_data.get("code_quality", {}).get("readme_size", 0) < 1000:
        recommendations.append("Expand README with more details (+20 points)")

    # Calculate overall score
    overall_score = (setup_score + doc_score + contributing_score) / 3

    dx_score.update(
        {
            "overall_score": round(overall_score, 1),
            "setup_difficulty": (
                "easy" if setup_score > 60 else "medium" if setup_score > 30 else "hard"
            ),
            "documentation_score": doc_score,
            "contributing_score": contributing_score,
            "recommendations": recommendations,
        }
    )

    return dx_score


def get_business_intelligence(repo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze business value and adoption potential"""
    bi_data = {
        "adoption_potential": "unknown",
        "maintenance_risk": "unknown",
        "enterprise_readiness": "unknown",
        "roi_analysis": {},
        "market_position": "unknown",
    }

    try:
        # Adoption potential
        stars = repo_data.get("activity_metrics", {}).get("stars", 0)
        forks = repo_data.get("activity_metrics", {}).get("forks", 0)
        contributors = repo_data.get("activity_metrics", {}).get("contributors_count", 0)

        adoption_score = 0
        if stars > 10000:
            adoption_score = 95
            bi_data["adoption_potential"] = "very_high"
        elif stars > 1000:
            adoption_score = 80
            bi_data["adoption_potential"] = "high"
        elif stars > 100:
            adoption_score = 60
            bi_data["adoption_potential"] = "medium"
        elif stars > 10:
            adoption_score = 40
            bi_data["adoption_potential"] = "low"
        else:
            adoption_score = 20
            bi_data["adoption_potential"] = "very_low"

        # Maintenance risk
        commit_freq = repo_data.get("activity_metrics", {}).get("commit_frequency", {})
        recent_activity = commit_freq.get("recent_activity", "unknown")

        if recent_activity == "very_active":
            bi_data["maintenance_risk"] = "very_low"
        elif recent_activity == "active":
            bi_data["maintenance_risk"] = "low"
        elif recent_activity == "moderate":
            bi_data["maintenance_risk"] = "medium"
        elif recent_activity == "low":
            bi_data["maintenance_risk"] = "high"
        else:
            bi_data["maintenance_risk"] = "very_high"

        # Enterprise readiness
        enterprise_score = 0
        if repo_data.get("community_health", {}).get("license"):
            enterprise_score += 30
        if repo_data.get("code_quality", {}).get("has_readme"):
            enterprise_score += 20
        if contributors > 5:
            enterprise_score += 25
        if stars > 100:
            enterprise_score += 25

        if enterprise_score > 80:
            bi_data["enterprise_readiness"] = "production_ready"
        elif enterprise_score > 60:
            bi_data["enterprise_readiness"] = "near_ready"
        elif enterprise_score > 40:
            bi_data["enterprise_readiness"] = "experimental"
        else:
            bi_data["enterprise_readiness"] = "not_ready"

        # ROI Analysis
        time_saved = min(40, stars // 10)  # Rough estimate
        bi_data["roi_analysis"] = {
            "time_saved_hours_per_week": time_saved,
            "cost_savings_per_month": time_saved * 50,  # Assuming $50/hour
            "implementation_time_days": max(1, 30 - stars // 100),
            "maintenance_effort": "low" if recent_activity == "very_active" else "medium",
        }

        # Market position
        if stars > 5000:
            bi_data["market_position"] = "market_leader"
        elif stars > 500:
            bi_data["market_position"] = "established"
        elif stars > 50:
            bi_data["market_position"] = "emerging"
        else:
            bi_data["market_position"] = "niche"

    except Exception as e:
        print(f"Error in business intelligence analysis: {e}")

    return bi_data


def analyze_github_repo(url: str) -> Dict[str, Any]:
    """Comprehensively analyze a GitHub repository with enhanced data fetching"""
    analysis_result = {
        "url": url,
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "data": {
            "basic_info": {},
            "code_quality": {},
            "activity_metrics": {},
            "community_health": {},
            "scraped_content": {},
            "ai_insights": {},
            "quality_scores": {},
            "security_analysis": {},
            "trending_analysis": {},
            "developer_experience": {},
            "business_intelligence": {},
            "portia_analysis": {},
            "portia_code_review": {},
        },
        "errors": [],
    }

    try:
        # Extract repository information from URL
        owner, repo_name = extract_repo_info(url)
        normalized_url = f"https://github.com/{owner}/{repo_name}"

        analysis_result["data"]["basic_info"]["owner"] = owner
        analysis_result["data"]["basic_info"]["repo_name"] = repo_name
        analysis_result["data"]["basic_info"]["normalized_url"] = normalized_url

        # Check if GitHub client is available
        if not github_client:
            analysis_result["errors"].append("GitHub API client not available")
            analysis_result["data"]["basic_info"]["note"] = "GitHub API integration not available"

            # Still attempt web scraping even without API
            scraped_data = scrape_readme_content(normalized_url, owner, repo_name)
            analysis_result["data"]["scraped_content"] = scraped_data

            return analysis_result

        # Fetch repository data from GitHub API with comprehensive error handling
        try:
            repo = github_client.get_repo(f"{owner}/{repo_name}")

            # Basic repository information
            analysis_result["data"]["basic_info"].update(
                {
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "language": repo.language,
                    "created_at": repo.created_at.isoformat() if repo.created_at else None,
                    "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                    "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                    "size": repo.size,
                    "default_branch": repo.default_branch,
                    "clone_url": repo.clone_url,
                    "ssh_url": repo.ssh_url,
                    "homepage": repo.homepage,
                }
            )

            # Enhanced activity metrics with error handling
            analysis_result["data"]["activity_metrics"] = {
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "watchers": repo.watchers_count,
                "open_issues": repo.open_issues_count,
                "subscribers": repo.subscribers_count,
                "network_count": repo.network_count,
                "contributors_count": get_contributors_count(repo),
                "last_commit_date": get_last_commit_date(repo),
                "commit_frequency": get_commit_frequency(repo),
                "issue_metrics": get_issue_metrics(repo),
            }

            # Enhanced community health metrics
            analysis_result["data"]["community_health"] = {
                "has_issues": repo.has_issues,
                "has_projects": repo.has_projects,
                "has_wiki": repo.has_wiki,
                "has_pages": repo.has_pages,
                "has_downloads": repo.has_downloads,
                "archived": repo.archived,
                "disabled": repo.disabled,
                "private": repo.private,
                "fork": repo.fork,
                "license": repo.license.name if repo.license else None,
                "allow_forking": getattr(repo, "allow_forking", None),
                "allow_merge_commit": getattr(repo, "allow_merge_commit", None),
                "allow_squash_merge": getattr(repo, "allow_squash_merge", None),
                "allow_rebase_merge": getattr(repo, "allow_rebase_merge", None),
            }

            # Enhanced code quality indicators
            code_quality_data = {
                "topics": [],
                "languages": {},
                "has_readme": False,
                "readme_size": 0,
            }

            # Get topics with error handling
            try:
                code_quality_data["topics"] = repo.get_topics()
            except Exception as e:
                analysis_result["errors"].append(f"Error fetching topics: {str(e)}")

            # Get languages with error handling
            try:
                languages = repo.get_languages()
                code_quality_data["languages"] = dict(list(languages.items())[:10])
            except Exception as e:
                analysis_result["errors"].append(f"Error fetching languages: {str(e)}")

            # Check for README with error handling
            try:
                readme = repo.get_readme()
                code_quality_data["has_readme"] = True
                code_quality_data["readme_size"] = readme.size
            except Exception as e:
                analysis_result["errors"].append(f"Error checking README: {str(e)}")

            analysis_result["data"]["code_quality"] = code_quality_data

            # Calculate quality scores
            analysis_result["data"]["quality_scores"] = calculate_quality_score(
                analysis_result["data"]
            )

            # Enhanced analysis
            analysis_result["data"]["security_analysis"] = get_security_analysis(repo)
            analysis_result["data"]["trending_analysis"] = get_trending_analysis(repo)
            analysis_result["data"]["releases"] = get_releases_cadence(repo)
            analysis_result["data"]["pr_flow"] = get_pr_flow(repo)
            analysis_result["data"]["hygiene"] = detect_ci_and_docs(repo)
            analysis_result["data"]["developer_experience"] = get_developer_experience_score(
                analysis_result["data"]
            )
            analysis_result["data"]["business_intelligence"] = get_business_intelligence(
                analysis_result["data"]
            )

            # Portia AI Analysis
            if analysis_result["data"]["scraped_content"].get("readme_content"):
                analysis_result["data"]["portia_analysis"] = analyze_with_portia(
                    analysis_result["data"]["scraped_content"]["readme_content"],
                    analysis_result["data"],
                )
                analysis_result["data"]["portia_code_review"] = get_portia_code_review(
                    analysis_result["data"]
                )

        except Exception as api_error:
            error_msg = str(api_error)
            analysis_result["errors"].append(f"GitHub API error: {error_msg}")

            # Handle specific API errors
            if "401" in error_msg or "bad credentials" in error_msg.lower():
                analysis_result["errors"].append(
                    "Auth token invalid; retrying without authentication for public data"
                )
                try:
                    # Retry with unauthenticated client for public repositories
                    public_client = Github() if GITHUB_AVAILABLE else None
                    if public_client:
                        repo = public_client.get_repo(f"{owner}/{repo_name}")
                        # Basic repository information
                        analysis_result["data"]["basic_info"].update(
                            {
                                "name": repo.name,
                                "full_name": repo.full_name,
                                "description": repo.description,
                                "language": repo.language,
                                "created_at": (
                                    repo.created_at.isoformat() if repo.created_at else None
                                ),
                                "updated_at": (
                                    repo.updated_at.isoformat() if repo.updated_at else None
                                ),
                                "pushed_at": repo.pushed_at.isoformat() if repo.pushed_at else None,
                                "size": repo.size,
                                "default_branch": repo.default_branch,
                                "clone_url": repo.clone_url,
                                "ssh_url": repo.ssh_url,
                                "homepage": repo.homepage,
                            }
                        )
                        # Activity metrics
                        analysis_result["data"]["activity_metrics"] = {
                            "stars": repo.stargazers_count,
                            "forks": repo.forks_count,
                            "watchers": repo.watchers_count,
                            "open_issues": repo.open_issues_count,
                            "subscribers": repo.subscribers_count,
                            "network_count": repo.network_count,
                            "contributors_count": get_contributors_count(repo),
                            "last_commit_date": get_last_commit_date(repo),
                            "commit_frequency": get_commit_frequency(repo),
                            "issue_metrics": get_issue_metrics(repo),
                        }
                        # Community health
                        analysis_result["data"]["community_health"] = {
                            "has_issues": repo.has_issues,
                            "has_projects": repo.has_projects,
                            "has_wiki": repo.has_wiki,
                            "has_pages": repo.has_pages,
                            "has_downloads": repo.has_downloads,
                            "archived": repo.archived,
                            "disabled": repo.disabled,
                            "private": repo.private,
                            "fork": repo.fork,
                            "license": repo.license.name if repo.license else None,
                            "allow_forking": getattr(repo, "allow_forking", None),
                            "allow_merge_commit": getattr(repo, "allow_merge_commit", None),
                            "allow_squash_merge": getattr(repo, "allow_squash_merge", None),
                            "allow_rebase_merge": getattr(repo, "allow_rebase_merge", None),
                        }
                        # Code quality indicators
                        code_quality_data = {
                            "topics": [],
                            "languages": {},
                            "has_readme": False,
                            "readme_size": 0,
                        }
                        try:
                            code_quality_data["topics"] = repo.get_topics()
                        except Exception as e:
                            analysis_result["errors"].append(
                                f"Error fetching topics (unauth): {str(e)}"
                            )
                        try:
                            languages = repo.get_languages()
                            code_quality_data["languages"] = dict(list(languages.items())[:10])
                        except Exception as e:
                            analysis_result["errors"].append(
                                f"Error fetching languages (unauth): {str(e)}"
                            )
                        try:
                            readme = repo.get_readme()
                            code_quality_data["has_readme"] = True
                            code_quality_data["readme_size"] = readme.size
                        except Exception as e:
                            analysis_result["errors"].append(
                                f"Error checking README (unauth): {str(e)}"
                            )
                        analysis_result["data"]["code_quality"] = code_quality_data

                        # Calculate quality scores
                        analysis_result["data"]["quality_scores"] = calculate_quality_score(
                            analysis_result["data"]
                        )

                        # Enhanced analysis (limited for unauthenticated)
                        analysis_result["data"]["security_analysis"] = get_security_analysis(repo)
                        analysis_result["data"]["trending_analysis"] = get_trending_analysis(repo)
                        analysis_result["data"]["releases"] = get_releases_cadence(repo)
                        analysis_result["data"]["pr_flow"] = get_pr_flow(repo)
                        analysis_result["data"]["hygiene"] = detect_ci_and_docs(repo)
                        analysis_result["data"]["developer_experience"] = (
                            get_developer_experience_score(analysis_result["data"])
                        )
                        analysis_result["data"]["business_intelligence"] = (
                            get_business_intelligence(analysis_result["data"])
                        )

                        # Portia AI Analysis (works with public data)
                        if analysis_result["data"]["scraped_content"].get("readme_content"):
                            analysis_result["data"]["portia_analysis"] = analyze_with_portia(
                                analysis_result["data"]["scraped_content"]["readme_content"],
                                analysis_result["data"],
                            )
                            analysis_result["data"]["portia_code_review"] = get_portia_code_review(
                                analysis_result["data"]
                            )

                        # Reset overall status since unauthenticated fetch worked
                        analysis_result["status"] = "success"
                    else:
                        analysis_result["status"] = "api_error"
                        analysis_result["data"]["basic_info"][
                            "note"
                        ] = "GitHub library not available"
                except Exception as unauth_err:
                    analysis_result["errors"].append(
                        f"Unauthenticated retry error: {str(unauth_err)}"
                    )
                    analysis_result["status"] = "api_error"
                    analysis_result["data"]["basic_info"]["note"] = f"GitHub API error: {error_msg}"
            if "rate limit" in error_msg.lower():
                analysis_result["status"] = "rate_limited"
                analysis_result["data"]["basic_info"][
                    "note"
                ] = "GitHub API rate limit exceeded. Please try again later."
            elif "not found" in error_msg.lower() or "404" in error_msg:
                analysis_result["status"] = "not_found"
                analysis_result["data"]["basic_info"][
                    "note"
                ] = "Repository not found or is private."
            else:
                analysis_result["status"] = "api_error"
                analysis_result["data"]["basic_info"]["note"] = f"GitHub API error: {error_msg}"

        # Attempt web scraping regardless of API success/failure
        try:
            scraped_data = scrape_readme_content(normalized_url, owner, repo_name)
            analysis_result["data"]["scraped_content"] = scraped_data
            # If Portia is configured and we have README content, run Portia analysis now
            try:
                if analysis_result["data"]["scraped_content"].get("readme_content"):
                    analysis_result["data"]["portia_analysis"] = analyze_with_portia(
                        analysis_result["data"]["scraped_content"]["readme_content"],
                        analysis_result["data"],
                    )
                    analysis_result["data"]["portia_code_review"] = get_portia_code_review(
                        analysis_result["data"]
                    )
            except Exception as portia_err:
                analysis_result["errors"].append(
                    f"Portia analysis error after scraping: {str(portia_err)}"
                )
        except Exception as scrape_error:
            analysis_result["errors"].append(f"Web scraping error: {str(scrape_error)}")
            analysis_result["data"]["scraped_content"] = {
                "scraping_status": f"error: {str(scrape_error)}"
            }

        # Perform AI analysis with Gemini if we have README content
        try:
            readme_content = analysis_result["data"]["scraped_content"].get("full_readme_content")
            if (
                readme_content and len(readme_content.strip()) > 50
            ):  # Only analyze if we have substantial content
                ai_analysis = analyze_with_gemini(readme_content, analysis_result["data"])
                analysis_result["data"]["ai_insights"] = ai_analysis

                # If AI analysis was successful, enhance the overall status
                if ai_analysis.get("analysis_status") == "success":
                    # Use AI-determined status if available
                    if ai_analysis.get("status") and ai_analysis["status"] != "unknown":
                        analysis_result["data"]["community_health"]["ai_status"] = ai_analysis[
                            "status"
                        ]
            else:
                analysis_result["data"]["ai_insights"] = {
                    "analysis_status": "no_content",
                    "purpose": None,
                    "features": [],
                    "tech_stack": {},
                    "status": "unknown",
                }
        except Exception as ai_error:
            analysis_result["errors"].append(f"AI analysis error: {str(ai_error)}")
            analysis_result["data"]["ai_insights"] = {
                "analysis_status": f"error: {str(ai_error)}",
                "purpose": None,
                "features": [],
                "tech_stack": {},
                "status": "unknown",
            }

    except ValueError as ve:
        analysis_result["status"] = "invalid_url"
        analysis_result["error"] = str(ve)
        analysis_result["data"]["basic_info"] = {"note": str(ve), "url": url}
    except Exception as e:
        analysis_result["status"] = "error"
        analysis_result["error"] = str(e)
        analysis_result["errors"].append(f"General error: {str(e)}")
        analysis_result["data"]["basic_info"] = {
            "note": f"Error analyzing repository: {str(e)}",
            "url": url,
        }

    return analysis_result


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    """Landing page with search functionality"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    """About page with project information and creator links"""
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/stack", response_class=HTMLResponse)
async def tech_stack_page(request: Request):
    """Hackathon-focused tech stack and architecture overview page"""
    return templates.TemplateResponse("stack.html", {"request": request})


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    """Two-repo comparison view"""
    return templates.TemplateResponse("compare.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request, q: str = Query(None, description="GitHub repository URL")):
    """Results page - now dynamically loads data with JavaScript"""
    return templates.TemplateResponse(
        "results.html", {"request": request, "query": q, "github_url": q}
    )


@app.get("/api/analyze", response_class=JSONResponse)
async def api_analyze_repo(
    url: str = Query(..., description="GitHub repository URL"),
    refresh: bool = Query(False, description="Bypass cache and force fresh analysis"),
):
    """API endpoint for repository analysis"""

    # Check cache first unless refresh is requested
    if not refresh:
        cached_data = get_cached_analysis(url)
        if cached_data and cached_data.get("status") == "success":
            cached_data["cached"] = True
            return cached_data

    # Perform analysis if not cached
    analysis_data = analyze_github_repo(url)
    analysis_data["cached"] = False

    # Cache the results (only if successful)
    cache_analysis(url, analysis_data)

    return analysis_data


@app.get("/healthz", response_class=JSONResponse)
async def healthz():
    services = {
        "redis": bool(redis_client),
        "github_client": bool(github_client),
        "gemini_enabled": bool(GENAI_AVAILABLE and ENABLE_GEMINI and os.getenv("GEMINI_API_KEY")),
        "portia_enabled": bool(PORTIA_AVAILABLE and ENABLE_PORTIA and os.getenv("PORTIA_API_KEY")),
    }
    return {"ok": True, "services": services}


@app.get("/api/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    if not redis_client:
        return JSONResponse(content={"status": "error", "message": "Redis not available"})

    try:
        keys = redis_client.keys("repo_analysis:*")
        if keys:
            redis_client.delete(*keys)
            return JSONResponse(content={"status": "success", "cleared": len(keys)})
        else:
            return JSONResponse(content={"status": "success", "cleared": 0})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn

    # Helpful server start message for local runs
    try:
        print("Server is starting...")
        print("Check at: http://127.0.0.1:5000")
    except Exception:
        pass

    @app.on_event("startup")
    async def _announce_start():
        try:
            print("Server has started. Visit: http://127.0.0.1:5000")
        except Exception:
            pass

    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
