# 🚀 GitHubWatcher

> **Intelligent GitHub Repository Analysis & Health Monitoring**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116.1+-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 **The Problem We Solve**

Evaluating open-source projects is time-consuming and subjective. Developers spend hours manually reviewing repositories to assess:
- **Code Quality**: Is the code well-structured and maintainable?
- **Project Health**: Is the project actively maintained?
- **Documentation**: Is the README comprehensive and helpful?
- **Community**: How engaged is the development community?

**GitHubWatcher** eliminates this manual work by providing instant, data-driven insights into any GitHub repository.

## ✨ **Key Features**

### 🔍 **Comprehensive Repository Analysis**
- **Activity Metrics**: Commit frequency, recent updates, contributor activity
- **Code Quality Indicators**: Repository size, language distribution, file structure
- **Documentation Assessment**: README quality, wiki presence, issue templates
- **Community Health**: Star count, fork activity, open issues/PRs

### 🤖 **AI-Powered Insights**
- **Gemini AI Integration**: Advanced code analysis and project assessment
- **Portia AI Analysis**: Deep repository understanding and recommendations
- **Smart Summaries**: Automated generation of project health reports

### 📊 **Interactive Dashboard**
- **Real-time Data**: Live GitHub API integration with caching
- **Visual Analytics**: Clean, intuitive charts and metrics
- **Responsive Design**: Works seamlessly on desktop and mobile
- **Comparison Tools**: Side-by-side repository analysis

### ⚡ **Performance & Reliability**
- **Redis Caching**: Fast response times and reduced API calls
- **Rate Limit Management**: Intelligent GitHub API usage
- **Error Handling**: Graceful degradation when services are unavailable

## 🛠 **Technology Stack**

### **Backend**
- **FastAPI**: Modern, fast web framework for building APIs
- **Uvicorn**: Lightning-fast ASGI server
- **Redis**: High-performance caching layer

### **Frontend**
- **Jinja2**: Powerful templating engine
- **Tailwind CSS**: Utility-first CSS framework
- **JavaScript**: Interactive dashboard functionality

### **Integrations**
- **PyGithub**: Official GitHub API client
- **Google Generative AI**: Advanced AI analysis via Gemini
- **Portia AI**: Enterprise-grade repository intelligence
- **Trafilatura**: Web content extraction

### **Infrastructure**
- **Python 3.8+**: Modern Python with type hints
- **Docker Ready**: Containerized deployment
- **Environment Config**: Flexible configuration management

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- Redis (optional, for enhanced performance)
- GitHub API token (optional, for higher rate limits)

### **Installation**

```bash
# Clone the repository
git clone https://github.com/Lkshayyadav/GitHubWatcher.git
cd GitHubWatcher

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### **Environment Variables**

```bash
# Required for full functionality
GITHUB_API_KEY=ghp_your_github_token_here
GEMINI_API_KEY=your_gemini_api_key_here
PORTIA_API_KEY=your_portia_api_key_here

# Optional Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### **Run Locally**

```bash
# Start the server
python main.py

# Visit the application
open http://127.0.0.1:5000
```

## 📱 **Usage**

1. **Enter Repository URL**: Paste any GitHub repository URL
2. **Get Instant Analysis**: View comprehensive metrics and insights
3. **AI-Powered Insights**: Read AI-generated project assessments
4. **Compare Repositories**: Side-by-side analysis of multiple projects
5. **Export Reports**: Save analysis results for team sharing

## 🏗 **Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   External      │
│   (Jinja2 +     │◄──►│   Backend       │◄──►│   APIs          │
│   Tailwind)     │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    │ • GitHub API   │
                                              │ • Gemini AI    │
┌─────────────────┐    ┌─────────────────┐    │ • Portia AI    │
│   Redis Cache   │◄──►│   Data Layer    │    └─────────────────┘
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

## 🚀 **Deployment**

### **Render (Recommended)**
```bash
# Build Command
pip install -r requirements.txt

# Start Command
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### **Docker**
```bash
docker build -t githubwatcher .
docker run -p 5000:5000 githubwatcher
```

### **Heroku/Railway**
- Connect your GitHub repository
- Set environment variables
- Deploy automatically on push

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **GitHub** for their excellent API
- **Google** for Gemini AI capabilities
- **Portia** for advanced repository intelligence
- **FastAPI** team for the amazing framework

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/Lkshayyadav/GitHubWatcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Lkshayyadav/GitHubWatcher/discussions)
- **Email**: [Your Email]

---

**Built with ❤️ for the open-source community**

*GitHubWatcher - Making repository analysis intelligent, fast, and accessible.*
