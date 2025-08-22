// Main JavaScript for GitHub Repository Checker

document.addEventListener('DOMContentLoaded', function() {
    console.log('GitHub Repository Checker loaded');
    
    // Initialize application
    initializeApp();
    
    // Initialize results page functionality if we're on results page
    if (window.location.pathname === '/results') {
        initializeResultsPage();
    }
});

function initializeApp() {
    // Initialize search functionality
    initializeSearch();
    
    // Initialize navigation
    initializeNavigation();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Initialize animations
    initializeAnimations();
}

function initializeResultsPage() {
    // This function is called from the results template
    // The template handles the dynamic loading logic
    console.log('Results page initialized');
}

function initializeSearch() {
    const searchForms = document.querySelectorAll('.search-form');
    
    searchForms.forEach(form => {
        const input = form.querySelector('input[name="q"]');
        const button = form.querySelector('button[type="submit"]');
        
        if (input && button) {
            // Add real-time validation
            input.addEventListener('input', function() {
                validateGitHubUrl(this);
            });
            
            // Add form submission handling
            form.addEventListener('submit', function(e) {
                if (!validateGitHubUrl(input)) {
                    e.preventDefault();
                    showValidationError(input, 'Please enter a valid GitHub repository URL');
                }
            });
            
            // Add keyboard shortcuts
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    if (validateGitHubUrl(this)) {
                        form.submit();
                    }
                }
            });
        }
    });
}

function validateGitHubUrl(input) {
    const value = input.value.trim();
    
    if (!value) {
        clearValidationError(input);
        return false;
    }
    
    // GitHub URL pattern
    const githubPattern = /^https?:\/\/(www\.)?github\.com\/[\w\-\.]+\/[\w\-\.]+\/?$/;
    
    if (githubPattern.test(value)) {
        showValidationSuccess(input);
        return true;
    } else {
        showValidationError(input, 'Please enter a valid GitHub repository URL (e.g., https://github.com/username/repo)');
        return false;
    }
}

function showValidationError(input, message) {
    clearValidationError(input);
    
    input.classList.add('border-red-500', 'focus:border-red-500');
    input.classList.remove('border-green-500', 'focus:border-green-500');
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'validation-error mt-2 text-red-600 text-sm';
    errorDiv.textContent = message;
    
    input.closest('.relative').insertAdjacentElement('afterend', errorDiv);
}

function showValidationSuccess(input) {
    clearValidationError(input);
    
    input.classList.add('border-green-500', 'focus:border-green-500');
    input.classList.remove('border-red-500', 'focus:border-red-500');
}

function clearValidationError(input) {
    input.classList.remove('border-red-500', 'focus:border-red-500', 'border-green-500', 'focus:border-green-500');
    
    const existingError = input.closest('.relative').parentNode.querySelector('.validation-error');
    if (existingError) {
        existingError.remove();
    }
}

function initializeNavigation() {
    // Add active state to current navigation item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a[href]');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath === '/' && href === '/')) {
            link.classList.add('text-blue-600', 'font-semibold');
            link.classList.remove('text-gray-600');
        }
    });
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Theme toggle logic
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
        toggle.addEventListener('click', () => {
            const root = document.documentElement;
            const isDark = root.classList.toggle('dark');
            try { localStorage.setItem('theme', isDark ? 'dark' : 'light'); } catch (e) {}
        });
    }
}

function initializeFormValidation() {
    // Add enhanced form validation for all forms
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input[required]');
        
        inputs.forEach(input => {
            input.addEventListener('blur', function() {
                if (this.value.trim() === '') {
                    this.classList.add('border-red-500');
                } else {
                    this.classList.remove('border-red-500');
                    this.classList.add('border-green-500');
                }
            });
            
            input.addEventListener('focus', function() {
                this.classList.remove('border-red-500', 'border-green-500');
            });
        });
    });
}

function initializeAnimations() {
    // Add intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate in
    const animateElements = document.querySelectorAll('.bg-white, .glass-card, .grid > div, .space-y-8 > div, .feature-card');
    animateElements.forEach(el => {
        observer.observe(el);
    });
    
    // Add hover effects to interactive elements
    const interactiveElements = document.querySelectorAll('button, .bg-white, .glass-card, a[href]');
    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        el.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

// Utility functions
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `fixed top-4 right-4 z-50 p-4 rounded-lg shadow-lg max-w-sm transition-all duration-300 transform translate-x-full`;
    
    const bgColors = {
        'success': 'bg-green-500',
        'error': 'bg-red-500',
        'warning': 'bg-yellow-500',
        'info': 'bg-blue-500'
    };
    
    notification.className += ` ${bgColors[type] || bgColors.info} text-white`;
    notification.innerHTML = `
        <div class="flex items-center justify-between">
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" class="ml-2 text-white hover:text-gray-200">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.classList.remove('translate-x-full');
    }, 100);
    
    // Auto remove
    if (duration > 0) {
        setTimeout(() => {
            notification.classList.add('translate-x-full');
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 300);
        }, duration);
    }
}

// GitHub API helper functions (for future implementation)
function extractRepoInfo(githubUrl) {
    const match = githubUrl.match(/github\.com\/([^\/]+)\/([^\/]+)/);
    if (match) {
        return {
            owner: match[1],
            repo: match[2].replace(/\.git$/, '')
        };
    }
    return null;
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function timeAgo(date) {
    const now = new Date();
    const diff = now - new Date(date);
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    const months = Math.floor(days / 30);
    const years = Math.floor(days / 365);
    
    if (years > 0) return `${years} year${years > 1 ? 's' : ''} ago`;
    if (months > 0) return `${months} month${months > 1 ? 's' : ''} ago`;
    if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
    if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
    if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
    return 'Just now';
}

// Add CSS for fade-in animation
if (!document.querySelector('#fade-in-styles')) {
    const style = document.createElement('style');
    style.id = 'fade-in-styles';
    style.textContent = `
        .animate-fade-in {
            animation: fadeIn 0.6s ease-out forwards;
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;
    document.head.appendChild(style);
}

// Export functions for use in other modules (if needed)
window.RepoChecker = {
    validateGitHubUrl,
    showNotification,
    extractRepoInfo,
    formatNumber,
    timeAgo
};
