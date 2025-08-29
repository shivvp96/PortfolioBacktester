# Authentication Configuration for Portfolio Backtesting Tool
# This file contains user credentials and authentication settings

# Demo user accounts
# In production, you would store these in a secure database with proper hashing
DEMO_USERS = {
    "admin": {
        "password": "admin123",
        "role": "administrator",
        "permissions": ["full_access", "manage_users"]
    },
    "user": {
        "password": "password123", 
        "role": "user",
        "permissions": ["portfolio_access"]
    },
    "demo": {
        "password": "demo",
        "role": "demo",
        "permissions": ["limited_access"]
    },
    "analyst": {
        "password": "analyst2024",
        "role": "analyst", 
        "permissions": ["portfolio_access", "advanced_analytics"]
    }
}

# Security settings
SESSION_TIMEOUT_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
REQUIRE_HTTPS = False  # Set to True in production

# Features by role
ROLE_FEATURES = {
    "administrator": {
        "max_portfolios": None,  # Unlimited
        "advanced_charts": True,
        "export_data": True,
        "benchmark_comparison": True
    },
    "analyst": {
        "max_portfolios": 10,
        "advanced_charts": True,
        "export_data": True,
        "benchmark_comparison": True
    },
    "user": {
        "max_portfolios": 5,
        "advanced_charts": True,
        "export_data": True,
        "benchmark_comparison": True
    },
    "demo": {
        "max_portfolios": 2,
        "advanced_charts": False,
        "export_data": False,
        "benchmark_comparison": True
    }
}