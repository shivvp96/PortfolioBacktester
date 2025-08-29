# Authentication System - Portfolio Backtesting Tool

## Overview

The portfolio backtesting application now includes a comprehensive authentication system with role-based access control and user registration capabilities. This ensures secure access to different features based on user roles and permissions, while allowing new users to create their own accounts.

## Features

### 🔐 Secure Login System
- Password-based authentication
- Session management with automatic logout
- Login attempt limiting (max 5 attempts)
- Secure password handling

### 📝 User Registration System
- Self-service account creation
- Email validation and username uniqueness checking
- Password strength recommendations
- Terms of Service and Privacy Policy integration
- Automatic role assignment (Regular User level)
- Persistent user storage in file system

### 👥 Role-Based Access Control
Four distinct user roles with different permission levels:

#### 🔵 Demo User
- **Username:** `demo` | **Password:** `demo`
- **Permissions:** Limited access
- **Features:**
  - Basic portfolio backtesting (max 2 portfolios)
  - Standard charts only
  - No data export
  - Benchmark comparison available

#### 🟢 Regular User  
- **Username:** `user` | **Password:** `password123`
- **Permissions:** Standard access
- **Features:**
  - Portfolio backtesting (max 5 portfolios)
  - Advanced interactive charts
  - CSV data export
  - Full benchmark comparison

#### 🟡 Analyst
- **Username:** `analyst` | **Password:** `analyst2024`
- **Permissions:** Advanced access
- **Features:**
  - Portfolio backtesting (max 10 portfolios)
  - Advanced interactive charts
  - CSV data export
  - Full benchmark comparison
  - Enhanced analytics features

#### 🔴 Administrator
- **Username:** `admin` | **Password:** `admin123`
- **Permissions:** Full access
- **Features:**
  - Unlimited portfolio backtesting
  - All advanced features
  - CSV data export
  - Full benchmark comparison
  - User management capabilities

## Security Features

### Session Management
- Automatic session handling via Streamlit's session state
- Secure logout functionality
- User role tracking throughout the session

### Input Validation
- Username and password validation
- Protection against brute force attacks
- Session timeout handling

### Data Protection
- Role-based feature restrictions
- Secure access to export functionality
- Protected advanced analytics features

## Usage Instructions

### For New Users
1. **Create Account:** Click "Sign Up" on the login page
2. **Fill Registration Form:** Enter username, email, and password
3. **Accept Terms:** Review and agree to Terms of Service and Privacy Policy
4. **Confirm Registration:** Complete account creation
5. **Login:** Switch to Login tab and use your new credentials

### For Existing Users
1. **Access the Application:** Navigate to the portfolio backtesting tool
2. **Login:** Use your registered credentials or demo accounts
3. **Explore Features:** Access features based on your role permissions
4. **Logout:** Use the logout button in the top-right corner when finished

### For Developers
1. **User Management:** Edit `auth_config.py` to add/modify users
2. **Role Configuration:** Modify `ROLE_FEATURES` in `auth_config.py` to adjust permissions
3. **Security Settings:** Update security parameters in the configuration file

## File Structure

```
├── app.py                 # Main application with authentication integration
├── auth_config.py         # User credentials and role configuration
└── AUTHENTICATION.md      # This documentation file
```

## Configuration

### Adding New Users
Edit `auth_config.py` and add entries to the `DEMO_USERS` dictionary:

```python
"new_user": {
    "password": "secure_password",
    "role": "user",
    "permissions": ["portfolio_access"]
}
```

### Modifying Role Permissions
Update the `ROLE_FEATURES` dictionary in `auth_config.py`:

```python
"custom_role": {
    "max_portfolios": 3,
    "advanced_charts": True,
    "export_data": False,
    "benchmark_comparison": True
}
```

## Security Considerations

### Production Deployment
- **Change Default Passwords:** Replace all demo passwords with secure credentials
- **Environment Variables:** Store sensitive configuration in environment variables
- **HTTPS:** Enable HTTPS for production deployments
- **Database Integration:** Consider migrating to a proper user database for production use

### Current Limitations
- Passwords are stored in plain text in the configuration file
- No password recovery mechanism
- Limited to demo accounts for this version
- Session management relies on Streamlit's built-in capabilities

## Future Enhancements

### Planned Features
- Database-backed user management
- Password encryption and hashing
- Email-based password recovery
- OAuth integration (Google, GitHub)
- Admin panel for user management
- Audit logging for security events
- Multi-factor authentication (MFA)

### Integration Possibilities
- LDAP/Active Directory integration
- Single Sign-On (SSO) support
- API key authentication for programmatic access
- Role-based API endpoints

## Support

For questions about the authentication system:
1. Check the demo credentials in the login form
2. Review role permissions in `auth_config.py`
3. Ensure proper session management in the application

---

**Note:** This authentication system is designed for demonstration and development purposes. For production deployment, implement proper security measures including encrypted password storage, secure session management, and comprehensive user administration.