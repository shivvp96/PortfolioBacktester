import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import hashlib
import hmac
from auth_config import DEMO_USERS, ROLE_FEATURES
from terms_of_service import get_terms_of_service, get_privacy_policy

# Set page config
st.set_page_config(
    page_title="Portfolio Backtesting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Authentication functions

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against provided password"""
    return stored_password == provided_password

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)

def get_user_role():
    """Get current user's role"""
    username = st.session_state.get('username', '')
    
    # Check demo users first
    if username in DEMO_USERS:
        return DEMO_USERS[username]['role']
    
    # Check registered users
    registered_users = load_registered_users()
    if username in registered_users:
        return registered_users[username]['role']
    
    return 'demo'

def get_user_permissions():
    """Get current user's permissions"""
    role = get_user_role()
    return ROLE_FEATURES.get(role, ROLE_FEATURES['demo'])

def save_user_to_file(username, password, email, role="user"):
    """Save new user to a persistent file"""
    import os
    user_file = "registered_users.txt"
    
    # Create file if it doesn't exist
    if not os.path.exists(user_file):
        with open(user_file, 'w') as f:
            f.write("# Registered Users\n")
    
    # Append new user
    with open(user_file, 'a') as f:
        f.write(f"{username}|{password}|{email}|{role}\n")

def load_registered_users():
    """Load registered users from file"""
    import os
    user_file = "registered_users.txt"
    registered_users = {}
    
    if os.path.exists(user_file):
        with open(user_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('|')
                    if len(parts) >= 4:
                        username, password, email, role = parts[:4]
                        registered_users[username] = {
                            'password': password,
                            'email': email,
                            'role': role,
                            'permissions': ['portfolio_access']
                        }
    return registered_users

# Removed - integrated into login_form function

def login_form():
    """Display professional trading-style login interface"""
    # Initialize session state
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    
    # Show success message if just signed up
    if st.session_state.get('signup_success', False):
        st.session_state['signup_success'] = False
        st.session_state.show_signup = False
    
    # Professional header
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #00D4AA; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;'>
            Portfolio Backtesting Platform
        </h1>
        <p style='color: #8B949E; font-size: 1.1rem; margin-bottom: 2rem;'>
            Professional-grade portfolio backtesting and risk analysis tool
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Centered navigation with professional styling
    col1, col2, col3, col4, col5 = st.columns([2.5, 1, 0.5, 1, 2.5])
    with col2:
        if st.button("Login", type="secondary" if st.session_state.show_signup else "primary", use_container_width=True):
            st.session_state.show_signup = False
            st.rerun()
    with col4:
        if st.button("Sign Up", type="primary" if st.session_state.show_signup else "secondary", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the form content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.session_state.show_signup:
            # Clean signup form without box
            st.markdown("<h4 style='text-align: center;'>Create New Account</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Get access to advanced portfolio analysis tools</p>", unsafe_allow_html=True)
            
            with st.form("signup_form", clear_on_submit=True):
                new_username = st.text_input("Username", placeholder="Choose a username", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Username</small>", unsafe_allow_html=True)
                
                new_email = st.text_input("Email", placeholder="Enter your email address", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Email Address</small>", unsafe_allow_html=True)
                
                new_password = st.text_input("Password", type="password", placeholder="Create a secure password", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Choose Password</small>", unsafe_allow_html=True)
                
                confirm_password = st.text_input("Confirm", type="password", placeholder="Confirm your password", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Confirm Password</small>", unsafe_allow_html=True)
                
                agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
                signup_button = st.form_submit_button("Create Account", use_container_width=True)
                
                if signup_button:
                    # Validation
                    registered_users = load_registered_users()
                    all_users = {**DEMO_USERS, **registered_users}
                    
                    if not new_username or not new_email or not new_password:
                        st.error("Please fill in all fields.")
                    elif new_username in all_users:
                        st.error("Username already exists.")
                    elif len(new_username) < 3:
                        st.error("Username must be at least 3 characters.")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif '@' not in new_email:
                        st.error("Please enter a valid email address.")
                    elif not agree_terms:
                        st.error("Please agree to the Terms of Service.")
                    else:
                        try:
                            save_user_to_file(new_username, new_password, new_email, "user")
                            st.success("Account created successfully!")
                            st.info(f"Welcome aboard, {new_username}!")
                            st.session_state['show_signup'] = False
                            st.session_state['signup_success'] = True
                        except Exception as e:
                            st.error(f"Error creating account: {str(e)}")
        
        else:
            # Clean login form without box
            st.markdown("<h4 style='text-align: center;'>Welcome Back</h4>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Sign in to access your portfolio analytics</p>", unsafe_allow_html=True)
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Username</small>", unsafe_allow_html=True)
                
                password = st.text_input("Password", type="password", placeholder="Enter your password", label_visibility="collapsed")
                st.markdown("<small style='color: #8B949E;'>Password</small>", unsafe_allow_html=True)
                
                login_button = st.form_submit_button("Sign In", use_container_width=True)
                
                if login_button:
                    if st.session_state.login_attempts >= 5:
                        st.error("Too many login attempts. Please refresh the page.")
                        return
                    
                    # Check both demo users and registered users
                    registered_users = load_registered_users()
                    all_users = {**DEMO_USERS, **registered_users}
                    
                    if username in all_users:
                        if verify_password(all_users[username]['password'], password):
                            st.session_state['authenticated'] = True
                            st.session_state['username'] = username
                            st.session_state['user_role'] = all_users[username]['role']
                            st.session_state.login_attempts = 0
                            st.success("Welcome back!")
                            st.rerun()
                        else:
                            st.session_state.login_attempts += 1
                            st.error("Invalid credentials")
                    else:
                        st.session_state.login_attempts += 1
                        st.error("Invalid credentials")

def logout():
    """Logout user"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()

@st.cache_data
def fetch_prices(tickers, start_date, end_date):
    """
    Fetch adjusted close prices for given tickers from Yahoo Finance.
    Falls back to close prices if adjusted close is not available.
    Handles invalid tickers gracefully.
    """
    prices = pd.DataFrame()
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                invalid_tickers.append(ticker)
                continue
            
            # Try to get adjusted close, fall back to close
            if 'Adj Close' in data.columns and not bool(data['Adj Close'].isna().all()):
                price_series = data['Adj Close']
            else:
                price_series = data['Close']
            
            prices[ticker] = price_series
            
        except Exception as e:
            invalid_tickers.append(ticker)
            st.warning(f"Failed to fetch data for {ticker}: {str(e)}")
    
    if invalid_tickers:
        st.error(f"Invalid or unavailable tickers: {', '.join(invalid_tickers)}")
    
    return prices.dropna()

def compute_returns(prices, frequency):
    """
    Resample prices to specified frequency and compute percentage returns.
    """
    if frequency == 'D':
        # Daily - no resampling needed
        resampled_prices = prices
    elif frequency == 'W':
        # Weekly - Friday end
        resampled_prices = prices.resample('W-FRI').last()
    elif frequency == 'M':
        # Monthly - month end
        resampled_prices = prices.resample('M').last()
    else:
        resampled_prices = prices
    
    # Compute percentage returns
    returns = resampled_prices.pct_change().dropna()
    return returns

def rebalance_points(index, freq):
    """
    Return rebalancing dates based on frequency.
    """
    if freq == "None":
        return []
    
    rebalance_dates = []
    
    if freq == "Monthly":
        # Last business day of each month
        monthly_ends = pd.date_range(start=index.min(), end=index.max(), freq='ME')
        for date in monthly_ends:
            # Find the closest date in the index
            closest_date = index[index <= date].max()
            if pd.notna(closest_date) and closest_date not in rebalance_dates:
                rebalance_dates.append(closest_date)
    
    elif freq == "Quarterly":
        # Last business day of each quarter
        quarterly_ends = pd.date_range(start=index.min(), end=index.max(), freq='Q')
        for date in quarterly_ends:
            closest_date = index[index <= date].max()
            if pd.notna(closest_date) and closest_date not in rebalance_dates:
                rebalance_dates.append(closest_date)
    
    elif freq == "Yearly":
        # Last business day of each year
        yearly_ends = pd.date_range(start=index.min(), end=index.max(), freq='Y')
        for date in yearly_ends:
            closest_date = index[index <= date].max()
            if pd.notna(closest_date) and closest_date not in rebalance_dates:
                rebalance_dates.append(closest_date)
    
    return sorted(rebalance_dates)

def apply_vol_targeting(port_returns, target_annual, window=63, lev_cap=3.0):
    """
    Apply volatility targeting overlay to portfolio returns.
    
    Parameters:
    port_returns: Series of portfolio returns
    target_annual: Target annual volatility (as percentage, e.g., 15 for 15%)
    window: Rolling window for volatility estimation (default 63 days ~ 3 months)
    lev_cap: Maximum leverage cap (default 3.0)
    
    Returns:
    vol_targeted_returns: Series of volatility-targeted returns
    """
    if port_returns.empty or target_annual is None or target_annual <= 0:
        return port_returns.copy()
    
    # Convert annual target to decimal
    target_annual_decimal = target_annual / 100
    
    # Calculate rolling realized volatility (shifted by 1 day to avoid look-ahead)
    realized_vol = port_returns.shift(1).rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
    
    # Calculate leverage multiplier
    leverage = target_annual_decimal / realized_vol
    leverage = leverage.clip(0, lev_cap)  # Apply leverage cap
    
    # Apply volatility targeting
    vol_targeted_returns = port_returns * leverage
    
    # Handle NaNs at the beginning
    vol_targeted_returns = vol_targeted_returns.fillna(port_returns)
    
    return vol_targeted_returns

def calculate_capture_ratios(portfolio_returns, benchmark_returns):
    """
    Calculate upside/downside capture ratios and hit rates.
    """
    if portfolio_returns.empty or benchmark_returns.empty:
        return {}
    
    # Align dates
    aligned_data = pd.DataFrame({
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }).dropna()
    
    if aligned_data.empty:
        return {}
    
    port_ret = aligned_data['portfolio']
    bench_ret = aligned_data['benchmark']
    
    # Upside/Downside periods
    upside_mask = bench_ret > 0
    downside_mask = bench_ret < 0
    
    # Capture ratios
    upside_capture = 0
    downside_capture = 0
    
    if upside_mask.any():
        port_up = port_ret[upside_mask].mean()
        bench_up = bench_ret[upside_mask].mean()
        if bench_up != 0:
            upside_capture = (port_up / bench_up) * 100
    
    if downside_mask.any():
        port_down = port_ret[downside_mask].mean()
        bench_down = bench_ret[downside_mask].mean()
        if bench_down != 0:
            downside_capture = (port_down / bench_down) * 100
    
    # Hit rates
    daily_hit_rate = (port_ret > 0).mean() * 100
    
    # Monthly win rate (resample to month-end)
    monthly_returns = port_ret.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly_win_rate = (monthly_returns > 0).mean() * 100 if len(monthly_returns) > 0 else 0
    
    return {
        'Upside Capture (%)': upside_capture,
        'Downside Capture (%)': downside_capture,
        'Daily Hit Rate (%)': daily_hit_rate,
        'Monthly Win Rate (%)': monthly_win_rate
    }

def calculate_rolling_metrics(portfolio_returns, benchmark_returns, risk_free_rate, window):
    """
    Calculate rolling Sharpe, Max DD, and Beta.
    """
    if portfolio_returns.empty:
        return pd.DataFrame()
    
    # Rolling Sharpe
    excess_returns = portfolio_returns - (risk_free_rate / 252)
    rolling_sharpe = (excess_returns.rolling(window).mean() / 
                     excess_returns.rolling(window).std() * np.sqrt(252))
    
    # Rolling Max Drawdown
    rolling_cum = portfolio_returns.rolling(window).apply(lambda x: (1 + x).prod(), raw=False)
    rolling_peak = rolling_cum.rolling(window, min_periods=1).max()
    rolling_dd = ((rolling_cum - rolling_peak) / rolling_peak * 100).rolling(window).min()
    
    # Rolling Beta (if benchmark available)
    rolling_beta = pd.Series(index=portfolio_returns.index, dtype=float)
    if benchmark_returns is not None and not benchmark_returns.empty:
        for i in range(window, len(portfolio_returns)):
            port_window = portfolio_returns.iloc[i-window:i]
            bench_window = benchmark_returns.iloc[i-window:i]
            if len(port_window) == len(bench_window) and bench_window.var() > 0:
                rolling_beta.iloc[i] = port_window.cov(bench_window) / bench_window.var()
    
    results = pd.DataFrame({
        'Rolling_Sharpe': rolling_sharpe,
        'Rolling_MaxDD': rolling_dd,
        'Rolling_Beta': rolling_beta if benchmark_returns is not None else None
    }, index=portfolio_returns.index)
    
    return results

def simulate_portfolio(returns, target_weights, rebalance, fees_bps, rebalance_band=0.0):
    """
    Simulate portfolio returns with weight drift and rebalancing.
    Returns portfolio returns, weight history, and drawdown.
    """
    if returns.empty:
        return pd.Series(), pd.DataFrame(), pd.Series()
    
    # Initialize
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    weight_history = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    # Get rebalancing dates
    rebalance_dates = rebalance_points(returns.index, rebalance)
    
    # Initial weights
    current_weights = pd.Series(target_weights, index=returns.columns)
    current_weights = current_weights / current_weights.sum()  # Normalize
    
    portfolio_value = 1.0
    
    for i, date in enumerate(returns.index):
        # Record current weights
        weight_history.loc[date] = current_weights
        
        # Calculate portfolio return for this period
        period_return = (returns.loc[date] * current_weights).sum()
        portfolio_returns.loc[date] = period_return
        
        # Update portfolio value
        portfolio_value *= (1 + period_return)
        
        # Update weights due to price changes (drift)
        asset_returns = returns.loc[date]
        current_weights = current_weights * (1 + asset_returns)
        current_weights = current_weights / current_weights.sum()
        
        # Check if rebalancing is needed
        if date in rebalance_dates:
            target_weights_series = pd.Series(target_weights, index=returns.columns)
            target_weights_series = target_weights_series / target_weights_series.sum()
            
            # Drift-band rebalancing logic
            if rebalance_band > 0:
                # Check which assets exceed the drift band
                weight_diffs = abs(current_weights - target_weights_series)
                rebalance_mask = weight_diffs > (rebalance_band / 100)
                
                if rebalance_mask.any():
                    # Calculate turnover only for assets exceeding the band
                    turnover = weight_diffs[rebalance_mask].sum()
                    transaction_cost = turnover * fees_bps / 10000
                    
                    # Apply transaction costs
                    portfolio_value *= (1 - transaction_cost)
                    
                    # Rebalance only assets that exceed the band
                    current_weights[rebalance_mask] = target_weights_series[rebalance_mask]
                # If no assets exceed band, no rebalancing occurs
            else:
                # Traditional rebalancing (band = 0)
                turnover = abs(current_weights - target_weights_series).sum()
                transaction_cost = turnover * fees_bps / 10000
                
                # Apply transaction costs
                portfolio_value *= (1 - transaction_cost)
                
                # Reset to target weights
                current_weights = target_weights_series.copy()
    
    # Calculate drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    return portfolio_returns, weight_history, drawdown

def performance_stats(port_ret, rf_annual, freq):
    """
    Compute performance statistics.
    """
    if port_ret.empty:
        return {}
    
    # Adjust risk-free rate for frequency
    if freq == 'D':
        rf_period = rf_annual / 252
    elif freq == 'W':
        rf_period = rf_annual / 52
    elif freq == 'M':
        rf_period = rf_annual / 12
    else:
        rf_period = rf_annual / 252
    
    # Total return
    total_return = (1 + port_ret).prod() - 1
    
    # Annualized return (CAGR)
    years = len(port_ret) / (252 if freq == 'D' else 52 if freq == 'W' else 12)
    if years > 0:
        cagr = (1 + total_return) ** (1/years) - 1
    else:
        cagr = 0
    
    # Volatility (annualized)
    vol_multiplier = np.sqrt(252 if freq == 'D' else 52 if freq == 'W' else 12)
    volatility = port_ret.std() * vol_multiplier
    
    # Sharpe ratio
    excess_returns = port_ret - rf_period
    if excess_returns.std() != 0:
        sharpe = excess_returns.mean() / excess_returns.std() * vol_multiplier
    else:
        sharpe = 0
    
    # Max drawdown
    cumulative = (1 + port_ret).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    return {
        'Total Return (%)': total_return * 100,
        'CAGR (%)': cagr * 100,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown * 100
    }

def plot_equity_curve(port_ret, bench_ret=None, vol_targeted_ret=None):
    """
    Plot equity curve showing growth of $1.
    """
    fig = go.Figure()
    
    if not port_ret.empty:
        portfolio_growth = (1 + port_ret).cumprod()
        fig.add_trace(go.Scatter(
            x=portfolio_growth.index,
            y=portfolio_growth.values,
            mode='lines',
            name='Base Portfolio',
            line=dict(color='#1f77b4', width=2)
        ))
    
    # Volatility-targeted overlay if provided
    if vol_targeted_ret is not None and not vol_targeted_ret.empty:
        vol_growth = (1 + vol_targeted_ret).cumprod()
        fig.add_trace(go.Scatter(
            x=vol_growth.index,
            y=vol_growth.values,
            mode='lines',
            name='Vol-Targeted',
            line=dict(color='#00D4AA', width=2, dash='dash')
        ))
    
    if bench_ret is not None and not bench_ret.empty:
        benchmark_growth = (1 + bench_ret).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_growth.index,
            y=benchmark_growth.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2)
        ))
    
    fig.update_layout(
        title='Portfolio Equity Curve (Growth of $1)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        showlegend=True,
        template='plotly_dark'
    )
    
    return fig

def plot_drawdown(drawdown):
    """
    Plot drawdown chart.
    """
    fig = go.Figure()
    
    if not drawdown.empty:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
    
    fig.update_layout(
        title='Portfolio Drawdown',
        xaxis_title='Date',
        yaxis_title='Drawdown (%)',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

# Main Streamlit App
def main():
    # Professional header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div style='padding: 1rem 0;'>
            <h1 style='color: #00D4AA; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem;'>
                Portfolio Analytics
            </h1>
            <p style='color: #8B949E; font-size: 1rem; margin: 0;'>
                Advanced backtesting and risk analysis platform
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if check_authentication():
            user_role = get_user_role()
            role_colors = {"administrator": "#FF6B6B", "analyst": "#4ECDC4", "user": "#45B7D1", "demo": "#96CEB4"}
            role_color = role_colors.get(user_role, "#8B949E")
            
            # User info and sign out button in aligned layout
            st.markdown(f"""
            <div style='text-align: right; padding: 1rem 0;'>
                <div style='display: inline-block; background: {role_color}20; 
                           padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid {role_color}40; margin-bottom: 0.5rem;'>
                    <span style='color: {role_color}; font-weight: 600;'>{st.session_state.get('username', 'User')}</span>
                    <span style='color: #8B949E; font-size: 0.9rem;'> • {user_role.title()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Align the sign out button to the right
            col_spacer, col_btn = st.columns([3, 1])
            with col_btn:
                if st.button("Sign Out", key="logout_btn", use_container_width=True):
                    logout()
    
    # Professional sidebar styling
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0; border-bottom: 1px solid #30363d; margin-bottom: 1rem;'>
        <h3 style='color: #00D4AA; margin: 0;'>Analysis Parameters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Date selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*3),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # Frequency selection
    frequency = st.sidebar.selectbox(
        "Sampling Frequency",
        options=['D', 'W', 'M'],
        format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
        index=0
    )
    
    # Rebalancing frequency
    rebalance_freq = st.sidebar.selectbox(
        "Rebalancing Frequency",
        options=['None', 'Monthly', 'Quarterly', 'Yearly'],
        index=1
    )
    
    # Transaction costs
    transaction_costs = st.sidebar.number_input(
        "Transaction Costs (basis points)",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=1.0,
        help="Transaction costs in basis points (1 bp = 0.01%)"
    )
    
    # Risk-free rate
    risk_free_rate = st.sidebar.number_input(
        "Risk-free Rate (annual %)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        step=0.1,
        help="Annual risk-free rate for Sharpe ratio calculation"
    ) / 100
    
    # Benchmark selection
    benchmark = st.sidebar.selectbox(
        "Benchmark",
        options=['None', 'SPY', 'QQQ', 'DIA'],
        index=0
    )
    
    # Advanced Features Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Features")
    
    # Drift-Band Rebalancing
    rebalance_band = st.sidebar.slider(
        "Rebalance Band (±%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help="Only rebalance when asset weight drifts outside this band. 0% = rebalance on schedule."
    )
    
    # Volatility Targeting
    target_vol = st.sidebar.number_input(
        "Target Vol (annual %)",
        min_value=0.0,
        max_value=50.0,
        value=None,
        step=1.0,
        help="Target annual volatility for portfolio overlay"
    )
    
    show_vol_overlay = st.sidebar.toggle(
        "Show Vol-Targeted Overlay",
        value=False,
        help="Enable volatility-targeted portfolio overlay"
    )
    
    # Portfolio composition with professional styling
    st.markdown("""
    <div style='padding: 1rem 0; border-bottom: 1px solid #30363d; margin-bottom: 1.5rem;'>
        <h2 style='color: #E6EDF3; font-size: 1.5rem; margin: 0;'>Portfolio Composition</h2>
        <p style='color: #8B949E; font-size: 0.95rem; margin: 0.5rem 0 0 0;'>
            Configure your investment portfolio allocation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for portfolio data
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', ''],
            'Weight': [0.4, 0.3, 0.3, 0.0]
        })
    
    # Show role-based information with professional styling
    permissions = get_user_permissions()
    max_portfolios = permissions.get('max_portfolios')
    if max_portfolios:
        st.markdown(f"""
        <div style='background: #00D4AA20; padding: 1rem; border-radius: 8px; border-left: 4px solid #00D4AA; margin-bottom: 1rem;'>
            <strong style='color: #00D4AA;'>Account Level:</strong> 
            <span style='color: #E6EDF3;'>Up to {max_portfolios} portfolio simulations</span><br>
            <strong style='color: #00D4AA;'>Features:</strong> 
            <span style='color: #E6EDF3;'>Advanced Analytics {'Enabled' if permissions.get('advanced_charts') else 'Limited'}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Data editor for portfolio
    edited_data = st.data_editor(
        st.session_state.portfolio_data,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol"),
            "Weight": st.column_config.NumberColumn("Weight", min_value=0.0, max_value=1.0, step=0.01)
        }
    )
    
    # Process portfolio data
    portfolio_data = edited_data.copy()
    # Remove rows with None, NaN, or empty ticker values
    portfolio_data = portfolio_data.dropna(subset=['Ticker'])
    
    # Filter out empty and 'none' values
    valid_mask = (
        (portfolio_data['Ticker'].astype(str).str.strip() != '') &
        (portfolio_data['Ticker'].astype(str).str.strip().str.lower() != 'none')
    )
    portfolio_data = portfolio_data[valid_mask]
    
    # Remove duplicates and convert to uppercase
    portfolio_data = portfolio_data.drop_duplicates(subset=['Ticker'])
    portfolio_data['Ticker'] = portfolio_data['Ticker'].astype(str).str.upper()
    
    # Normalize weights
    if not portfolio_data.empty and portfolio_data['Weight'].sum() > 0:
        portfolio_data['Weight'] = portfolio_data['Weight'] / portfolio_data['Weight'].sum()
        
        # Display normalized weights with professional styling
        st.markdown("""
        <div style='margin: 1.5rem 0 1rem 0;'>
            <h4 style='color: #E6EDF3; margin-bottom: 0.5rem;'>Normalized Allocation</h4>
        </div>
        """, unsafe_allow_html=True)
        weight_df = portfolio_data.copy()
        weight_df['Weight (%)'] = weight_df['Weight'] * 100
        st.dataframe(weight_df[['Ticker', 'Weight (%)']], use_container_width=True)
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if portfolio_data.empty:
            st.error("Please add at least one ticker to your portfolio.")
            return
        
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            return
        
        with st.spinner("Running backtest..."):
            try:
                # Fetch price data
                tickers = portfolio_data['Ticker'].tolist()
                prices = fetch_prices(tickers, start_date, end_date)
                
                if prices.empty:
                    st.error("No valid price data found for the selected tickers and date range.")
                    return
                
                # Filter portfolio data to only include valid tickers
                valid_tickers = prices.columns.tolist()
                portfolio_data = portfolio_data[portfolio_data['Ticker'].isin(valid_tickers)]
                
                if portfolio_data.empty:
                    st.error("No valid tickers found in your portfolio.")
                    return
                
                # Compute returns
                returns = compute_returns(prices, frequency)
                
                if returns.empty:
                    st.error("Insufficient data to compute returns.")
                    return
                
                # Prepare target weights
                target_weights = {}
                for _, row in portfolio_data.iterrows():
                    target_weights[row['Ticker']] = row['Weight']
                
                # Ensure all tickers in returns have weights
                for ticker in returns.columns:
                    if ticker not in target_weights:
                        target_weights[ticker] = 0.0
                
                # Run portfolio simulation with drift-band rebalancing
                portfolio_returns, weight_history, drawdown = simulate_portfolio(
                    returns, target_weights, rebalance_freq, transaction_costs, rebalance_band
                )
                
                # Fetch benchmark data if selected
                benchmark_returns = None
                if benchmark != 'None':
                    bench_prices = fetch_prices([benchmark], start_date, end_date)
                    if not bench_prices.empty:
                        benchmark_returns = compute_returns(bench_prices, frequency)[benchmark]
                
                # Apply volatility targeting if enabled
                vol_targeted_returns = None
                if show_vol_overlay and target_vol is not None and target_vol > 0:
                    vol_targeted_returns = apply_vol_targeting(portfolio_returns, target_vol)
                
                # Calculate performance statistics
                portfolio_stats = performance_stats(portfolio_returns, risk_free_rate, frequency)
                vol_targeted_stats = None
                if vol_targeted_returns is not None:
                    vol_targeted_stats = performance_stats(vol_targeted_returns, risk_free_rate, frequency)
                
                # Display results
                st.header("📊 Backtest Results")
                
                # Show configuration summary
                config_summary = f"Rebalancing: {rebalance_freq}"
                if rebalance_band > 0:
                    config_summary += f" | Drift Band: ±{rebalance_band}%"
                if vol_targeted_returns is not None:
                    config_summary += f" | Vol Target: {target_vol}%"
                
                st.markdown(f"**Configuration:** {config_summary}")
                st.markdown("---")
                
                # Performance metrics comparison
                if vol_targeted_stats is not None:
                    # Side-by-side comparison table
                    st.subheader("📈 Performance Comparison")
                    
                    comparison_data = {
                        'Metric': list(portfolio_stats.keys()),
                        'Base Portfolio': [f"{v:.2f}%" if 'Ratio' not in k else f"{v:.3f}" 
                                         for k, v in portfolio_stats.items()],
                        'Vol-Targeted': [f"{v:.2f}%" if 'Ratio' not in k else f"{v:.3f}" 
                                        for k, v in vol_targeted_stats.items()]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Upside/Downside Capture Analysis (only if benchmark is selected)
                if benchmark_returns is not None:
                    st.subheader("📊 Upside/Downside Capture Analysis")
                    
                    capture_metrics = calculate_capture_ratios(portfolio_returns, benchmark_returns)
                    
                    if capture_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Upside Capture", 
                                f"{capture_metrics['Upside Capture (%)']:.1f}%",
                                help="Portfolio performance vs benchmark during up periods"
                            )
                        
                        with col2:
                            st.metric(
                                "Downside Capture", 
                                f"{capture_metrics['Downside Capture (%)']:.1f}%",
                                help="Portfolio performance vs benchmark during down periods"
                            )
                        
                        with col3:
                            st.metric(
                                "Daily Hit Rate", 
                                f"{capture_metrics['Daily Hit Rate (%)']:.1f}%",
                                help="Percentage of days with positive returns"
                            )
                        
                        with col4:
                            st.metric(
                                "Monthly Win Rate", 
                                f"{capture_metrics['Monthly Win Rate (%)']:.1f}%",
                                help="Percentage of months with positive returns"
                            )
                else:
                    # Regular performance display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Portfolio Performance")
                        for metric, value in portfolio_stats.items():
                            if 'Ratio' in metric:
                                st.metric(metric, f"{value:.3f}")
                            else:
                                st.metric(metric, f"{value:.2f}%")
                    
                    with col2:
                        if benchmark_returns is not None:
                            benchmark_stats = performance_stats(benchmark_returns, risk_free_rate, frequency)
                            st.subheader(f"{benchmark} Benchmark Performance")
                            for metric, value in benchmark_stats.items():
                                if 'Ratio' in metric:
                                    st.metric(metric, f"{value:.3f}")
                                else:
                                    st.metric(metric, f"{value:.2f}%")
                
                # Charts
                st.subheader("📈 Equity Curve")
                equity_fig = plot_equity_curve(portfolio_returns, benchmark_returns, vol_targeted_returns)
                st.plotly_chart(equity_fig, use_container_width=True)
                
                st.subheader("📉 Drawdown Chart")
                drawdown_fig = plot_drawdown(drawdown)
                st.plotly_chart(drawdown_fig, use_container_width=True)
                
                # Rolling Metrics Section
                st.subheader("📈 Rolling Metrics")
                
                # Rolling window selector
                col1, col2 = st.columns([1, 3])
                with col1:
                    rolling_window = st.selectbox(
                        "Rolling Window",
                        options=[63, 126, 252],
                        format_func=lambda x: f"{x} days",
                        index=0,
                        key="rolling_window"
                    )
                
                # Calculate rolling metrics
                rolling_metrics = calculate_rolling_metrics(
                    portfolio_returns, benchmark_returns, risk_free_rate, rolling_window
                )
                
                if not rolling_metrics.empty:
                    # Create rolling metrics charts
                    fig_rolling = go.Figure()
                    
                    # Rolling Sharpe
                    fig_rolling.add_trace(go.Scatter(
                        x=rolling_metrics.index,
                        y=rolling_metrics['Rolling_Sharpe'],
                        mode='lines',
                        name=f'Rolling Sharpe ({rolling_window}d)',
                        line=dict(color='#00D4AA', width=2),
                        yaxis='y1'
                    ))
                    
                    # Rolling Max Drawdown
                    fig_rolling.add_trace(go.Scatter(
                        x=rolling_metrics.index,
                        y=rolling_metrics['Rolling_MaxDD'],
                        mode='lines',
                        name=f'Rolling Max DD ({rolling_window}d)',
                        line=dict(color='#FF6B6B', width=2),
                        yaxis='y2'
                    ))
                    
                    # Rolling Beta (if benchmark available)
                    if benchmark_returns is not None and 'Rolling_Beta' in rolling_metrics.columns:
                        fig_rolling.add_trace(go.Scatter(
                            x=rolling_metrics.index,
                            y=rolling_metrics['Rolling_Beta'],
                            mode='lines',
                            name=f'Rolling Beta ({rolling_window}d)',
                            line=dict(color='#45B7D1', width=2),
                            yaxis='y3'
                        ))
                    
                    # Update layout for multiple y-axes
                    fig_rolling.update_layout(
                        title=f'Rolling Metrics ({rolling_window}-Day Window)',
                        xaxis_title='Date',
                        yaxis=dict(
                            title='Sharpe Ratio',
                            titlefont=dict(color='#00D4AA'),
                            tickfont=dict(color='#00D4AA'),
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Max Drawdown (%)',
                            titlefont=dict(color='#FF6B6B'),
                            tickfont=dict(color='#FF6B6B'),
                            overlaying='y',
                            side='right'
                        ),
                        yaxis3=dict(
                            title='Beta',
                            titlefont=dict(color='#45B7D1'),
                            tickfont=dict(color='#45B7D1'),
                            overlaying='y',
                            side='right',
                            position=0.97
                        ) if benchmark_returns is not None else None,
                        hovermode='x unified',
                        template='plotly_dark',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_rolling, use_container_width=True)
                
                # Final weights
                if not weight_history.empty:
                    st.subheader("📋 Final Portfolio Weights")
                    final_weights = weight_history.iloc[-1] * 100
                    final_weights_df = pd.DataFrame({
                        'Ticker': final_weights.index,
                        'Final Weight (%)': final_weights.values
                    })
                    st.dataframe(final_weights_df, use_container_width=True)
                
                # CSV download (role-based access)
                permissions = get_user_permissions()
                if permissions.get('export_data', False):
                    st.subheader("💾 Download Results")
                    
                    # Prepare data for download
                    download_data = pd.DataFrame(index=portfolio_returns.index)
                    download_data['Portfolio_Return'] = portfolio_returns
                    download_data['Portfolio_Cumulative'] = (1 + portfolio_returns).cumprod()
                    
                    # Include volatility-targeted returns if enabled
                    if vol_targeted_returns is not None:
                        download_data['Vol_Targeted_Return'] = vol_targeted_returns
                        download_data['Vol_Targeted_Cumulative'] = (1 + vol_targeted_returns).cumprod()
                    
                    # Include benchmark data and capture metrics if available
                    if benchmark_returns is not None:
                        download_data['Benchmark_Return'] = benchmark_returns
                        download_data['Benchmark_Cumulative'] = (1 + benchmark_returns).cumprod()
                        
                        # Add up/down day indicators for capture analysis
                        download_data['is_up_day'] = portfolio_returns > 0
                        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                        download_data['is_up_month'] = portfolio_returns.index.to_period('M').map(
                            lambda x: x in monthly_returns[monthly_returns > 0].index
                        )
                    
                    # Convert to CSV
                    csv_buffer = io.StringIO()
                    download_data.to_csv(csv_buffer)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download Daily Returns (CSV)",
                        data=csv_data,
                        file_name=f"backtest_results_{start_date}_{end_date}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("💡 Data export is not available for your account level. Please upgrade for CSV download functionality.")
                
            except Exception as e:
                st.error(f"An error occurred during backtesting: {str(e)}")

# Authentication and main app logic
# Check authentication and run appropriate function
if not check_authentication():
    login_form()
else:
    main()
