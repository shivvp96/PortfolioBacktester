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
    layout="wide"
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

def signup_form():
    """Display signup form"""
    st.title("📝 Create New Account")
    st.markdown("Sign up for a new account to access the portfolio backtesting tool.")
    
    with st.form("signup_form"):
        st.subheader("Sign Up")
        new_username = st.text_input("Choose Username")
        new_email = st.text_input("Email Address")
        new_password = st.text_input("Choose Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        # Terms and conditions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 View Terms of Service"):
                st.session_state['show_terms'] = True
        with col2:
            if st.button("🔒 View Privacy Policy"):
                st.session_state['show_privacy'] = True
        
        agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        
        # Show terms modal
        if st.session_state.get('show_terms', False):
            with st.expander("📋 Terms of Service", expanded=True):
                st.markdown(get_terms_of_service())
                if st.button("Close Terms"):
                    st.session_state['show_terms'] = False
                    st.rerun()
        
        # Show privacy modal  
        if st.session_state.get('show_privacy', False):
            with st.expander("🔒 Privacy Policy", expanded=True):
                st.markdown(get_privacy_policy())
                if st.button("Close Privacy Policy"):
                    st.session_state['show_privacy'] = False
                    st.rerun()
        
        signup_button = st.form_submit_button("Create Account")
        
        if signup_button:
            # Validation
            registered_users = load_registered_users()
            all_users = {**DEMO_USERS, **registered_users}
            
            if not new_username or not new_email or not new_password:
                st.error("Please fill in all fields.")
                return
            
            if new_username in all_users:
                st.error("Username already exists. Please choose a different username.")
                return
            
            if len(new_username) < 3:
                st.error("Username must be at least 3 characters long.")
                return
            
            if len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
                return
            
            # Enhanced password validation
            if not any(c.isdigit() for c in new_password):
                st.warning("Password should contain at least one number for better security.")
            
            if not any(c.isupper() for c in new_password):
                st.warning("Password should contain at least one uppercase letter for better security.")
            
            if new_password != confirm_password:
                st.error("Passwords do not match.")
                return
            
            if '@' not in new_email:
                st.error("Please enter a valid email address.")
                return
            
            if not agree_terms:
                st.error("Please agree to the Terms of Service.")
                return
            
            # Save new user
            try:
                save_user_to_file(new_username, new_password, new_email, "user")
                st.success("🎉 Welcome to Portfolio Backtesting Tool!")
                st.info(f"""
                **Account Created Successfully!**
                
                👤 Username: {new_username}
                📧 Email: {new_email}
                🎯 Role: Regular User
                
                **Your Benefits:**
                - Create up to 5 portfolios
                - Advanced interactive charts
                - CSV data export
                - Full benchmark comparison
                
                You can now switch to the Login tab to access your account.
                """)
                st.session_state['show_signup'] = False
                st.session_state['signup_success'] = True
            except Exception as e:
                st.error(f"Error creating account: {str(e)}")

def login_form():
    """Display login form with signup option"""
    # Initialize session state
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    
    # Show success message if just signed up
    if st.session_state.get('signup_success', False):
        st.session_state['signup_success'] = False
        st.session_state.show_signup = False
    
    # Toggle between login and signup
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔐 Login", type="primary" if not st.session_state.show_signup else "secondary"):
            st.session_state.show_signup = False
            st.rerun()
    with col2:
        if st.button("📝 Sign Up", type="primary" if st.session_state.show_signup else "secondary"):
            st.session_state.show_signup = True
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.show_signup:
        signup_form()
    else:
        st.title("🔐 Portfolio Backtesting Tool - Login")
        st.markdown("Welcome! Please log in to access the portfolio backtesting tool.")
        
        with st.form("login_form"):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if st.session_state.login_attempts >= 5:
                    st.error("Too many login attempts. Please refresh the page to try again.")
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
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.session_state.login_attempts += 1
                        st.error("Invalid password")
                else:
                    st.session_state.login_attempts += 1
                    st.error("Invalid username")
        
        # Demo credentials info
        # Show registered users count
        registered_users = load_registered_users()
        if registered_users:
            st.success(f"✅ {len(registered_users)} registered user(s) in the system")
        
        with st.expander("📋 Demo Credentials"):
            st.info("""
            **Available Demo Accounts:**
            
            🔵 **Demo User** (Limited features)
            - Username: `demo` | Password: `demo`
            
            🟢 **Regular User** (Standard features)  
            - Username: `user` | Password: `password123`
            
            🟡 **Analyst** (Advanced features)
            - Username: `analyst` | Password: `analyst2024`
            
            🔴 **Administrator** (Full access)
            - Username: `admin` | Password: `admin123`
            """)

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

def simulate_portfolio(returns, target_weights, rebalance, fees_bps):
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
            # Calculate turnover for transaction costs
            target_weights_series = pd.Series(target_weights, index=returns.columns)
            target_weights_series = target_weights_series / target_weights_series.sum()
            
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

def plot_equity_curve(port_ret, bench_ret=None):
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
            name='Portfolio',
            line=dict(color='blue', width=2)
        ))
    
    if bench_ret is not None and not bench_ret.empty:
        benchmark_growth = (1 + bench_ret).cumprod()
        fig.add_trace(go.Scatter(
            x=benchmark_growth.index,
            y=benchmark_growth.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title='Portfolio Equity Curve (Growth of $1)',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        showlegend=True
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
    # Header with user info and logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 Portfolio Backtesting Tool")
        st.markdown("A comprehensive tool for backtesting portfolio strategies with historical data from Yahoo Finance.")
    with col2:
        if check_authentication():
            user_role = get_user_role()
            role_emoji = {"administrator": "🔴", "analyst": "🟡", "user": "🟢", "demo": "🔵"}
            st.write(f"{role_emoji.get(user_role, '👤')} **{st.session_state.get('username', 'User')}** ({user_role.title()})")
            if st.button("🚪 Logout", key="logout_btn"):
                logout()
    
    # Sidebar for parameters
    st.sidebar.header("Backtest Parameters")
    
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
        options=['None', 'SPY', 'QQQ'],
        index=0
    )
    
    # Portfolio composition
    st.header("Portfolio Composition")
    
    # Initialize session state for portfolio data
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = pd.DataFrame({
            'Ticker': ['AAPL', 'MSFT', 'GOOGL', ''],
            'Weight': [0.4, 0.3, 0.3, 0.0]
        })
    
    # Show role-based information
    permissions = get_user_permissions()
    max_portfolios = permissions.get('max_portfolios')
    if max_portfolios:
        st.info(f"📊 Your account allows up to {max_portfolios} portfolio simulations. Advanced features: {'✅' if permissions.get('advanced_charts') else '❌'}")
    
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
        
        # Display normalized weights
        st.subheader("Normalized Portfolio Weights")
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
                
                # Run portfolio simulation
                portfolio_returns, weight_history, drawdown = simulate_portfolio(
                    returns, target_weights, rebalance_freq, transaction_costs
                )
                
                # Fetch benchmark data if selected
                benchmark_returns = None
                if benchmark != 'None':
                    bench_prices = fetch_prices([benchmark], start_date, end_date)
                    if not bench_prices.empty:
                        benchmark_returns = compute_returns(bench_prices, frequency)[benchmark]
                
                # Calculate performance statistics
                portfolio_stats = performance_stats(portfolio_returns, risk_free_rate, frequency)
                
                # Display results
                st.header("📊 Backtest Results")
                
                # Performance metrics
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
                equity_fig = plot_equity_curve(portfolio_returns, benchmark_returns)
                st.plotly_chart(equity_fig, use_container_width=True)
                
                st.subheader("📉 Drawdown Chart")
                drawdown_fig = plot_drawdown(drawdown)
                st.plotly_chart(drawdown_fig, use_container_width=True)
                
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
                    
                    if benchmark_returns is not None:
                        download_data['Benchmark_Return'] = benchmark_returns
                        download_data['Benchmark_Cumulative'] = (1 + benchmark_returns).cumprod()
                    
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
