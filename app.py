import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io

# Set page config
st.set_page_config(
    page_title="Portfolio Backtesting Tool",
    page_icon="📈",
    layout="wide"
)

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
        monthly_ends = pd.date_range(start=index.min(), end=index.max(), freq='M')
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
    st.title("📈 Portfolio Backtesting Tool")
    st.markdown("A comprehensive tool for backtesting portfolio strategies with historical data from Yahoo Finance.")
    
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
    portfolio_data = portfolio_data[portfolio_data['Ticker'].str.strip() != '']  # Remove empty tickers
    portfolio_data = portfolio_data.drop_duplicates(subset='Ticker')  # Remove duplicates
    portfolio_data['Ticker'] = portfolio_data['Ticker'].str.upper()  # Convert to uppercase
    
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
                
                # CSV download
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
                
            except Exception as e:
                st.error(f"An error occurred during backtesting: {str(e)}")

if __name__ == "__main__":
    main()
