# Portfolio Backtesting Tool

A comprehensive web application for backtesting investment portfolio strategies using historical stock market data. Built with Streamlit, this tool provides professional-grade portfolio analysis with interactive visualizations and detailed performance metrics.

## Features

### Core Functionality
- **Historical Data Fetching**: Automatically downloads stock price data from Yahoo Finance using the yfinance library
- **Portfolio Simulation**: Simulates portfolio performance with configurable rebalancing strategies
- **Performance Analytics**: Calculates key metrics including total return, CAGR, volatility, Sharpe ratio, and maximum drawdown
- **Interactive Visualizations**: Plotly-powered charts for equity curves and drawdown analysis
- **Transaction Cost Modeling**: Incorporates realistic transaction costs based on portfolio turnover
- **Benchmark Comparison**: Compare portfolio performance against SPY or QQQ benchmarks

### User Interface
- **Interactive Portfolio Editor**: Easy-to-use table for inputting ticker symbols and weights
- **Flexible Parameters**: Configurable sampling frequency, rebalancing frequency, transaction costs, and risk-free rates
- **Real-time Results**: Instant backtesting with comprehensive performance statistics
- **Data Export**: Download detailed results as CSV files for further analysis

## Getting Started

### Prerequisites
This application requires the following Python packages:
- streamlit
- pandas
- numpy
- yfinance
- plotly

### Running the Application

1. **Local Development**:
   ```bash
   streamlit run app.py --server.port 3000 --server.address 0.0.0.0
   ```

2. **Replit Environment**:
   The application is configured to run automatically in Replit using the `.replit` configuration file.

### Using the Application

1. **Set Parameters**: Use the sidebar to configure:
   - Start and end dates for the backtest period
   - Sampling frequency (Daily, Weekly, Monthly)
   - Rebalancing frequency (None, Monthly, Quarterly, Yearly)
   - Transaction costs in basis points
   - Annual risk-free rate for Sharpe ratio calculation
   - Benchmark for comparison (None, SPY, QQQ)

2. **Build Your Portfolio**: 
   - Use the interactive table to add ticker symbols and weights
   - Weights are automatically normalized to sum to 100%
   - Empty tickers and duplicates are automatically filtered out
   - Supports any Yahoo Finance symbol (stocks, ETFs, indices)

3. **Run Backtest**: Click the "Run Backtest" button to:
   - Fetch historical price data
   - Simulate portfolio performance with rebalancing
   - Calculate comprehensive performance metrics
   - Generate interactive charts
   - Provide downloadable results

## How It Works

### Weight Normalization
Portfolio weights are automatically normalized to ensure they sum to 100%. For example:
- Input: AAPL: 40%, MSFT: 30%, GOOGL: 20%
- Normalized: AAPL: 44.4%, MSFT: 33.3%, GOOGL: 22.2%

### Rebalancing Strategies
- **None**: Buy and hold strategy with no rebalancing
- **Monthly**: Rebalance on the last trading day of each month
- **Quarterly**: Rebalance on the last trading day of each quarter
- **Yearly**: Rebalance on the last trading day of each year

### Transaction Costs
Transaction costs are calculated based on portfolio turnover:
- Turnover = Sum of absolute differences between current and target weights
- Cost = Turnover × (Basis Points / 10,000)
- Applied only on rebalancing dates

### Performance Metrics
- **Total Return**: Cumulative return over the entire period
- **CAGR**: Compound Annual Growth Rate
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return using the specified risk-free rate
- **Max Drawdown**: Largest peak-to-trough decline

## Supported Assets

The application works with any symbol available on Yahoo Finance, including:
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, etc.
- **ETFs**: SPY, QQQ, VTI, BND, etc.
- **International Markets**: ASML (NASDAQ), 7203.T (Tokyo), etc.
- **Cryptocurrencies**: BTC-USD, ETH-USD, etc.
- **Commodities**: GC=F (Gold), CL=F (Oil), etc.

## Technical Details

### Data Handling
- Automatically handles missing data and invalid tickers
- Uses adjusted close prices when available, falls back to close prices
- Graceful error handling for data fetching issues
- Caching implemented to avoid repeated downloads during the same session

### Performance Optimization
- `@st.cache_data` decorator for efficient data fetching
- Optimized pandas operations for large datasets
- Responsive UI with real-time feedback

## Future Enhancements

Potential improvements and feature additions:

### Advanced Rebalancing
- **Drift-band Rebalancing**: Rebalance only when weights drift beyond specified thresholds
- **Volatility-based Rebalancing**: Adjust rebalancing frequency based on market volatility
- **Tax-aware Rebalancing**: Optimize for tax efficiency in taxable accounts

### Analytics Enhancements
- **Rolling Metrics**: Calculate performance metrics over rolling time windows
- **Risk Analytics**: Value at Risk (VaR), Conditional VaR, maximum entropy portfolios
- **Factor Analysis**: Exposure to common risk factors (market, size, value, momentum)

### Portfolio Management
- **Multiple Portfolio Comparison**: Compare different portfolio strategies side-by-side
- **Save/Load Portfolios**: Persistent storage for favorite portfolio configurations
- **Optimization Tools**: Mean-variance optimization, Black-Litterman model integration

### Data and Universe
- **Quick-select Lists**: Pre-built lists of S&P 500, NASDAQ 100, or sector constituents
- **Alternative Data**: Integration with fundamental data, earnings estimates, insider trading
- **International Markets**: Enhanced support for global markets and currencies

### User Experience
- **Advanced Charting**: Additional chart types, technical indicators, correlation heatmaps
- **Report Generation**: PDF reports with professional formatting
- **API Integration**: Connect with brokerage APIs for live portfolio tracking

## Error Handling

The application includes comprehensive error handling for:
- Invalid ticker symbols
- Insufficient historical data
- Network connectivity issues
- Invalid date ranges
- Empty portfolios
- Data processing errors

## Limitations

- Data is sourced from Yahoo Finance and subject to their terms of service
- Historical data may have survivorship bias
- Transaction cost model is simplified and may not reflect all real-world costs
- Past performance does not guarantee future results

## Support

For issues, suggestions, or contributions, please refer to the application's error messages which provide specific guidance for resolution.

---

**Disclaimer**: This tool is for educational and research purposes only. It should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions.
