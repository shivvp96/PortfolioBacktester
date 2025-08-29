# Portfolio Backtesting Tool

## Overview

This is a comprehensive portfolio backtesting application built with Streamlit that allows users to analyze investment portfolio strategies using historical stock market data. The tool provides professional-grade portfolio analysis capabilities including performance metrics calculation, benchmark comparison, transaction cost modeling, and interactive visualizations. Users can configure portfolios with custom ticker symbols and weights, set rebalancing frequencies, and analyze results through detailed charts and statistics.

**Authentication System** - The application includes a secure login system with role-based access control, allowing different user types (Demo, User, Analyst, Administrator) to access features based on their permission levels.

**Advanced Features** - The platform now includes sophisticated portfolio management tools:
- **Volatility Targeting Overlay**: Dynamically scales portfolio returns to achieve a target annual volatility using rolling window estimation and leverage constraints
- **Drift-Band Rebalancing**: Only rebalances assets when they drift outside specified tolerance bands, reducing transaction costs and turnover

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the primary web framework, providing a reactive single-page application structure. The UI is organized with a sidebar for parameter configuration and a main content area for portfolio input and results display. The interface includes:
- **Authentication Layer**: Secure login form with role-based access control
- Interactive data editor for portfolio composition
- Parameter controls in sidebar for backtest configuration
- Real-time result updates triggered by user interactions
- Plotly-based interactive charts for data visualization
- **User Dashboard**: Role-specific header showing user information and logout functionality

### Backend Architecture
The backend follows a **functional programming** approach with modular components:
- **Authentication Module**: Manages user login, session state, and role-based permissions
- **Data Fetching Layer**: Handles external API calls to Yahoo Finance with caching
- **Computation Engine**: Processes returns, simulates portfolios, and calculates performance metrics
- **Advanced Portfolio Management**: 
  - Volatility targeting with rolling window estimation and leverage capping
  - Drift-band rebalancing with configurable tolerance thresholds
  - Transaction cost optimization through reduced turnover
- **Visualization Layer**: Generates interactive charts using Plotly with overlay support
- **State Management**: Leverages Streamlit's built-in state management for reactive updates and user sessions
- **Authorization System**: Controls feature access based on user roles and permissions

### Data Processing Pipeline
The application implements a **sequential data processing pipeline**:
1. **Price Data Acquisition**: Fetches historical stock prices with error handling
2. **Return Calculation**: Converts prices to returns with configurable sampling frequency
3. **Portfolio Simulation**: Models portfolio performance with weight drift and advanced rebalancing strategies
4. **Volatility Overlay Processing**: Applies dynamic leverage scaling based on rolling volatility estimation
5. **Performance Analysis**: Computes comprehensive financial metrics for base and enhanced portfolios
6. **Comparison Analytics**: Side-by-side performance analysis between base and vol-targeted strategies
7. **Visualization Generation**: Creates interactive charts with multiple strategy overlays

### Caching Strategy
Implements **Streamlit's native caching** (`@st.cache_data`) for the price fetching function to minimize API calls and improve performance. This prevents redundant data downloads when users modify parameters without changing the underlying ticker symbols or date ranges.

### Error Handling
The system uses **graceful degradation** patterns:
- Invalid tickers are filtered out with user notifications
- Missing data is handled with fallback mechanisms (Adjusted Close to Close prices)
- API failures are caught and reported without breaking the application flow

## External Dependencies

### Data Services
- **Yahoo Finance API** (via yfinance library): Primary data source for historical stock prices, dividend-adjusted close prices, and market data
- **Market Benchmarks**: SPY and QQQ data for portfolio performance comparison

### Core Libraries
- **Streamlit**: Web application framework and UI components
- **yfinance**: Yahoo Finance API wrapper for financial data retrieval
- **pandas**: Data manipulation and time series analysis
- **numpy**: Numerical computations and array operations
- **plotly**: Interactive charting and data visualization

### Runtime Environment
- **Python 3.x**: Primary runtime environment
- **Replit**: Cloud hosting platform with automatic deployment configuration
- **Web Browser**: Client-side rendering of Streamlit interface and Plotly charts

The application is designed to be self-contained with minimal external service dependencies, relying primarily on Yahoo Finance for market data and Streamlit's built-in capabilities for the web interface.