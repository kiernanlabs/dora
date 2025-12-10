# Kalshi Market Risk Analysis Dashboard

An exploratory Streamlit dashboard for analyzing market risk and trade patterns on Kalshi prediction markets.

## Features

- **Real-time Market Data**: Fetch up to 2000 recent trades for any market ticker
- **Trade Price Visualization**: Track price movements over time
- **Signed Volume Analysis**: Visualize buy/sell pressure (Yes trades = positive, No trades = negative)
- **Market Statistics**: View comprehensive price and volume statistics
- **Demo & Production Support**: Toggle between demo and production environments

## Setup

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

### 2. Configure API Credentials

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Get your Kalshi API credentials:
   - Sign up at [Kalshi](https://kalshi.com) or [Kalshi Demo](https://demo.kalshi.com)
   - Generate API keys from your account settings
   - Download your private key file

3. Edit `.env` and add your credentials:
```env
DEMO_KEYID=your_actual_key_id
DEMO_KEYFILE=/path/to/your/private_key.pem
```

### 3. Run the Application

```bash
streamlit run market_risk_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage

1. **Select Environment**: Choose Demo or Production mode in the sidebar
2. **Enter Market Ticker**: Input a valid Kalshi market ticker (e.g., `INXD-25JAN31-T4850`)
3. **Set Trade Limit**: Choose how many recent trades to fetch (100-2000)
4. **Fetch Data**: Click the "Fetch Data" button to load and visualize the market

## Dashboard Components

### Market Summary
- Total number of trades
- Latest trade price
- Total and net volume metrics

### Visualizations
- **Trade Price Over Time**: Line chart showing price evolution
- **Signed Volume Over Time**: Bar chart with color-coded buy/sell activity
  - Green bars = Yes/Buy trades
  - Red bars = No/Sell trades

### Statistics
- Price statistics (min, max, mean, median, std dev)
- Volume statistics (min, max, mean, median)
- Raw trade data table (expandable)

## Project Structure

```
kalshi/
├── kalshi-starter-code-python-main/  # Original Kalshi API client code
│   ├── clients.py                     # HTTP and WebSocket clients
│   └── main.py                        # Example usage
├── kalshi_service.py                  # Service layer for API interactions
├── market_risk_app.py                 # Main Streamlit application
├── .env.example                       # Environment variables template
└── README.md                          # This file
```

## Technical Details

### Architecture

- **kalshi_service.py**: Wraps the Kalshi API client with convenience methods
  - Handles authentication and API initialization
  - Provides DataFrame-based trade data with computed fields
  - Calculates signed volume for directional analysis

- **market_risk_app.py**: Streamlit dashboard implementation
  - Interactive UI with sidebar controls
  - Plotly-based interactive charts
  - Real-time data fetching and visualization

### Data Fields

The application processes trade data with these key fields:
- `timestamp`: Trade execution time
- `price`: Yes price in cents
- `count`: Trade volume (number of contracts)
- `side`: Trade direction ('yes' or 'no')
- `signed_volume`: Calculated field (positive for yes, negative for no)

## Troubleshooting

### "Missing API credentials" Error
Make sure your `.env` file exists and contains valid `DEMO_KEYID` and `DEMO_KEYFILE` values.

### "Private key file not found" Error
Check that the path in `DEMO_KEYFILE` points to your actual private key PEM file.

### "No trades found" Error
- Verify the market ticker exists in your selected environment (Demo vs Prod)
- Some markets may not have recent trade activity
- Try a different ticker or check Kalshi's market listings

## Notes

- The demo environment may have limited market data
- API rate limits apply (100ms between requests is enforced)
- Trade data is fetched in real-time; historical data depends on API availability
