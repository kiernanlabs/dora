# Dora Bot Streamlit Dashboard

A real-time dashboard for monitoring the Dora Bot trading system. This Streamlit application provides read-only access to DynamoDB tables to visualize trading activity, positions, P&L, and execution logs.

## Features

### Home Page
- **P&L Over Time**: Interactive chart showing cumulative and daily P&L
- **Exposure by Market**: Visual breakdown of position exposure across markets
- **Recent Activity**: Most recent decision and execution log entries
- **Active Markets Table**: Comprehensive view of all active markets with:
  - Current order book (best bid/ask, spread)
  - Net positions
  - 24-hour position changes
  - Realized P&L and 24-hour P&L changes
  - Click-through to market deep dive

### Market Deep Dive Page
- **Decision Logs**: Historical record of strategy decisions including:
  - Order book snapshots
  - Inventory levels
  - Target quotes generated
  - Drill-down into individual decisions
- **Execution Logs**: Detailed execution history with:
  - Order placement, results, and cancellations
  - Fill events
  - Latency tracking
  - Error information
- **Analytics**:
  - Execution timeline by event type
  - Latency distribution charts
  - Performance metrics (avg, P50, P95 latency)

## Installation

1. Install dependencies:
```bash
cd /home/joe/dora/kalshi/dora_streamlit_ui
pip install -r requirements.txt
```

2. Configure AWS credentials:
```bash
# Ensure you have AWS credentials configured
aws configure
# OR set environment variables:
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your browser (typically at http://localhost:8501).

### Navigation

1. **Sidebar Controls**:
   - **Environment**: Switch between `demo` and `prod` environments
   - **AWS Region**: Select the AWS region (default: us-east-1)
   - **Page Selection**: Navigate between Home and Market Deep Dive pages

2. **Home Page**:
   - View overall P&L and exposure metrics
   - Monitor all active markets
   - Click on any market in the table and press "View Market Deep Dive" to drill down

3. **Market Deep Dive**:
   - Select a market from the dropdown
   - Choose a lookback period (1-30 days)
   - Browse decision logs, execution logs, and analytics in separate tabs

## Deployment to Streamlit Cloud

To deploy this dashboard to Streamlit Cloud:

1. **Push to GitHub**: Ensure your code is pushed to a GitHub repository

2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch, and set the main file path to `kalshi/dora_streamlit_ui/app.py`

3. **Configure Secrets**:
   - In your Streamlit Cloud app settings, go to "Secrets"
   - Add your AWS credentials in TOML format:

```toml
AWS_ACCESS_KEY_ID = "your_access_key_id"
AWS_SECRET_ACCESS_KEY = "your_secret_access_key"
```

4. **Deploy**: Click "Deploy" and your app will be live!

### Local Testing with Secrets

To test locally with the same secrets format, create a `.streamlit/secrets.toml` file:

```bash
mkdir -p .streamlit
cat > .streamlit/secrets.toml << EOF
AWS_ACCESS_KEY_ID = "your_access_key_id"
AWS_SECRET_ACCESS_KEY = "your_secret_access_key"
EOF
```

**Note**: The `.streamlit/` directory is already in `.gitignore` to prevent accidentally committing secrets.

## AWS Permissions

The application requires **read-only** access to the following DynamoDB tables:

- `dora_market_config_{demo|prod}`
- `dora_state_{demo|prod}`
- `dora_trade_log_{demo|prod}`
- `dora_decision_log_{demo|prod}`
- `dora_execution_log_{demo|prod}`

### Recommended IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan"
      ],
      "Resource": [
        "arn:aws:dynamodb:us-east-1:*:table/dora_*_demo",
        "arn:aws:dynamodb:us-east-1:*:table/dora_*_prod",
        "arn:aws:dynamodb:us-east-1:*:table/dora_*_demo/index/*",
        "arn:aws:dynamodb:us-east-1:*:table/dora_*_prod/index/*"
      ]
    }
  ]
}
```

## Architecture

```
dora_streamlit_ui/
├── app.py                  # Main application entry point
├── db_client.py           # Read-only DynamoDB client
├── views/
│   ├── __init__.py
│   ├── home.py            # Home page dashboard
│   └── market_deep_dive.py # Market detail page
├── requirements.txt       # Python dependencies
├── run.sh                 # Convenience script to run locally
├── .gitignore            # Git ignore file (includes .streamlit/)
└── README.md             # This file
```

## Data Flow

1. **AWS Credentials** → Either from Streamlit secrets (Cloud) or default credential chain (local)
2. **DynamoDB Tables** → Read-only queries via boto3
3. **db_client.py** → Provides clean API for data access with automatic credential detection
4. **Views** → Render data using Streamlit components
5. **Plotly** → Interactive charts and visualizations
6. **Pandas** → Data manipulation and table display

## Performance Considerations

- **Caching**: Consider adding `@st.cache_data` decorators for expensive queries
- **Pagination**: Large result sets may need pagination for better performance
- **Refresh Rate**: Use the refresh button to reload data on demand
- **Query Optimization**: Queries use date-based partitioning for efficiency

## Troubleshooting

### Connection Issues
```
Error: Unable to locate credentials
```
**Solution**: Configure AWS credentials using `aws configure` or environment variables

### Empty Data
```
No positions/markets/logs found
```
**Solution**:
- Verify the environment (demo/prod) matches your data
- Check that the bot is running and writing to DynamoDB
- Verify table names match the expected format

### Slow Queries
**Solution**:
- Reduce the lookback period
- Consider adding DynamoDB read capacity if needed
- Use date-based filtering when possible

## Development

To extend the dashboard:

1. **Add a new metric**: Update `db_client.py` with a new query method
2. **Add a new page**: Create a new file in `pages/` directory
3. **Modify visualization**: Update the respective page component
4. **Add caching**: Use `@st.cache_data` for expensive operations

## Future Enhancements

- [ ] Real-time updates with auto-refresh
- [ ] Trade execution reconciliation view
- [ ] Risk metrics and alerts
- [ ] Export data to CSV/Excel
- [ ] Historical backtesting visualization
- [ ] Multi-market comparison views
- [ ] Custom date range selectors

## Support

For issues or questions:
1. Check the main Dora Bot documentation in `/home/joe/dora/kalshi/dora_bot/README.md`
2. Review the logging project plan in `/home/joe/dora/kalshi/logging_project_plan.md`
3. Verify AWS permissions and connectivity
