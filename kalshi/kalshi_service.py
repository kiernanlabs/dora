import os
from typing import Optional, Dict, Any, List
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
import pandas as pd

# Import from the starter code
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'kalshi-starter-code-python-main'))
from clients import KalshiHttpClient, Environment

class KalshiService:
    """Service class for interacting with Kalshi API."""

    def __init__(self, use_demo: bool = True):
        """Initialize the Kalshi service.

        Args:
            use_demo: If True, use demo environment. Otherwise use production.
        """
        load_dotenv()

        self.env = Environment.DEMO if use_demo else Environment.PROD
        keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
        keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

        if not keyid or not keyfile:
            raise ValueError(f"Missing API credentials. Please set {'DEMO' if use_demo else 'PROD'}_KEYID and {'DEMO' if use_demo else 'PROD'}_KEYFILE in .env file")

        try:
            with open(keyfile, "rb") as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Private key file not found at {keyfile}")
        except Exception as e:
            raise Exception(f"Error loading private key: {str(e)}")

        self.client = KalshiHttpClient(
            key_id=keyid,
            private_key=private_key,
            environment=self.env
        )

    def get_trades(
        self,
        ticker: str,
        limit: int = 1000,
        cursor: Optional[str] = None,
        max_ts: Optional[int] = None,
        min_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get trades for a specific market ticker.

        Args:
            ticker: The market ticker symbol
            limit: Maximum number of trades to retrieve (default 1000)
            cursor: Pagination cursor
            max_ts: Maximum timestamp filter
            min_ts: Minimum timestamp filter

        Returns:
            Dictionary containing trades data from API
        """
        return self.client.get_trades(
            ticker=ticker,
            limit=limit,
            cursor=cursor,
            max_ts=max_ts,
            min_ts=min_ts
        )

    def get_trades_dataframe(
        self,
        ticker: str,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """Get trades as a pandas DataFrame with computed fields.

        Args:
            ticker: The market ticker symbol
            limit: Maximum number of trades to retrieve (default 1000)

        Returns:
            DataFrame with columns: timestamp, price, volume, side, signed_volume, etc.
        """
        result = self.get_trades(ticker=ticker, limit=limit)
        trades = result.get('trades', [])

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Convert timestamp to datetime
        if 'created_time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['created_time'])

        # Set price based on yes_price
        if 'yes_price' in df.columns:
            df['price'] = df['yes_price']

        # Determine trade action (buy/sell) and calculate signed volume
        if 'taker_side' in df.columns and 'count' in df.columns:
            # taker_side indicates who initiated the trade: 'yes' or 'no'
            # For market risk analysis:
            # - taker_side='yes' means someone bought yes (positive pressure)
            # - taker_side='no' means someone bought no (negative pressure)
            df['action'] = df['taker_side'].apply(lambda x: 'Buy Yes' if x == 'yes' else 'Buy No')
            df['signed_volume'] = df.apply(
                lambda row: row['count'] if row['taker_side'] == 'yes' else -row['count'],
                axis=1
            )
        elif 'side' in df.columns and 'count' in df.columns:
            # Fallback to 'side' if 'taker_side' is not available
            df['action'] = df['side'].apply(lambda x: 'Buy Yes' if x == 'yes' else 'Buy No')
            df['signed_volume'] = df.apply(
                lambda row: row['count'] if row['side'] == 'yes' else -row['count'],
                axis=1
            )

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        return df

    def get_balance(self) -> Dict[str, Any]:
        """Get account balance.

        Returns:
            Dictionary containing balance information
        """
        return self.client.get_balance()

    def get_exchange_status(self) -> Dict[str, Any]:
        """Get exchange status.

        Returns:
            Dictionary containing exchange status
        """
        return self.client.get_exchange_status()
