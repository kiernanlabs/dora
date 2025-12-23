import os
from typing import Optional, Dict, Any, List
from cryptography.hazmat.primitives import serialization
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from openai import OpenAI

# Import from dora_bot
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'dora_bot'))
from kalshi_client import KalshiHttpClient, Environment

class KalshiService:
    """Service class for interacting with Kalshi API."""

    def __init__(self, use_demo: bool = True):
        """Initialize the Kalshi service.

        Args:
            use_demo: If True, use demo environment. Otherwise use production.
        """
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

        self.env = Environment.DEMO if use_demo else Environment.PROD
        keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
        keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')
        if keyfile:
            keyfile = os.path.expanduser(keyfile)
            if not os.path.isabs(keyfile):
                cwd_candidate = os.path.abspath(keyfile)
                if os.path.exists(cwd_candidate):
                    keyfile = cwd_candidate
                else:
                    module_candidate = os.path.join(os.path.dirname(__file__), keyfile)
                    if os.path.exists(module_candidate):
                        keyfile = module_candidate

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

        # Set price based on yes_price (this is the raw price from API)
        if 'yes_price' in df.columns:
            df['price'] = df['yes_price']

        # Determine trade action (buy/sell) and calculate signed volume
        # Translate "Buy No" to "Sell Yes" with adjusted pricing
        if 'taker_side' in df.columns and 'count' in df.columns:
            # taker_side indicates who initiated the trade: 'yes' or 'no'
            # For market risk analysis:
            # - taker_side='yes' means someone bought yes at yes_price (positive pressure)
            # - taker_side='no' means someone bought no at (100 - yes_price) = selling yes at yes_price (negative pressure)
            df['action'] = df['taker_side'].apply(lambda x: 'Buy Yes' if x == 'yes' else 'Sell Yes')
            df['signed_volume'] = df.apply(
                lambda row: row['count'] if row['taker_side'] == 'yes' else -row['count'],
                axis=1
            )
            # Adjusted price:
            # - Buy Yes: use yes_price directly
            # - Sell Yes: also use yes_price (they're selling Yes at this price, which equals buying No at 100-yes_price)
            df['adjusted_price'] = df['price']  # Same for both, since API gives us yes_price

        elif 'side' in df.columns and 'count' in df.columns:
            # Fallback to 'side' if 'taker_side' is not available
            df['action'] = df['side'].apply(lambda x: 'Buy Yes' if x == 'yes' else 'Sell Yes')
            df['signed_volume'] = df.apply(
                lambda row: row['count'] if row['side'] == 'yes' else -row['count'],
                axis=1
            )
            # Adjusted price: use yes_price for both
            df['adjusted_price'] = df['price']

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

    def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get market metadata for a specific ticker.

        Args:
            ticker: The market ticker symbol

        Returns:
            Dictionary containing market metadata (title, resolution criteria, dates, etc.)
        """
        return self.client.get(f"/trade-api/v2/markets/{ticker}")

    def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """Get current orderbook for a specific ticker.

        Args:
            ticker: The market ticker symbol

        Returns:
            Dictionary containing orderbook data with yes/no bid levels
        """
        return self.client.get(f"/trade-api/v2/markets/{ticker}/orderbook")

    def calculate_trade_risk_metrics(
        self,
        df: pd.DataFrame,
        risk_aversion_k: float = 0.5
    ) -> Dict[str, Any]:
        """Calculate risk metrics for trades including exit prices, P&L, adverse selection, and inventory risk.

        For each trade, looks forward to find the first trade with the same sign (Buy/Sell)
        over the following 24 hours, excluding trades within 10 minutes. This is the "exit price".

        Args:
            df: DataFrame with trade data (must have timestamp, action, price, count columns)
            risk_aversion_k: Risk aversion constant for inventory risk calculation (default 0.5)

        Returns:
            Dictionary containing:
            - df_with_metrics: DataFrame with exit_price and pnl columns added
            - adverse_selection_per_unit: Average P&L across all trades
            - inventory_risk_per_unit: Standard deviation of P&L * risk_aversion_k
            - required_half_spread: Required half-spread to cover costs
            - required_full_spread: Required full spread (2x half-spread)
        """
        if df.empty or 'timestamp' not in df.columns or 'action' not in df.columns:
            return {
                'df_with_metrics': df,
                'adverse_selection_per_unit': 0,
                'inventory_risk_per_unit': 0,
                'required_half_spread': 0,
                'required_full_spread': 0,
                'num_trades_with_exits': 0,
                'num_trades_total': 0
            }

        # Create a copy to add new columns
        df_result = df.copy()
        df_result['exit_price'] = np.nan
        df_result['exit_time'] = pd.Series(dtype='datetime64[ns, UTC]')
        df_result['pnl'] = np.nan
        df_result['trade_sign'] = df_result['action'].apply(lambda x: 1 if x == 'Buy Yes' else -1)

        # For each trade, find the exit price
        pnls = []

        for idx, trade in df_result.iterrows():
            trade_time = trade['timestamp']
            trade_action = trade['action']
            trade_price = trade['price']
            trade_sign = trade['trade_sign']

            # Look forward 24 hours, excluding first 10 minutes
            min_time = trade_time + pd.Timedelta(minutes=10)
            max_time = trade_time + pd.Timedelta(hours=24)

            # Find future trades with same action
            future_trades = df_result[
                (df_result['timestamp'] > min_time) &
                (df_result['timestamp'] <= max_time) &
                (df_result['action'] == trade_action)
            ]

            if len(future_trades) > 0:
                # Take the first matching trade as exit
                exit_trade = future_trades.iloc[0]
                exit_price = exit_trade['price']
                exit_time = exit_trade['timestamp']

                # Calculate P&L for the maker of the original trade
                # If original trade was a buy (trade_sign = 1), maker sold at trade_price and buys back at exit_price
                # If original trade was a sell (trade_sign = -1), maker bought at trade_price and sells at exit_price
                # pnl = -trade_sign * (exit_price - trade_price)
                pnl = -trade_sign * (exit_price - trade_price)

                df_result.at[idx, 'exit_price'] = float(exit_price)
                df_result.at[idx, 'exit_time'] = pd.Timestamp(exit_time)
                df_result.at[idx, 'pnl'] = float(pnl)
                pnls.append(pnl)

        # Calculate risk metrics
        if len(pnls) > 0:
            adverse_selection_per_unit = np.mean(pnls)
            inventory_risk_per_unit = np.std(pnls) * risk_aversion_k
        else:
            adverse_selection_per_unit = 0
            inventory_risk_per_unit = 0

        # Calculate required spread
        required_half_spread = adverse_selection_per_unit + inventory_risk_per_unit
        required_full_spread = 2.0 * required_half_spread

        return {
            'df_with_metrics': df_result,
            'adverse_selection_per_unit': adverse_selection_per_unit,
            'inventory_risk_per_unit': inventory_risk_per_unit,
            'required_half_spread': required_half_spread,
            'required_full_spread': required_full_spread,
            'num_trades_with_exits': len(pnls),
            'num_trades_total': len(df_result)
        }

    def assess_information_risk(
        self,
        market_title: str,
        current_price: float,
        market_subtitle: Optional[str] = None,
        rules: Optional[str] = None
    ) -> Dict[str, Any]:
        """Assess the likelihood of market-moving information being released in the next 7 days.

        Uses OpenAI API to evaluate information risk for a prediction market.

        Args:
            market_title: The title of the market
            current_price: Current market price (0-100)
            market_subtitle: Optional subtitle providing additional context
            rules: Optional resolution rules for the market

        Returns:
            Dictionary containing:
            - probability: Likelihood percentage (0-100)
            - rationale: 2-3 sentence explanation
            - error: Error message if API call fails
        """
        try:
            # Get OpenAI API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    'probability': None,
                    'rationale': 'OpenAI API key not configured',
                    'error': 'Missing OPENAI_API_KEY in environment variables'
                }

            # Initialize OpenAI client
            client = OpenAI(api_key=api_key)

            # Build context for the prompt
            context = f"The market is: {market_title}"
            if market_subtitle:
                context += f"\n{market_subtitle}"
            if rules:
                context += f"\n\nResolution rules: {rules[:500]}"  # Limit rules length
            context += f"\n\nThe current price is ~{current_price:.0f}%"

            # Create the prompt
            prompt = f"""You are a market risk assessment expert for prediction markets on Kalshi. Your job is to evaluate the likelihood that market moving information will be released in the next 7 days that would move the current pricing more than 20% in either direction.

Please return your assessment in the form of a likelihood percentage (number from 0-100%) and 2-3 sentence rationale.

{context}

Your response should be only a JSON dictionary e.g. {{"probability": "XX%", "rationale": "XXXX"}}"""

            # Call OpenAI API with gpt-5.1 using medium reasoning and web search tool access
            response = client.responses.create(
                model="gpt-5.1",
                reasoning={"effort": "medium"},
                tools=[{"type": "web_search"}],
                input=prompt,
            )

            # Parse the response
            response_text = getattr(response, "output_text", None)
            if not response_text:
                # Fallback to pulling from structured content if output_text is unavailable
                try:
                    response_text = response.output[0].content[0].text
                except Exception:
                    raise ValueError("Unable to parse response content from OpenAI API")
            response_text = response_text.strip()

            # Try to extract JSON if wrapped in markdown code blocks
            if response_text.startswith('```'):
                # Remove markdown code blocks
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            result = json.loads(response_text)

            return {
                'probability': result.get('probability', 'N/A'),
                'rationale': result.get('rationale', 'No rationale provided'),
                'error': None
            }

        except json.JSONDecodeError as e:
            return {
                'probability': None,
                'rationale': f'Failed to parse API response: {str(e)}',
                'error': 'JSON parsing error'
            }
        except Exception as e:
            return {
                'probability': None,
                'rationale': f'Error calling OpenAI API: {str(e)}',
                'error': str(e)
            }
