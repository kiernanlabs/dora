"""Exchange client for interacting with Kalshi API."""

import sys
import os
import time
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from requests.exceptions import HTTPError, RequestException

# Add parent directory to path to import kalshi starter code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kalshi-starter-code-python-main'))
from clients import KalshiHttpClient, Environment

from models import OrderBook, Order, Fill, Balance

logger = logging.getLogger(__name__)


def parse_kalshi_timestamp(timestamp_str: str) -> datetime:
    """Parse Kalshi timestamp format handling variable microsecond precision.

    Kalshi sometimes returns timestamps with 5-digit microseconds (e.g., '2025-12-13T02:15:43.42496+00:00')
    which Python's fromisoformat() doesn't handle. This function pads microseconds to 6 digits.

    Args:
        timestamp_str: ISO format timestamp string from Kalshi API

    Returns:
        Parsed datetime object
    """
    # Replace 'Z' with '+00:00' for timezone
    timestamp_str = timestamp_str.replace('Z', '+00:00')

    # Handle variable-length microseconds
    if '.' in timestamp_str and '+' in timestamp_str:
        # Split into parts: datetime, microseconds, timezone
        date_part, rest = timestamp_str.split('.')
        microseconds, timezone = rest.split('+')

        # Pad or truncate microseconds to 6 digits
        microseconds = microseconds.ljust(6, '0')[:6]

        # Reconstruct timestamp
        timestamp_str = f"{date_part}.{microseconds}+{timezone}"

    return datetime.fromisoformat(timestamp_str)


class KalshiExchangeClient:
    """Wrapper around Kalshi API for market making operations."""

    def __init__(self, client: KalshiHttpClient, max_retries: int = 3, retry_delay: float = 1.0):
        """Initialize the exchange client.

        Args:
            client: Authenticated KalshiHttpClient instance
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries in seconds
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _retry_request(self, func, *args, **kwargs) -> Any:
        """Execute a request with retries on failure.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Raises:
            Exception: If all retries fail
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (HTTPError, RequestException) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")

        raise last_error

    def get_order_book(self, market_id: str) -> OrderBook:
        """Fetch order book for a market.

        Args:
            market_id: Market ticker symbol

        Returns:
            OrderBook object with current market state
        """
        def _fetch():
            response = self.client.get(f"/trade-api/v2/markets/{market_id}/orderbook")
            orderbook_data = response.get('orderbook', {})

            # Extract bid/ask levels from yes and no sides
            yes_bids = orderbook_data.get('yes', [])
            no_asks = orderbook_data.get('no', [])

            # Log raw order book data for debugging
            logger.debug(f"{market_id} Raw orderbook: yes_bids={yes_bids}, no_asks={no_asks}")

            # In Kalshi order book API:
            # - 'yes' array contains YES orders (people buying YES)
            # - 'no' array contains NO orders (people buying NO = selling YES)
            # - All prices in the API are YES prices (the yes_price field)
            # So: best_bid = highest YES price, best_ask = lowest NO price (already in YES terms!)

            # Process bid levels (YES bids, up to 3 levels)
            # Sort YES bids by price descending (highest first = best bid)
            bid_levels = []
            best_bid = None
            bid_size = 0

            if yes_bids:
                yes_bids_sorted = sorted(yes_bids, key=lambda x: x[0], reverse=True)
                for i, level in enumerate(yes_bids_sorted[:3]):  # Take up to 3 levels
                    price = level[0] / 100.0  # Convert cents to decimal
                    size = level[1]
                    bid_levels.append((price, size))

                    if i == 0:
                        best_bid = price
                        bid_size = size
            else:
                # No bids means nobody wants to buy YES at any price -> bid is effectively $0.01
                logger.info(f"{market_id} No YES bids in order book, using $0.01 as best bid")
                best_bid = 0.01
                bid_size = 0

            # Process ask levels (NO orders = people selling YES, up to 3 levels)
            # Sort NO orders by price ascending (lowest YES price first = best YES ask)
            # NOTE: Prices in 'no' array are already YES prices from the API!
            ask_levels = []
            best_ask = None
            ask_size = 0

            if no_asks:
                no_asks_sorted = sorted(no_asks, key=lambda x: x[0])
                for i, level in enumerate(no_asks_sorted[:3]):  # Take up to 3 levels
                    # Price is already the YES price (yes_price field from Kalshi API)
                    # NO order at yes_price=33 means selling YES at $0.33
                    yes_price = level[0] / 100.0  # Convert cents to decimal
                    size = level[1]
                    ask_levels.append((yes_price, size))

                    if i == 0:
                        best_ask = yes_price
                        ask_size = size
            else:
                # No NO asks means nobody wants to sell YES at any price -> ask is effectively $0.99
                logger.info(f"{market_id} No NO asks in order book, using $0.99 as best ask")
                best_ask = 0.99
                ask_size = 0

            return OrderBook(
                market_id=market_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                timestamp=datetime.utcnow()
            )

        return self._retry_request(_fetch)

    def get_open_orders(self, market_id: Optional[str] = None) -> List[Order]:
        """Fetch all open orders.

        Args:
            market_id: Optional market filter

        Returns:
            List of open Order objects
        """
        def _fetch():
            params = {}
            if market_id:
                params['ticker'] = market_id

            response = self.client.get("/trade-api/v2/portfolio/orders", params=params)
            orders_data = response.get('orders', [])

            orders = []
            for order_data in orders_data:
                # Only include active orders
                status = order_data.get('status', '').lower()
                if status not in ['resting', 'pending']:
                    continue

                # Store exactly as Kalshi returns it:
                # side = 'yes' or 'no' (Kalshi's format)
                # price = yes_price (what Kalshi stores, always the YES price regardless of side)
                orders.append(Order(
                    order_id=order_data.get('order_id'),
                    market_id=order_data.get('ticker'),
                    side=order_data.get('side'),  # 'yes' or 'no' (Kalshi format)
                    price=order_data.get('yes_price', 0) / 100.0,  # Always YES price
                    size=order_data.get('remaining_count', order_data.get('count', 0)),
                    filled_size=order_data.get('count', 0) - order_data.get('remaining_count', 0),
                    status=status,
                    created_at=parse_kalshi_timestamp(order_data.get('created_time', '')),
                    tif=order_data.get('tif', 'gtc')
                ))

            return orders

        return self._retry_request(_fetch)

    def get_fills(self, since: Optional[datetime] = None, market_id: Optional[str] = None) -> List[Fill]:
        """Fetch recent fills.

        Args:
            since: Only return fills after this timestamp
            market_id: Optional market filter

        Returns:
            List of Fill objects
        """
        def _fetch():
            params = {}
            if market_id:
                params['ticker'] = market_id
            if since:
                # Kalshi expects timestamp in milliseconds
                params['min_ts'] = int(since.timestamp() * 1000)

            response = self.client.get("/trade-api/v2/portfolio/fills", params=params)
            fills_data = response.get('fills', [])

            fills = []
            for fill_data in fills_data:
                fills.append(Fill(
                    fill_id=fill_data.get('trade_id'),
                    order_id=fill_data.get('order_id'),
                    market_id=fill_data.get('ticker'),
                    side=fill_data.get('side'),
                    price=fill_data.get('yes_price', 0) / 100.0,
                    size=fill_data.get('count', 0),
                    timestamp=parse_kalshi_timestamp(fill_data.get('created_time', '')),
                    fees=fill_data.get('fees', 0) / 100.0
                ))

            return fills

        return self._retry_request(_fetch)

    def get_balance(self) -> Balance:
        """Fetch account balance.

        Returns:
            Balance object
        """
        def _fetch():
            response = self.client.get_balance()
            return Balance(
                balance=response.get('balance', 0) / 100.0,  # Convert cents to dollars
                payout=response.get('payout', 0) / 100.0
            )

        return self._retry_request(_fetch)

    def place_order(
        self,
        market_id: str,
        side: str,
        price: float,
        size: int,
        tif: str = "gtc"
    ) -> Optional[Order]:
        """Place a new order.

        Args:
            market_id: Market ticker
            side: 'bid' or 'ask' (in YES terms)
            price: Price in decimal (0.01 to 0.99), always in YES terms
            size: Number of contracts
            tif: Time in force ('gtc', 'ioc', 'fok')

        Returns:
            Order object if successful, None otherwise
        """
        def _place():
            # Convert bid/ask to Kalshi's yes/no format
            # IMPORTANT: yes_price in Kalshi API is ALWAYS the YES price, even for NO orders!
            # bid (buy YES at X) -> side="yes", yes_price=X
            # ask (sell YES at X) -> side="no", yes_price=X (same YES price!)
            if side == "bid":
                kalshi_side = "yes"
                yes_price = int(price * 100)
            else:  # side == "ask"
                kalshi_side = "no"
                # Selling YES at X means buying NO at yes_price=X (NOT 1-X!)
                yes_price = int(price * 100)

            payload = {
                "ticker": market_id,
                "action": "buy",  # Always "buy" - side determines yes/no
                "side": kalshi_side,
                "count": int(size),  # Ensure integer type
                "yes_price": yes_price,
                "type": "limit",
                "tif": tif
            }

            # Log in YES price terms
            order_type = "BID" if side == "bid" else "ASK"
            logger.debug(f"Placing order: {market_id} {order_type} {size}@{price:.2f} (yes_price={yes_price}, kalshi_side={kalshi_side})")

            try:
                response = self.client.post("/trade-api/v2/portfolio/orders", payload)
                order_data = response.get('order', {})

                logger.info(f"Placed order: {market_id} {order_type} {size}@{price:.2f} - ID: {order_data.get('order_id')}")

                # Return Order with Kalshi's format:
                # - side is "yes" or "no" (Kalshi's format)
                # - price is always YES price (what Kalshi stores in yes_price field)
                return Order(
                    order_id=order_data.get('order_id'),
                    market_id=market_id,
                    side=kalshi_side,  # "yes" or "no" (Kalshi format)
                    price=price,  # Always YES price (matches the input price parameter)
                    size=size,
                    filled_size=0,
                    status='pending',
                    created_at=datetime.utcnow(),
                    tif=tif
                )
            except HTTPError as e:
                logger.error(f"HTTP error placing order {market_id} {order_type} {size}@{price:.2f}: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"API error details: {error_detail}")
                    except:
                        logger.error(f"Response text: {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"Failed to place order {market_id} {order_type} {size}@{price:.2f}: {e}", exc_info=True)
                return None

        return self._retry_request(_place)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        def _cancel():
            try:
                self.client.delete(f"/trade-api/v2/portfolio/orders/{order_id}")
                logger.info(f"Cancelled order: {order_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")
                return False

        return self._retry_request(_cancel)

    def cancel_all_orders(self, market_id: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by market.

        Args:
            market_id: Optional market filter

        Returns:
            Number of orders cancelled
        """
        open_orders = self.get_open_orders(market_id=market_id)
        cancelled_count = 0

        for order in open_orders:
            if self.cancel_order(order.order_id):
                cancelled_count += 1
            time.sleep(0.1)  # Small delay to avoid rate limiting

        logger.info(f"Cancelled {cancelled_count} orders" + (f" for {market_id}" if market_id else ""))
        return cancelled_count

    def get_market_info(self, market_id: str) -> Dict[str, Any]:
        """Get market metadata.

        Args:
            market_id: Market ticker

        Returns:
            Dictionary with market information
        """
        def _fetch():
            return self.client.get(f"/trade-api/v2/markets/{market_id}")

        return self._retry_request(_fetch)

    def get_exchange_status(self) -> Dict[str, Any]:
        """Get exchange status.

        Returns:
            Dictionary with exchange status
        """
        def _fetch():
            return self.client.get_exchange_status()

        return self._retry_request(_fetch)
