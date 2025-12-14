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

                # Try to extract API error details
                error_details = ""
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_details = f" [API Error: {error_data.get('code', 'N/A')} - {error_data.get('message', 'N/A')}"
                        if 'details' in error_data:
                            error_details += f", Details: {error_data['details']}"
                        if 'service' in error_data:
                            error_details += f", Service: {error_data['service']}"
                        error_details += "]"
                    except:
                        # If JSON parsing fails, include raw response text
                        try:
                            error_details = f" [Response: {e.response.text[:200]}]"
                        except:
                            pass

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}{error_details}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}{error_details}")

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

            # Extract raw data from Kalshi API
            # 'yes' array = YES bids (people buying YES at a YES price)
            # 'no' array = NO bids (people buying NO at a NO price)
            yes_bids_raw = orderbook_data.get('yes', [])
            no_bids_raw = orderbook_data.get('no', [])

            # Convert NO bids to YES asks immediately
            # If someone bids X cents for NO, they're effectively asking (100-X) cents for YES
            # Example: NO bid at 58¢ = YES ask at 42¢
            yes_asks_converted = [(100 - level[0], level[1]) for level in no_bids_raw]

            # Log converted order book data for debugging
            logger.info(f"{market_id} Orderbook: yes_bids={yes_bids_raw}, yes_asks={yes_asks_converted} (converted from no_bids={no_bids_raw})")

            # Process bid levels (YES bids, up to 3 levels)
            # Sort YES bids by price descending (highest first = best bid)
            bid_levels = []
            best_bid = None
            bid_size = 0

            if yes_bids_raw:
                yes_bids_sorted = sorted(yes_bids_raw, key=lambda x: x[0], reverse=True)
                for i, level in enumerate(yes_bids_sorted[:3]):  # Take up to 3 levels
                    price = level[0] / 100.0  # Convert cents to decimal
                    size = level[1]
                    bid_levels.append((price, size))

                    if i == 0:
                        best_bid = price
                        bid_size = size
            else:
                # No bids means nobody wants to buy YES at any price -> bid is effectively $0.01
                logger.info(f"{market_id} No YES bids in order book, using $0.00 as best bid")
                best_bid = 0.00
                bid_size = 0

            # Process ask levels (YES asks converted from NO bids, up to 3 levels)
            # Sort YES asks by price ascending (lowest first = best ask)
            ask_levels = []
            best_ask = None
            ask_size = 0

            if yes_asks_converted:
                yes_asks_sorted = sorted(yes_asks_converted, key=lambda x: x[0])

                for i, level in enumerate(yes_asks_sorted[:3]):  # Take up to 3 levels
                    yes_price = level[0] / 100.0  # Convert cents to decimal
                    size = level[1]
                    ask_levels.append((yes_price, size))

                    if i == 0:
                        best_ask = yes_price
                        ask_size = size
            else:
                # No NO bids means nobody wants to buy NO at any price -> ask is effectively $0.99
                logger.info(f"{market_id} No NO bids in order book (no YES asks), using $1.00 as best ask")
                best_ask = 1.0
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

    def get_balance(self) -> Optional[Balance]:
        """Fetch account balance.

        Returns:
            Balance object, or None if the request fails
        """
        def _fetch():
            response = self.client.get_balance()
            return Balance(
                balance=response.get('balance', 0) / 100.0,  # Convert cents to dollars
                payout=response.get('payout', 0) / 100.0
            )

        try:
            return self._retry_request(_fetch)
        except Exception as e:
            logger.warning(f"Failed to fetch balance, continuing without it: {e}")
            return None

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
        try:
            self.client.delete(f"/trade-api/v2/portfolio/orders/{order_id}")
            logger.info(f"Cancelled order: {order_id}")
            return True
        except HTTPError as e:
            # Handle 404 - order doesn't exist, verify by checking open orders
            if e.response is not None and e.response.status_code == 404:
                logger.info(f"Order {order_id} returned 404, verifying it doesn't exist...")

                # Fetch all open orders to confirm the order is truly gone
                try:
                    open_orders = self.get_open_orders()
                    order_exists = any(order.order_id == order_id for order in open_orders)

                    if not order_exists:
                        logger.info(f"Confirmed order {order_id} does not exist (already filled/cancelled), treating as successful cancel")
                        return True
                    else:
                        logger.error(f"Order {order_id} returned 404 but still appears in open orders list - unexpected state")
                        return False
                except Exception as verify_error:
                    logger.error(f"Failed to verify if order {order_id} exists: {verify_error}")
                    return False

            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

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
