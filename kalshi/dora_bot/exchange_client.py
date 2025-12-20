"""Exchange client for interacting with Kalshi API."""

import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from requests.exceptions import HTTPError, RequestException

from kalshi_client import KalshiHttpClient, Environment

from models import OrderBook, Order, Fill, Balance
from structured_logger import get_logger, EventType

logger = get_logger(__name__)


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

    def __init__(
        self,
        client: KalshiHttpClient,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        bot_run_id: Optional[str] = None
    ):
        """Initialize the exchange client.

        Args:
            client: Authenticated KalshiHttpClient instance
            max_retries: Maximum number of retries for failed requests
            retry_delay: Base delay between retries in seconds
            bot_run_id: Bot run ID for log correlation
        """
        self.client = client
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.bot_run_id = bot_run_id

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

                # Extract API error details
                error_code = None
                error_message = str(e)
                status_code = None
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    try:
                        error_data = e.response.json()
                        error_code = error_data.get('code')
                        error_message = error_data.get('message', str(e))
                    except:
                        pass

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning("API request failed, retrying", extra={
                        "event_type": EventType.LOG,
                        "attempt": attempt + 1,
                        "max_retries": self.max_retries,
                        "retry_delay": delay,
                        "error_type": type(e).__name__,
                        "error_code": error_code,
                        "error_msg": error_message,
                        "status_code": status_code,
                    })
                    time.sleep(delay)
                else:
                    logger.error("API request failed after all retries", extra={
                        "event_type": EventType.ERROR,
                        "attempts": self.max_retries,
                        "error_type": type(e).__name__,
                        "error_code": error_code,
                        "error_msg": error_message,
                        "status_code": status_code,
                    })

        raise last_error

    def get_order_book(self, market_id: str, exclude_own_orders: bool = True) -> OrderBook:
        """Fetch order book for a market.

        Args:
            market_id: Market ticker symbol
            exclude_own_orders: If True, filter out the bot's own orders from the order book

        Returns:
            OrderBook object with current market state
        """
        def _fetch():
            response = self.client.get(f"/trade-api/v2/markets/{market_id}/orderbook")
            orderbook_data = response.get('orderbook', {})

            # Extract raw data from Kalshi API
            # 'yes' array = YES bids (people buying YES at a YES price)
            # 'no' array = NO bids (people buying NO at a NO price)
            yes_bids_raw = orderbook_data.get('yes', []) or []
            no_bids_raw = orderbook_data.get('no', []) or []

            # Get bot's own orders to filter them out
            own_orders_by_price = {'yes': {}, 'no': {}}
            if exclude_own_orders:
                try:
                    own_orders = self.get_open_orders(market_id=market_id)
                    for order in own_orders:
                        # Group by side and price (in cents)
                        price_cents = int(order.price * 100)
                        if order.side in own_orders_by_price:
                            if price_cents not in own_orders_by_price[order.side]:
                                own_orders_by_price[order.side][price_cents] = 0
                            own_orders_by_price[order.side][price_cents] += order.size
                except Exception as e:
                    logger.warning("Failed to fetch own orders for filtering", extra={
                        "event_type": EventType.LOG,
                        "market": market_id,
                        "error_msg": str(e),
                    })

            # Filter out bot's own YES bids
            yes_bids_filtered = []
            for price_cents, size in yes_bids_raw:
                own_size = own_orders_by_price['yes'].get(price_cents, 0)
                remaining_size = size - own_size
                if remaining_size > 0:
                    yes_bids_filtered.append([price_cents, remaining_size])

            # Filter out bot's own NO bids
            no_bids_filtered = []
            for price_cents, size in no_bids_raw:
                own_size = own_orders_by_price['no'].get(100-price_cents, 0)
                remaining_size = size - own_size
                if remaining_size > 0:
                    no_bids_filtered.append([price_cents, remaining_size])

            # Use filtered data
            yes_bids_raw = yes_bids_filtered
            no_bids_raw = no_bids_filtered

            # Convert NO bids to YES asks immediately
            # If someone bids X cents for NO, they're effectively asking (100-X) cents for YES
            # Example: NO bid at 58¢ = YES ask at 42¢
            yes_asks_converted = [(100 - level[0], level[1]) for level in no_bids_raw]

            # Log order book for debugging (at debug level to reduce noise)
            logger.debug("Orderbook fetched", extra={
                "event_type": EventType.LOG,
                "market": market_id,
                "yes_bids": yes_bids_raw,
                "yes_asks": yes_asks_converted,
            })

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
            logger.warning("Failed to fetch balance", extra={
                "event_type": EventType.LOG,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
            return None

    def place_order(
        self,
        market_id: str,
        side: str,
        price: float,
        size: int,
        tif: str = "gtc",
        decision_id: Optional[str] = None
    ) -> Optional[Order]:
        """Place a new order.

        Args:
            market_id: Market ticker
            side: 'bid' or 'ask' (in YES terms)
            price: Price in decimal (0.01 to 0.99), always in YES terms
            size: Number of contracts
            tif: Time in force ('gtc', 'ioc', 'fok')
            decision_id: Decision ID for correlation

        Returns:
            Order object if successful, None otherwise
        """
        def _place():
            start_time = time.time()

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

            # Log order placement attempt
            order_type = "bid" if side == "bid" else "ask"
            logger.info("Placing order", extra={
                "event_type": EventType.ORDER_PLACE,
                "market": market_id,
                "decision_id": decision_id,
                "side": order_type,
                "price": price,
                "size": size,
            })

            try:
                response = self.client.post("/trade-api/v2/portfolio/orders", payload)
                order_data = response.get('order', {})
                latency_ms = int((time.time() - start_time) * 1000)

                order_id = order_data.get('order_id')
                logger.info("Order placed", extra={
                    "event_type": EventType.ORDER_RESULT,
                    "market": market_id,
                    "decision_id": decision_id,
                    "order_id": order_id,
                    "side": order_type,
                    "price": price,
                    "size": size,
                    "status": "ACCEPTED",
                    "latency_ms": latency_ms,
                })

                # Return Order with Kalshi's format:
                # - side is "yes" or "no" (Kalshi's format)
                # - price is always YES price (what Kalshi stores in yes_price field)
                return Order(
                    order_id=order_id,
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
                latency_ms = int((time.time() - start_time) * 1000)
                error_code = None
                error_msg = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_code = error_data.get('code')
                        error_msg = error_data.get('message', str(e))
                    except:
                        pass

                logger.error("Order placement failed", extra={
                    "event_type": EventType.ORDER_RESULT,
                    "market": market_id,
                    "decision_id": decision_id,
                    "side": order_type,
                    "price": price,
                    "size": size,
                    "status": "REJECTED",
                    "latency_ms": latency_ms,
                    "error_type": "HTTPError",
                    "error_code": error_code,
                    "error_msg": error_msg,
                })
                return None
            except Exception as e:
                latency_ms = int((time.time() - start_time) * 1000)
                logger.error("Order placement failed", extra={
                    "event_type": EventType.ORDER_RESULT,
                    "market": market_id,
                    "decision_id": decision_id,
                    "side": order_type,
                    "price": price,
                    "size": size,
                    "status": "ERROR",
                    "latency_ms": latency_ms,
                    "error_type": type(e).__name__,
                    "error_msg": str(e),
                }, exc_info=True)
                return None

        return self._retry_request(_place)

    def cancel_order(self, order_id: str, decision_id: Optional[str] = None) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel
            decision_id: Decision ID for correlation

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        logger.info("Cancelling order", extra={
            "event_type": EventType.ORDER_CANCEL,
            "order_id": order_id,
            "decision_id": decision_id,
        })

        try:
            self.client.delete(f"/trade-api/v2/portfolio/orders/{order_id}")
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("Order cancelled", extra={
                "event_type": EventType.ORDER_RESULT,
                "order_id": order_id,
                "decision_id": decision_id,
                "status": "CANCELLED",
                "latency_ms": latency_ms,
            })
            return True
        except HTTPError as e:
            latency_ms = int((time.time() - start_time) * 1000)
            # Handle 404 - order doesn't exist, verify by checking open orders
            if e.response is not None and e.response.status_code == 404:
                # Fetch all open orders to confirm the order is truly gone
                try:
                    open_orders = self.get_open_orders()
                    order_exists = any(order.order_id == order_id for order in open_orders)

                    if not order_exists:
                        logger.info("Order already cancelled/filled", extra={
                            "event_type": EventType.ORDER_RESULT,
                            "order_id": order_id,
                            "decision_id": decision_id,
                            "status": "ALREADY_GONE",
                            "latency_ms": latency_ms,
                        })
                        return True
                    else:
                        logger.error("Order 404 but still in open orders", extra={
                            "event_type": EventType.ORDER_RESULT,
                            "order_id": order_id,
                            "decision_id": decision_id,
                            "status": "ERROR",
                            "error_msg": "Order returned 404 but still appears in open orders",
                            "latency_ms": latency_ms,
                        })
                        return False
                except Exception as verify_error:
                    logger.error("Failed to verify order status", extra={
                        "event_type": EventType.ERROR,
                        "order_id": order_id,
                        "decision_id": decision_id,
                        "error_type": type(verify_error).__name__,
                        "error_msg": str(verify_error),
                    })
                    return False

            logger.error("Order cancellation failed", extra={
                "event_type": EventType.ORDER_RESULT,
                "order_id": order_id,
                "decision_id": decision_id,
                "status": "ERROR",
                "latency_ms": latency_ms,
                "error_type": "HTTPError",
                "error_msg": str(e),
            })
            return False
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("Order cancellation failed", extra={
                "event_type": EventType.ORDER_RESULT,
                "order_id": order_id,
                "decision_id": decision_id,
                "status": "ERROR",
                "latency_ms": latency_ms,
                "error_type": type(e).__name__,
                "error_msg": str(e),
            })
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

        logger.info("Cancelled all orders", extra={
            "event_type": EventType.ORDER_CANCEL,
            "market": market_id,
            "cancelled_count": cancelled_count,
            "total_orders": len(open_orders),
        })
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
