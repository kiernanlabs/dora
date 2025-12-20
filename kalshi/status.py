#!/usr/bin/env python3
"""Display current positions, P&L, and resting orders with best bid/ask comparison."""

import os
import sys
import argparse
from typing import Dict, List, Optional
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dora_bot'))

from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
from kalshi_client import KalshiHttpClient, Environment
from exchange_client import KalshiExchangeClient


def get_client(use_demo: bool = True) -> KalshiExchangeClient:
    """Initialize Kalshi client."""
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

    env = Environment.DEMO if use_demo else Environment.PROD
    keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
    keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

    if not keyid or not keyfile:
        env_type = "DEMO" if use_demo else "PROD"
        print(f"Error: Missing {env_type}_KEYID or {env_type}_KEYFILE in .env")
        sys.exit(1)

    with open(keyfile, "rb") as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    http_client = KalshiHttpClient(key_id=keyid, private_key=private_key, environment=env)
    return KalshiExchangeClient(http_client)


def get_positions(client: KalshiExchangeClient) -> Dict:
    """Fetch portfolio positions from Kalshi API."""
    response = client.client.get("/trade-api/v2/portfolio/positions")
    return response.get('market_positions', [])


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def format_price(price: float) -> str:
    """Format price as cents."""
    return f"${price:.2f}"


def main():
    parser = argparse.ArgumentParser(description='Display Kalshi positions and orders status')
    parser.add_argument('--prod', action='store_true', help='Use production environment (default: demo)')
    args = parser.parse_args()

    use_demo = not args.prod
    env_name = "DEMO" if use_demo else "PROD"

    print(f"Connecting to Kalshi ({env_name})...")
    client = get_client(use_demo=use_demo)

    # Get balance
    balance = client.get_balance()
    if balance:
        print(f"\nAccount Balance: ${balance.balance:.2f} | Payout: ${balance.payout:.2f} | Total: ${balance.total:.2f}")

    # ============ POSITIONS & P&L ============
    print_header("POSITIONS & P&L")

    positions = get_positions(client)
    if not positions:
        print("  No open positions")
    else:
        total_unrealized = 0.0
        total_realized = 0.0

        for pos in positions:
            ticker = pos.get('ticker', 'Unknown')
            position_qty = pos.get('position', 0)  # Positive = long YES, negative = short YES
            total_traded = pos.get('total_traded', 0)
            resting_orders_count = pos.get('resting_orders_count', 0)
            realized_pnl = pos.get('realized_pnl', 0) / 100.0  # Convert cents to dollars
            fees_paid = pos.get('fees_paid', 0) / 100.0

            # Get market price for unrealized P&L
            try:
                orderbook = client.get_order_book(ticker, exclude_own_orders=False)
                mid = orderbook.mid_price or 0.5

                if position_qty > 0:
                    # Long YES - profit if price goes up
                    # We can sell at best_bid
                    exit_price = orderbook.best_bid or mid
                elif position_qty < 0:
                    # Short YES (long NO) - profit if price goes down
                    # We need to buy back at best_ask
                    exit_price = orderbook.best_ask or mid
                else:
                    exit_price = mid

                # Calculate unrealized (this is approximate since we don't have avg cost from API)
                # Just show current market value
                market_value = abs(position_qty) * exit_price if position_qty != 0 else 0
            except:
                mid = 0.5
                market_value = 0

            # Position direction
            if position_qty > 0:
                direction = "LONG YES"
            elif position_qty < 0:
                direction = "SHORT YES"
            else:
                direction = "FLAT"

            total_realized += realized_pnl

            print(f"\n  {ticker}")
            print(f"    Position: {position_qty:+d} ({direction})")
            print(f"    Market Mid: {format_price(mid)} | Est. Exit Value: {format_price(market_value)}")
            print(f"    Realized P&L: {format_price(realized_pnl)} | Fees Paid: {format_price(fees_paid)}")
            print(f"    Total Traded: {total_traded} | Resting Orders: {resting_orders_count}")

        print(f"\n  --- TOTAL REALIZED P&L: {format_price(total_realized)} ---")

    # ============ RESTING ORDERS ============
    print_header("RESTING ORDERS")

    orders = client.get_open_orders()
    if not orders:
        print("  No resting orders")
    else:
        # Group orders by market
        orders_by_market: Dict[str, List] = defaultdict(list)
        for order in orders:
            orders_by_market[order.market_id].append(order)

        for market_id, market_orders in orders_by_market.items():
            # Get orderbook for this market (including own orders to see true market)
            try:
                orderbook = client.get_order_book(market_id, exclude_own_orders=False)
                best_bid = orderbook.best_bid
                best_ask = orderbook.best_ask
            except Exception as e:
                print(f"\n  {market_id} (failed to get orderbook: {e})")
                best_bid = None
                best_ask = None

            print(f"\n  {market_id}")
            print(f"    Market: Bid {format_price(best_bid) if best_bid else 'N/A'} / Ask {format_price(best_ask) if best_ask else 'N/A'}")
            if best_bid and best_ask:
                spread = best_ask - best_bid
                print(f"    Spread: {format_price(spread)} ({spread*100:.0f}c)")

            print(f"    {'Side':<8} {'Price':>8} {'Size':>6} {'Status':>10} {'At Best?':>10}")
            print(f"    {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")

            for order in sorted(market_orders, key=lambda o: (o.side, -o.price)):
                # Determine if bid or ask
                if order.side == 'yes':
                    side_str = "BID"
                    is_at_best = best_bid is not None and abs(order.price - best_bid) < 0.005
                else:
                    side_str = "ASK"
                    is_at_best = best_ask is not None and abs(order.price - best_ask) < 0.005

                best_indicator = "YES" if is_at_best else "no"

                # Color coding (using ANSI if terminal supports it)
                if is_at_best:
                    best_str = f"\033[92m{best_indicator:>10}\033[0m"  # Green
                else:
                    best_str = f"\033[93m{best_indicator:>10}\033[0m"  # Yellow

                print(f"    {side_str:<8} {format_price(order.price):>8} {order.size:>6} {order.status:>10} {best_str}")

    print()


if __name__ == "__main__":
    main()
