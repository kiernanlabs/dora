"""Simple script to test Kalshi API connection."""

import sys
import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'kalshi-starter-code-python-main'))
from clients import KalshiHttpClient, Environment

from exchange_client import KalshiExchangeClient


def test_connection(use_demo: bool = True):
    """Test connection to Kalshi API.

    Args:
        use_demo: If True, use demo environment
    """
    print(f"\n{'='*60}")
    print(f"Testing Kalshi API Connection ({'DEMO' if use_demo else 'PROD'} mode)")
    print(f"{'='*60}\n")

    # Get the parent kalshi directory (where .env is located)
    kalshi_dir = os.path.join(os.path.dirname(__file__), '..')

    # Load environment
    load_dotenv(os.path.join(kalshi_dir, '.env'))

    env = Environment.DEMO if use_demo else Environment.PROD
    keyid = os.getenv('DEMO_KEYID') if use_demo else os.getenv('PROD_KEYID')
    keyfile = os.getenv('DEMO_KEYFILE') if use_demo else os.getenv('PROD_KEYFILE')

    if not keyid or not keyfile:
        print("❌ ERROR: Missing API credentials in .env file")
        print(f"   Need: {'DEMO' if use_demo else 'PROD'}_KEYID and {'DEMO' if use_demo else 'PROD'}_KEYFILE")
        return False

    print(f"✓ Found API credentials")
    print(f"  Key ID: {keyid[:8]}...")
    print(f"  Key file: {keyfile}")

    # Resolve keyfile path relative to kalshi directory
    keyfile_path = os.path.join(kalshi_dir, keyfile)

    # Load private key
    try:
        with open(keyfile_path, "rb") as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=None
            )
        print(f"✓ Loaded private key from {keyfile_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Private key file not found: {keyfile_path}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Failed to load private key: {e}")
        return False

    # Create client
    try:
        kalshi_client = KalshiHttpClient(
            key_id=keyid,
            private_key=private_key,
            environment=env
        )
        exchange = KalshiExchangeClient(kalshi_client)
        print(f"✓ Created exchange client")
    except Exception as e:
        print(f"❌ ERROR: Failed to create client: {e}")
        return False

    # Test exchange status
    print("\nTesting API calls...")
    try:
        status = exchange.get_exchange_status()
        print(f"✓ Exchange status: {status.get('exchange_active', 'unknown')}")
    except Exception as e:
        print(f"❌ ERROR: Failed to get exchange status: {e}")
        return False

    # Test balance
    try:
        balance = exchange.get_balance()
        print(f"✓ Account balance: ${balance.balance:.2f}")
        print(f"  Total (balance + payout): ${balance.total:.2f}")
    except Exception as e:
        print(f"❌ ERROR: Failed to get balance: {e}")
        return False

    # Test getting open orders
    try:
        orders = exchange.get_open_orders()
        print(f"✓ Open orders: {len(orders)}")
        if orders:
            print(f"  Sample order: {orders[0].market_id} {orders[0].side} {orders[0].size}@{orders[0].price:.2f}")
    except Exception as e:
        print(f"❌ ERROR: Failed to get open orders: {e}")
        return False

    # Test getting order book for a sample market
    print("\nTo test order book, provide a market ticker:")
    print("  (or press Enter to skip)")
    market_id = input("Market ticker: ").strip()

    if market_id:
        try:
            order_book = exchange.get_order_book(market_id)
            print(f"✓ Order book for {market_id}:")
            print(f"  Best bid: {order_book.best_bid:.2f} ({order_book.bid_size} contracts)")
            print(f"  Best ask: {order_book.best_ask:.2f} ({order_book.ask_size} contracts)")
            print(f"  Spread: {order_book.spread:.3f}")
            print(f"  Mid: {order_book.mid_price:.2f}")

            # Display depth
            if order_book.bid_levels:
                print(f"\n  Bid depth ({len(order_book.bid_levels)} levels, {order_book.total_bid_depth} total):")
                for i, (price, size) in enumerate(order_book.bid_levels, 1):
                    print(f"    L{i}: {price:.2f} x {size}")

            if order_book.ask_levels:
                print(f"\n  Ask depth ({len(order_book.ask_levels)} levels, {order_book.total_ask_depth} total):")
                for i, (price, size) in enumerate(order_book.ask_levels, 1):
                    print(f"    L{i}: {price:.2f} x {size}")
        except Exception as e:
            print(f"❌ ERROR: Failed to get order book: {e}")

    print(f"\n{'='*60}")
    print(f"✓ Connection test complete!")
    print(f"{'='*60}\n")
    return True


if __name__ == "__main__":
    use_demo = True
    if len(sys.argv) > 1 and sys.argv[1] == '--prod':
        use_demo = False

    success = test_connection(use_demo)
    sys.exit(0 if success else 1)
