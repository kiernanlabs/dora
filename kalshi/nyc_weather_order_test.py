#!/usr/bin/env python3
"""
Single-file Kalshi demo:
1) Find today's NYC weather market via the API
2) Fetch market data + orderbook
3) Place and cancel a 1-lot order
"""

import argparse
import base64
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from dotenv import load_dotenv

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    ZoneInfo = None  # type: ignore


DEMO_BASE_URL = "https://demo-api.kalshi.co"
PROD_BASE_URL = "https://api.elections.kalshi.com"

NYC_KEYWORDS = ("new york city", "nyc", "new york")
WEATHER_KEYWORDS = (
    "weather",
    "temperature",
    "temp",
    "rain",
    "snow",
    "precip",
    "forecast",
    "wind",
    "humidity",
)

NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else None


class KalshiHttpClient:
    def __init__(self, key_id: str, private_key: rsa.RSAPrivateKey, base_url: str) -> None:
        self.key_id = key_id
        self.private_key = private_key
        self.base_url = base_url
        self.last_api_call = datetime.utcnow()

    def _rate_limit(self) -> None:
        # Minimal rate limiter to avoid tripping default thresholds.
        if (datetime.utcnow() - self.last_api_call).total_seconds() < 0.1:
            time.sleep(0.1)
        self.last_api_call = datetime.utcnow()

    def _sign(self, text: str) -> str:
        signature = self.private_key.sign(
            text.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        path_no_params = path.split("?")[0]
        signature = self._sign(timestamp + method + path_no_params)
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._rate_limit()
        response = requests.get(
            self.base_url + path,
            headers=self._headers("GET", path),
            params=params or {},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        self._rate_limit()
        response = requests.post(
            self.base_url + path,
            headers=self._headers("POST", path),
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def delete(self, path: str) -> Dict[str, Any]:
        self._rate_limit()
        response = requests.delete(
            self.base_url + path,
            headers=self._headers("DELETE", path),
            timeout=30,
        )
        response.raise_for_status()
        return response.json()


def _load_private_key(keyfile: str) -> rsa.RSAPrivateKey:
    with open(keyfile, "rb") as handle:
        return serialization.load_pem_private_key(handle.read(), password=None)


def load_client(use_demo: bool) -> KalshiHttpClient:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)

    key_id = os.getenv("DEMO_KEYID" if use_demo else "PROD_KEYID")
    keyfile = os.getenv("DEMO_KEYFILE" if use_demo else "PROD_KEYFILE")
    if not key_id or not keyfile:
        raise ValueError("Missing API credentials in .env")

    keyfile = os.path.expanduser(keyfile)
    if not os.path.isabs(keyfile):
        cwd_candidate = os.path.abspath(keyfile)
        module_candidate = os.path.join(os.path.dirname(__file__), keyfile)
        if os.path.exists(cwd_candidate):
            keyfile = cwd_candidate
        elif os.path.exists(module_candidate):
            keyfile = module_candidate

    private_key = _load_private_key(keyfile)
    base_url = DEMO_BASE_URL if use_demo else PROD_BASE_URL
    return KalshiHttpClient(key_id=key_id, private_key=private_key, base_url=base_url)


def _matches_keywords(text: str, keywords: Iterable[str]) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)


def _parse_kalshi_time(timestamp: str) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _today_ny_date() -> datetime.date:
    if NY_TZ is None:
        return datetime.utcnow().date()
    return datetime.now(NY_TZ).date()


def _market_closes_today(market: Dict[str, Any], today_ny: datetime.date) -> bool:
    close_time = market.get("close_time") or market.get("expiration_time") or ""
    close_dt = _parse_kalshi_time(close_time)
    if close_dt is None:
        return False
    if NY_TZ is not None:
        close_dt = close_dt.astimezone(NY_TZ)
    return close_dt.date() == today_ny


def fetch_events(client: KalshiHttpClient, status: str = "open", limit: int = 200, max_pages: int = 10) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    for _ in range(max_pages):
        params: Dict[str, Any] = {"status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/events", params=params)
        events.extend(data.get("events", []))
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return events


def fetch_markets_for_event(
    client: KalshiHttpClient,
    event_ticker: str,
    status: str = "open",
    limit: int = 200,
    max_pages: int = 10,
) -> List[Dict[str, Any]]:
    markets: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    for _ in range(max_pages):
        params: Dict[str, Any] = {"event_ticker": event_ticker, "status": status, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/markets", params=params)
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break
    return markets


def find_nyc_weather_market(client: KalshiHttpClient) -> Tuple[str, Dict[str, Any]]:
    today_ny = _today_ny_date()
    events = fetch_events(client)
    candidate_events: List[Dict[str, Any]] = []

    for event in events:
        title = event.get("title", "")
        subtitle = event.get("subtitle", "")
        event_text = f"{title} {subtitle} {event.get('ticker', '')}"
        if _matches_keywords(event_text, NYC_KEYWORDS) and _matches_keywords(event_text, WEATHER_KEYWORDS):
            candidate_events.append(event)

    for event in candidate_events:
        event_ticker = event.get("ticker") or event.get("event_ticker")
        if not event_ticker:
            continue
        markets = fetch_markets_for_event(client, event_ticker=event_ticker)
        for market in markets:
            if _market_closes_today(market, today_ny):
                return market.get("ticker"), market

    # Fallback: scan markets directly with a lightweight filter
    cursor: Optional[str] = None
    for _ in range(10):
        params: Dict[str, Any] = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = client.get("/trade-api/v2/markets", params=params)
        markets = data.get("markets", [])
        for market in markets:
            title = market.get("title", "")
            subtitle = market.get("subtitle", "")
            text = f"{title} {subtitle} {market.get('event_ticker', '')} {market.get('ticker', '')}"
            if _matches_keywords(text, NYC_KEYWORDS) and _matches_keywords(text, WEATHER_KEYWORDS):
                if _market_closes_today(market, today_ny):
                    return market.get("ticker"), market
        cursor = data.get("cursor") or data.get("next_cursor")
        if not cursor:
            break

    raise RuntimeError("Could not find an open NYC weather market for today.")


def _extract_best_prices(orderbook: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    orderbook_data = orderbook.get("orderbook", {}) if orderbook else {}
    yes_bids = orderbook_data.get("yes", []) or []
    no_bids = orderbook_data.get("no", []) or []

    best_bid = max((price for price, _ in yes_bids), default=None)
    # NO bids represent asks for YES at (100 - no_price).
    best_ask = min((100 - price for price, _ in no_bids), default=None)
    return best_bid, best_ask


def _pick_test_order(best_bid: Optional[int], best_ask: Optional[int]) -> Tuple[str, int]:
    if best_ask is not None:
        price = min(99, max(1, best_ask + 5))
        return "ask", price
    if best_bid is not None:
        price = max(1, min(99, best_bid - 5))
        return "bid", price
    return "bid", 1


def place_order(client: KalshiHttpClient, ticker: str, side: str, price_cents: int, size: int) -> str:
    if side not in ("bid", "ask"):
        raise ValueError("side must be 'bid' or 'ask'")
    kalshi_side = "yes" if side == "bid" else "no"
    payload = {
        "ticker": ticker,
        "action": "buy",
        "side": kalshi_side,
        "count": int(size),
        "yes_price": int(price_cents),
        "type": "limit",
        "tif": "gtc",
    }
    response = client.post("/trade-api/v2/portfolio/orders", payload)
    order = response.get("order", {})
    order_id = order.get("order_id")
    if not order_id:
        raise RuntimeError("Order placement succeeded but no order_id returned.")
    return order_id


def cancel_order(client: KalshiHttpClient, order_id: str) -> None:
    client.delete(f"/trade-api/v2/portfolio/orders/{order_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Place/cancel a 1-lot order on today's NYC weather market.")
    parser.add_argument("--prod", action="store_true", help="Use production environment.")
    parser.add_argument("--ticker", help="Override market ticker (skips NYC weather search).")
    args = parser.parse_args()

    client = load_client(use_demo=not args.prod)

    if args.ticker:
        market_ticker = args.ticker
        market_data = client.get(f"/trade-api/v2/markets/{market_ticker}")
        market_info = market_data.get("market", market_data)
    else:
        market_ticker, market_info = find_nyc_weather_market(client)

    print(f"Market: {market_ticker}")
    print(f"Title: {market_info.get('title')}")
    print(f"Close time: {market_info.get('close_time')}")

    orderbook = client.get(f"/trade-api/v2/markets/{market_ticker}/orderbook")
    best_bid, best_ask = _extract_best_prices(orderbook)
    print(f"Best bid: {best_bid}c, Best ask: {best_ask}c")

    side, price_cents = _pick_test_order(best_bid, best_ask)
    print(f"Placing 1-lot {side} at {price_cents}c...")
    order_id = place_order(client, market_ticker, side, price_cents, size=1)
    print(f"Order placed: {order_id}")

    print("Cancelling order...")
    cancel_order(client, order_id)
    print("Order cancelled.")


if __name__ == "__main__":
    main()
