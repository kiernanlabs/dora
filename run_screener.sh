#!/bin/bash
# Wrapper script for market_screener that sets PYTHONPATH
cd "$(dirname "$0")"
PYTHONPATH=/home/joe/dora/kalshi python -m dora_bot.market_screener "$@"
