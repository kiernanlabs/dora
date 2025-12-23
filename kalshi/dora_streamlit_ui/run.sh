#!/bin/bash
# Simple script to run the Dora Bot Streamlit Dashboard

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run streamlit
streamlit run app.py
