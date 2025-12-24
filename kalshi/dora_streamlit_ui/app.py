"""
Dora Bot Streamlit UI - Main Application
"""
import streamlit as st
from datetime import datetime, timedelta

# Configure the page
st.set_page_config(
    page_title="Dora Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'environment' not in st.session_state:
    st.session_state.environment = 'prod'
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

# Check for navigation flag BEFORE rendering sidebar
if st.session_state.get('navigate_to_deep_dive', False):
    st.session_state['current_page'] = 'Market Deep Dive'
    st.session_state['navigate_to_deep_dive'] = False

# Sidebar configuration
with st.sidebar:
    st.title("🤖 Dora Bot")
    st.markdown("---")

    # Environment selector
    environment = st.selectbox(
        "Environment",
        options=['demo', 'prod'],
        index=0 if st.session_state.environment == 'demo' else 1,
        key='env_selector'
    )
    st.session_state.environment = environment

    # AWS Region selector
    region = st.selectbox(
        "AWS Region",
        options=['us-east-1', 'us-west-2'],
        index=0,
        key='region_selector'
    )

    st.markdown("---")
    st.markdown("### Navigation")

    # Page selection
    page_options = ["Home", "Market Deep Dive"]
    default_index = page_options.index(st.session_state.get('current_page', 'Home'))

    page = st.radio(
        "Select Page",
        options=page_options,
        index=default_index,
        key='page_selector'
    )

    # Update current page if user changed selection
    if page != st.session_state.get('current_page'):
        st.session_state['current_page'] = page

    st.markdown("---")
    st.caption(f"Environment: **{environment.upper()}**")
    st.caption(f"Region: **{region}**")

# Route to the appropriate page
if page == "Home":
    from views import home
    home.render(environment, region)
elif page == "Market Deep Dive":
    from views import market_deep_dive
    market_deep_dive.render(environment, region)
