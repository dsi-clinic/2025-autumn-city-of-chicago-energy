"""Modularizing configurations for each page"""

import streamlit as st


def apply_page_config() -> None:
    """Standardizing page apperance"""
    st.set_page_config(
        page_title="Chicago Energy Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
