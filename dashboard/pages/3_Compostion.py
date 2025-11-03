"""Compostion of the data such as buildings Page for dashboard"""

import streamlit as st

from utils.dashboard_utils import apply_page_config

apply_page_config()

st.text(
    "Explaining how the data composed and what decisions were made to decide which observations were kept"
)
