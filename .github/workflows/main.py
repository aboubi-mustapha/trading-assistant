import streamlit as st
from ta.momentum import RSIIndicator

st.title("Test TA Library")
st.write("TA version:", RSIIndicator.__version__)
