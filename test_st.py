import streamlit as st
import numpy as np
img = np.zeros((100, 100, 3), dtype=np.uint8)
try:
    st.image(img, width="stretch")
except Exception as e:
    print(f"Error: {e}")
