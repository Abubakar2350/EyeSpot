import streamlit as st
import numpy as np

img = np.zeros((300, 300, 3), dtype=np.uint8)

res = st.image(img, "Click me!")
st.write("Result:", type(res), res)
