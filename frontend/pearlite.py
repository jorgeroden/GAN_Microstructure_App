import io
import requests
import streamlit as st
from PIL import Image
from config import BACKEND_URL

def pearlite():
    st.title("Pearlite Generation")
    if st.button("Generate"):
        res = requests.get(f"{BACKEND_URL}/pearlite")
        image = Image.open(io.BytesIO(res.content))
        st.image(image, caption="Pearlite")