import io
import requests
import streamlit as st
from PIL import Image
from config import BACKEND_URL

def martensite():
    st.title("Martensite Generation")
    if st.button("Generate"):
        res = requests.get(f"{BACKEND_URL}/martensite")
        image = Image.open(io.BytesIO(res.content))
        st.image(image, caption="Martensite")