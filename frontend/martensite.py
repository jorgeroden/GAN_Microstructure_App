import io
import requests
import streamlit as st
from PIL import Image
from config import BACKEND_URL

def martensite():
    st.title("Martensite Generation")
    if st.button("Generate"):
        try:
       
            res = requests.get(f"{BACKEND_URL}/martensite")
            res.raise_for_status() 
            image = Image.open(io.BytesIO(res.content))

       
            st.image(image, caption="Martensite")

       
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

       
            st.download_button(
                label="Download Image as PNG",
                data=buf,
                file_name="martensite.png",
                mime="image/png"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except IOError:
            st.error("Failed to process the image.")

