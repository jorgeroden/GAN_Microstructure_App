import io
import requests
import streamlit as st
from PIL import Image
from config import BACKEND_URL

def pearlite():
    st.title("Pearlite Generation")
    if st.button("Generate"):
        try:
            
            res = requests.get(f"{BACKEND_URL}/pearlite")
            res.raise_for_status()  # Verifica si la solicitud fue exitosa
            image = Image.open(io.BytesIO(res.content))

           
            st.image(image, caption="Pearlite")

          
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

       
            st.download_button(
                label="Download Image as PNG",
                data=buf,
                file_name="pearlite.png",
                mime="image/png"
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except IOError:
            st.error("Failed to process the image.")

