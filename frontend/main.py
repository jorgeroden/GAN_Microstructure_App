import streamlit as st
from martensite import martensite
from pearlite import pearlite

st.sidebar.title("Models")
option = st.sidebar.radio("Select", ["Martensite", "Pearlite"])
if option == "Martensite":
    martensite()
elif option == "Pearlite":
    pearlite()