import streamlit as st
import pandas as pd
import pickle

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
    
    )
    st.write("Hello World")

if __name__ == "__main__":
    main()