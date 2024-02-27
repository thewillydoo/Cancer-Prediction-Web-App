import streamlit as st
import pandas as pd
import pickle

def add_slider():
    st.sidebar.header("Cell Nuclei Measurements")

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    
    )
    
    add_slider()

    with st.container():
        st.title("Breast Cancer Prediction App")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from the cytosis lab. You can also update the measurements by using the sliders in the sidebar.")
    

    col1, col2 = st.columns([4, 1])

    with col1:
        st.write("this is col1")
    with col2:
        st.write("this is col2")

if __name__ == "__main__":
    main()