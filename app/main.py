import streamlit as st
import pandas as pd
import pickle

def get_clean_data():
    # Load the data
    data = pd.read_csv('data/data.csv')

    # Drop the columns that are not required
    data = data.drop(['id', 'Unnamed: 32'], axis=1)

    # Convert the diagnosis column to binary
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})


    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave Points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal Dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave Points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal Dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave Points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal Dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value = float(0),
            max_value = float(data[key].max()),
            value = float(data[key].mean())
        )
    
    return input_dict()

def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction App",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    
    )
    
    input_data = add_sidebar()

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