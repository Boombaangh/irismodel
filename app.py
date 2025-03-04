import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Configure page layout
st.set_page_config(layout="wide")

# Title of the app
st.title("Iris Flower Species Predictor")

# Load the trained model
iris_model = pickle.load(open('iris_trained_model.pkl', 'rb'))

# User input fields for flower measurements
slength = st.number_input("Enter Sepal Length (cm)")
swidth = st.number_input("Enter Sepal Width (cm)")
plength = st.number_input("Enter Petal Length (cm)")
pwidth = st.number_input("Enter Petal Width (cm)")

# Creating a dictionary for user inputs
input_data = {
    'sepal length (cm)': slength,
    'sepal width (cm)': swidth,
    'petal length (cm)': plength,
    'petal width (cm)': pwidth
}

# Convert input dictionary into a DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Prediction button
if st.button("Classify"):
    predicted_class = iris_model.predict(input_df)
    
    # Extract prediction result
    species_result = predicted_class[0]

    # Display the predicted species
    st.success(f"The model predicts this flower belongs to the **{species_result}** species.")
