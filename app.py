import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Load the trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Diabetes Prediction App')
st.markdown("Model has been trained with data from 'National Institute of Diabetes and Digestive and Kidney Diseases' and has shown 85% accuracy")

# Collect input features with placeholders and validation checks
pregnancies = st.number_input('No. of Pregnancies',min_value=0,  max_value=20, step=1, key="pregnancies")
glucose = st.number_input('Glucose Level', min_value=0.1, key="glucose")
bp = st.number_input('Blood Pressure (mm Hg)', min_value=0.1, key="bp")
skin_thickness = st.number_input('Tricep Skin Thickness (mm)', min_value=0.1, key="skin_thickness")
insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0.1, key="insulin")
bmi = st.number_input('BMI', min_value=0.1, key="bmi")
age = st.number_input('Age', min_value=1, max_value=100, step=1, key="age")


# Predict the outcome only if all inputs are valid
if st.button('Predict'):
    input_data = np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi, age])
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.write('The patient is diabetic.')
    else:
        st.write('The patient is not diabetic.')

