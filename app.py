# app.py
import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the trained model
# file=open('diabetes_model.pkl','rb')
# model = pickle.load(file)

import requests 
url='https://raw.githubusercontent.com/imShashanKashyap/Diabetic_Model/main/diabetes_model.pkl'
response = requests.get(url)
model = pickle.loads(response.content)

st.title('Diabetes Prediction App using Machine Learning')
st.markdown("The model has been trained on a dataset from the 'National Institute of Diabetes and Digestive and Kidney Diseases' and has demonstrated an accuracy of 85%.")

# Collect input features with placeholders and validation checks
pregnancies = st.number_input('No. of Pregnancies',min_value=0,  max_value=20, step=1, key="pregnancies")
glucose = st.number_input('Glucose Level (mg/dL)', min_value=0.1, key="glucose")
bp = st.number_input('Blood Pressure (mm Hg)', min_value=0.1, key="bp")
skin_thickness = st.number_input('Triceps Skinfold Thickness (mm)', min_value=0.1, key="skin_thickness")
insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0.1, key="insulin")
bmi = st.number_input('BMI', min_value=0.1, key="bmi")
age = st.number_input('Age', min_value=1, max_value=100, step=1, key="age")

# Check if any input is empty or invalid
# inputs_provided = all([
#     pregnancies >= 0,
#     glucose > 0,
#     bp > 0,
#     skin_thickness > 0,
#     insulin > 0,
#     bmi > 0,
#     age > 0
# ])

# # Display validation messages for incorrect inputs
# if pregnancies < 0:
#     st.error('No. of Pregnancies cannot be negative.')
# if glucose <= 0:
#     st.error('Glucose Level must be greater than 0.')
# if bp <= 0:
#     st.error('Blood Pressure must be greater than 0.')
# if skin_thickness <= 0:
#     st.error('Tricep Skin Thickness must be greater than 0.')
# if insulin <= 0:
#     st.error('Insulin Level must be greater than 0.')
# if bmi <= 0:
#     st.error('BMI must be greater than 0.')
# if age <= 0:
#     st.error('Age must be greater than 0.')

# Predict the outcome only if all inputs are valid
# if inputs_provided:
#     if st.button('Predict'):
#         input_data = np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi, age])
#         prediction = model.predict([input_data])
#         if prediction[0] == 1:
#             st.write('The patient is diabetic.')
#         else:
#             st.write('The patient is not diabetic.')


# Predict the outcome only if all inputs are valid

if st.button('Predict'):
    input_data = np.array([pregnancies, glucose, bp, skin_thickness, insulin, bmi, age])
    prediction = model.predict([input_data])
    if prediction[0] == 1:
        st.write('The patient is diabetic.')
    else:
        st.write('The patient is not diabetic.')
    st.markdown('Find the full code on my Github:  https://github.com/imShashanKashyap/Diabetes_Model')
st.markdown('Visit to know more about me: https://imshashankashyap.github.io/')
# Optionally, display the confusion matrix
# Note: You'll need to have the confusion matrix saved in a variable or file for this to work.
# For now, I've commented this section out.
# if st.checkbox('Show Confusion Matrix'):
#     st.write(confusion_matrix)  # Assuming you have the confusion matrix saved in this variable
