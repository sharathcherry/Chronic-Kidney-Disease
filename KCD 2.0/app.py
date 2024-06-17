import streamlit as st
import pandas as pd
import joblib

# Load the model and label encoders
model = joblib.load("C:\\Users\\Sharath Chandra\\Desktop\\KCD 2.0\\kidneydisease.pkl")
label_encoders = joblib.load("C:\\Users\\Sharath Chandra\\Desktop\\KCD 2.0\\kidneydisea1se.pkl")

# Create the user interface
st.title("Kidney Disease Prediction App")

st.sidebar.header('Patient Data')
def user_input_features():
    age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
    bp = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    sg = st.sidebar.number_input("Specific Gravity", min_value=1.000, max_value=1.050, value=1.025)
    al = st.sidebar.number_input("Albumin", min_value=0.0, max_value=5.0, value=0.0)
    su = st.sidebar.number_input("Sugar", min_value=0.0, max_value=5.0, value=0.0)
    rbc = st.sidebar.selectbox("Red Blood Cells", ("normal", "abnormal"))
    pc = st.sidebar.selectbox("Pus Cell", ("normal", "abnormal"))
    pcc = st.sidebar.selectbox("Pus Cell Clumps", ("notpresent", "present"))
    ba = st.sidebar.selectbox("Bacteria", ("notpresent", "present"))
    bgr = st.sidebar.number_input("Blood Glucose Random", min_value=0.0, max_value=500.0, value=100.0)
    bu = st.sidebar.number_input("Blood Urea", min_value=0.0, max_value=200.0, value=40.0)
    sc = st.sidebar.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.2)
    sod = st.sidebar.number_input("Sodium", min_value=0.0, max_value=200.0, value=135.0)
    pot = st.sidebar.number_input("Potassium", min_value=0.0, max_value=10.0, value=4.5)
    hemo = st.sidebar.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=15.0)
    pcv = st.sidebar.number_input("Packed Cell Volume", min_value=0.0, max_value=60.0, value=45.0)
    wc = st.sidebar.number_input("White Blood Cell Count", min_value=0.0, max_value=30000.0, value=8000.0)
    rc = st.sidebar.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=5.0)
    htn = st.sidebar.selectbox("Hypertension", ("yes", "no"))
    dm = st.sidebar.selectbox("Diabetes Mellitus", ("yes", "no"))
    cad = st.sidebar.selectbox("Coronary Artery Disease", ("yes", "no"))
    appet = st.sidebar.selectbox("Appetite", ("good", "poor"))
    pe = st.sidebar.selectbox("Pedal Edema", ("yes", "no"))
    ane = st.sidebar.selectbox("Anemia", ("yes", "no"))

    data = {
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': sc,
        'sod': sod,
        'pot': pot,
        'hemo': hemo,
        'pcv': pcv,
        'wc': wc,
        'rc': rc,
        'htn': htn,
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Encode categorical variables
for col in label_encoders:
    if col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
kidney_disease = 'notckd' if prediction[0] == 1 else 'ckd'
st.write(kidney_disease)

st.subheader('Prediction Probability')
st.write(prediction_proba)
