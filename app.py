import streamlit as st
import numpy as np
import pickle

# Load model
loaded_model = pickle.load(open('model_uas.pkl', 'rb'))

# Streamlit App
st.title("Nama: Sekar Septi Ardiyati")
st.title("NIM: 2021230003")
st.title("Prediksi Premi Asuransi")

# Input Form
age = st.number_input("Umur (Age)", min_value=0, max_value=120, value=30, step=1)
sex = st.selectbox("Jenis Kelamin (Sex)", ["Perempuan (0)", "Laki-Laki (1)"])
bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Jumlah Anak (Children)", min_value=0, max_value=10, value=0, step=1)
smoker = st.selectbox("Perokok (Smoker)", ["Tidak (0)", "Ya (1)"])

# Convert Inputs
sex = int(sex.split()[-1].strip("()"))
smoker = int(smoker.split()[-1].strip("()"))

# Prediction Button
if st.button("Prediksi"):
    X = np.array([age, sex, bmi, children, smoker]).reshape(1, -1)
    prediction = loaded_model.predict(X)
    st.write(f"Prediksi Pembayaran Premi: ${prediction[0]:,.2f}")
