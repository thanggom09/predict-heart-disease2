import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd

# Import the dataset
dataset = pd.read_csv('heart.csv')
x = dataset.iloc[:, 0:13].values

# Standardize the data
sc = StandardScaler()
x = sc.fit_transform(x)

# Load the pre-trained model
loaded_model = tf.keras.models.load_model("heart_disease_model.h5")

# Mapping dictionaries
chest_pain_mapping = {
    "0.Không đau ngực": 0,
    "1.Đau thắt ngực ổn định": 1,
    "2.Đau thắt ngực không ổn định": 2,
    "3.Biến thể đau thắt ngực": 3,
    "4.Đau thắt ngực vi mạch": 4
}

gender_mapping = {
    "0.Nữ": 0,
    "1.Nam": 1
}

blood_sugar_mapping = {
    "0.<= 120mg/dl": 0,
    "1.> 120mg/dl": 1
}

electro_results_mapping ={
    "0. Bình thường": 0,
    "1. Có sóng ST-T biến đổi không bình thường": 1,
    "2. Có sóng ST-T bất thường": 2
}

angina_mapping = {
    "Không": 0,
    "Có": 1
}

thal_mapping = {
    "0. Không bị": 0,
    "1. Bị nhẹ": 1,
    "2. Tổn thương khổn thể khắc phục": 2,
    "3. Tổn thương có thể khắc phục": 3
}

def predict(age, gender_str, chest_pain_str, blood_pressure, cholesterol, blood_sugar_str,
            electro_results_str, max_heart_rate, angina_str, oldpeak, slope, vessels_colored, thal_str):
    # Check if any input values are empty
    if not all([age, gender_str, chest_pain_str, blood_pressure, cholesterol, blood_sugar_str,
                electro_results_str, max_heart_rate, angina_str, oldpeak, slope, vessels_colored, thal_str]):
        return None, "Vui lòng nhập đầy đủ các thông tin !!!"

    # Convert input values to appropriate data types
    age = int(age)
    gender = gender_mapping.get(gender_str, 0)
    chest_pain = chest_pain_mapping.get(chest_pain_str, 0)
    blood_pressure = int(blood_pressure)
    cholesterol = int(cholesterol)
    blood_sugar = blood_sugar_mapping.get(blood_sugar_str, 0)
    electro_results = electro_results_mapping.get(electro_results_str, 0)
    max_heart_rate = int(max_heart_rate)
    angina = angina_mapping.get(angina_str, 0)
    oldpeak = int(oldpeak)
    slope = int(slope)
    vessels_colored = int(vessels_colored)
    thal = thal_mapping.get(thal_str, 0)

    # Create a new sample array
    new_sample = np.array([[age, gender, chest_pain, blood_pressure, cholesterol, blood_sugar,
                            electro_results, max_heart_rate, angina, oldpeak, slope,
                            vessels_colored, thal]])

    # Standardize the input data
    scaled_sample = sc.transform(new_sample)

    # Make predictions with the loaded model
    prediction = loaded_model.predict(scaled_sample)
    binary_prediction = "Có" if prediction > 0.5 else "Không"

    return prediction, binary_prediction


# Streamlit UI
st.title("Dự đoán bệnh tim")

# Input widgets
age = st.number_input("Tuổi", value=0)
gender_str = st.selectbox("Giới tính", list(gender_mapping.keys()))
chest_pain_str = st.selectbox("Loại đau ngực", list(chest_pain_mapping.keys()))
blood_pressure = st.number_input("Huyết áp tâm trương", value=0)
cholesterol = st.number_input("Cholesterol trong huyết thanh", value=0)
blood_sugar_str = st.selectbox("Đo lượng đường trong máu", list(blood_sugar_mapping.keys()))
electro_results_str = st.selectbox("Chỉ số huyết áp", list(electro_results_mapping.keys()))
max_heart_rate = st.number_input("Nhịp tim tối đa trên phút", value=0)
angina_str = st.selectbox("Đau thắt ngực do vận động", list(angina_mapping.keys()))
oldpeak = st.number_input("Chỉ số Oldpeak", value=0)
slope = st.number_input("Độ dốc đỉnh đoạn ST của bài kiểm tra tăng cường", value=0)
vessels_colored = st.number_input("Số mạch máu chính (0-3) được nhuộm bằng fluoroscopy", value=0)
thal_str = st.selectbox("Bệnh Thalassemia (thal)", list(thal_mapping.keys()))

# Update the predict() function to handle missing inputs and return an error message if any input is missing
if st.button("Dự đoán"):
    prediction, binary_prediction = predict(age, gender_str, chest_pain_str, blood_pressure, cholesterol, blood_sugar_str,
                                             electro_results_str, max_heart_rate, angina_str, oldpeak, slope,
                                             vessels_colored, thal_str)
    if prediction is None:
        st.write(binary_prediction)  # Display the error message
    else:
        st.write(f"Tỉ lệ mắc bệnh tim : {prediction[0][0]*100:.2f}%")
        st.write(f"Dự đoán mắc bệnh tim: {binary_prediction}")
