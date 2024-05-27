import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
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

electro_results_mapping = {
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
    "2. Tổn thương không thể khắc phục": 2,
    "3. Tổn thương có thể khắc phục": 3
}

def predict():
    # Get input values from the user
    age = age_entry.get()
    gender_str = gender_combobox.get()
    chest_pain_str = chest_pain_combobox.get()
    blood_pressure = blood_pressure_entry.get()
    cholesterol = cholesterol_entry.get()
    blood_sugar_str = blood_sugar_combobox.get()
    electro_results_str = electro_results_combobox.get()
    max_heart_rate = max_heart_rate_entry.get()
    angina_str = angina_combobox.get()
    oldpeak = oldpeak_entry.get()
    slope = slope_combobox.get()
    vessels_colored = vessels_colored_entry.get()
    thal_str = thal_combobox.get()

    # Check if any of the input values are empty
    if not all([age, gender_str, chest_pain_str, blood_pressure, cholesterol, blood_sugar_str,
                electro_results_str, max_heart_rate, angina_str, oldpeak, slope, vessels_colored, thal_str]):
        result_label.config(text="Vui lòng nhập đầy đủ các thông tin !!!")
    else:
        # Convert the input values to the appropriate data types
        try:
            age = float(age)
            gender = gender_mapping.get(gender_str, 0)
            chest_pain = chest_pain_mapping.get(chest_pain_str, 0)
            blood_pressure = float(blood_pressure)
            cholesterol = float(cholesterol)
            blood_sugar = blood_sugar_mapping.get(blood_sugar_str, 0)
            electro_results = electro_results_mapping.get(electro_results_str, 0)
            max_heart_rate = float(max_heart_rate)
            angina = angina_mapping.get(angina_str, 0)
            oldpeak = float(oldpeak)
            slope = float(slope)
            vessels_colored = float(vessels_colored)
            thal = thal_mapping.get(thal_str, 0)

            # Create a new sample array
            new_sample = np.array([[age, gender, chest_pain, blood_pressure, cholesterol, blood_sugar,
                                    electro_results, max_heart_rate, angina, oldpeak, slope,
                                    vessels_colored, thal]])

            # Standardize the input data
            scaled_sample = sc.transform(new_sample)

            # Make predictions with the loaded model
            prediction = loaded_model.predict(scaled_sample)
            percentage_disease = prediction[0][0] * 100
            binary_prediction = "Có" if prediction > 0.5 else "Không"

            # Display the results
            result_label.config(text=f"Tỉ lệ mắc bệnh tim : {percentage_disease:.2f}%\nDự đoán mắc bệnh tim: {binary_prediction}")

            # Plot the pie chart for the percentage of disease
            fig, ax = plt.subplots(figsize=(4, 4))  # Smaller size
            ax.pie([percentage_disease, 100 - percentage_disease], labels=['Bị bệnh', 'Không bị bệnh'], autopct='%1.1f%%', startangle=90)
            ax.set_title('Tỉ lệ mắc bệnh tim')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

            # Embed the plot in the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.draw()
            canvas.get_tk_widget().grid(row=1, column=2, rowspan=15, padx=10, pady=5, sticky="n")

        except ValueError as e:
            result_label.config(text="Lỗi nhập liệu, vui lòng kiểm tra lại!")

# Create the main window
root = tk.Tk()    
root.title("Dự đoán bệnh tim")

# Create input labels and entry widgets
info_header_label = tk.Label(root, text="Thông tin người bệnh", font=('Helvetica', 14, 'bold'))
info_header_label.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

age_label = tk.Label(root, text="Tuổi:")
age_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
age_entry = tk.Entry(root)
age_entry.grid(row=1, column=1, padx=10, pady=5)

gender_label = tk.Label(root, text="Giới tính:")
gender_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
gender_combobox = ttk.Combobox(root, values=list(gender_mapping.keys()), state="readonly")
gender_combobox.grid(row=2, column=1, padx=10, pady=5)

chest_pain_label = tk.Label(root, text="Loại đau ngực:")
chest_pain_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
chest_pain_combobox = ttk.Combobox(root, values=list(chest_pain_mapping.keys()), state="readonly")
chest_pain_combobox.grid(row=3, column=1, padx=10, pady=5)

blood_pressure_label = tk.Label(root, text="Huyết áp tâm trương:")
blood_pressure_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")
blood_pressure_entry = tk.Entry(root)
blood_pressure_entry.grid(row=4, column=1, padx=10, pady=5)

cholesterol_label = tk.Label(root, text="Cholesterol trong huyết thanh:")
cholesterol_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")
cholesterol_entry = tk.Entry(root)
cholesterol_entry.grid(row=5, column=1, padx=10, pady=5)

blood_sugar_label = tk.Label(root, text="Đo lượng đường trong máu:")
blood_sugar_label.grid(row=6, column=0, padx=10, pady=5, sticky="w")
blood_sugar_combobox = ttk.Combobox(root, values=list(blood_sugar_mapping.keys()), state="readonly")
blood_sugar_combobox.grid(row=6, column=1, padx=10, pady=5)

electro_results_label = tk.Label(root, text="Chỉ số điện tâm đồ:")
electro_results_label.grid(row=7, column=0, padx=10, pady=5, sticky="w")
electro_results_combobox = ttk.Combobox(root, values=list(electro_results_mapping.keys()), state="readonly")
electro_results_combobox.grid(row=7, column=1, padx=10, pady=5)

max_heart_rate_label = tk.Label(root, text="Nhịp tim tối đa trên phút:")
max_heart_rate_label.grid(row=8, column=0, padx=10, pady=5, sticky="w")
max_heart_rate_entry = tk.Entry(root)
max_heart_rate_entry.grid(row=8, column=1, padx=10, pady=5)

angina_label = tk.Label(root, text="Đau thắt ngực do vận động:")
angina_label.grid(row=9, column=0, padx=10, pady=5, sticky="w")
angina_combobox = ttk.Combobox(root, values=list(angina_mapping.keys()), state="readonly")
angina_combobox.grid(row=9, column=1, padx=10, pady=5)

oldpeak_label = tk.Label(root, text="Chỉ số Oldpeak:")
oldpeak_label.grid(row=10, column=0, padx=10, pady=5, sticky="w")
oldpeak_entry = tk.Entry(root)
oldpeak_entry.grid(row=10, column=1, padx=10, pady=5)

slope_label = tk.Label(root, text="Độ dốc đoạn ST:")
slope_label.grid(row=11, column=0, padx=10, pady=5, sticky="w")
slope_combobox = ttk.Combobox(root, values=["0", "1", "2"], state="readonly")
slope_combobox.grid(row=11, column=1, padx=10, pady=5)

vessels_colored_label = tk.Label(root, text="Số mạch màu:")
vessels_colored_label.grid(row=12, column=0, padx=10, pady=5, sticky="w")
vessels_colored_entry = tk.Entry(root)
vessels_colored_entry.grid(row=12, column=1, padx=10, pady=5)

thal_label = tk.Label(root, text="Thalassemia:")
thal_label.grid(row=13, column=0, padx=10, pady=5, sticky="w")
thal_combobox = ttk.Combobox(root, values=list(thal_mapping.keys()), state="readonly")
thal_combobox.grid(row=13, column=1, padx=10, pady=5)

# Create the predict button
predict_button = tk.Button(root, text="Dự đoán", command=predict)
predict_button.grid(row=14, column=0, columnspan=2, pady=10)

# Create the result label
result_label = tk.Label(root, text="", font=('Helvetica', 12))
result_label.grid(row=15, column=0, columnspan=2, pady=10)

# Start the main loop
root.mainloop()
