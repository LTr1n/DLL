import pandas as pd
import joblib
import numpy as np

def predict_stage(input_data):
    # Load model
    model = joblib.load("ml/lung_cancer_model.pkl")
    
    # Lấy danh sách đặc trưng từ mô hình đã huấn luyện
    features = model.feature_names_in_
    
    # Đảm bảo input_data có đầy đủ các cột, nếu thiếu thì gán giá trị mặc định là 0
    for feature in features:
        if feature not in input_data:
            input_data[feature] = 0  # Giả định bệnh nền mặc định là không có (0)
    
    # Chuyển dữ liệu đầu vào thành DataFrame, đảm bảo cột đúng thứ tự
    input_df = pd.DataFrame([input_data])[list(features)]
    
    # Đảm bảo kiểu dữ liệu nhất quán với khi huấn luyện
    input_df = input_df.astype(np.float64)
    
    # Dự đoán giai đoạn ung thư
    prediction = model.predict(input_df)
    return prediction[0]

# Ví dụ sử dụng
data_example = {
    'age': 65, 'tumor_size_mm': 35, 'survival_months': 12, 'performance_status': 1,
    'blood_pressure_systolic': 120, 'blood_pressure_diastolic': 80, 'blood_pressure_pulse': 40,
    'hemoglobin_level': 13.5, 'white_blood_cell_count': 6.2, 'platelet_count': 250,
    'albumin_level': 3.8, 'alkaline_phosphatase_level': 90, 'alanine_aminotransferase_level': 25,
    'aspartate_aminotransferase_level': 30, 'creatinine_level': 1.1, 'ldh_level': 180,
    'calcium_level': 9.5, 'phosphorus_level': 3.5, 'glucose_level': 100, 'potassium_level': 4.2,
    'sodium_level': 140, 'smoking_pack_years': 20, 'gender': 1, 'smoking_history': 1,
    'tumor_location': 2, 'treatment': 1, 'ethnicity': 0, 'insurance_type': 1, 'family_history': 0,
    'comorbidity_autoimmune_disease': 0, 'comorbidity_chronic_lung_disease': 0,
    'comorbidity_diabetes': 0, 'comorbidity_heart_disease': 0, 'comorbidity_hypertension': 0,
    'comorbidity_kidney_disease': 0, 'comorbidity_other': 0
}

predicted_stage = predict_stage(data_example)
print(f'Giai đoạn ung thư dự đoán: {predicted_stage}')
