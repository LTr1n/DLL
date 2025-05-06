from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

app = Flask(__name__)

# Load data
data_file = "data/lung_cancer_data.csv"
data = pd.read_csv(data_file)

# Load model
model = joblib.load("ml/lung_cancer_model.pkl")

# File lưu lịch sử dự đoán
prediction_history_file = "data/prediction_history.csv"

@app.route('/')
def index():
    # Đọc lịch sử dự đoán nếu file tồn tại
    prediction_history = pd.DataFrame()
    if os.path.exists(prediction_history_file):
        prediction_history = pd.read_csv(prediction_history_file)
    
    return render_template('index.html', 
                         tables=[data.to_html(classes='data', index=False)], 
                         titles=data.columns.values,
                         prediction_history=prediction_history.to_html(classes='data', index=False) if not prediction_history.empty else None,
                         prediction_table=None,  # Đặt mặc định là None khi vào trang chính
                         patient_data=None,
                         prediction_result=None,
                         prediction_time=None,
                         confidence=None)

@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        new_patient = {col: request.form.get(col, 0) for col in data.columns}
        
        # Chuyển đổi kiểu dữ liệu phù hợp
        for key in new_patient.keys():
            try:
                new_patient[key] = float(new_patient[key])  # Chuyển thành số nếu có thể
            except ValueError:
                pass  # Giữ nguyên nếu không thể chuyển đổi
        
        new_patient_df = pd.DataFrame([new_patient])
        new_patient_df.to_csv(data_file, mode='a', header=False, index=False)
        return redirect(url_for('index'))
    
    return render_template('add_patient.html')

@app.route('/plot')
def plot():
    plt.figure(figsize=(8, 6))
    sns.histplot(data['age'], bins=20, kde=True)
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')

    # Tạo thư mục static nếu chưa tồn tại
    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/age_distribution.png")
    return redirect(url_for('index'))

@app.route('/visualize')
def visualize():
    images = [
        "age_distribution.png",
        "treatment_distribution.png",
        "tumor_size_survival.png",
        "stage_distribution.png"
    ]

    # Chỉ hiển thị các hình ảnh tồn tại
    existing_images = [img for img in images if os.path.exists(os.path.join("static", img))]

    return render_template('visualize.html', images=existing_images)

@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))
    
    # Đọc file
    ext = file.filename.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        return redirect(url_for('index'))

    # Kiểm tra cột đầu vào
    required_cols = ['age', 'gender', 'smoking_history', 'tumor_size_mm', 'tumor_location', 'treatment', 
                     'survival_months', 'ethnicity', 'insurance_type', 'family_history', 'comorbidity_diabetes', 
                     'comorbidity_hypertension', 'comorbidity_heart_disease', 'comorbidity_chronic_lung_disease', 
                     'comorbidity_kidney_disease', 'comorbidity_autoimmune_disease', 'comorbidity_other', 
                     'performance_status', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                     'blood_pressure_pulse', 'hemoglobin_level', 'white_blood_cell_count', 'platelet_count', 
                     'albumin_level', 'alkaline_phosphatase_level', 'alanine_aminotransferase_level', 
                     'aspartate_aminotransferase_level', 'creatinine_level', 'ldh_level', 'calcium_level', 
                     'phosphorus_level', 'glucose_level', 'potassium_level', 'sodium_level', 'smoking_pack_years']
    
    if not all(col in df.columns for col in required_cols):
        return "File không đúng định dạng!", 400

    # Tiền xử lý dữ liệu
    df.fillna(0, inplace=True)

    # Kiểm tra model trước khi dự đoán
    if model is None:
        return "Mô hình chưa được tải!", 500

    # Dự đoán
    predictions = model.predict(df)
    
    # Lấy dữ liệu của bệnh nhân đầu tiên trong file (giả sử file chỉ chứa 1 bệnh nhân)
    patient_data = df.iloc[0]
    prediction_result = predictions[0]
    prediction_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence = 95  # Giả lập độ tin cậy (có thể thay bằng logic thực tế từ model nếu có)

    # Thêm kết quả dự đoán vào DataFrame lịch sử
    df['predicted_stage'] = predictions
    df['prediction_time'] = prediction_time

    # Lưu kết quả dự đoán vào lịch sử
    if not os.path.exists(prediction_history_file):
        df.to_csv(prediction_history_file, index=False)
    else:
        df.to_csv(prediction_history_file, mode='a', header=False, index=False)

    # Đọc lại lịch sử dự đoán
    prediction_history = pd.read_csv(prediction_history_file)

    # Render lại trang dự đoán ở Tab 2 mà không chuyển sang Tab 1
    return render_template('index.html', 
                           tables=[df.to_html(classes='data', index=False)], 
                           titles=df.columns.values,
                           prediction_table=df.to_html(classes='data', index=False),
                           patient_data=patient_data,
                           prediction_result=prediction_result,
                           prediction_time=prediction_time,
                           confidence=confidence,
                           prediction_history=prediction_history.to_html(classes='data', index=False),
                           active_tab=1)  # Thêm biến active_tab để giữ tab 2


if __name__ == '__main__':
    app.run(debug=True, port=5004)