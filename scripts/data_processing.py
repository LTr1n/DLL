import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def clean_data(input_file, output_file):
    # Đọc dữ liệu từ CSV
    df = pd.read_csv(input_file)
    
    # Xóa dòng có giá trị NULL
    df.dropna(inplace=True)
    
    # Chuẩn hóa tên cột
    df.columns = df.columns.str.strip().str.lower()
    
    # Đảm bảo cột age là numeric và giữ nguyên giá trị
    if 'age' in df.columns:
        age_values = df['age'].copy()  # Lưu giữ giá trị gốc của age
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # Loại bỏ cột patient_id nếu tồn tại
    df.drop(columns=[col for col in ['patient_id'] if col in df.columns], inplace=True)
    
    # Định dạng đúng các cột số (trừ age)
    numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'age']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Chuẩn hóa các cột số (trừ age)
    if numeric_columns:  # Chỉ chuẩn hóa nếu có cột số nào khác age
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Xử lý các cột phân loại
    categorical_mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'smoking_history': {'Never Smoked': 0, 'Former Smoker': 1, 'Current Smoker': 2},
        'stage': {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    categorical_columns = ['tumor_location', 'treatment', 'ethnicity', 'insurance_type', 'family_history']
    for col in categorical_columns:
        if col in df.columns:
            df[col], _ = pd.factorize(df[col])
    
    # Xử lý cột comorbidity
    comorbidity_columns = [col for col in df.columns if col.startswith('comorbidity_')]
    for col in comorbidity_columns:
        if col in df.columns:
            df[col] = df[col].map({'No': 0, 'Yes': 1})
    
    # Xóa các dòng có giá trị NULL sau khi xử lý
    df.dropna(inplace=True)
    
    # Khôi phục giá trị gốc của age nếu cần
    if 'age' in df.columns and 'age_values' in locals():
        df['age'] = age_values[df.index]  # Gán lại giá trị gốc cho age
    
    # Kiểm tra phân bố nhãn 'stage'
    if 'stage' in df.columns:
        print("Phân bố nhãn 'stage' trong dữ liệu gốc:")
        print(df['stage'].value_counts(normalize=True) * 100)
    
    # Kiểm tra age có bị thay đổi không
    if 'age' in df.columns:
        print("\nKiểm tra giá trị age:")
        print("Giá trị age đầu tiên trong dữ liệu gốc:", age_values.iloc[0])
        print("Giá trị age đầu tiên sau xử lý:", df['age'].iloc[0])
    
    # Lưu dữ liệu đã làm sạch
    df.to_csv(output_file, index=False)
    print(f"\nDữ liệu đã được làm sạch và lưu vào {output_file}")
    print("Cột age đã được giữ nguyên không thay đổi")
    
    return df

if __name__ == "__main__":
    input_file = "data/lung_cancer_data.csv"
    output_file = "data/lung_cancer_data_cleaned.csv"
    df_cleaned = clean_data(input_file, output_file)