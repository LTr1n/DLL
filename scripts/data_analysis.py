import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_data(df):
    if 'age' in df.columns:
        print("Thống kê cơ bản về tuổi:")
        print(df['age'].describe())
    
    if 'gender' in df.columns:
        print("\nTỷ lệ giới tính (0: Female, 1: Male):")
        print(df['gender'].value_counts(normalize=True) * 100)
    
    if 'smoking_history' in df.columns:
        print("\nTỷ lệ tiền sử hút thuốc (0: Never, 1: Former, 2: Current):")
        print(df['smoking_history'].value_counts(normalize=True) * 100)
    
    if 'tumor_size_mm' in df.columns:
        print("\nKích thước khối u trung bình:")
        print(df['tumor_size_mm'].mean())
    
    if 'stage' in df.columns:
        print("\nPhân bố giai đoạn ung thư:")
        print(df['stage'].value_counts(normalize=True) * 100)
    
    if 'treatment' in df.columns:
        print("\nPhương pháp điều trị phổ biến:")
        print(df['treatment'].value_counts(normalize=True) * 100)
    
    if 'survival_months' in df.columns:
        print("\nSố tháng sống sót trung bình:")
        print(df['survival_months'].mean())

def visualize_data(df):
    if 'stage' in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(x=df['stage'], palette='coolwarm')
        plt.title("Phân bố giai đoạn ung thư phổi")
        plt.xlabel("Giai đoạn")
        plt.ylabel("Số lượng bệnh nhân")
        plt.show()
    
    if 'age' in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df['age'], bins=20, kde=True, color='blue')
        plt.title("Phân bố tuổi của bệnh nhân")
        plt.xlabel("Tuổi")
        plt.ylabel("Số lượng bệnh nhân")
        plt.show()
    
    if {'tumor_size_mm', 'survival_months', 'stage'}.issubset(df.columns):
        plt.figure(figsize=(8,5))
        sample_df = df.sample(frac=0.1, random_state=42)
        sns.scatterplot(x=sample_df['tumor_size_mm'], y=sample_df['survival_months'], 
                        hue=sample_df['stage'], palette='coolwarm')
        plt.title("Tương quan giữa kích thước khối u và số tháng sống sót")
        plt.xlabel("Kích thước khối u")
        plt.ylabel("Số tháng sống sót")
        plt.show()
    
    if 'treatment' in df.columns:
        plt.figure(figsize=(8,5))
        sns.countplot(x=df['treatment'], palette='coolwarm')
        plt.title("Phân bố phương pháp điều trị")
        plt.xlabel("Phương pháp điều trị")
        plt.ylabel("Số lượng bệnh nhân")
        plt.show()
if __name__ == "__main__":
    input_file = "data/lung_cancer_data_cleaned.csv"
    df = pd.read_csv(input_file)
    analyze_data(df)
    visualize_data(df)