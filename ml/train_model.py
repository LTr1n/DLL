import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def train_model(data_path, model_path):
    # Đọc dữ liệu từ CSV
    df = pd.read_csv(data_path)
    
    # Giữ nguyên toàn bộ dữ liệu thay vì lấy mẫu 50%
    X = df.drop(columns=['stage'])
    y = df['stage']
    
    # Kiểm tra phân bố nhãn 'stage' trong dữ liệu
    print("Phân bố nhãn 'stage' trong dữ liệu:")
    print(df['stage'].value_counts(normalize=True) * 100)
    
    # Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Xử lý mất cân bằng lớp bằng SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)  # Giảm k_neighbors để phù hợp hơn
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_orig, y_train_orig)
    
    # Kiểm tra phân bố nhãn sau SMOTE
    print("\nPhân bố nhãn 'stage' sau SMOTE (tập huấn luyện):")
    print(pd.Series(y_train_balanced).value_counts(normalize=True) * 100)
    
    # Giữ nguyên các đặc trưng đầu vào
    X_train_selected = X_train_balanced
    X_test_selected = X_test_orig
    
    # Tìm tham số tối ưu bằng GridSearchCV với không gian tham số hợp lý hơn
    param_grid = {
        'n_estimators': [100, 200],  # Thử nghiệm số cây khác nhau
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_selected, y_train_balanced)
    
    # Lấy mô hình tốt nhất
    model = grid_search.best_estimator_
    print(f"\nTham số tốt nhất: {grid_search.best_params_}")
    
    # Đánh giá mô hình trên tập kiểm tra
    y_pred = model.predict(X_test_selected)
    accuracy = accuracy_score(y_test_orig, y_pred)
    print(f"Độ chính xác của mô hình trên tập kiểm tra: {accuracy:.2f}")
    
    # In báo cáo phân loại chi tiết
    print("\nBáo cáo phân loại chi tiết trên tập kiểm tra:")
    print(classification_report(y_test_orig, y_pred, target_names=['Stage I', 'Stage II', 'Stage III', 'Stage IV']))
    
    # Đánh giá trên toàn bộ tập dữ liệu gốc thay vì chỉ lấy mẫu 20%
    y_pred_orig = model.predict(X)
    accuracy_orig = accuracy_score(y, y_pred_orig)
    print(f"Độ chính xác trên toàn bộ tập dữ liệu: {accuracy_orig * 100:.2f}%")
    
    # Lưu mô hình
    joblib.dump(model, model_path)
    print(f"Mô hình đã được lưu tại {model_path}")
    
    return model

if __name__ == "__main__":
    data_file = "data/lung_cancer_data_cleaned.csv"  # Cập nhật đường dẫn file dữ liệu
    model_file = "ml/lung_cancer_model.pkl"
    train_model(data_file, model_file)