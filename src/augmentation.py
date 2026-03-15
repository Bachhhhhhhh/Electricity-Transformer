# src/augmentation.py
import numpy as np
import pandas as pd

class DataAugmenter:
    def __init__(self, noise_level=0.01, seed=42):
        """
        Khởi tạo bộ Augmentation.
        :param noise_level: Độ lệch chuẩn của nhiễu (0.01 ~ 1%).
        :param seed: Random seed để cố định kết quả.
        """
        self.noise_level = noise_level
        self.seed = seed

    def add_gaussian_noise(self, df):
        """
        Thêm nhiễu Gaussian vào các cột dữ liệu số.
        """
        np.random.seed(self.seed)
        df_noisy = df.copy()
        
        # Lọc các cột số (trừ date và các cột category nếu có)
        # Giả sử bạn đã scale dữ liệu rồi thì toàn bộ là số
        numeric_cols = df_noisy.select_dtypes(include=[np.number]).columns
        
        # Nếu có cột target là số nguyên (classification) thì phải loại ra
        # Nhưng bài này là dự báo nhiệt độ (regression) nên cộng nhiễu vào target cũng OK
        
        for col in numeric_cols:
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=len(df_noisy))
            df_noisy[col] = df_noisy[col] + noise
            
        return df_noisy

    def run(self, train_df, auto_concat=True):
        """
        Chạy quy trình Augmentation.
        :param train_df: DataFrame tập Train gốc.
        :param auto_concat: Nếu True, tự động gộp (Gốc + Nhiễu). Nếu False, chỉ trả về tập Nhiễu.
        """
        print(f"--- Bắt đầu Data Augmentation (Noise Level={self.noise_level}) ---")
        print(f"Kích thước gốc: {train_df.shape}")
        
        # Tạo bản sao có nhiễu
        noisy_df = self.add_gaussian_noise(train_df)
        
        if auto_concat:
            # Gộp lại: Dữ liệu gốc + Dữ liệu nhiễu
            augmented_df = pd.concat([train_df, noisy_df], axis=0).reset_index(drop=True)
            print(f"-> Đã gộp dữ liệu. Kích thước mới: {augmented_df.shape}")
            return augmented_df
        else:
            print(f"-> Trả về dữ liệu nhiễu riêng lẻ.")
            return noisy_df