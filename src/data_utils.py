import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from src.augmentation import DataAugmenter

def feature_engineering(df):
    """Tạo đặc trưng nâng cao: Cyclical time, Lags, EWM, Interactions"""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)
    
    # 1. Cyclical Time
    timestamp_s = data['date'].map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    week = 7 * day
    data['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    data['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    data['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    data['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))

    # 2. Physics & Load
    data['Total_Load'] = data['HUFL'] + data['HULL'] + data['MUFL'] + \
                         data['MULL'] + data['LUFL'] + data['LULL']
    data['Total_Load_Sq'] = data['Total_Load'] ** 2
    
    # 3. Lags
    for i in [1, 2, 3]:
        data[f'OT_lag_{i}h'] = data['OT'].shift(i)
    data['OT_lag_24h'] = data['OT'].shift(24)

    # 4. Rolling
    data['OT_roll_mean_6h'] = data['OT'].shift(1).rolling(window=6).mean()
    data['OT_roll_std_6h']  = data['OT'].shift(1).rolling(window=6).std()
    data['Load_roll_mean_3h'] = data['Total_Load'].shift(1).rolling(window=3).mean()

    # 5. Clean up
    data.drop(['date', 'index'], axis=1, errors='ignore', inplace=True)
    data.dropna(inplace=True)
    
    # Đưa target về cuối
    cols = [c for c in data.columns if c != 'OT'] + ['OT']
    return data[cols]

def create_sliding_window(dataset, input_width, prediction_width, offset):
    """Tạo X, y cho Time Series"""
    X, y = [], []
    target_col_idx = -1 
    start_index = 0
    end_index = len(dataset) - input_width - offset - prediction_width
    
    for i in range(start_index, end_index + 1):
        indices_x = range(i, i + input_width)
        start_y = i + input_width + offset
        end_y = start_y + prediction_width
        X.append(dataset[indices_x])
        y.append(dataset[start_y : end_y, target_col_idx])
        
    return np.array(X), np.array(y)

def prepare_dataloaders(df, input_width, pred_width, offset, batch_size=32, augment=True, noise_level=0.01):
    """Pipeline đầy đủ: Split -> Scale -> Window -> Loader"""
    # 1. Split
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # --- ÁP DỤNG AUGMENTATION Ở ĐÂY (CHỈ CHO TRAIN) ---
    if augment:
        augmenter = DataAugmenter(noise_level=0.01, seed=42) # Khởi tạo
        train_df = augmenter.run(train_df, auto_concat=True) # Gấp đôi dữ liệu train
    
    # 2. Scale (Sau khi đã augment)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df) # Fit trên dữ liệu đã gộp
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # 3. Create Windows
    X_train, y_train = create_sliding_window(train_scaled, input_width, pred_width, offset)
    X_val, y_val = create_sliding_window(val_scaled, input_width, pred_width, offset)
    X_test, y_test = create_sliding_window(test_scaled, input_width, pred_width, offset)
    
    # 4. To Tensor
    train_data = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    val_data = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    test_data = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    
    # 5. Loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler