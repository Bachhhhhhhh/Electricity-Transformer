import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import DEVICE

# ==============================================================================
# 1. TRAINING LOOP (DÙNG CHUNG CHO CẢ 2 TASK)
# ==============================================================================
def train_model(model, train_loader, val_loader, num_epochs=100, patience=15, lr=0.0005, model_name='best_model.pth', use_teacher_forcing=False):
    """
    Args:
        use_teacher_forcing (bool): 
            - False: Train bình thường (Dùng cho Task 1 hoặc Task 2 kiểu cũ).
            - True: Train với Scheduled Sampling (Chỉ dùng cho Task 2).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    # Cấu hình Scheduler cho Teacher Forcing
    tf_ratio = 0.9
    gamma = 0.9 
    
    print(f"--- Starting Training: {model_name} (Teacher Forcing={use_teacher_forcing}) ---")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            # --- XỬ LÝ LINH HOẠT ---
            if use_teacher_forcing:
                # Chỉ model Task 2 mới nhận tham số này
                outputs = model(inputs, target=labels, teacher_forcing_ratio=tf_ratio)
            else:
                # Task 1 hoặc Task 2 chạy thường (Autoregressive thuần túy)
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        
        # Giảm TF Ratio nếu đang dùng
        if use_teacher_forcing:
            tf_ratio = tf_ratio * gamma
        
        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                # Khi validate luôn tắt Teacher Forcing (dự báo thật)
                outputs = model(inputs) 
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_running_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}')
        
        # Early Stopping (Giữ nguyên)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
                
    model.load_state_dict(torch.load(model_name))
    
    # Plotting (Giữ nguyên)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title(f'Loss: {model_name}')
    plt.legend()
    plt.show()
    return model

# ==============================================================================
# 2. HELPER: INVERSE SCALING
# ==============================================================================
def inverse_transform_target(y_scaled, scaler, target_col_idx=-1):
    """
    Chuyển đổi dữ liệu từ dạng Scale [0,1] về dạng gốc.
    Hỗ trợ cả dạng 1 chiều (Task 1) và 2 chiều (Task 2).
    """
    # Lưu lại shape ban đầu (VD: N, 5) hoặc (N, 1)
    original_shape = y_scaled.shape
    
    # Flatten về dạng (N_total, 1) để xử lý
    y_flat = y_scaled.reshape(-1, 1)
    
    # Tạo ma trận giả (Dummy Matrix) với số cột bằng số features lúc fit scaler
    n_features = scaler.n_features_in_
    dummy = np.zeros((y_flat.shape[0], n_features))
    
    # Gán dữ liệu cần inverse vào đúng cột target
    dummy[:, target_col_idx] = y_flat[:, 0]
    
    # Inverse toàn bộ ma trận giả
    inversed_dummy = scaler.inverse_transform(dummy)
    
    # Trích xuất lại cột target và reshape về kích thước ban đầu
    return inversed_dummy[:, target_col_idx].reshape(original_shape)

def calculate_metrics(y_true, y_pred):
    """Tính các chỉ số đánh giá cơ bản"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # NSE (Nash-Sutcliffe Efficiency)
    numerator = np.sum((y_true_flat - y_pred_flat) ** 2)
    denominator = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    nse = 1 - (numerator / denominator)
    
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2 = r2_score(y_true_flat, y_pred_flat)
    
    return nse, mae, rmse, r2

# ==============================================================================
# 3. EVALUATION FOR TASK 1 (SINGLE STEP)
# ==============================================================================
def evaluate_task1(model, test_loader, scaler):
    """Đánh giá và vẽ biểu đồ cho Task 1 (Dự báo t+2)"""
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            # Output model Task 1 shape: (Batch) hoặc (Batch, 1)
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            actuals.append(labels.numpy())
            
    # Gộp batch
    y_pred_scaled = np.concatenate(preds, axis=0)
    y_true_scaled = np.concatenate(actuals, axis=0)
    
    # Inverse Scale
    y_pred = inverse_transform_target(y_pred_scaled, scaler)
    y_true = inverse_transform_target(y_true_scaled, scaler)
    
    # Tính Metrics
    nse, mae, rmse, r2 = calculate_metrics(y_true, y_pred)
    
    print(f"\n=== TASK 1 RESULTS (t+2 Prediction) ===")
    print(f"NSE:  {nse:.4f}")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f} °C")
    print(f"RMSE: {rmse:.4f} °C")
    
    # Vẽ biểu đồ đường liên tục (Zoom 300 điểm đầu tiên)
    plt.figure(figsize=(15, 5))
    plot_len = 300 # Số điểm muốn vẽ
    plt.plot(y_true[:plot_len], label='Ground Truth', color='black')
    plt.plot(y_pred[:plot_len], label='Prediction', color='green', linestyle='--')
    plt.title(f'Task 1: Single Step Prediction (First {plot_len} hours) - NSE: {nse:.3f}')
    plt.xlabel('Time Step')
    plt.ylabel('Oil Temperature')
    plt.legend()
    plt.show()

# ==============================================================================
# 4. EVALUATION FOR TASK 2 (SEQ2SEQ)
# ==============================================================================
def evaluate_task2(model, test_loader, scaler):
    """Đánh giá và vẽ biểu đồ Trajectory cho Task 2 (Seq 5 steps)"""
    model.eval()
    preds, actuals = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            # Output model Task 2 shape: (Batch, 5)
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            actuals.append(labels.numpy())
            
    y_pred = inverse_transform_target(np.concatenate(preds, axis=0), scaler)
    y_true = inverse_transform_target(np.concatenate(actuals, axis=0), scaler)
    
    # Global Metrics
    nse, mae, rmse, r2 = calculate_metrics(y_true, y_pred)
    
    print(f"\n=== TASK 2 RESULTS (Seq2Seq 5-Steps) ===")
    print(f"Global NSE:  {nse:.4f}")
    print(f"Global MAE:  {mae:.4f} °C")
    print(f"Global RMSE: {rmse:.4f} °C")
    
    # In chi tiết từng bước
    print("\n--- Step-wise Errors ---")
    steps = ["t+1", "t+2", "t+3", "t+4", "t+5"]
    for i in range(5):
        step_mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        print(f"Step {steps[i]}: MAE = {step_mae:.4f}")

    # Vẽ biểu đồ Trajectory (3 mẫu ngẫu nhiên)
    # Chọn mẫu 0, 100, 200
    indices = [0, 100, 200]
    plt.figure(figsize=(15, 4))
    
    for i, idx in enumerate(indices):
        plt.subplot(1, 3, i+1)
        # Vẽ chuỗi 5 bước
        x_axis = range(1, 6)
        plt.plot(x_axis, y_true[idx], 'o-k', label='Real')
        plt.plot(x_axis, y_pred[idx], 'x--r', label='Pred')
        plt.title(f'Sample Index: {idx}')
        plt.xlabel('Horizon (1-5)')
        if i==0: plt.ylabel('Oil Temp')
        if i==0: plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()