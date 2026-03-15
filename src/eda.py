# ts_eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

class OTEDA:
    def __init__(self, df, style='whitegrid', palette='viridis'):
        """
        Khởi tạo class phân tích Time Series.
        :param df: DataFrame đã clean, index phải là DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Lỗi: Index của DataFrame phải là định dạng Datetime.")
        
        self.df = df
        sns.set(style=style, palette=palette)
        # Cấu hình font chữ cho rõ ràng
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
    
    

    def plot_monthly(self, columns, title="Tổng giá trị theo tháng"):
        """
        Vẽ biểu đồ theo tháng. 
        - Chỉ hiển thị Năm ở trục X.
        - Chỉ giữ lại Grid line NGANG (axis='y').
        """
        # 1. Xử lý input
        if isinstance(columns, str):
            columns = [columns]

        # 2. Gom nhóm dữ liệu
        monthly_data = self.df[columns].resample('MS').mean()
        y_label = "Giá trị trung bình"
        
        # 3. Khởi tạo
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # 4. Vẽ biểu đồ
        for col in monthly_data.columns:
            ax.plot(monthly_data.index, monthly_data[col], 
                    marker='o',       
                    linewidth=2.5,    
                    label=col)        
        
        # 5. Xử lý trục X (Chỉ hiện Năm)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=12)

        # 6. XỬ LÝ GRID LINE (CHỈ GIỮ NGANG)
        # axis='y': Chỉ hiện lưới ngang
        # axis='x': Chỉ hiện lưới dọc
        # axis='both': Hiện cả hai (mặc định)
        ax.grid(visible=True, axis='y', linestyle='--', alpha=0.5, color='gray')
        
        # Bỏ khung viền trên và phải
        sns.despine() 

        # 7. Trang trí
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel(y_label, fontsize=12)
        plt.xlabel("") 
        
        # Chú thích
        plt.legend(title="", frameon=False, fontsize=12)
        
        plt.tight_layout()
        plt.show()

    def plot_resample(self, column, rule='D', agg_func='mean', title="Biểu đồ Resample"):
        """
        Vẽ biểu đồ sau khi gom nhóm dữ liệu (ví dụ: trung bình theo ngày, tháng).
        :param rule: 'D' (Ngày), 'W' (Tuần), 'M' (Tháng).
        """
        data_resampled = self.df[column].resample(rule).agg(agg_func)
        
        plt.figure(figsize=(15, 6))
        data_resampled.plot(linewidth=2)
        plt.title(f"{title} - Gom nhóm theo {rule} ({agg_func})", fontsize=16)
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

    def plot_seasonal_boxplot(self, column, by='hour', title="Phân phối theo chu kỳ"):
        """
        Vẽ Boxplot để xem tính mùa vụ (ví dụ: gió mạnh vào giờ nào trong ngày).
        :param by: 'hour' (giờ trong ngày), 'month' (tháng trong năm), 'weekday'.
        """
        df_temp = self.df.copy()
        
        if by == 'hour':
            df_temp['Time_Unit'] = df_temp.index.hour
            xlabel = "Giờ trong ngày (0-23)"
        elif by == 'month':
            df_temp['Time_Unit'] = df_temp.index.month
            xlabel = "Tháng trong năm (1-12)"
        elif by == 'weekday':
            df_temp['Time_Unit'] = df_temp.index.weekday
            xlabel = "Thứ trong tuần (0=Thứ 2)"
        else:
            raise ValueError("Tham số 'by' chỉ nhận: 'hour', 'month', 'weekday'")

        plt.figure(figsize=(14, 7))
        sns.boxplot(x='Time_Unit', y=column, data=df_temp)
        plt.title(f"{title} - {column} theo {by}", fontsize=16)
        plt.xlabel(xlabel)
        plt.ylabel(column)
        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, title="Ma trận tương quan"):
        """Vẽ Heatmap để xem tương quan giữa các biến."""
        plt.figure(figsize=(10, 8))
        corr = self.df.corr()
        mask = corr.isnull() # Che các giá trị NaN nếu có
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, mask=mask)
        plt.title(title, fontsize=16)
        plt.show()

    def plot_scatter(self, col_x, col_y, title="Biểu đồ tán xạ (Scatter Plot)"):
        """
        Vẽ scatter plot. Cực quan trọng để vẽ Power Curve (Gió vs Điện).
        """
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.df[col_x], y=self.df[col_y], alpha=0.5, edgecolor=None)
        plt.title(f"{title}: {col_x} vs {col_y}", fontsize=16)
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def plot_decomposition(self, column, period=None, resample_rule=None, model='additive'):
        """
        Phân rã chuỗi thời gian. Hỗ trợ Resample trước khi phân tích để nhìn xu hướng rõ hơn.
        
        :param period: Số chu kỳ. Ví dụ: dữ liệu tháng thì period=12 (1 năm).
        :param resample_rule: Quy tắc gom nhóm. 'D' (Ngày), 'W' (Tuần), 'M' (Tháng).
                              Nếu None thì dùng dữ liệu gốc (Hourly).
        """
        # 1. Xử lý dữ liệu đầu vào (Resample nếu cần)
        if resample_rule:
            # Nếu gom nhóm, ta tính trung bình
            series = self.df[column].resample(resample_rule).mean().dropna()
            title_prefix = f" (Gom theo {resample_rule})"
        else:
            series = self.df[column].dropna()
            title_prefix = ""

        # 2. Tự động đoán Period nếu người dùng không nhập
        if period is None:
            if resample_rule == 'M': # Dữ liệu tháng -> Chu kỳ năm
                period = 12
            elif resample_rule == 'D': # Dữ liệu ngày -> Chu kỳ năm
                period = 365
            elif resample_rule == 'W': # Dữ liệu tuần -> Chu kỳ năm
                period = 52
            else: # Dữ liệu giờ -> Chu kỳ ngày
                period = 24 

        print(f"Đang phân rã với period = {period}...")

        # 3. Thực hiện phân rã
        result = seasonal_decompose(series, model=model, period=period)
        
        # 4. Vẽ biểu đồ
        fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        # Vẽ Observed
        result.observed.plot(ax=ax1, color='#442c75')
        ax1.set_ylabel('Observed')

        # Vẽ Trend
        result.trend.plot(ax=ax2, color='#442c75')
        ax2.set_ylabel('Trend')

        # Vẽ Residual
        result.resid.plot(ax=ax4, color='#6b5b95', linestyle='None', marker='.')
        ax4.set_ylabel('Residual')

        plt.title("Phân tích dài hạn (Long-term)")
        plt.show()

        # --- HÌNH 2: CHI TIẾT MÙA VỤ (Seasonal) ---
        plt.figure(figsize=(15, 4))
        # Chỉ vẽ 1 tuần (168 giờ) để thấy rõ sóng ngày đêm
        result.seasonal.iloc[:720].plot(color='#442c75')
        plt.title("Chi tiết chu kỳ Mùa vụ trong 1 tháng (Zoom in)")
        plt.ylabel('Seasonal')
        plt.grid(True)
        plt.show()
    def check_stationarity(self, column):
        """Kiểm tra tính dừng (Stationarity) bằng Augmented Dickey-Fuller Test."""
        print(f"--- Kết quả kiểm định ADF cho cột: {column} ---")
        result = adfuller(self.df[column].dropna())
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
            
        if result[1] <= 0.05:
            print("=> Kết luận: Chuỗi dữ liệu CÓ tính dừng (Stationary).")
        else:
            print("=> Kết luận: Chuỗi dữ liệu KHÔNG có tính dừng (Non-Stationary).")
        print("-" * 50)

    def plot_acf_pacf(self, column, lags=48):
        """Vẽ biểu đồ tự tương quan (ACF) và tự tương quan riêng phần (PACF)."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        plot_acf(self.df[column].dropna(), lags=lags, ax=ax1)
        ax1.set_title(f'Autocorrelation (ACF) - {column}')
        
        plot_pacf(self.df[column].dropna(), lags=lags, ax=ax2, method='ywm')
        ax2.set_title(f'Partial Autocorrelation (PACF) - {column}')
        
        plt.tight_layout()
        plt.show()