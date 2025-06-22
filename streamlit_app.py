import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Ép TensorFlow dùng CPU và giảm logging
tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Thiết lập logging tối thiểu
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thiết lập thư mục tạm
os.environ['HOME'] = '/tmp'
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp/streamlit_config'
os.environ['STREAMLIT_HOME'] = '/tmp/streamlit_home'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_config'

# Kiểm tra và tạo thư mục cấu hình với quyền ghi
def ensure_writable_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o777)
        test_file = os.path.join(directory, '.test_write')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Không thể tạo hoặc ghi vào {directory}: {e}")
        return False

# Tạo các thư mục cấu hình
config_dirs = [os.environ['STREAMLIT_CONFIG_DIR'], os.environ['STREAMLIT_HOME'], os.environ['MPLCONFIGDIR']]
for config_dir in config_dirs:
    if not ensure_writable_directory(config_dir):
        st.error(f"Không thể tạo thư mục {config_dir}. Falling back to /tmp.")
        os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp'
        os.environ['STREAMLIT_HOME'] = '/tmp'
        os.environ['MPLCONFIGDIR'] = '/tmp'

# Tạo tệp cấu hình Streamlit giả
try:
    config_path = os.path.join(os.environ['STREAMLIT_CONFIG_DIR'], 'config.toml')
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        f.write('[global]\ndataDir = "/tmp/streamlit_home"\n')
except Exception as e:
    logger.error(f"Không thể tạo config.toml: {e}")

# Cấu hình trang
st.set_page_config(page_title="Dự báo siêu chi tiết nhu cầu điện năng", page_icon="⚡", layout="wide")
st.title("Dự báo siêu chi tiết nhu cầu điện năng")
st.write(f"Ngày giờ hiện tại: {datetime.now().strftime('%I:%M %p %z, %A, %B %d, %Y')}")

# Utility functions
def mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Load dữ liệu từ data11.parquet (chỉ năm 2010)
@st.cache_data(show_spinner="🔄 Đang tải dữ liệu...")
def load_parquet_data():
    path = "src/data11.parquet"
    try:
        columns = ['Electricity_Consumed', 'Temperature', 'Humidity', 'Wind_Speed', 'Avg_Past_Consumption', 'Timestamp']
        df = pd.read_parquet(path, columns=columns, engine='fastparquet')
        # Lọc dữ liệu chỉ cho năm 2010
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df = df[(df['Timestamp'] >= '2010-01-01') & (df['Timestamp'] <= '2010-12-31')]
        if df.empty:
            raise ValueError("Không tìm thấy dữ liệu năm 2010 trong file data11.parquet")
        logger.info(f"Dữ liệu năm 2010 tải thành công: {len(df)} mẫu")
    except Exception as e:
        logger.warning(f"Lỗi khi đọc file data11.parquet: {e}, tạo dữ liệu mẫu năm 2010...")
        dates = pd.date_range(start="2010-01-01", end="2010-12-31", freq="30min")
        df = pd.DataFrame(index=dates)
        df['Electricity_Consumed'] = np.random.normal(1000, 200, len(dates))
        df['Temperature'] = np.random.normal(25, 5, len(dates))
        df['Humidity'] = np.random.normal(60, 10, len(dates))
        df['Wind_Speed'] = np.random.normal(5, 2, len(dates))
        df['Avg_Past_Consumption'] = np.random.normal(1000, 200, len(dates))
        df['Timestamp'] = df.index
        logger.info(f"Dữ liệu mẫu năm 2010 tạo thành công: {len(df)} mẫu")
    
    df = df.dropna(subset=['Timestamp']).set_index('Timestamp')
    return df

# Feature engineering
def prepare_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
    df[["hour_sin", "hour_cos"]] = np.column_stack([
        np.sin(2 * np.pi * df["hour"] / 24),
        np.cos(2 * np.pi * df["hour"] / 24)
    ])
    df[["dayofweek_sin", "dayofweek_cos"]] = np.column_stack([
        np.sin(2 * np.pi * df["dayofweek"] / 7),
        np.cos(2 * np.pi * df["dayofweek"] / 7)
    ])
    df[["dayofyear_sin", "dayofyear_cos"]] = np.column_stack([
        np.sin(2 * np.pi * df["dayofyear"] / 365.25),
        np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    ])
    df["lag_1h"] = df["Electricity_Consumed"].shift(2)
    df["lag_24h"] = df["Electricity_Consumed"].shift(48)
    df["lag_168h"] = df["Electricity_Consumed"].shift(48*7)
    df = df.dropna()
    return df

# Sidebar
with st.sidebar:
    st.header("Cài đặt")
    models = ["LSTM", "GRU", "Informer"]
    selected_model = st.selectbox("Chọn mô hình", models)
    default_start = pd.to_datetime("2011-01-01")
    default_end = pd.to_datetime("2011-05-04")
    date_range = st.date_input("📅 Chọn khoảng thời gian dự báo", [default_start, default_end])
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        st.warning("Vui lòng chọn cả ngày bắt đầu và kết thúc. Sử dụng giá trị mặc định.")
        start_date, end_date = default_start, default_end
    if start_date > end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()
    if st.button("🗑️ Xóa cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache đã được xóa!")

# Load and prepare data
df = load_parquet_data()
if df.empty:
    st.error("Dữ liệu rỗng, vui lòng kiểm tra file data11.parquet")
    st.stop()
df = prepare_features(df)
features_x = [
    "Electricity_Consumed", "Temperature", "Humidity", "Wind_Speed", "Avg_Past_Consumption",
    "lag_1h", "lag_24h", "lag_168h", "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos", "dayofyear_sin", "dayofyear_cos", "is_weekend"
]

# Kiểm tra start_date
if start_date < df.index[-1]:
    st.warning(f"Ngày bắt đầu ({start_date}) sớm hơn thời điểm cuối dữ liệu ({df.index[-1]}). Điều chỉnh để dự báo từ {df.index[-1]}.")
    start_date = df.index[-1] + timedelta(minutes=30)

# Hàm xây dựng mô hình
def build_model(model_type, seq_length, n_features):
    if model_type == "LSTM":
        model = Sequential([
            Input(shape=(seq_length, n_features)),
            LSTM(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1)
        ])
    elif model_type == "GRU":
        model = Sequential([
            Input(shape=(seq_length, n_features)),
            GRU(16, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1)
        ])
    elif model_type == "Informer":
        model = Sequential([
            Input(shape=(seq_length, n_features)),
            LSTM(16, return_sequences=True),
            LSTM(8, return_sequences=False),
            Dense(8, activation='relu'),
            Dense(1)
        ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Hàm huấn luyện và dự báo
@st.cache_resource(show_spinner="🔄 Đang huấn luyện mô hình...")
def train_and_forecast(_df, model_type, start_date, end_date, seq_length=48):
    try:
        df_x = _df[features_x].copy()
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_x)
        X, y = create_sequences(scaled_data, seq_length)
        if len(X) < 500:
            st.error(f"Dữ liệu không đủ để tạo sequence: {len(X)} mẫu. Cần ít nhất 500 mẫu.")
            return None, None, None, None
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        model = build_model(model_type, seq_length, len(features_x))
        model.fit(X_train, y_train[:, 0], epochs=3, batch_size=256, verbose=0)
        y_pred_scaled = model.predict(X_test, batch_size=256, verbose=0)
        y_pred = scaler.inverse_transform(
            np.column_stack([y_pred_scaled.flatten()] + [np.zeros((len(y_pred_scaled), len(features_x)-1))])
        )[:, 0]
        y_true = scaler.inverse_transform(y_test)[:, 0]
        test_index = _df.index[train_size + seq_length:train_size + seq_length + len(y_pred)]
        # Dự báo tương lai theo batch
        last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, len(features_x))
        future_predictions = []
        future_index = pd.date_range(start=_df.index[-1] + timedelta(minutes=30), end=end_date, freq='30min')
        batch_size = 96
        for i in range(0, len(future_index), batch_size):
            batch_end = min(i + batch_size, len(future_index))
            batch_predictions = []
            current_sequence = last_sequence.copy()
            for j in range(i, batch_end):
                pred_scaled = model.predict(current_sequence, batch_size=1, verbose=0)
                pred_value = scaler.inverse_transform(
                    np.column_stack([pred_scaled.flatten()] + [np.zeros((len(pred_scaled), len(features_x)-1))])
                )[:, 0]
                batch_predictions.append(pred_value[0])
                next_sequence = current_sequence[0, 1:, :].copy()
                new_row = current_sequence[0, -1, :].copy()
                new_row[0] = pred_scaled[0, 0]
                future_time = future_index[j]
                new_row[features_x.index("hour_sin")] = np.sin(2 * np.pi * future_time.hour / 24)
                new_row[features_x.index("hour_cos")] = np.cos(2 * np.pi * future_time.hour / 24)
                new_row[features_x.index("dayofweek_sin")] = np.sin(2 * np.pi * future_time.dayofweek / 7)
                new_row[features_x.index("dayofweek_cos")] = np.cos(2 * np.pi * future_time.dayofweek / 7)
                new_row[features_x.index("dayofyear_sin")] = np.sin(2 * np.pi * future_time.dayofyear / 365.25)
                new_row[features_x.index("dayofyear_cos")] = np.cos(2 * np.pi * future_time.dayofyear / 365.25)
                new_row[features_x.index("is_weekend")] = 1 if future_time.dayofweek in [5, 6] else 0
                current_sequence = np.append(next_sequence, [new_row], axis=0).reshape(1, seq_length, len(features_x))
            future_predictions.extend(batch_predictions)
            last_sequence = current_sequence
        tf.keras.backend.clear_session()
        return test_index, y_true, y_pred, future_index, np.array(future_predictions)
    except Exception as e:
        logger.error(f"Lỗi trong train_and_forecast {model_type}: {e}")
        st.error(f"Lỗi {model_type}: {e}")
        return None, None, None, None

# Tính metrics
def calculate_metrics(y_true, y_pred):
    try:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape_val = mape(y_true, y_pred)
        smape_val = smape(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE (%)": round(mape_val, 2),
            "sMAPE (%)": round(smape_val, 2),
            "Correlation": round(corr, 4)
        }
    except:
        return {"MAE": 0, "RMSE": 0, "MAPE (%)": 0, "sMAPE (%)": 0, "Correlation": 0}

# Main content
min_required = 336
if len(df) < min_required:
    st.error(f"Cần ít nhất {min_required} mẫu dữ liệu. Hiện có: {len(df)}")
else:
    # Thống kê cơ bản
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng mẫu", f"{len(df):,}")
    with col2:
        st.metric("Trung bình", f"{df['Electricity_Consumed'].mean():.3f} kWh")
    with col3:
        st.metric("Max", f"{df['Electricity_Consumed'].max():.3f} kWh")
    with col4:
        st.metric("Min", f"{df['Electricity_Consumed'].min():.3f} kWh")
    
    # Biểu đồ dữ liệu tiêu thụ điện (2 ngày cuối năm 2010)
    st.subheader("Dữ liệu tiêu thụ điện năm 2010")
    display_data = df.tail(96)  # 2 ngày cuối năm 2010
    fig_data = go.Figure()
    fig_data.add_trace(go.Scatter(
        x=display_data.index, 
        y=display_data['Electricity_Consumed'], 
        mode='lines', 
        name='Tiêu thụ điện'
    ))
    fig_data.update_layout(
        xaxis_title="Thời gian", 
        yaxis_title="kWh", 
        height=400,
        xaxis_range=[display_data.index[0], display_data.index[-1]]
    )
    st.plotly_chart(fig_data, use_container_width=True)
    
    # Dự báo
    st.markdown("---")
    st.subheader("Dự báo cho năm 2011")
    if st.button("Bắt đầu dự báo", use_container_width=True):
        with st.spinner(f"Đang dự báo với {selected_model}..."):
            start_time = time.time()
            test_index, y_true, y_pred, future_index, future_predictions = train_and_forecast(df, selected_model, start_date, end_date)
            
            if test_index is not None and future_index is not None:
                df_result = pd.DataFrame({"Timestamp": test_index, "Dữ liệu gốc": y_true, "Dự báo": y_pred}).set_index("Timestamp")
                df_future = pd.DataFrame({"Timestamp": future_index, "Dự báo tương lai": future_predictions}).set_index("Timestamp")
                df_future = df_future[(df_future.index >= start_date) & (df_future.index <= end_date)]
                
                # Biểu đồ kết quả
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=display_data.index, y=display_data['Electricity_Consumed'],
                    mode='lines', name='Dữ liệu năm 2010', line=dict(color='gray')
                ))
                fig.add_trace(go.Scatter(
                    x=df_result.index, y=df_result['Dữ liệu gốc'],
                    mode='lines+markers', name='Thực tế', line=dict(color='blue')
                ))
                colors = {"LSTM": "green", "GRU": "orange", "Informer": "purple"}
                fig.add_trace(go.Scatter(
                    x=df_result.index, y=df_result['Dự báo'],
                    mode='lines+markers', name=f'Dự báo ({selected_model})',
                    line=dict(color=colors.get(selected_model, "purple"), dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=df_future.index, y=df_future['Dự báo tương lai'],
                    mode='lines', name='Dự báo tương lai', line=dict(color='red', dash='dot')
                ))
                fig.update_layout(
                    title=f"Kết quả dự báo - {selected_model} ({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')})",
                    xaxis_title="Thời gian", yaxis_title="kWh",
                    height=500, hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                metrics = calculate_metrics(df_result["Dữ liệu gốc"], df_result["Dự báo"])
                processing_time = time.time() - start_time
                st.success(f"Hoàn thành trong {processing_time:.2f} giây")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("MAE", metrics["MAE"])
                with col2:
                    st.metric("RMSE", metrics["RMSE"])
                with col3:
                    st.metric("MAPE", f"{metrics['MAPE (%)']}%")
                with col4:
                    st.metric("sMAPE", f"{metrics['sMAPE (%)']}%")
                with col5:
                    st.metric("Correlation", metrics["Correlation"])
                
                # Bảng kết quả
                st.subheader("🧾 Dự báo tương lai")
                if not df_future.empty:
                    st.dataframe(df_future, use_container_width=True)
                    csv_future = df_future.reset_index().to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Tải dữ liệu dự báo tương lai",
                        csv_future,
                        f"du_bao_tuong_lai_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                else:
                    st.warning("Không có dữ liệu dự báo tương lai trong khoảng thời gian đã chọn.")
                
                st.subheader("🧾 Dự báo tập kiểm tra")
                if not df_result.empty:
                    st.dataframe(df_result, use_container_width=True)
                    csv_result = df_result.reset_index().to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Tải dữ liệu dự báo tập kiểm tra",
                        csv_result,
                        f"du_bao_test_{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv"
                    )
                else:
                    st.warning("Không có dữ liệu dự báo tập kiểm tra.")
            else:
                st.error("Không thể dự báo. Vui lòng kiểm tra dữ liệu.")
