import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
import tensorflow as tf
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# === Cấu hình Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Đảm bảo cổng cho Hugging Face Spaces ===
import os
os.environ["PORT"] = "7860"  # Cổng mặc định của Hugging Face Spaces

# === Cấu hình giao diện Streamlit ===
st.set_page_config(page_title="Dự báo Nhu cầu Điện", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #F9FAFC; }
    .sidebar .sidebar-content { background-color: #F0F2F6; padding: 10px }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stSelectbox, .stDateInput { font-size: 16px; }
    .stWarning { color: #FFA500; }
    .stError { color: #FF0000; }
    </style>
""", unsafe_allow_html=True)

# === Tiêu đề ===
st.title("🔌 Dự báo Siêu Chi Tiết Nhu cầu Điện Năng")
st.markdown("64TTNT2")

# === Sidebar: Chọn mô hình & thời gian ===
with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_options = {
        "LSTM": "lstm_full_model.h5",
        "GRU": "gru_full_model.h5",
        "Informer": "informer_full_model"
    }
    selected_model_name = st.selectbox("🔍 Chọn mô hình", list(model_options.keys()))
    model_path_key = model_options[selected_model_name]
    dataset_name = "Yu08/DemoTS"  # Thay bằng tên dataset của bạn
    parquet_path = "data/data11gb.parquet"  # Đường dẫn trong dataset
    default_start = pd.to_datetime("2020-01-01")
    default_end = pd.to_datetime("2020-12-31")
    date_range = st.date_input("📅 Chọn khoảng thời gian dự báo", [default_start, default_end])
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# === Load dữ liệu từ Hugging Face Datasets ===
@st.cache_data(show_spinner="🔄 Đang tải dữ liệu (có thể mất vài phút)...", persist=True)
def load_data_parquet(dataset_name, parquet_path, start_date, end_date):
    try:
        start_time = pd.Timestamp.now()
        dataset = load_dataset(dataset_name, data_files=parquet_path, streaming=True)
        df = pd.DataFrame(dataset["train"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df[(df["Timestamp"] >= start_date - timedelta(days=2)) & 
                (df["Timestamp"] <= end_date)]
        df.set_index("Timestamp", inplace=True)
        numeric_cols = df.select_dtypes(include=['number']).columns
        df = df[numeric_cols].resample("30min").mean().dropna()
        end_time = pd.Timestamp.now()
        logger.info(f"Tổng thời gian đọc và xử lý: {(end_time - start_time).total_seconds()} giây")
        if df.empty:
            st.warning("⚠️ Không có dữ liệu trong khoảng thời gian được chọn.")
            st.stop()
        return df
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        st.error(f"⚠️ Lỗi tải dữ liệu: {e}")
        st.stop()

df = load_data_parquet(dataset_name, parquet_path, start_date, end_date)

# === Tiền xử lý đặc trưng ===
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["dayofyear"] = df.index.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
df["lag_1h"] = df["Electricity_Consumed"].shift(2)
df["lag_24h"] = df["Electricity_Consumed"].shift(48)
df["lag_168h"] = df["Electricity_Consumed"].shift(48*7)
df.dropna(inplace=True)

# === Đặc trưng đầu vào và target ===
features_x = [
    "Electricity_Consumed",
    "lag_1h", "lag_24h", "lag_168h",
    "hour_sin", "hour_cos",
    "dayofweek_sin", "dayofweek_cos",
    "dayofyear_sin", "dayofyear_cos"
]
target = "Electricity_Consumed"

# === Load hoặc huấn luyện scaler_y ===
scaler_y_path = hf_hub_download(repo_id=dataset_name, 
                                filename=f"models/{selected_model_name.lower()}_scaler_y.pkl", 
                                repo_type="dataset")
try:
    scaler_y = joblib.load(scaler_y_path)
    logger.info(f"Đã tải scaler_y từ {scaler_y_path}.")
except FileNotFoundError:
    logger.warning(f"Không tìm thấy scaler_y: {scaler_y_path}. Đang huấn luyện lại...")
    st.warning("Scaler_y không tồn tại. Đang huấn luyện lại scaler_y...")
    scaler_y = StandardScaler()
    scaler_y.fit(df[[target]])
    joblib.dump(scaler_y, scaler_y_path)  # Lưu tạm trên Spaces (không ổn định, xem lưu ý)
    logger.info(f"Đã huấn luyện và lưu scaler_y.")

# === Load mô hình ===
if selected_model_name in ["LSTM", "GRU"]:
    try:
        model_path = hf_hub_download(repo_id=dataset_name, 
                                    filename=f"models/{selected_model_name.lower()}_full_model.h5", 
                                    repo_type="dataset")
        model = load_model(model_path)
        logger.info(f"Đã tải mô hình {selected_model_name} từ {model_path}.")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình {selected_model_name}: {e}")
        st.error(f"⚠️ Lỗi tải mô hình {selected_model_name}: {e}")
        st.stop()
elif selected_model_name == "Informer":
    try:
        model_path = hf_hub_download(repo_id=dataset_name, 
                                    filename="models/informer_full_model", 
                                    repo_type="dataset")
        model = tf.saved_model.load(model_path)
        logger.info(f"Đã tải mô hình Informer từ {model_path}.")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình Informer: {e}")
        st.error(f"⚠️ Lỗi tải mô hình Informer: {e}")
        st.stop()

# === Hàm tạo đặc trưng cho dự báo tương lai ===
def generate_future_features(df, start_date, end_date, seq_length=48):
    future_index = pd.date_range(start=start_date, end=end_date, freq="30min")
    future_df = pd.DataFrame(index=future_index)
    
    future_df["hour"] = future_df.index.hour
    future_df["dayofweek"] = future_df.index.dayofweek
    future_df["dayofyear"] = future_df.index.dayofyear
    future_df["hour_sin"] = np.sin(2 * np.pi * future_df["hour"] / 24)
    future_df["hour_cos"] = np.cos(2 * np.pi * future_df["hour"] / 24)
    future_df["dayofweek_sin"] = np.sin(2 * np.pi * future_df["dayofweek"] / 7)
    future_df["dayofweek_cos"] = np.cos(2 * np.pi * future_df["dayofweek"] / 7)
    future_df["dayofyear_sin"] = np.sin(2 * np.pi * future_df["dayofyear"] / 365.25)
    future_df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    
    last_year_data = df[df.index.year == df.index.max().year][["Electricity_Consumed"]]
    if len(last_year_data) < seq_length:
        st.error(f"Dữ liệu năm {df.index.max().year} không đủ {seq_length} mẫu để tạo đặc trưng!")
        st.stop()
    if len(last_year_data) < 48*7:
        logger.warning(f"Dữ liệu năm {df.index.max().year} không đủ {48*7} mẫu cho lag_168h.")
    
    future_df["Electricity_Consumed"] = np.nan
    future_df["lag_1h"] = np.nan
    future_df["lag_24h"] = np.nan
    future_df["lag_168h"] = np.nan
    
    for i in range(len(future_df)):
        if i < 2 and i-2 < len(last_year_data):
            future_df.iloc[i, future_df.columns.get_loc("lag_1h")] = last_year_data.iloc[-2+i]["Electricity_Consumed"]
        elif i >= 2:
            future_df.iloc[i, future_df.columns.get_loc("lag_1h")] = future_df["Electricity_Consumed"].iloc[i-2] if pd.notna(future_df["Electricity_Consumed"].iloc[i-2]) else last_year_data.iloc[-2+i]["Electricity_Consumed"] if i-2 < len(last_year_data) else np.nan
        
        if i < 48 and i-48 < len(last_year_data):
            future_df.iloc[i, future_df.columns.get_loc("lag_24h")] = last_year_data.iloc[-48+i]["Electricity_Consumed"]
        elif i >= 48:
            future_df.iloc[i, future_df.columns.get_loc("lag_24h")] = future_df["Electricity_Consumed"].iloc[i-48] if pd.notna(future_df["Electricity_Consumed"].iloc[i-48]) else last_year_data.iloc[-48+i]["Electricity_Consumed"] if i-48 < len(last_year_data) else np.nan
        
        if i < 48*7 and i-48*7 < len(last_year_data):
            future_df.iloc[i, future_df.columns.get_loc("lag_168h")] = last_year_data.iloc[-48*7+i]["Electricity_Consumed"] if i-48*7 < len(last_year_data) else np.nan
        elif i >= 48*7:
            future_df.iloc[i, future_df.columns.get_loc("lag_168h")] = future_df["Electricity_Consumed"].iloc[i-48*7] if pd.notna(future_df["Electricity_Consumed"].iloc[i-48*7]) else last_year_data.iloc[-48*7+i]["Electricity_Consumed"] if i-48*7 < len(last_year_data) else np.nan
    
    return future_df[features_x]

# === Dự báo ===
try:
    seq_length = 48
    latest_year = df.index.max().year
    is_future_forecast = start_date.year > latest_year
    
    if is_future_forecast:
        st.warning(f"⚠️ Dự báo cho tương lai ({start_date.year}). Sử dụng dữ liệu năm {latest_year} làm cơ sở.")
        df_x = generate_future_features(df, start_date, end_date, seq_length)
        time_idx = df_x.index
    else:
        df_x = df[features_x].copy()
        df_y = df[[target]].copy()
        time_idx = df_x.index[seq_length:len(df_x) - seq_length]

    X_raw = df_x.values
    data_seq_x = np.array([X_raw[i:i+seq_length] for i in range(len(X_raw) - seq_length)])

    logger.info(f"Shape of data_seq_x: {data_seq_x.shape}")

    if selected_model_name in ["LSTM", "GRU"]:
        n_inputs = len(model.inputs)
        logger.info(f"Số đầu vào mong đợi bởi mô hình: {n_inputs}")
        if n_inputs == 2:
            decoder_input = np.zeros((data_seq_x.shape[0], 24, 1), dtype=np.float32)
            y_pred_scaled = model.predict([data_seq_x, decoder_input], batch_size=32)
        elif n_inputs == 1:
            y_pred_scaled = model.predict(data_seq_x, batch_size=32)
        else:
            raise ValueError(f"Mô hình yêu cầu {n_inputs} đầu vào, không được hỗ trợ.")
    elif selected_model_name == "Informer":
        data_seq_x_tensor = tf.convert_to_tensor(data_seq_x, dtype=tf.float32)
        logger.info(f"Shape of data_seq_x_tensor: {data_seq_x_tensor.shape}")
        try:
            signatures = list(model.signatures.keys())
            logger.info(f"Signatures available: {signatures}")
            if "serving_default" in signatures:
                y_pred_scaled = model.signatures["serving_default"](inputs=data_seq_x_tensor)
                y_pred_scaled = y_pred_scaled['output_0']
            else:
                raise ValueError("Không tìm thấy signature 'serving_default'")
        except Exception as e:
            logger.error(f"Lỗi khi gọi mô hình Informer: {e}")
            st.error(f"⚠️ Lỗi gọi mô hình Informer: {e}")
            st.stop()
        y_pred_scaled = y_pred_scaled.numpy()

    logger.info(f"Shape of y_pred_scaled: {y_pred_scaled.shape}")

    if y_pred_scaled.ndim == 3:
        y_pred_scaled_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    else:
        y_pred_scaled_2d = y_pred_scaled
    y_pred = scaler_y.inverse_transform(y_pred_scaled_2d)
    y_pred = y_pred.flatten()[:len(time_idx)]

    df_result = pd.DataFrame({"Timestamp": time_idx, "Dự báo": y_pred}).set_index("Timestamp")
    df_result = df_result[(df_result.index >= start_date) & (df_result.index <= end_date)]
    
    df_result["Ngày/Tháng"] = df_result.index.strftime("%d/%m")
    df_result = df_result[["Ngày/Tháng", "Dự báo"]]

    logger.info(f"✅ Dự báo thành công với {len(df_result)} dòng.")
except Exception as e:
    logger.error(f"Lỗi dự báo: {e}")
    st.error(f"⚠️ Lỗi dự báo: {e}")
    st.stop()

# === Hiển thị kết quả ===
st.success(f"✅ Đã dự báo thành công với mô hình {selected_model_name}")

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("📈 Biểu đồ Dự báo")
    df_historical = df[[target]].rename(columns={target: "Dữ liệu gốc"}) if not is_future_forecast else pd.DataFrame()
    df_result_plot = pd.DataFrame({"Dự báo": df_result["Dự báo"]}, index=df_result.index)
    df_combined = pd.concat([df_historical, df_result_plot]) if not df_historical.empty else df_result_plot
    fig, ax = plt.subplots(figsize=(12, 5))
    df_combined.plot(ax=ax, color={"Dữ liệu gốc": "green", "Dự báo": "royalblue"}, label=["Dữ liệu gốc", "Dự báo"])
    ax.set_ylabel("Nhu cầu điện (kWh)")
    ax.set_xlabel("Thời gian")
    ax.set_title(f"Dự báo nhu cầu điện ({start_date.year} - {end_date.year})")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Thống kê dự báo")
    st.dataframe(df_result.describe().T, use_container_width=True)
    st.subheader("🧾 Toàn bộ dữ liệu dự báo")
    st.dataframe(df_result, use_container_width=True)
    csv = df_result.reset_index().to_csv(index=False, encoding="utf-8").encode("utf-8")
    st.download_button("📥 Tải dữ liệu dự báo", data=csv, file_name="du_bao_nhu_cau_dien.csv", mime="text/csv")