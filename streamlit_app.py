import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
import os
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from datasets import load_dataset

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Streamlit UI settings ===
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

# === Title ===
st.title("🔌 Dự báo Siêu Chi Tiết Nhu cầu Điện Năng")
st.markdown("64TTNT2")

# === Sidebar: chọn model & thời gian ===
with st.sidebar:
    st.header("⚙️ Cấu hình")

    model_options = {
        "LSTM": "lstm_full_model.h5",
        "GRU": "gru_full_model.h5",
        "Informer": "informer_full_model"
    }
    selected_model_name = st.selectbox("🔍 Chọn mô hình", list(model_options.keys()))
    model_path = model_options[selected_model_name]

    file_path = "datathugon.csv"
    default_start = pd.to_datetime("2020-01-01")
    default_end = pd.to_datetime("2020-12-31")
    date_range = st.date_input("📅 Chọn khoảng thời gian dự báo", [default_start, default_end])
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

# === Load dữ liệu từ Hugging Face (Parquet) ===
@st.cache_data(show_spinner="🔄 Đang tải dữ liệu từ Hugging Face...")
def load_data_chunked(dataset_name, file_path="datathugon.csv", forecast_start=None, forecast_end=None):
    dtypes = {
        "Electricity_Consumed": "float32",
        "Temperature": "float32",
        "Humidity": "float32",
        "Wind_Speed": "float32",
        "Avg_Past_Consumption": "float32"
    }
    parse_dates = ["Timestamp"]
    usecols = list(dtypes.keys()) + ["Timestamp"]
    historical_start = pd.to_datetime("2006-01-01")
    historical_end = pd.to_datetime("2025-01-01")

    if not os.path.exists(file_path):
        try:
            logger.info(f"Đang tải dataset {dataset_name} từ Hugging Face...")
            dataset = load_dataset(dataset_name, split="train")
            df = dataset.to_pandas()
            if df.empty:
                raise ValueError("Dataset trống sau khi tải!")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
            df = df[usecols].dropna()
            df = df[(df["Timestamp"] >= historical_start) & (df["Timestamp"] <= historical_end)]
            df = df[(df["Timestamp"] >= pd.to_datetime("2020-01-01")) & (df["Timestamp"] <= pd.to_datetime("2025-01-01"))]
            df.to_csv(file_path, index=False)
            logger.info(f"Đã lưu dữ liệu vào {file_path} với {len(df)} dòng.")
        except Exception as e:
            logger.error(f"Lỗi khi tải dataset: {e}")
            st.error(f"⚠️ Lỗi tải dataset: {e}")
            return pd.DataFrame()
    else:
        logger.info(f"Đang đọc file {file_path} đã lưu...")

    df = pd.read_csv(file_path, dtype=dtypes, parse_dates=parse_dates)
    df.set_index("Timestamp", inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df = df[numeric_cols].resample("30min").mean().dropna()

    if forecast_start and forecast_end:
        mask = (df.index >= forecast_start) & (df.index <= forecast_end)
        df = df[mask]

    if df.empty:
        logger.warning("Không có dữ liệu trong khoảng thời gian được chọn.")
        st.warning("⚠️ Không có dữ liệu trong khoảng thời gian được chọn.")

    return df

df = load_data_chunked("Yu08/TS", forecast_start=start_date, forecast_end=end_date)

if df.empty:
    st.warning("⚠️ Không có dữ liệu trong khoảng thời gian được chọn.")
    st.stop()

latest_year = df.index.max().year
if start_date.year > latest_year:
    st.warning(f"⚠️ Dữ liệu chỉ đến năm {latest_year}. Điều chỉnh dự báo.")
    start_date = pd.to_datetime(f"{latest_year}-01-01")
    end_date = pd.to_datetime(f"{latest_year}-12-31")

# === Tiền xử lý đặc trưng ===
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["dayofyear"] = df.index.dayofyear
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

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
    "dayofyear_sin", "dayofyear_cos",
    "is_weekend"
]
target = "Electricity_Consumed"

# === Load hoặc huấn luyện scaler_y ===
scaler_y_path = f"{selected_model_name.lower()}_scaler_y.pkl"

try:
    scaler_y = joblib.load(scaler_y_path)
    logger.info(f"Đã tải scaler_y từ {scaler_y_path}.")
except FileNotFoundError:
    logger.warning(f"Không tìm thấy scaler_y: {scaler_y_path}. Đang huấn luyện lại...")
    st.warning("Scaler_y không tồn tại. Đang huấn luyện lại scaler_y...")
    scaler_y = StandardScaler()
    scaler
