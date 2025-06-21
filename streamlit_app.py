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
        "Informer": "informer_full_model"  # Đường dẫn đến thư mục chứa .pb
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
    historical_start = pd.to_datetime("2006-01-01")  # Điều chỉnh theo dataset Yu08/TS
    historical_end = pd.to_datetime("2025-01-01")

    if not os.path.exists(file_path):
        try:
            logger.info(f"Đang tải dataset {dataset_name} từ Hugging Face...")
            dataset = load_dataset(dataset_name, split="train")
            df = dataset.to_pandas()
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df[usecols]
            df = df[(df["Timestamp"] >= historical_start) & (df["Timestamp"] <= historical_end)]
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
    scaler_y.fit(df[[target]])
    joblib.dump(scaler_y, scaler_y_path)
    logger.info(f"Đã huấn luyện và lưu scaler_y.")

# === Load mô hình với cache ===
@st.cache_resource
def load_model_cached(model_path, model_name):
    if model_name in ["LSTM", "GRU"]:
        return load_model(model_path)
    elif model_name == "Informer":
        return tf.saved_model.load(model_path)
    return None

model = load_model_cached(model_path, selected_model_name)
if model is None:
    st.error(f"⚠️ Lỗi tải mô hình {selected_model_name} từ {model_path}")
    st.stop()

# === Dự báo ===
try:
    # Chuẩn bị dữ liệu
    df_x = df[features_x].copy()
    df_y = df[[target]].copy()
    X_raw = df_x.values
    y_scaled = scaler_y.transform(df_y.values)

    seq_length = 48
    data_seq_x = np.array([X_raw[i:i+seq_length] for i in range(len(X_raw) - seq_length)])
    data_seq_y = np.array([y_scaled[i:i+seq_length] for i in range(len(y_scaled) - seq_length)])

    logger.info(f"Shape of data_seq_x: {data_seq_x.shape}")
    logger.info(f"Shape of data_seq_y: {data_seq_y.shape}")

    time_idx = df_x.index[seq_length:seq_length + len(data_seq_x)]

    if selected_model_name in ["LSTM", "GRU"]:
        n_inputs = len(model.inputs)
        logger.info(f"Số đầu vào mong đợi bởi mô hình: {n_inputs}")

        if n_inputs == 2:  # Encoder-Decoder (GRU)
            decoder_input = np.zeros((data_seq_x.shape[0], 24, 1), dtype=np.float32)
            with st.spinner("Đang dự báo..."):
                y_pred_scaled = model.predict([data_seq_x, decoder_input], batch_size=32)
        elif n_inputs == 1:  # Chỉ encoder (LSTM)
            with st.spinner("Đang dự báo..."):
                y_pred_scaled = model.predict(data_seq_x, batch_size=32)
        else:
            raise ValueError(f"Mô hình yêu cầu {n_inputs} đầu vào, không được hỗ trợ.")
    elif selected_model_name == "Informer":
        data_seq_x_tensor = tf.convert_to_tensor(data_seq_x, dtype=tf.float32)
        try:
            signatures = list(model.signatures.keys())
            logger.info(f"Signatures available: {signatures}")
            if "serving_default" in signatures:
                with st.spinner("Đang dự báo..."):
                    y_pred_scaled = model.signatures["serving_default"](data_seq_x_tensor)
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
    df_historical = df[[target]].rename(columns={target: "Dữ liệu gốc"})
    df_combined = pd.concat([df_historical, df_result])
    fig, ax = plt.subplots(figsize=(12, 5))
    df_combined.plot(ax=ax, color={"Dữ liệu gốc": "green", "Dự báo": "royalblue"}, label=["Dữ liệu gốc", "Dự báo"])
    ax.set_ylabel("Nhu cầu điện (kWh)")
    ax.set_xlabel("Thời gian")
    ax.set_title(f"Dự báo nhu cầu điện ({start_date.year} - {end_date.year})")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("📊 Thống kê dự báo")
    st.dataframe(df_result.describe().T, use_container_width=True)

    st.subheader("🧾 Toàn bộ dữ liệu dự báo")
    st.dataframe(df_result, use_container_width=True)

    csv = df_result.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải dữ liệu dự báo", data=csv, file_name="du_bao_nhu_cau_dien.csv", mime="text/csv")
