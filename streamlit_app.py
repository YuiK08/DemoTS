import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import tensorflow as tf
import os
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
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

# === Load dữ liệu từ CSV theo chunk ===
@st.cache_data(show_spinner="🔄 Đang tải dữ liệu...")
def load_data_chunked(file_path, forecast_start, forecast_end):
    dtypes = {
        "Electricity_Consumed": "float32",
        "Temperature": "float32",
        "Humidity": "float32",
        "Wind_Speed": "float32",
        "Avg_Past_Consumption": "float32"
    }
    parse_dates = ["Timestamp"]
    usecols = list(dtypes.keys()) + ["Timestamp"]
    historical_start = pd.to_datetime("2000-01-01")
    historical_end = pd.to_datetime("2025-01-01")
    chunks = []

    try:
        logger.info(f"Đang đọc file: {file_path}")
        if not os.path.exists(file_path):
            st.error(f"Tệp dữ liệu {file_path} không tồn tại!")
            st.stop()
        reader = pd.read_csv(file_path, dtype=dtypes, parse_dates=parse_dates,
                             usecols=usecols, chunksize=100_000,
                             on_bad_lines='skip', low_memory=False)
        for chunk in reader:
            mask = (chunk["Timestamp"] >= historical_start - timedelta(days=2)) & \
                   (chunk["Timestamp"] <= historical_end)
            chunk = chunk[mask]
            if not chunk.empty:
                chunk.set_index("Timestamp", inplace=True)
                numeric_cols = chunk.select_dtypes(include=['number']).columns
                chunk = chunk[numeric_cols].resample("30min").mean().dropna()
                chunks.append(chunk)
        if chunks:
            df = pd.concat(chunks).groupby(level=0).mean().asfreq("30min").dropna()
            logger.info(f"Đã tải thành công dữ liệu với {len(df)} dòng.")
            return df
        else:
            logger.warning("Không có dữ liệu trong khoảng thời gian 2000-2018.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        st.error(f"⚠️ Lỗi tải dữ liệu: {e}")
        return pd.DataFrame()

df = load_data_chunked(file_path, start_date, end_date)

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
]  # 11 features
target = "Electricity_Consumed"

# === Load hoặc huấn luyện scaler_y (chỉ giữ scaler_y) ===
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

# === Load mô hình ===
if selected_model_name in ["LSTM", "GRU"]:
    model = load_model(model_path)
    logger.info(f"Đã tải mô hình {selected_model_name} từ {model_path}.")
elif selected_model_name == "Informer":
    try:
        # Đường dẫn đến thư mục SavedModel
        model = tf.saved_model.load(model_path)
        logger.info(f"Đã tải mô hình Informer từ {model_path}.")
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình Informer: {e}")
        st.error(f"⚠️ Lỗi tải mô hình Informer: {e}")
        st.stop()

# === Dự báo ===
try:
    # Chuẩn bị dữ liệu (dùng dữ liệu thô đã chuẩn hóa)
    df_x = df[features_x].copy()
    df_y = df[[target]].copy()
    X_raw = df_x.values  # Dữ liệu thô đã chuẩn hóa
    y_scaled = scaler_y.transform(df_y.values)  # Chuẩn hóa target ban đầu

    seq_length = 48
    data_seq_x = np.array([X_raw[i:i+seq_length] for i in range(len(X_raw) - seq_length)])
    data_seq_y = np.array([y_scaled[i:i+seq_length] for i in range(len(y_scaled) - seq_length)])

    logger.info(f"Shape of data_seq_x: {data_seq_x.shape}")
    logger.info(f"Shape of data_seq_y: {data_seq_y.shape}")

    # Tính time_idx trước khi dự đoán
    time_idx = df_x.index[seq_length:seq_length + len(data_seq_x)]

    if selected_model_name in ["LSTM", "GRU"]:
        n_inputs = len(model.inputs)
        logger.info(f"Số đầu vào mong đợi bởi mô hình: {n_inputs}")

        if n_inputs == 2:  # Encoder-Decoder (GRU)
            decoder_input = np.zeros((data_seq_x.shape[0], 24, 1), dtype=np.float32)
            y_pred_scaled = model.predict([data_seq_x, decoder_input], batch_size=32)
        elif n_inputs == 1:  # Chỉ encoder (LSTM)
            y_pred_scaled = model.predict(data_seq_x, batch_size=32)
        else:
            raise ValueError(f"Mô hình yêu cầu {n_inputs} đầu vào, không được hỗ trợ.")
    elif selected_model_name == "Informer":
        # Chuyển dữ liệu sang định dạng TensorFlow
        data_seq_x_tensor = tf.convert_to_tensor(data_seq_x, dtype=tf.float32)
        # Gọi mô hình SavedModel (cần kiểm tra signature_def)
        try:
            # Kiểm tra signature có sẵn
            signatures = list(model.signatures.keys())
            logger.info(f"Signatures available: {signatures}")
            if "serving_default" in signatures:
                y_pred_scaled = model.signatures["serving_default"](data_seq_x_tensor)
                y_pred_scaled = y_pred_scaled['output_0']  # Điều chỉnh tên output
            else:
                raise ValueError("Không tìm thấy signature 'serving_default'")
        except Exception as e:
            logger.error(f"Lỗi khi gọi mô hình Informer: {e}")
            st.error(f"⚠️ Lỗi gọi mô hình Informer: {e}")
            st.stop()
        y_pred_scaled = y_pred_scaled.numpy()  # Chuyển về numpy array

    logger.info(f"Shape of y_pred_scaled: {y_pred_scaled.shape}")

    # Chuyển ngược về giá trị gốc (chỉ áp dụng cho đầu ra 2D)
    if y_pred_scaled.ndim == 3:
        y_pred_scaled_2d = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    else:
        y_pred_scaled_2d = y_pred_scaled
    y_pred = scaler_y.inverse_transform(y_pred_scaled_2d)
    y_pred = y_pred.flatten()[:len(time_idx)]  # Cắt về đúng số mẫu

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
