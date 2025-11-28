# demo_streamlit.py
import datetime
import streamlit as st
import pandas as pd
import joblib
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64 # Import cho Base64 encoding

# --- 1. SET PAGE CONFIG (ENABLE DARK MODE) ---
st.set_page_config(
    page_title="🤖 Dự đoán giá & Phát hiện bất thường - Xe máy công nghệ", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. INJECT CUSTOM CSS FOR ENHANCED FUTURISTIC/MECHANICAL THEME ---
st.markdown(
    """
    <style>
    /* Global Background (Dark/Mechanical) */
    .stApp {
        background-color: #0d1117; /* Dark Background */
        color: #c9d1d9; /* Light gray text */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Highlight/Primary Color (Deep Neon Cyan) */
    :root {
        --primary-color: #00bcd4; /* Cyan/Teal Neon */
        --secondary-color: #00e5ff; /* Brighter Cyan */
        --text-glow: 0 0 15px rgba(0, 229, 255, 0.9), 0 0 25px rgba(0, 229, 255, 0.4);
    }
    
    /* Headers (H1, H2, H3, H4) in Content - NO GLOW, highlight with color and border */
    h1, h2, h3, h4 {
        color: var(--secondary-color);
        text-shadow: none; /* ĐÃ XÓA GLOW */
        border-bottom: 2px solid rgba(0, 188, 212, 0.4);
        padding-bottom: 8px;
        margin-top: 20px;
    }
    
    /* Global Content Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* --- CUSTOM HEADER OVERLAY STYLE (Dùng Background CSS) --- */
    /* Container for the image and text overlay */
    .cover-header {
        position: relative; 
        height: 300px; /* Chiều cao cố định */
        margin-bottom: 30px;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 188, 212, 0.5);
        overflow: hidden; 
        
        /* Cần thiết cho Background Scaling */
        background-position: center; 
    }
    
    /* Text Overlay Container */
    .cover-text-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 10; 
        
        /* Centering the H1 text */
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    /* Styling the H1 title overlay on the cover (GLOW KEPT ONLY HERE) */
    .cover-header h1 {
        z-index: 10;
        color: white; /* Màu trắng cho tiêu đề trên ảnh */
        text-shadow: var(--text-glow); /* GLOW DUY NHẤT */
        font-size: 3em;
        padding: 0;
        margin: 0;
        border-bottom: none;
        width: 90%; 
        text-align: center;
    }
    
    /* New CSS for notes at the bottom right of the cover */
    .cover-notes {
        position: absolute;
        bottom: 15px; 
        right: 20px; 
        z-index: 10;
        color: rgba(255, 255, 255, 0.9); 
        font-size: 0.85em;
        line-height: 1.4;
        text-align: left; 
        text-shadow: 0 0 5px rgba(0, 0, 0, 0.8); 
        width: 300px; 
    }
    /* Style for pending rows in Admin table */
    .pending-row {
        background-color: rgba(255, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------- Load data (mẫu) + allow upload ----------
DATA_PATH = "./data_motobikes.xlsx"
df = None

def load_default_data(path=DATA_PATH):
    if os.path.exists(path):
        try:
            return pd.read_excel(path, engine="openpyxl")
        except Exception as e:
            st.error(f"❌ Lỗi đọc file mẫu {path}: {e}") 
            return None
    return None

def preprocess_df_before_predict(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # ---- XỬ LÝ GIÁ ----
    if "Giá" in df.columns:
        df["Giá"] = (
            df["Giá"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
        )

        df["Giá"] = pd.to_numeric(df["Giá"], errors="coerce")

    # ---- XỬ LÝ NĂM ĐĂNG KÝ ----
    if "Năm đăng ký" in df.columns:
        df["Năm đăng ký"] = df["Năm đăng ký"].astype(str).str.strip()

        df["Năm đăng ký"] = df["Năm đăng ký"].apply(
            lambda x: 1980 if "trước" in x.lower() else x
        )

        df["Năm đăng ký"] = pd.to_numeric(df["Năm đăng ký"], errors="coerce")
        df["Năm đăng ký"] = df["Năm đăng ký"].fillna(1980)

    # ---- XỬ LÝ SỐ KM ----
    if "Số Km đã đi" in df.columns:
        df["Số Km đã đi"] = (
            df["Số Km đã đi"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
        )
        df["Số Km đã đi"] = pd.to_numeric(df["Số Km đã đi"], errors="coerce")

    return df


df = load_default_data()


# Helper function để hiển thị profile image với scaling và cropping (100x100)
def display_profile_image(image_path, caption_text):
    
    img_src = ""
    # CSS cho thẻ chứa 100x100 và ảnh bên trong
    style_css = """
        width: 100px;
        height: 100px;
        border-radius: 50%; /* Làm tròn để nhìn giống profile */
        overflow: hidden;
        margin-bottom: 10px;
        border: 2px solid #00bcd4;
        display: inline-block;
    """
    
    # Placeholder HTML nếu không tìm thấy ảnh
    placeholder_html = f"""
        <div style="{style_css} background-color:#161b22; display: flex; align-items: center; justify-content: center;">
            <p style="color: #c9d1d9; font-size: 0.8em; text-align: center;">[{caption_text}]</p>
        </div>
    """

    if os.path.exists(image_path):
        try:
            # Đọc và chuyển đổi ảnh sang Base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            mime_type = "image/jpeg" 
            img_src = f"data:{mime_type};base64,{encoded_string}"
            
            # HTML cho ảnh, sử dụng object-fit: cover để scaling và crop
            image_html = f"""
                <div style="{style_css}">
                    <img src="{img_src}" style="width: 100%; height: 100%; object-fit: cover;">
                </div>
            """
            st.markdown(image_html, unsafe_allow_html=True)
            return
            
        except Exception:
            # Fallback nếu có lỗi Base64
            pass
            
    # Hiển thị Placeholder nếu ảnh không tồn tại hoặc lỗi
    st.markdown(placeholder_html, unsafe_allow_html=True)

# ---------- Sidebar (3 tabs) ----------
st.sidebar.title("🛠️ **HỆ THỐNG MENU**")
menu = ["Tổng quan", "Dự đoán giá", "Phát hiện bất thường"]
choice = st.sidebar.selectbox("Chọn tính năng", menu)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("⬆️ Upload File Data (CSV/XLSX)", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.sidebar.success("✅ File Data đã được load thành công!")
    except Exception as e:
        st.sidebar.error(f"❌ Lỗi khi đọc file upload: {e}")
        df = None

# ---------- Load model once ----------
MODEL_PATH = "car_price_gbr_pipeline.pkl"
model = None
model_load_error = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model_load_error = e
else:
    model_load_error = FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

# Helper function for Image Overlay (Sử dụng Base64 Encoding làm CSS Background)
def display_title_overlay(title_text, image_path, notes_html=""):
    
    background_style = ""
    # Lớp phủ tối 15% (tương đương filter: brightness(0.85))
    dark_filter = "linear-gradient(rgba(0,0,0,0.15), rgba(0,0,0,0.15))" 
    
    # Fallback box style nếu không tìm thấy ảnh
    fallback_style = "background-color: #161b22; border: 2px dashed #00bcd4;"
    
    if os.path.exists(image_path):
        try:
            # Đọc và chuyển đổi ảnh sang Base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode()
            
            # Xây dựng Base64 URI
            mime_type = "image/jpeg" 
            img_src = f"url(data:{mime_type};base64,{encoded_string})"
            
            # SỬA LỖI REPEAT VÀ ĐẢM BẢO SCALING (Sử dụng longhand properties)
            # 1. Background Image: Filter (lớp 1) và Ảnh (lớp 2)
            background_style += f"background-image: {dark_filter}, {img_src};"
            # 2. Background Repeat: no-repeat cho cả hai lớp (Chặn lặp lại)
            background_style += "background-repeat: no-repeat, no-repeat;"
            # 3. Background Position: center cho cả hai lớp (Lấy phần trung tâm)
            background_style += "background-position: center, center;"
            # 4. Background Size: auto cho filter, cover cho ảnh (Scaling ra vừa khung)
            background_style += "background-size: auto, cover;"
            
            # Reset fallback style nếu ảnh được load qua Base64/CSS
            fallback_style = "" 
            
        except Exception:
            # Giữ nguyên fallback style nếu có lỗi Base64
            pass
            
    # HTML structure now uses inline style for background
    html_content = f"""
    <div class="cover-header" style="{fallback_style} {background_style}">
        <div class="cover-text-overlay">
            <h1>{title_text}</h1>
            <div class="cover-notes">{notes_html}</div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# ---------- Pages ----------
if choice == "Tổng quan":
    
    # Notes for the title page (bottom right, left aligned internally, bỏ dấu **)
    notes_content = """
    Giảng viên: Khuat Thuy Phuong<br>
    Nhóm 6: Tran Thien Thanh & Nguyen Quoc Thinh<br>
    Ngày báo cáo: 29/11/2025
    """
    
    # Use Title Overlay for the main page with line break in title (xuống hàng sau Project:, dòng 2 không wrap)
    display_title_overlay(
        "Final Data Science Project:<br><span style='white-space: nowrap;'>Price Prediction and Anomaly Detection</span>", 
        "hero_bike.jpg", 
        notes_html=notes_content
    )
    
    # Define main tabs
    tab_titles = ["Mục tiêu nghiệp vụ", "Thu thập dữ liệu", "EDA", "SKlearn", "Pyspark", "Phân công công việc", "Bài học kinh nghiệm"]
    tabs = st.tabs(tab_titles)

    # --- 1. Mục tiêu nghiệp vụ ---
    with tabs[0]:
        st.header("🎯 Mục tiêu nghiệp vụ")
        
        st.markdown("""
        Dự án xây dựng hai tính năng cốt lõi dựa trên Machine Learning để nâng cao độ tin cậy và minh bạch cho nền tảng giao dịch xe máy cũ: 
        """)
        
        st.markdown("##### 💰 1. Định Giá Thị Trường (Price Prediction)")
        st.markdown("""
        * **Mục tiêu**: Phát triển mô hình hồi quy (Regression Model) để ước tính **Giá Tham Chiếu Công Bằng** (Fair Market Price) cho xe máy cũ.
        * **Giá trị**: Giúp người bán định giá hợp lý, người mua có cơ sở tham khảo chính xác.
        """)
        
        st.markdown("##### 🚨 2. Cảnh Báo Gian Lận (Anomaly Detection)")
        st.markdown("""
        * **Mục tiêu**: Sử dụng các phương pháp thống kê hoặc học máy (dựa trên residual của mô hình giá) để xác định các giao dịch có giá **quá thấp** (nguy cơ lừa đảo, lỗi nhập liệu) hoặc **quá cao** (thổi phồng giá).
        * **Giá trị**: Tăng cường **Độ Tin Cậy** và **Minh Bạch** của sàn giao dịch.
        """)
        
        st.markdown("---")
        st.subheader("Phạm vi & Công nghệ")
        st.info("""
        * **Phạm vi Data**: Dữ liệu giao dịch xe máy cũ tại TP.HCM.
        * **Công nghệ ML**: Thử nghiệm và so sánh giữa thư viện **Scikit-learn (SKlearn)** và **PySpark MLlib** để đánh giá hiệu suất trên tập dữ liệu.
        """)


    # --- 2. Thu thập dữ liệu ---
    with tabs[1]:
        st.header("🛠️ Thu thập dữ liệu")
        st.markdown("""
        Dữ liệu được thu thập thông qua Web Scraping từ một nền tảng giao dịch xe máy cũ lớn, tập trung vào thị trường **TP.HCM**.
        
        ### 📊 Tóm tắt Data Set
        """)
        st.info("""
        * **Kích thước ban đầu**: 7208 rows và 18 columns.
        * **Các cột chính**: `Giá` (Target), `Thương hiệu`, `Dòng xe`, `Năm đăng ký`, `Số Km đã đi`, `Tình trạng`, `Loại xe`, `Dung tích xe`, `Xuất xứ`.
        * **Định dạng thô**: Các cột `Giá`, `Năm đăng ký`, `Số Km đã đi` cần được xử lý/chuẩn hóa vì chứa chuỗi ký tự không phải số (`trước năm 1980`, đơn vị tiền tệ, v.v.).
        """)
        st.subheader("🧹 Data Cleaning")
        st.code("""
# Xử lý cột 'Giá'
df["Giá"] = df["Giá"].astype(str).str.replace(r"[^0-9]", "", regex=True)
df["Giá"] = pd.to_numeric(df["Giá"], errors="coerce")

# Xử lý cột 'Năm đăng ký'
df["Năm đăng ký"] = df["Năm đăng ký"].apply(
    lambda x: 1980 if "trước" in str(x).lower() else x
)
df["Năm đăng ký"] = pd.to_numeric(df["Năm đăng ký"], errors="coerce").fillna(1980)

# Xử lý Outlier: IQR method được áp dụng cho cột 'Giá' và 'Số Km đã đi' để loại bỏ các giá trị cực đoan.
""", language='python')


    # --- 3. EDA (Exploratory Data Analysis) ---
    with tabs[2]:
        st.header("🔍 EDA - Phân tích Dữ liệu Khám phá")
        st.markdown("""
        Phân tích EDA nhằm hiểu rõ phân bố dữ liệu, tìm kiếm mối quan hệ giữa các biến, và phát hiện outliers.
        """)
        
        # Tạo Biểu đồ 1: Phân bố Giá (Log Transformed)
        st.subheader("1. 📈 Phân bố biến mục tiêu (Giá)")
        if df is not None and 'Giá' in df.columns:
            # Tạo DataFrame sạch để vẽ biểu đồ (chỉ cho mục đích trực quan)
            df_eda = df.copy()
            df_eda = preprocess_df_before_predict(df_eda)
            
            # Loại bỏ NaNs và lọc giá trị hợp lý (tránh lỗi Log)
            df_eda = df_eda.dropna(subset=['Giá'])
            df_eda = df_eda[df_eda['Giá'] > 0]
            
            if not df_eda.empty:
                # Log Transform (để hình ảnh trực quan tốt hơn)
                df_eda['Log Giá'] = np.log1p(df_eda['Giá'])
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot 1: Original Distribution (Price)
                sns.histplot(df_eda['Giá'], ax=ax[0], bins=50, kde=True, color='#00e5ff')
                ax[0].set_title('Phân bố Giá gốc (Lệch phải)', color='white')
                ax[0].tick_params(colors='white')
                ax[0].set_xlabel('Giá (VND)', color='white')
                ax[0].set_ylabel('Tần suất', color='white')

                # Plot 2: Log-Transformed Distribution
                sns.histplot(df_eda['Log Giá'], ax=ax[1], bins=50, kde=True, color='#00bcd4')
                ax[1].set_title('Phân bố Log Giá (Gần chuẩn)', color='white')
                ax[1].tick_params(colors='white')
                ax[1].set_xlabel('Log(Giá)', color='white')
                ax[1].set_ylabel('Tần suất', color='white')
                
                # Theme adjustments for dark mode
                fig.patch.set_facecolor('#0d1117')
                ax[0].set_facecolor('#161b22')
                ax[1].set_facecolor('#161b22')
                ax[0].spines['top'].set_color('white')
                ax[0].spines['bottom'].set_color('white')
                ax[0].spines['left'].set_color('white')
                ax[0].spines['right'].set_color('white')
                ax[1].spines['top'].set_color('white')
                ax[1].spines['bottom'].set_color('white')
                ax[1].spines['left'].set_color('white')
                ax[1].spines['right'].set_color('white')
                
                plt.tight_layout()
                st.pyplot(fig)
                st.info("Biểu đồ cho thấy cột Giá gốc bị lệch phải nghiêm trọng, việc Log-Transformation giúp phân bố gần Normal hơn, rất quan trọng cho các mô hình hồi quy tuyến tính.")
            else:
                st.warning("Không đủ dữ liệu hợp lệ (Giá > 0) để vẽ biểu đồ.")
        else:
            st.warning("Dataframe không được tải hoặc thiếu cột 'Giá'.")

        
        # Tạo Biểu đồ 2: Ma trận Tương quan (Correlation Heatmap)
        st.subheader("2. 🔗 Ma trận Tương quan giữa các biến Số")
        numerical_cols = ['Giá', 'Năm đăng ký', 'Số Km đã đi']
        if df is not None and all(col in df.columns for col in numerical_cols):
            df_corr = df.copy()
            df_corr = preprocess_df_before_predict(df_corr)
            df_corr = df_corr.select_dtypes(include=np.number).dropna()
            
            if not df_corr.empty and len(df_corr.columns) >= 2:
                corr_matrix = df_corr.corr()
                
                fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt=".2f", 
                    linewidths=.5, 
                    linecolor='#0d1117',
                    cbar_kws={'label': 'Hệ số tương quan'},
                    ax=ax_corr
                )
                ax_corr.set_title('Ma trận Tương quan', color='white')
                fig_corr.patch.set_facecolor('#0d1117')
                ax_corr.set_facecolor('#161b22')
                ax_corr.tick_params(colors='white')
                
                plt.tight_layout()
                st.pyplot(fig_corr)
                st.info("Ma trận tương quan cho thấy 'Giá' có mối tương quan âm mạnh với 'Năm đăng ký' (xe càng cũ, giá càng giảm) và 'Số Km đã đi' (chạy càng nhiều, giá càng giảm).")
            else:
                st.warning("Không đủ biến số hợp lệ để tính toán ma trận tương quan.")
        else:
            st.warning("Dataframe không được tải hoặc thiếu các cột số cần thiết.")

        st.markdown("---")


    # --- 4. SKlearn (Traditional ML) ---
    with tabs[3]:
        st.header("⚙️ SKlearn - Mô hình Machine Learning Truyền thống")
        
        tab_sk_pred, tab_sk_anom = st.tabs(["Mô hình Dự đoán Giá (Regression)", "Mô hình Phát hiện Bất thường (Anomaly)"])
        
        with tab_sk_pred:
            st.subheader("🤖 Dự đoán Giá (Regression)")
            st.markdown("""
            Thử nghiệm 4 mô hình hồi quy phổ biến sau khi tiền xử lý dữ liệu (Log-Transformation, Scaling, One-Hot Encoding).
            """)
            
            # Bảng so sánh mô hình SKlearn (Giữ nguyên)
            st.table(
            pd.DataFrame({
                "Mô hình": ["Linear Regression", "Random Forest Regressor", "**Gradient Boosting Regressor (GBR)**", "XGBoost Regressor"],
                "RMSE (triệu VND)": ["9.39", "8.92", "**8.86**", "8.81"],
                "MAE (triệu VND)": ["5.88", "5.42", "**5.22**", "5.29"],
                "R²": ["0.62", "0.66", "**0.66**", "0.66"],
                "Ghi chú": ["Cơ bản", "Tốt", "**Tốt nhất MAE**", "Tốt, nhanh"]
            })
            )
            st.success("""
            **Kết luận & Lựa chọn**: **Gradient Boosting Regressor (GBR)** được chọn để triển khai API/GUI. Mặc dù RMSE hơi cao hơn XGBoost, nhưng **MAE (Sai số tuyệt đối trung bình)** thấp nhất (**5.22 triệu VND**) cho thấy mô hình dự đoán giá chính xác hơn đối với phần lớn giao dịch.
            """)
            
            # Hình ảnh sơ đồ Pipeline
            if os.path.exists("ml_pipeline.jpg"):
                 st.image("ml_pipeline.jpg", caption="ML Pipeline Architecture", use_container_width=True)
            else:
                 # FIX: Sử dụng triple quotes
                 st.markdown("""<div style="background-color:#161b22; height: 150px; border-radius: 10px; border: 2px dashed #00bcd4; display: flex; align-items: center; justify-content: center;"><h5 style="color: #c9d1d9;">[PLACEHOLDER: ml_pipeline.jpg - Sơ đồ quy trình ML]</h5></div>""", unsafe_allow_html=True)


        with tab_sk_anom:
            st.subheader("⚠️ Phát hiện Bất thường (Anomaly Detection)")
            st.markdown("""
            **Phương pháp Residual-based**: Sử dụng mô hình **GBR** đã huấn luyện để ước tính giá trị thị trường $ \hat{y} $ của một giao dịch. Bất thường được phát hiện dựa trên độ lớn của **phần dư (residual)**: $ |y - \hat{y}| $.
            """)
            st.code(r"Anomaly = True \text{ if } |Giá thực tế - Giá dự đoán| > Threshold", language='text')
            st.info("""
            * **Phần dư Dương ($y - \hat{y} > 0$):** Giá thực tế **cao hơn** giá thị trường -> **Cảnh báo giá quá cao** (thổi phồng/xe hiếm).
            * **Phần dư Âm ($y - \hat{y} < 0$):** Giá thực tế **thấp hơn** giá thị trường -> **Cảnh báo giá quá thấp** (lỗi nhập liệu/gian lận).
            * **Ngưỡng ($Threshold$)**: Được đặt bằng **1.5 lần độ lệch chuẩn (Standard Deviation)** của residuals trên tập Train/Validation để xác định một giao dịch là bất thường.
            """)


    # --- 5. Pyspark (Big Data ML) ---
    with tabs[4]:
        st.header("☁️ PySpark - Xử lý & Mô hình PySpark MLlib")
        st.markdown("""
        PySpark được sử dụng để mô phỏng khả năng mở rộng xử lý dữ liệu (ETL) và huấn luyện mô hình trên môi trường Big Data (Spark Cluster).
        """)
        tab_spark_pred, tab_spark_anom = st.tabs(["Mô hình Dự đoán Giá (Regression)", "Mô hình Phát hiện Bất thường (Anomaly)"])
        
        with tab_spark_pred:
            st.subheader("🚀 Dự đoán Giá (PySpark Regression)")
            st.markdown("""
            Thử nghiệm với các mô hình PySpark MLlib sau khi xử lý dữ liệu bằng **VectorAssembler**, **StringIndexer** và **OneHotEncoder**.
            """)
            
            # Bảng so sánh mô hình PySpark (Giữ nguyên)
            st.table(
            pd.DataFrame({
                "Mô hình": ["Linear Regression (PySpark)", "Decision Tree Regressor", "**Gradient Boosted Tree Regressor (GBT)**", "Random Forest Regressor"],
                "RMSE (triệu VND)": ["10.21", "10.05", "**8.95**", "9.01"],
                "MAE (triệu VND)": ["6.15", "6.12", "**5.30**", "5.45"],
                "Ghi chú": ["Hiệu suất thấp", "Tốt", "**Tốt nhất PySpark**", "Tốt"]
            })
            )
            st.success("""
            **Kết luận & Lựa chọn (PySpark)**: **Gradient Boosted Tree Regressor (GBT)** cho thấy hiệu suất cao nhất trong môi trường PySpark, với MAE là **5.30 triệu VND**, gần bằng với GBR của SKlearn. Mô hình này được chọn cho quy trình xử lý Big Data.
            """)
            
            # Hình ảnh sơ đồ Big Data Workflow
            if os.path.exists("mechanical_bg.jpg"):
                 st.image("mechanical_bg.jpg", caption="PySpark GBT Workflow", use_container_width=True)
            else:
                 # FIX: Sử dụng triple quotes
                 st.markdown("""<div style="background-color:#161b22; height: 150px; border-radius: 10px; border: 2px dashed #00bcd4; display: flex; align-items: center; justify-content: center;"><h5 style="color: #c9d1d9;">[PLACEHOLDER: mechanical_bg.jpg - Sơ đồ quy trình Big Data]</h5></div>""", unsafe_allow_html=True)


        with tab_spark_anom:
            st.subheader("🚨 Phát hiện Bất thường (PySpark Anomaly Detection)")
            st.markdown("""
            **Phương pháp Residual-based**: Sử dụng mô hình **GBT (PySpark)** để tính residuals và xác định ngưỡng bất thường.
            """)
            st.code(r"PySpark Anomaly = True \text{ if } |Giá thực tế - GBT\_Giá dự đoán| > Threshold", language='text')
            st.info("""
            * **Ưu điểm PySpark**: Quá trình tính toán residuals và xác định ngưỡng (ví dụ: tính $\sigma$ của residuals) có thể được thực hiện song song trên cluster, rất hiệu quả cho lượng dữ liệu lớn.
            """)

    # --- 6. Phân công công việc ---
    with tabs[5]:
        st.header("👥 Phân công công việc")
        st.markdown("""
        Dự án được thực hiện bởi nhóm hai người với sự phân công chuyên môn hóa rõ ràng:
        """)
        
        col_thanh, col_thinh = st.columns(2)
        
        with col_thanh:
            st.subheader("👤 **Trần Thiện Thanh**")
            display_profile_image("profile_thanh.jpg", "Ảnh Thanh")
                 
            st.markdown("""
            * **Chuyên môn**: **Modelling** & **Deployment**.
            * **Công việc chính**:
                * Xây dựng và so sánh các Mô hình dự đoán **Regression** (SKlearn & PySpark).
                * Xây dựng Mô hình **Phát hiện Bất thường** (Anomaly Detection).
                * **Tối ưu hóa Hyperparameters** (GridSearch/RandomSearch).
                * **Đóng gói Model** (Joblib) và tích hợp vào Streamlit App.
            """)
            
        with col_thinh:
            st.subheader("👤 **Nguyễn Quốc Thịnh**")
            display_profile_image("profile_thinh.jpg", "Ảnh Thịnh")
                 
            st.markdown("""
            * **Chuyên môn**: **Data Analysis** & **GUI/UX**.
            * **Công việc chính**:
                * **Thu thập dữ liệu** (Web Scraping).
                * Thực hiện **EDA (Exploratory Data Analysis)** và Data Cleaning ban đầu.
                * **Thiết kế giao diện người dùng (GUI)** bằng Streamlit.
                * Đảm bảo tính **User Experience (UX)** và thẩm mỹ (Dark/Futuristic Theme).
            """)
            
        st.markdown("---")
        st.info("Sự kết hợp giữa chuyên môn ML/Deployment và Data Analysis/UX đảm bảo dự án có cả tính chính xác và tính ứng dụng cao.")

    # --- 7. Learning points ---
    with tabs[6]:
        st.header("🧠 Bài học kinh nghiệm")
        st.markdown("""
        Dự án đã mang lại nhiều bài học quan trọng trong việc triển khai giải pháp ML từ A đến Z:
        """)
        
        st.markdown("##### 🧪 1. Xử lý Dữ liệu Lệch (Skewed Data)")
        st.info("""
        * **Thử thách**: Biến Target (`Giá`) bị Right-Skewed nặng, làm giảm hiệu suất của các mô hình hồi quy tuyến tính.
        * **Bài học**: Việc áp dụng **Log-Transformation** cho biến Target là cực kỳ quan trọng đối với các mô hình tuyến tính và ensemble tree (dù ít nhạy cảm hơn) để đạt được phân bố gần Gaussian, cải thiện đáng kể chỉ số RMSE/MAE.
        """)
        
        st.markdown("##### ⚖️ 2. So sánh Công nghệ (SKlearn vs. PySpark)")
        st.info("""
        * **Thử thách**: Đánh giá sự cần thiết của môi trường Big Data (PySpark) so với môi trường truyền thống (SKlearn) trên một tập data trung bình.
        * **Bài học**: Mặc dù SKlearn (Python đơn) cho kết quả **MAE tốt hơn chút ít** (5.22 triệu VND so với 5.30 triệu VND của PySpark GBT), PySpark chứng minh khả năng xử lý **Mở Rộng** (Scalability) và quy trình **ETL song song** nhanh hơn khi khối lượng data tăng lên.
        """)
        
        st.markdown("##### 🎯 3. Anomaly Detection (Residual-based)")
        st.info("""
        * **Thử thách**: Xây dựng cơ chế phát hiện bất thường thực tế, hữu dụng cho Business.
        * **Bài học**: Phương pháp **Residual-based** (dựa trên sự khác biệt giữa giá thực tế và giá dự đoán của mô hình Regression) là một cách tiếp cận **hiệu quả và dễ giải thích** cho Business để phát hiện các giao dịch không hợp lý so với xu hướng thị trường.
        """)
        
        st.markdown("##### 🖥️ 4. Tích hợp & GUI/UX")
        st.info("""
        * **Thử thách**: Đóng gói mô hình và tạo giao diện trực quan, hấp dẫn cho người dùng cuối.
        * **Bài học**: Việc sử dụng **Streamlit** giúp triển khai nhanh chóng. Thiết kế **Dark Theme & Futuristic UX** không chỉ đẹp mắt mà còn cải thiện khả năng đọc và thu hút người dùng trong môi trường ứng dụng phân tích.
        """)


elif choice == "Dự đoán giá":
    # Use Title Overlay for the prediction page
    display_title_overlay("PRICE PREDICTION", "price_prediction.jpg")

    if df is None:
        st.error("⚠️ Hệ thống chưa có dữ liệu. Vui lòng **Upload File Data** ở Sidebar.")
        st.stop()

    # --- Hướng dẫn cho người dùng ---
    st.markdown("### 📋 **HƯỚNG DẪN SỬ DỤNG TÍNH NĂNG ĐỊNH GIÁ**")
    st.info("""
    AI sẽ tính toán **Giá Tham Chiếu Hợp Lý** (Fair Market Price) cho chiếc xe của bạn dựa trên dữ liệu thị trường đã huấn luyện.
    
    **Các bước:**
    1. **Chọn** tất cả các thông số kỹ thuật (Hãng xe, Dòng xe, Tình trạng, v.v.).
    2. **Nhập** chỉ số `Số Km đã đi` hiện tại của xe.
    3. Nhấn nút **TÍNH TOÁN GIÁ THỊ TRƯỜNG**.
    
    Kết quả sẽ hiển thị **GIÁ ƯỚC TÍNH HỢP LÝ** (VND), là mức giá thị trường bạn nên tham khảo.
    """)
    # --- Kết thúc Hướng dẫn ---

    # Inputs layout with columns
    st.subheader("⚙️ **NHẬP THÔNG SỐ XE**")
    try:
        col_cat1, col_cat2, col_cat3 = st.columns(3)
        with col_cat1:
            thuong_hieu = st.selectbox("Hãng xe", df['Thương hiệu'].dropna().unique())
            tinh_trang = st.selectbox("Tình trạng", df['Tình trạng'].dropna().unique())
        with col_cat2:
            dong_xe = st.selectbox("Dòng xe", df['Dòng xe'].dropna().unique())
            loai_xe = st.selectbox("Loại xe", df['Loại xe'].dropna().unique())
        with col_cat3:
            dung_tich_xe = st.selectbox("Dung tích xe (cc)", df['Dung tích xe'].dropna().unique())
            xuat_xu = st.selectbox("Xuất xứ", df['Xuất xứ'].dropna().unique())
            
        col_num1, col_num2 = st.columns(2)
        with col_num1:
            # Hoán đổi: Năm đăng ký -> Number Input
            nam_dang_ky = st.number_input("Năm đăng ký", min_value=1980, max_value=2025, value=2015, step=1)
        with col_num2:
            # Hoán đổi: Số Km đã đi -> Slider
            so_km_da_di = st.slider("Số Km đã đi", min_value=0, max_value=500000, value=50000, step=1000)

    except Exception:
        # Giữ nguyên source Code cho phần này theo yêu cầu của user
        st.error("❌ Data mẫu bị lỗi hoặc thiếu cột thông số xe.")
        st.stop()

    if model is None:
        st.warning(f"⚠️ Mô hình định giá chưa sẵn sàng ({model_load_error}).")

    st.markdown("---")
    du_doan_gia = st.button("✨ **TÍNH TOÁN GIÁ THỊ TRƯỜNG**", type="primary")
    
    if du_doan_gia:
        with st.spinner('Đang phân tích dữ liệu thị trường...'):
            if model is None:
                st.error("❌ Không thể dự đoán vì Mô hình không load được.")
            else:
                # FIX: Thêm cột 'Khoảng giá min' với giá trị 0 vì nó bị thiếu trong input_data nhưng cần cho model.
                input_data = pd.DataFrame([{
                    'Thương hiệu': thuong_hieu,
                    'Dòng xe': dong_xe,
                    'Tình trạng': tinh_trang,
                    'Loại xe': loai_xe,
                    'Dung tích xe': dung_tich_xe,
                    'Xuất xứ': xuat_xu,
                    'Năm đăng ký': nam_dang_ky,
                    'Số Km đã đi': so_km_da_di,
                    'Khoảng giá min': 0 # Cột bị thiếu trong lỗi
                }])
                try:
                    pred = model.predict(input_data)[0]
                    st.markdown("### 📈 **KẾT QUẢ ĐỊNH GIÁ**")
                    
                    st.metric(
                        label="GIÁ ƯỚC TÍNH HỢP LÝ (VND)",
                        value=f"{pred:,.0f}",
                        delta="Giá được đề xuất",
                        delta_color="normal"
                    )
                    st.success(f"🔑 Giá tham chiếu cho chiếc **{thuong_hieu} {dong_xe}** là **{pred:,.0f} VND**.")

                except Exception as e:
                    # Generic error message to handle the wide variety of missing columns
                    st.error("❌ Lỗi trong quá trình tính toán giá. Vui lòng kiểm tra lại dữ liệu đầu vào.")
                    # st.exception(e) # Dùng st.exception(e) để xem chi tiết lỗi nếu cần debug thêm.


elif choice == "Phát hiện bất thường":
    # Use Title Overlay for the anomaly page
    display_title_overlay("ANOMALY DETECTION", "anomaly_detection.jpg")

    if df is None:
        st.error("⚠️ Hệ thống chưa có dữ liệu. Vui lòng **Upload File Data** ở Sidebar.")
        st.stop()

    st.write("Cơ chế: So sánh **Giá Bạn Đăng** với **Giá Tham Chiếu** của hệ thống. Chênh lệch vượt **Ngưỡng Cho Phép** sẽ kích hoạt cảnh báo.")

    # Tạo 2 sub-tabs
    tab_user, tab_admin = st.tabs(["Người dùng (Kiểm tra bài đăng)", "Admin (Quản lý cảnh báo)"])

    with tab_user:
        st.subheader("📝 **Kiểm tra trước khi đăng bài**")
        # Inputs for user
        try:
            col_u_cat1, col_u_cat2, col_u_cat3 = st.columns(3)
            with col_u_cat1:
                thuong_hieu_a = st.selectbox("Hãng xe", df['Thương hiệu'].dropna().unique(), key="u1")
                tinh_trang_a = st.selectbox("Tình trạng", df['Tình trạng'].dropna().unique(), key="u3")
            with col_u_cat2:
                dong_xe_a = st.selectbox("Dòng xe", df['Dòng xe'].dropna().unique(), key="u2")
                loai_xe_a = st.selectbox("Loại xe", df['Loại xe'].dropna().unique(), key="u4")
            with col_u_cat3:
                dung_tich_a = st.selectbox("Dung tích xe (cc)", df['Dung tích xe'].dropna().unique(), key="u5")
                xuat_xu_a = st.selectbox("Xuất xứ", df['Xuất xứ'].dropna().unique(), key="u6")

            col_u_num1, col_u_num2 = st.columns(2)
            with col_u_num1:
                nam_dk_a = st.slider("Năm đăng ký", 1980, 2025, 2015, key="u7")
            with col_u_num2:
                so_km_a = st.number_input("Số Km đã đi", min_value=0, max_value=500000, value=50000, step=1000, key="u8")
        except Exception:
            # Giữ nguyên source Code cho phần này theo yêu cầu của user
            st.error("❌ Data mẫu bị lỗi hoặc thiếu cột thông số xe.")
            st.stop()

        gia_thuc_te = st.number_input("💲 **Giá thực tế (VND) bạn muốn đăng**", min_value=0, max_value=1_000_000_000, value=150_000_000, step=100_000)
        residual_threshold = st.number_input("📐 **Ngưỡng Chênh Lệch Tối Đa** (VND)", min_value=0, max_value=200_000_000, value=10_000_000, step=500_000)

        st.session_state.residual_threshold = residual_threshold

        btn_check_user = st.button("🔥 **KÍCH HOẠT KIỂM TRA HỆ THỐNG**", type="primary")
        if btn_check_user:
            if model is None:
                st.error(f"❌ Mô hình kiểm định chưa sẵn sàng ({model_load_error}).")
            else:
                # FIX: Thêm cột 'Khoảng giá min' với giá trị 0
                input_row = {
                    "Thương hiệu": thuong_hieu_a,
                    "Dòng xe": dong_xe_a,
                    "Tình trạng": tinh_trang_a,
                    "Loại xe": loai_xe_a,
                    "Dung tích xe": dung_tich_a,
                    "Xuất xứ": xuat_xu_a,
                    "Năm đăng ký": nam_dk_a,
                    "Số Km đã đi": so_km_a,
                    'Khoảng giá min': 0, # Cột bị thiếu trong lỗi
                    "Giá": gia_thuc_te
                }
                df_test = pd.DataFrame([input_row])

                def detect_residual_anomaly_single(df_single, model, threshold):
                    X = df_single.drop(columns=["Giá"])
                    pred_price = model.predict(X)[0]
                    residual = df_single["Giá"].iloc[0] - pred_price
                    is_anom = abs(residual) > threshold
                    return pred_price, residual, is_anom

                try:
                    pred_price, residual, is_anom = detect_residual_anomaly_single(df_test, model, residual_threshold)
                    
                    st.markdown("### **KẾT QUẢ KIỂM ĐỊNH**")
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Giá Tham Chiếu", f"{pred_price:,.0f} VND")
                    with col_res2:
                        delta_color = "inverse" if abs(residual) > residual_threshold else "normal"
                        st.metric("Chênh Lệch", f"{residual:,.0f} VND", delta=f"{residual:,.0f} VND", delta_color=delta_color)
                    
                    if 'anomaly_records' not in st.session_state:
                        st.session_state.anomaly_records = []

                    record = {
                        "Thời gian": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Hãng xe": thuong_hieu_a,
                        "Dòng xe": dong_xe_a,
                        "Giá thực tế": gia_thuc_te,
                        "Giá dự đoán": pred_price,
                        "Chênh lệch": residual,
                        "Status": "Pending" if is_anom else "Approved",
                        "Bất thường": is_anom
                    }

                    if is_anom:
                        delta = residual / 1000000
                        if residual > 0:
                            st.error(f"🚨 **CẢNH BÁO: GIÁ QUÁ CAO**! (Chênh **{delta:,.1f} triệu VND**). Bài đăng cần **Admin Phê Duyệt**. (Lý do: Thổi phồng giá).")
                        else:
                            st.error(f"🚨 **CẢNH BÁO: GIÁ QUÁ THẤP**! (Chênh **{abs(delta):,.1f} triệu VND**). Bài đăng cần **Admin Phê Duyệt**. (Lý do: Nghi vấn Lỗi nhập liệu/Gian lận).")
                        record["Bất thường loại"] = "Quá cao" if residual > 0 else "Quá thấp"
                        st.session_state.anomaly_records.append(record)
                    else:
                        st.success(f"✅ **GIAO DỊCH CHUẨN**: Giá nằm trong ngưỡng cho phép (± {residual_threshold:,} VND). Bài đăng được duyệt tự động.")
                        st.session_state.anomaly_records.append(record)
                except Exception as e:
                    st.error("❌ Lỗi trong quá trình kiểm tra. Vui lòng kiểm tra lại dữ liệu đầu vào.")
                    # st.exception(e) # Dùng st.exception(e) để xem chi tiết lỗi nếu cần debug thêm.

    with tab_admin:
        st.subheader("🛡️ **QUẢN LÝ CẢNH BÁO**")

        st.markdown("#### 1. Bài đăng **CHỜ DUYỆT** từ Người dùng")
        if 'anomaly_records' not in st.session_state or not st.session_state.anomaly_records:
            st.info("Chưa có cảnh báo nào từ người dùng.")
        else:
            df_admin_user = pd.DataFrame(st.session_state.anomaly_records)
            
            def highlight_pending(s):
                return ['background-color: rgba(255, 0, 0, 0.2)' if v == 'Pending' else '' for v in s]

            st.dataframe(df_admin_user.style.apply(highlight_pending, subset=['Status'], axis=0), use_container_width=True)

            total_anom_user = df_admin_user[df_admin_user['Bất thường'] == True].shape[0]
            st.write(f"Tổng số cảnh báo **Bất Thường** từ người dùng: **{total_anom_user}**.")

            st.markdown("##### 🔑 **CỔNG PHÊ DUYỆT**")
            col_select, col_app, col_rej = st.columns([2, 1, 1])
            with col_select:
                selected_index = st.selectbox("Chọn index cảnh báo (từ 0)", range(len(df_admin_user)))
            
            with col_app:
                if st.button("✅ CHẤP NHẬN", key="btn_app_user"):
                    st.session_state.anomaly_records[selected_index]["Status"] = "Approved"
                    st.success(f"Đã chấp nhận bài đăng {selected_index}.")
            with col_rej:
                if st.button("❌ TỪ CHỐI", key="btn_rej_user"):
                    st.session_state.anomaly_records[selected_index]["Status"] = "Rejected"
                    st.warning(f"Đã từ chối bài đăng {selected_index}.")

            st.markdown("---")
            st.write("Bảng **Cảnh Báo** cập nhật:")
            st.dataframe(pd.DataFrame(st.session_state.anomaly_records).style.apply(highlight_pending, subset=['Status'], axis=0), use_container_width=True)

        st.markdown("#### 2. Quét Anomaly trên **Dữ liệu Lớn**")
        admin_threshold = st.number_input("📐 Ngưỡng chênh lệch (VND) cho data load", min_value=0, max_value=200_000_000, value=st.session_state.get('residual_threshold', 10_000_000), step=500_000, key="admin_thres")
        
        btn_check_df = st.button("🔎 **QUÉT TOÀN BỘ DATASET**", type="secondary")
        if btn_check_df:
            if model is None:
                st.error(f"❌ Mô hình kiểm định chưa sẵn sàng ({model_load_error}).")
            else:
                with st.spinner('Đang kiểm tra toàn bộ Data Lake...'):
                    try:
                        df_clean = df.copy()
                        
                        # FIX: Thêm 'Khoảng giá min' vào cột yêu cầu
                        required_cols = ['Giá', 'Thương hiệu', 'Dòng xe', 'Tình trạng', 'Loại xe', 'Dung tích xe', 'Xuất xứ', 'Năm đăng ký', 'Số Km đã đi', 'Khoảng giá min']
                        
                        missing_cols = [col for col in required_cols if col not in df_clean.columns]
                        
                        if missing_cols:
                            # Tự động thêm cột bị thiếu nếu là 'Khoảng giá min'
                            if 'Khoảng giá min' in missing_cols:
                                df_clean['Khoảng giá min'] = 0
                                missing_cols.remove('Khoảng giá min')
                                st.warning("Cột 'Khoảng giá min' bị thiếu trong file upload. Đã đặt giá trị mặc định là 0 để Pipeline hoạt động.")
                            
                            if missing_cols:
                                st.error(f"❌ Dataframe thiếu cột quan trọng: {', '.join(missing_cols)}")
                                st.stop()
                        
                        # Fixes for prediction data quality
                        df_clean = preprocess_df_before_predict(df_clean)
                        
                        for col in ['Thương hiệu', 'Dòng xe', 'Tình trạng', 'Loại xe', 'Dung tích xe', 'Xuất xứ']:
                            mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                            df_clean[col].fillna(mode_val, inplace=True)
                        
                        km_median = df_clean['Số Km đã đi'].median()
                        df_clean['Số Km đã đi'].fillna(km_median, inplace=True)
                        gia_median = df_clean['Giá'].median()
                        df_clean['Giá'].fillna(gia_median, inplace=True)
                        
                        if df_clean.empty:
                            st.warning("⚠️ Dataframe rỗng sau xử lý.")
                        else:
                            X = df_clean.drop(columns=["Giá"])
                            pred_prices = model.predict(X)
                            residuals = df_clean["Giá"] - pred_prices
                            is_anom = abs(residuals) > admin_threshold
                            df_anom = df_clean[is_anom].copy()
                            df_anom["Giá dự đoán"] = pred_prices[is_anom]
                            df_anom["Chênh lệch"] = residuals[is_anom]
                            df_anom["Bất thường loại"] = ["Quá cao" if r > 0 else "Quá thấp" for r in residuals[is_anom]]
                            df_anom["Status"] = "Pending" 
                            
                            df_anom.reset_index(names=['Original Index'], inplace=True)

                            if df_anom.empty:
                                st.success("🎉 **SUCCESS**: Không tìm thấy giao dịch bất thường nào trong dataset này.")
                            else:
                                st.write(f"**KẾT QUẢ**: Tìm thấy **{df_anom.shape[0]}** giao dịch bất thường.")
                                st.dataframe(df_anom, use_container_width=True)
                                
                                if 'df_anom_records' not in st.session_state:
                                    st.session_state.df_anom_records = df_anom.to_dict('records')
                                else:
                                    st.session_state.df_anom_records = df_anom.to_dict('records')


                                st.markdown("##### 🔑 **CỔNG PHÊ DUYỆT (DATASET)**")
                                
                                col_df_select, col_df_app, col_df_rej = st.columns([2, 1, 1])
                                with col_df_select:
                                    selected_df_index = st.selectbox("Chọn index cảnh báo (từ 0)", range(len(st.session_state.df_anom_records)), key="select_df_anom")
                                
                                with col_df_app:
                                    if st.button("✅ CHẤP NHẬN (DF)", key="btn_app_df"):
                                        st.session_state.df_anom_records[selected_df_index]["Status"] = "Approved"
                                        st.success(f"Đã chấp nhận sản phẩm {selected_df_index}.")
                                with col_df_rej:
                                    if st.button("❌ TỪ CHỐI (DF)", key="btn_rej_df"):
                                        st.session_state.df_anom_records[selected_df_index]["Status"] = "Rejected"
                                        st.warning(f"Đã từ chối sản phẩm {selected_df_index}.")
                                
                                st.markdown("---")
                                st.write("Bảng **Cảnh Báo Dataset** cập nhật:")
                                st.dataframe(pd.DataFrame(st.session_state.df_anom_records), use_container_width=True)

                    except Exception as e:
                        st.error("❌ Lỗi trong quá trình quét dataset. Vui lòng kiểm tra lại data đầu vào hoặc file model.")
                        # st.exception(e) # Dùng st.exception(e) để xem chi tiết lỗi nếu cần debug thêm.