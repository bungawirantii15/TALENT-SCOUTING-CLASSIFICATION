import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

from preprocess import preprocess_dataset
from logistic_regression import train_logistic_regression
from optimasi import train_hybrid_ga_lr

# ==========================
#   PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="LR & GA Classifier",
    layout="wide",
)


# ==========================
#   CUSTOM CSS
# ==========================
st.markdown("""
    <style>

    body { background: #eef2f7 !important; }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72, #2a5298);
        color: white;
        padding: 20px;
    }
    .identitas-card {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #E0E0E5;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
    }
    .identitas-card img {
        border-radius: 50%;
        border: 3px solid #2c3e50;
    }
    .identitas-card h4 {
        font-weight: 700;
        font-size: 18px;
        color:  #2c3e50;
    }
    .identitas-card p {
        font-size: 14px;
        color: #6A6A6A;
        margin-top: -5px;
    }

    .sidebar-text {
        font-size: 16px;
        line-height: 1.6;
        color: white;
        text-align: center;
    }

    /* HEADER */
    .custom-header {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 20px;
    }

    .custom-header h2 {
        margin: 0;
        font-size: 30px;
        font-weight: 700;
    }

    .custom-header p {
        margin: 5px 0 0 0;
        font-size: 16px;
        opacity: 0.9;
    }

    /* CARD */
    .glass-card {
        background: rgba(255, 255, 255, 0.60);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(255,255,255,0.45);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    /* BUTTON */
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 30px;
        font-size: 17px;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(37,117,252,0.4);
        transition: 0.3s;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(37,117,252,0.6);
    }

    /* SECTION TITLE */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
        border-left: 6px solid #2575fc;
        padding-left: 10px;
    }

    /* DIVIDER */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #2575fc, transparent);
        margin: 25px 0;
    }

    </style>
""", unsafe_allow_html=True)



# ==========================
#   SIDEBAR
# ==========================
with st.sidebar:
    st.markdown("""
    <div class="identitas-card">
        <img src="https://www.seekpng.com/png/detail/380-3807393_unsri-vector-logo-logo-universitas-sriwijaya.png" width="75">
        <h4>Nama: Bunga Wiranti</h4>
        <p>NIM: 09021182227120</p>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("""
<br>
<div class='sidebar-text'>
<b>Nama:</b><br>
</div>
""", unsafe_allow_html=True)


# ==========================
#   HEADER
# ==========================
st.markdown("""
<div class="custom-header">
    <h2>🔍 Logistic Regression & Genetic Algorithm Classifier</h2>
    <p>Evaluasi Model Logistic Regression dengan optimasi Genetich Algorithm untuk Klasifikasi Talent Scouting Renang.</p>
</div>
""", unsafe_allow_html=True)


# ================================
#     UPLOAD DATA
# ================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📂 Upload Dataset (CSV)</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)



# =============================
# CEK FORMAT
# =============================
if uploaded:

    if not uploaded.name.endswith(".csv"):
        st.error("❌ Hanya file CSV yang diizinkan!")
        st.stop()

    df = pd.read_csv(uploaded)

    # PREPROCESS
    df_train_x, df_test_x, df_train_y, df_test_y = preprocess_dataset(df)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔎 Preview Dataset (After Preprocessing)</div>", unsafe_allow_html=True)
    preview_processed = pd.concat([df_train_x, df_train_y], axis=1)
    st.dataframe(preview_processed.head(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚙ Pengaturan Model</div>", unsafe_allow_html=True)

    model_choice = st.radio(
        "Pilih Model:",
        ["Logistic Regression", "Logistic Regression + GA"],
        horizontal=True
    )

    col1, col2 = st.columns(2)
    with col1:
        C_val = st.selectbox(
            "C Value",
            ["-- pilih C_value --", 0.01, 0.1, 1, 10, 100]
        )
    with col2:
        solver = st.selectbox(
            "Solver",
            ["-- pilih solver --", "liblinear", "lbfgs", "saga"]
        )

    # PARAMETER GA
    if model_choice == "Logistic Regression + GA":
        st.markdown("<br><div class='section-title'>⚙ Parameter Genetic Algorithm</div>", unsafe_allow_html=True)
        colA, colB, colC, colD = st.columns(4)

        with colA:
            pop = st.number_input("Population", 10, 200, 40)
        with colB:
            gen = st.number_input("Generations", 5, 200, 20)
        with colC:
            mutation = st.number_input("Mutation Rate", 0.01, 1.0, 0.1)
        with colD:
            crossover = st.number_input("Crossover Rate", 0.1, 1.0, 0.8)

    st.markdown("</div>", unsafe_allow_html=True)


    # =======================================
    #              TOMBOL RUN
    # =======================================
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    run_btn = st.button("🚀 RUN CLASSIFICATION")
    st.markdown("</div>", unsafe_allow_html=True)


    if run_btn:

        # VALIDASI
        if solver == "-- pilih solver --":
            st.warning("⚠ Harap pilih solver terlebih dahulu!")
            st.stop()

        if C_val == "-- pilih C_value --" or C_val <= 0:
            st.warning("⚠ Nilai C tidak valid!")
            st.stop()
        if model_choice == "Logistic Regression + GA":
            if pop <= 0 or gen <= 0 or mutation <= 0 or crossover <= 0:
                st.warning("⚠ Parameter GA tidak boleh kosong!")
                st.stop()

        # TRAINING MODEL
        if model_choice == "Logistic Regression":
            acc, pred = train_logistic_regression(
                df_train_x, df_test_x, df_train_y, df_test_y,
                C_val, solver
            )
            st.success(f"Accuracy Logistic Regression: {acc:.4f}")

        else:
            acc, pred, selected_features = train_hybrid_ga_lr(
                df_train_x, df_test_x, df_train_y, df_test_y,
                C_val, solver, pop, gen, mutation, crossover
            )
            st.success(f"Accuracy LR + GA: {acc:.4f}")


        # ==========================
        #     METRICS
        # ==========================
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📊 Evaluation Metrics</div>", unsafe_allow_html=True)

        st.write(f"Accuracy: {accuracy_score(df_test_y, pred):.4f}")
        st.write(f"Precision: {precision_score(df_test_y, pred, average='macro'):.4f}")
        st.write(f"Recall: {recall_score(df_test_y, pred, average='macro'):.4f}")
        st.write(f"F1 Score: {f1_score(df_test_y, pred, average='macro'):.4f}")

        st.markdown("</div>", unsafe_allow_html=True)


        # ==========================
        # CONFUSION MATRIX
        # ==========================
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔢 Confusion Matrix</div>", unsafe_allow_html=True)

        cm = confusion_matrix(df_test_y, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)


        # ==========================
        # TABEL HASIL
        # ==========================
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📄 Hasil Klasifikasi</div>", unsafe_allow_html=True)

        result_df = pd.DataFrame({
            "Actual": df_test_y.values,
            "Predicted": pred
        })

        st.dataframe(result_df, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("📂 Silakan upload file dataset berformat CSV untuk melanjutkan.")