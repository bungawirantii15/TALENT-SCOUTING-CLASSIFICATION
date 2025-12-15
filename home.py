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

st.title("Logistic Regression & GA Classifier")

uploaded = st.file_uploader("Upload Dataset", type=["csv"])

# ================================
#  ALERT: CEK FORMAT FILE
# ================================
if uploaded:
    if not uploaded.name.endswith(".csv"):
        st.error("❌ Hanya file CSV yang diizinkan!")
        st.stop()

    df = pd.read_csv(uploaded)
    st.write("Preview Dataset:")
    st.dataframe(df.head())

    # PREPROCESS
    df_train_x, df_test_x, df_train_y, df_test_y = preprocess_dataset(df)

    st.write("### Pilih Model")
    model_choice = st.radio("Model:", ["Logistic Regression", "Logistic Regression + GA"])

    st.write("---")
    st.write("### Hyperparameter")

    C_val = st.selectbox(
        "C Value",
        ["-- pilih C_value --", 0.01, 0.1, 1, 10, 100]
    )

    solver = st.selectbox(
        "Solver",
        ["-- pilih solver --", "liblinear", "lbfgs", "saga"]
    )

    # Parameter GA
    if model_choice == "Logistic Regression + GA":
        pop = st.number_input("Population Size", 10, 200, 40)
        gen = st.number_input("Generations", 5, 200, 20)
        mutation = st.number_input("Mutation Rate", 0.01, 1.0, 0.1)
        crossover = st.number_input("Crossover Rate", 0.1, 1.0, 0.8)

    st.write("---")

    if st.button("CLASSIFY"):

        # ============================
        # VALIDASI SOLVER
        # ============================
        if solver == "-- pilih solver --":
            st.warning("⚠ Harap pilih solver terlebih dahulu!")
            st.stop()

        # VALIDASI C VALUE
        if C_val <= 0:
            st.warning("⚠ Nilai C tidak boleh 0 atau negatif!")
            st.stop()

        # VALIDASI GA
        if model_choice == "Logistic Regression + GA":
            if pop <= 0 or gen <= 0 or mutation <= 0 or crossover <= 0:
                st.warning("⚠ Harap isi semua parameter GA dengan benar!")
                st.stop()

        # ==========================
        #  MODEL TRAINING
        # ==========================
        if model_choice == "Logistic Regression":
            acc, pred = train_logistic_regression(
                df_train_x, df_test_x, df_train_y, df_test_y, C_val, solver
            )
            st.success(f"Accuracy Logistic Regression: {acc:.4f}")

        else:
            acc, pred, selected_features = train_hybrid_ga_lr(
                df_train_x, df_test_x, df_train_y, df_test_y,
                C_val, solver, pop, gen, mutation, crossover
            )
            st.success(f"Accuracy Logistic Regression + GA: {acc:.4f}")

        # ==========================
        #  METRIK EVALUASI
        # ==========================
        st.write("### 📊 Evaluation Metrics")

        st.write(f"**Accuracy:** {accuracy_score(df_test_y, pred):.4f}")
        st.write(f"**Precision:** {precision_score(df_test_y, pred, average='macro'):.4f}")
        st.write(f"**Recall:** {recall_score(df_test_y, pred, average='macro'):.4f}")
        st.write(f"**F1 Score:** {f1_score(df_test_y, pred, average='macro'):.4f}")

        # ==========================
        #  CONFUSION MATRIX
        # ==========================
        st.write("### 🔢 Confusion Matrix")

        cm = confusion_matrix(df_test_y, pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ==========================
        #  TABEL HASIL
        # ==========================
        result_df = pd.DataFrame({
            "Actual": df_test_y.values,
            "Predicted": pred
        })

        st.write("### Hasil Klasifikasi")
        st.dataframe(result_df)

else:
    st.info("📂 Silakan upload file dataset berformat CSV untuk melanjutkan.")
