# ==========================================================
# DASHBOARD SEGMENTASI TINGKAT OVERSTOCK PERSEDIAAN PRODUK RETAIL
# Model: Agglomerative Clustering (Unsupervised Learning)
# Target Pengguna: Manajer Retail (Non-Teknis)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering

# ==========================================================
# 1. KONFIGURASI HALAMAN
# ==========================================================

st.set_page_config(
    page_title="Dashboard Segmentasi Overstock Retail",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = "retail_store_inventory.csv"

CLUSTER_NAME_MAP = {
    0: "High Overstock",
    1: "Moderate/Low Overstock"
}

CLUSTER_COLOR_MAP = {
    "High Overstock": "#E74C3C",
    "Moderate/Low Overstock": "#2ECC71"
}


# ==========================================================
# 2. LOAD DATA & PIPELINE CLUSTERING (BACKGROUND, TIDAK TAMPAK KE USER)
# ==========================================================

@st.cache_data(show_spinner="Memuat dan memproses data...")
def load_and_cluster_data(path: str) -> pd.DataFrame:
    """
    1. Membaca dataset mentah lokal.
    2. Membersihkan missing value & duplikat.
    3. Melakukan encoding kategorikal (khusus untuk kalkulasi model).
    4. Mengagregasi data pada level Store ID + Product ID (unit analisis produk).
    5. Menstandardisasi fitur numerik.
    6. Menjalankan AgglomerativeClustering (n_clusters=2, linkage='ward').
    7. Menggabungkan label Cluster & Segment kembali ke data transaksi asli
       agar tetap bisa difilter per baris (Region, Category, Date).
    """
    df_raw = pd.read_csv(path)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])

    df_clean = df_raw.copy()

    # --- Imputasi missing value ---
    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0 and df_clean[numeric_cols].isnull().sum().sum() > 0:
        imputer_num = SimpleImputer(strategy="median")
        df_clean[numeric_cols] = imputer_num.fit_transform(df_clean[numeric_cols])

    categorical_cols = df_clean.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0 and df_clean[categorical_cols].isnull().sum().sum() > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])

    # --- Hapus duplikat ---
    df_clean.drop_duplicates(inplace=True)

    # --- Pecah tanggal untuk kebutuhan fitur numerik model ---
    df_encoded = df_clean.copy()
    df_encoded["Year"] = df_encoded["Date"].dt.year
    df_encoded["Month"] = df_encoded["Date"].dt.month
    df_encoded["Day"] = df_encoded["Date"].dt.day

    # --- Label Encoding untuk fitur kategorikal (khusus perhitungan model) ---
    cat_columns_to_encode = [
        "Store ID", "Product ID", "Category", "Region",
        "Weather Condition", "Holiday/Promotion", "Seasonality"
    ]
    for col in cat_columns_to_encode:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + "_enc"] = le.fit_transform(df_encoded[col].astype(str))

    # --- Penanganan outlier (IQR clipping) pada fitur numerik utama ---
    outlier_cols = [
        "Inventory Level", "Units Sold", "Units Ordered",
        "Demand Forecast", "Price", "Discount", "Competitor Pricing"
    ]
    for col in outlier_cols:
        if col in df_encoded.columns:
            Q1 = df_encoded[col].quantile(0.25)
            Q3 = df_encoded[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_encoded[col] = df_encoded[col].clip(lower, upper)

    # --- Agregasi ke level Store ID + Product ID (unit analisis produk) ---
    product_df = (
        df_encoded
        .groupby(["Store ID", "Product ID"])
        .agg({
            "Category_enc": "first",
            "Region_enc": "first",
            "Inventory Level": "mean",
            "Units Sold": "sum",
            "Units Ordered": "sum",
            "Demand Forecast": "mean",
            "Price": "mean",
            "Discount": "mean",
            "Weather Condition_enc": "first",
            "Holiday/Promotion_enc": "mean",
            "Competitor Pricing": "mean",
            "Seasonality_enc": "first",
            "Year": "first",
            "Month": "mean",
            "Day": "mean",
        })
        .reset_index()
    )

    feature_cols = [
        "Category_enc", "Region_enc", "Inventory Level", "Units Sold",
        "Units Ordered", "Demand Forecast", "Price", "Discount",
        "Weather Condition_enc", "Holiday/Promotion_enc", "Competitor Pricing",
        "Seasonality_enc", "Year", "Month", "Day"
    ]

    X = product_df[feature_cols]

    # --- Standardisasi fitur ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Model Agglomerative Clustering (dijalankan otomatis di latar belakang) ---
    model = AgglomerativeClustering(n_clusters=2, linkage="ward")
    product_df["Cluster"] = model.fit_predict(X_scaled)
    product_df["Segment"] = product_df["Cluster"].map(CLUSTER_NAME_MAP)

    # --- Gabungkan label Cluster & Segment kembali ke data transaksi asli ---
    df_final = df_clean.merge(
        product_df[["Store ID", "Product ID", "Cluster", "Segment"]],
        on=["Store ID", "Product ID"],
        how="left"
    )

    return df_final


try:
    df = load_and_cluster_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"❌ File `{DATA_PATH}` tidak ditemukan. "
        f"Pastikan file dataset berada di folder yang sama dengan `app.py`."
    )
    st.stop()


# ==========================================================
# 3. SIDEBAR — FILTER
# ==========================================================

st.sidebar.title("🔎 Filter Dashboard")
st.sidebar.markdown("Gunakan filter berikut untuk menyesuaikan tampilan data.")

# --- Filter Region ---
region_options = sorted(df["Region"].dropna().unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Pilih Region",
    options=region_options,
    default=region_options
)

# --- Filter Category ---
category_options = sorted(df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Pilih Kategori Produk",
    options=category_options,
    default=category_options
)

# --- Filter Rentang Tanggal ---
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

selected_date_range = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
else:
    start_date, end_date = min_date, max_date

st.sidebar.markdown("---")
st.sidebar.caption(
    "Segmentasi dihasilkan otomatis oleh model **Agglomerative Clustering** "
    "berdasarkan pola Inventory Level, Units Sold, Units Ordered, dan fitur lainnya."
)

# --- Terapkan filter ---
mask = (
    df["Region"].isin(selected_regions)
    & df["Category"].isin(selected_categories)
    & (df["Date"].dt.date >= start_date)
    & (df["Date"].dt.date <= end_date)
)
df_filtered = df.loc[mask].copy()

if df_filtered.empty:
    st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter Anda.")
    st.stop()


# ==========================================================
# 4. HEADER
# ==========================================================

st.title("📦 Dashboard Segmentasi Tingkat Overstock Persediaan Produk Retail")
st.markdown(
    "Dashboard ini membantu **Manajer Retail** memantau produk yang berpotensi mengalami "
    "*overstock* berdasarkan hasil segmentasi model **Agglomerative Clustering**, "
    "sehingga keputusan operasional (diskon, penghentian order, distribusi ulang stok) "
    "dapat diambil lebih cepat dan berbasis data."
)
st.markdown("---")


# ==========================================================
# 5. KPI CARDS
# ==========================================================

total_high = int((df_filtered["Segment"] == "High Overstock").sum())
total_moderate = int((df_filtered["Segment"] == "Moderate/Low Overstock").sum())
avg_inventory = df_filtered["Inventory Level"].mean()
avg_units_sold = df_filtered["Units Sold"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔴 High Overstock", f"{total_high:,}", help="Jumlah baris data pada segmen High Overstock (Cluster 0)")

with col2:
    st.metric("🟢 Moderate/Low Overstock", f"{total_moderate:,}", help="Jumlah baris data pada segmen Moderate/Low Overstock (Cluster 1)")

with col3:
    st.metric("📊 Rata-rata Inventory Level", f"{avg_inventory:,.1f}")

with col4:
    st.metric("🛒 Rata-rata Units Sold", f"{avg_units_sold:,.1f}")

st.markdown("---")


# ==========================================================
# 6. VISUALISASI GRAFIK
# ==========================================================

row1_col1, row1_col2 = st.columns([1, 1.4])

# --- Proporsi Cluster (Pie Chart) ---
with row1_col1:
    st.subheader("Proporsi Segmentasi Overstock")
    segment_counts = df_filtered["Segment"].value_counts().reset_index()
    segment_counts.columns = ["Segment", "Jumlah"]

    fig_pie = px.pie(
        segment_counts,
        names="Segment",
        values="Jumlah",
        color="Segment",
        color_discrete_map=CLUSTER_COLOR_MAP,
        hole=0.45
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig_pie, use_container_width=True)

# --- High Overstock berdasarkan Category (Bar Chart) ---
with row1_col2:
    st.subheader("Kategori Produk dengan High Overstock Tertinggi")
    high_by_cat = (
        df_filtered[df_filtered["Segment"] == "High Overstock"]
        .groupby("Category")
        .size()
        .reset_index(name="Jumlah")
        .sort_values("Jumlah", ascending=False)
    )

    fig_bar_cat = px.bar(
        high_by_cat,
        x="Category",
        y="Jumlah",
        color="Jumlah",
        color_continuous_scale="Reds",
        text="Jumlah"
    )
    fig_bar_cat.update_layout(
        xaxis_title="Kategori Produk",
        yaxis_title="Jumlah Produk (High Overstock)",
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar_cat, use_container_width=True)

st.markdown("---")

# --- Scatter Plot Inventory vs Units Sold ---
st.subheader("Analisis Inventory Level vs Units Sold per Segmen")
st.caption(
    "Grafik ini membuktikan secara visual mengapa suatu produk dikategorikan Overstock: "
    "titik merah (High Overstock) umumnya memiliki Inventory Level tinggi namun Units Sold rendah."
)

fig_scatter = px.scatter(
    df_filtered,
    x="Inventory Level",
    y="Units Sold",
    color="Segment",
    color_discrete_map=CLUSTER_COLOR_MAP,
    size="Units Ordered",
    hover_data=["Store ID", "Product ID", "Category", "Region"],
    opacity=0.6
)
fig_scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10))
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")


# ==========================================================
# 7. TABEL DATA DETAIL — REKOMENDASI AKSI MANAJER
# ==========================================================

st.subheader("📋 Detail Produk High Overstock & Rekomendasi Aksi")

high_overstock_df = df_filtered[df_filtered["Segment"] == "High Overstock"].copy()


def generate_recommendation(row) -> str:
    """Menentukan rekomendasi aksi berdasarkan kondisi Units Sold & Units Ordered."""
    if row["Units Sold"] < row["Units Ordered"] * 0.5:
        return "Hentikan sementara Units Ordered & berikan diskon tambahan"
    elif row["Units Ordered"] > 0 and row["Units Sold"] / row["Units Ordered"] < 0.8:
        return "Pertimbangkan diskon tambahan untuk mempercepat penjualan"
    else:
        return "Pindahkan sebagian stok ke Region dengan Units Sold lebih tinggi"


if not high_overstock_df.empty:
    high_overstock_df["Rekomendasi Aksi Manajer"] = high_overstock_df.apply(
        generate_recommendation, axis=1
    )

    display_cols = [
        "Date", "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered",
        "Price", "Discount", "Segment", "Rekomendasi Aksi Manajer"
    ]

    st.dataframe(
        high_overstock_df[display_cols].sort_values("Inventory Level", ascending=False),
        use_container_width=True,
        height=420
    )

    csv_download = high_overstock_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Unduh Data High Overstock (CSV)",
        data=csv_download,
        file_name="high_overstock_products.csv",
        mime="text/csv"
    )
else:
    st.info("Tidak ada produk pada segmen High Overstock untuk kombinasi filter saat ini.")

st.markdown("---")
st.caption(
    "Dashboard ini dibuat untuk tugas Sistem Informasi Analitik Bisnis (SIAB) — "
    "Segmentasi dihasilkan oleh model Agglomerative Clustering (unsupervised learning) "
    "yang dijalankan otomatis di latar belakang saat aplikasi dimuat."
)