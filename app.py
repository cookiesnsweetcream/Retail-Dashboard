# -*- coding: utf-8 -*-
"""
Retail Inventory Optimization Dashboard
=========================================
Dashboard analitik bisnis statis untuk membantu manajer untuk memahami
hasil segmentasi tingkat overstock produk (Agglomerative Clustering,
Ward Linkage, n_clusters=2) dan mengambil keputusan terkait persediaan.

Fitur utama model   : Demand Gap, Inventory Level, Overstock Ratio
Sumber data          : Hasil_Agglomerative_Clustering.csv

Cara menjalankan:
    streamlit run app.py
"""

import os
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Retail Inventory Optimization Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = "Hasil_Agglomerative_Clustering.csv"

CLUSTER_COLORS = {
    "High Overstock": "#E76F51",
    "Moderate Overstock": "#2A9D8F",
}

REQUIRED_COLUMNS = [
    "Store ID",
    "Category",
    "Product ID",
    "Inventory Level",
    "Demand Gap",
    "Overstock Ratio",
    "Cluster",
    "Cluster Name",
    "Recommendation",
]


# ==========================================================
# FUNGSI BANTU
# ==========================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Membaca dataset hasil clustering dari file CSV."""
    return pd.read_csv(path)


def format_id(value: int) -> str:
    """Format angka dengan pemisah ribuan ala Indonesia (titik)."""
    return f"{value:,}".replace(",", ".")


def clean_text(value) -> str:
    """Merapikan teks rekomendasi multi-baris menjadi satu baris bersih."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# ==========================================================
# CUSTOM CSS - TAMPILAN PROFESIONAL
# ==========================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1D3557;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
        max-width: 900px;
    }
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 1rem 1rem 0.5rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# CEK & MUAT DATASET
# ==========================================================
if not os.path.exists(DATA_PATH):
    st.warning(
        f"""
        **⚠️ File data tidak ditemukan**

        Dashboard ini membutuhkan file **`{DATA_PATH}`** untuk dapat menampilkan hasil analisis.

        **Langkah yang harus dilakukan:**
        1. Pastikan file `{DATA_PATH}` berada pada folder/direktori yang sama dengan file `app.py` ini.
        2. Jalankan ulang aplikasi dengan perintah `streamlit run app.py`.
        3. Jika file belum tersedia, jalankan terlebih dahulu notebook pemodelan
           (Agglomerative Clustering) untuk menghasilkan file tersebut.
        """
    )
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"❌ Terjadi kesalahan saat membaca file `{DATA_PATH}`: {e}")
    st.stop()

missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_columns:
    st.error(
        "❌ Struktur dataset tidak sesuai. Kolom berikut tidak ditemukan: "
        f"{', '.join(missing_columns)}. Pastikan file CSV merupakan hasil output "
        "notebook Agglomerative Clustering yang benar."
    )
    st.stop()

if df.empty:
    st.warning("⚠️ Dataset berhasil dibaca, namun tidak berisi data (kosong).")
    st.stop()


# ==========================================================
# HEADER & RINGKASAN EKSEKUTIF
# ==========================================================
st.markdown(
    '<p class="main-header">📦 Retail Inventory Optimization Dashboard</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Dashboard ini membantu manajer untuk memantau status persediaan produk '
    "dan memberikan rekomendasi aksi bisnis berdasarkan hasil segmentasi tingkat overstock, "
    "sehingga keputusan pemesanan, promosi, dan alokasi stok dapat diambil lebih cepat dan tepat.</p>",
    unsafe_allow_html=True,
)

st.divider()

total_produk = len(df)
total_high = int((df["Cluster Name"] == "High Overstock").sum())
total_moderate = int((df["Cluster Name"] == "Moderate Overstock").sum())

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)

with kpi_col1:
    st.metric("📊 Total Produk Dianalisis", format_id(total_produk))
    st.caption("Jumlah seluruh produk dalam dataset hasil segmentasi")

with kpi_col2:
    pct_high = (total_high / total_produk * 100) if total_produk else 0
    st.metric("🔴 Produk High Overstock", format_id(total_high))
    st.caption(f"{pct_high:.1f}% dari total produk — perlu evaluasi pemesanan")

with kpi_col3:
    pct_moderate = (total_moderate / total_produk * 100) if total_produk else 0
    st.metric("🟢 Produk Moderate Overstock", format_id(total_moderate))
    st.caption(f"{pct_moderate:.1f}% dari total produk — kondisi stok terkendali")

st.divider()


# ==========================================================
# VISUALISASI UTAMA
# ==========================================================
st.subheader("Visualisasi Segmentasi Produk")

viz_col1, viz_col2 = st.columns([3, 2])

with viz_col1:
    st.caption("**Demand Gap vs Inventory Level** — sebaran produk berdasarkan segmen overstock")
    fig_scatter = px.scatter(
        df,
        x="Demand Gap",
        y="Inventory Level",
        color="Cluster Name",
        color_discrete_map=CLUSTER_COLORS,
        opacity=0.75,
        hover_data=["Store ID", "Category", "Product ID"],
        labels={
            "Demand Gap": "Demand Gap (Permintaan − Stok)",
            "Inventory Level": "Tingkat Persediaan",
            "Cluster Name": "Segmen",
        },
    )
    fig_scatter.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(
        legend_title_text="Segmen",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with viz_col2:
    st.caption("**Jumlah Produk per Segmen**")
    cluster_count = (
        df["Cluster Name"]
        .value_counts()
        .rename_axis("Cluster Name")
        .reset_index(name="Jumlah Produk")
    )
    fig_bar = px.bar(
        cluster_count,
        x="Cluster Name",
        y="Jumlah Produk",
        color="Cluster Name",
        color_discrete_map=CLUSTER_COLORS,
        text="Jumlah Produk",
        labels={"Cluster Name": "Segmen"},
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        plot_bgcolor="white",
        xaxis_title="",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()


# ==========================================================
# ANALISIS DISTRIBUSI OPERASIONAL
# ==========================================================
st.subheader("Analisis Distribusi Operasional")
st.caption("Sebaran segmen overstock berdasarkan lokasi toko dan kategori produk")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    st.markdown("**Distribusi Segmen per Store ID**")
    store_dist = (
        df.groupby(["Store ID", "Cluster Name"])
        .size()
        .reset_index(name="Jumlah Produk")
    )
    fig_store = px.bar(
        store_dist,
        x="Store ID",
        y="Jumlah Produk",
        color="Cluster Name",
        color_discrete_map=CLUSTER_COLORS,
        barmode="stack",
        labels={"Cluster Name": "Segmen"},
    )
    fig_store.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        plot_bgcolor="white",
        legend_title_text="Segmen",
    )
    st.plotly_chart(fig_store, use_container_width=True)

with dist_col2:
    st.markdown("**Distribusi Segmen per Category**")
    category_dist = (
        df.groupby(["Category", "Cluster Name"])
        .size()
        .reset_index(name="Jumlah Produk")
    )
    fig_category = px.bar(
        category_dist,
        x="Category",
        y="Jumlah Produk",
        color="Cluster Name",
        color_discrete_map=CLUSTER_COLORS,
        barmode="stack",
        labels={"Cluster Name": "Segmen"},
    )
    fig_category.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
        plot_bgcolor="white",
        legend_title_text="Segmen",
    )
    st.plotly_chart(fig_category, use_container_width=True)

with st.expander("📋 Lihat Tabel Ringkasan Distribusi (Store ID × Segmen)"):
    crosstab_store = pd.crosstab(df["Store ID"], df["Cluster Name"])
    st.dataframe(crosstab_store, use_container_width=True)

with st.expander("📋 Lihat Tabel Ringkasan Distribusi (Category × Segmen)"):
    crosstab_category = pd.crosstab(df["Category"], df["Cluster Name"])
    st.dataframe(crosstab_category, use_container_width=True)

st.divider()


# ==========================================================
# TABEL STRATEGI & AKSI REKOMENDASI
# ==========================================================
st.subheader("Strategi & Aksi Rekomendasi")

tab1, tab2 = st.tabs(
    ["🔴 Produk High Overstock — Perlu Aksi", "📋 Preview Data Hasil Segmentasi"]
)

with tab1:
    st.markdown(
        "Produk berikut termasuk dalam segmen **High Overstock**. "
        "Segera lakukan evaluasi pemesanan, promosi, atau bundling sesuai rekomendasi "
        "di bawah ini agar stok tidak semakin menumpuk."
    )

    high_overstock_df = df.loc[
        df["Cluster Name"] == "High Overstock",
        [
            "Store ID",
            "Category",
            "Product ID",
            "Inventory Level",
            "Demand Gap",
            "Overstock Ratio",
            "Recommendation",
        ],
    ].copy()

    high_overstock_df["Recommendation"] = high_overstock_df["Recommendation"].apply(clean_text)
    high_overstock_df["Inventory Level"] = high_overstock_df["Inventory Level"].round(1)
    high_overstock_df["Demand Gap"] = high_overstock_df["Demand Gap"].round(1)
    high_overstock_df["Overstock Ratio"] = high_overstock_df["Overstock Ratio"].round(2)

    high_overstock_df = high_overstock_df.rename(
        columns={
            "Inventory Level": "Tingkat Persediaan",
            "Overstock Ratio": "Rasio Overstock",
            "Recommendation": "Rekomendasi Bisnis",
        }
    ).sort_values("Tingkat Persediaan", ascending=False)

    st.dataframe(
        high_overstock_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    csv_download = high_overstock_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Unduh Daftar Produk High Overstock (CSV)",
        data=csv_download,
        file_name="produk_high_overstock.csv",
        mime="text/csv",
    )

with tab2:
    st.markdown("Berikut adalah preview 20 data teratas hasil segmentasi seluruh produk.")
    preview_columns = [
        "Store ID",
        "Category",
        "Product ID",
        "Inventory Level",
        "Demand Gap",
        "Overstock Ratio",
        "Cluster Name",
    ]
    st.dataframe(
        df[preview_columns].head(20),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.caption(
    "Dashboard ini dibangun berdasarkan hasil model **Agglomerative Clustering** "
    "(Ward Linkage, n_clusters=2) dengan fitur utama **Demand Gap**, **Inventory Level**, "
    "dan **Overstock Ratio** yang dipilih melalui Sequential Forward Selection."
)
