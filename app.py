"""
==================================================================
DASHBOARD ANALITIK BISNIS - MANAJEMEN OVERSTOCK PERSEDIAAN RETAIL
Berbasis Hasil Segmentasi Agglomerative Clustering
==================================================================
Ditujukan untuk: Manajer Retail
Tujuan: Mendukung pengambilan keputusan strategis terkait
        manajemen persediaan produk & mitigasi overstock.
==================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================================
# 1. KONFIGURASI HALAMAN
# ==================================================================

st.set_page_config(
    page_title="Dashboard Manajemen Overstock Persediaan",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Sedikit styling tambahan agar tampilan lebih rapi & profesional
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background-color: #f8f9fb;
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .block-container {
        padding-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================================
# 2. FUNGSI LOAD DATA
# ==================================================================

@st.cache_data
def load_data(file) -> pd.DataFrame:
    """
    Membaca dataset persediaan retail hasil akhir clustering.
    Dataset diasumsikan sudah memiliki kolom 'Cluster' (0 / 1)
    hasil dari notebook Agglomerative Clustering.
    """
    df = pd.read_csv(file)

    # Pastikan kolom Date bertipe datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Pastikan kolom Cluster bertipe integer
    if "Cluster" in df.columns:
        df["Cluster"] = df["Cluster"].astype(int)

    return df


def map_cluster_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Melakukan pemetaan label segmen (High Overstock / Moderate-Low Overstock)
    secara DINAMIS berdasarkan rata-rata Inventory Level tiap cluster —
    bukan hardcode 0/1 — karena label numerik hasil Agglomerative
    Clustering bersifat arbitrer dan bisa berbeda tiap kali model dilatih ulang.
    Cluster dengan rata-rata Inventory Level LEBIH TINGGI relatif terhadap
    Units Sold akan diberi label "High Overstock".
    """
    df = df.copy()

    cluster_profile = (
        df.groupby("Cluster")
        .agg(
            avg_inventory=("Inventory Level", "mean"),
            avg_sold=("Units Sold", "mean")
        )
        .reset_index()
    )

    # Rasio persediaan terhadap penjualan -> makin besar rasio, makin "overstock"
    cluster_profile["overstock_ratio"] = (
        cluster_profile["avg_inventory"] / cluster_profile["avg_sold"].replace(0, np.nan)
    )
    cluster_profile["overstock_ratio"] = cluster_profile["overstock_ratio"].fillna(
        cluster_profile["avg_inventory"]
    )

    cluster_profile = cluster_profile.sort_values("overstock_ratio", ascending=False).reset_index(drop=True)

    label_map = {}
    for i, row in cluster_profile.iterrows():
        if i == 0:
            label_map[row["Cluster"]] = "High Overstock"
        else:
            label_map[row["Cluster"]] = "Moderate / Low Overstock"

    df["Segmen"] = df["Cluster"].map(label_map)
    return df, label_map


# ==================================================================
# 3. HEADER
# ==================================================================

st.title("📦 Dashboard Analitik Persediaan Retail")
st.caption(
    "Sistem Informasi Analitik Bisnis — Segmentasi Overstock Produk "
    "menggunakan Agglomerative Clustering"
)
st.divider()


# ==================================================================
# 4. UPLOAD / LOAD DATASET
# ==================================================================

uploaded_file = st.sidebar.file_uploader(
    "📁 Unggah Dataset (CSV)",
    type=["csv"],
    help="Dataset harus sudah memiliki kolom 'Cluster' hasil dari model Agglomerative Clustering."
)

st.sidebar.divider()

if uploaded_file is None:
    st.info(
        "⬆️ Silakan unggah dataset CSV pada sidebar untuk memulai analisis. "
        "Dataset harus memiliki kolom **'Cluster'** hasil dari proses clustering."
    )
    st.stop()

df_raw = load_data(uploaded_file)

required_cols = [
    "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered",
    "Demand Forecast", "Cluster"
]
missing_cols = [c for c in required_cols if c not in df_raw.columns]

if missing_cols:
    st.error(
        f"❌ Kolom berikut tidak ditemukan pada dataset: {', '.join(missing_cols)}. "
        "Pastikan dataset sudah sesuai format hasil clustering."
    )
    st.stop()

df_raw, label_map = map_cluster_segment(df_raw)


# ==================================================================
# 5. SIDEBAR FILTER
# ==================================================================

st.sidebar.header("🔎 Filter Data")

# --- Filter Region ---
region_options = sorted(df_raw["Region"].dropna().unique().tolist())
selected_region = st.sidebar.multiselect(
    "Wilayah (Region)",
    options=region_options,
    default=region_options
)

# --- Filter Category ---
category_options = sorted(df_raw["Category"].dropna().unique().tolist())
selected_category = st.sidebar.multiselect(
    "Kategori Produk",
    options=category_options,
    default=category_options
)

# --- Filter Store ID ---
store_options = sorted(df_raw["Store ID"].dropna().unique().tolist())
selected_store = st.sidebar.multiselect(
    "ID Toko (Store ID)",
    options=store_options,
    default=store_options
)

# --- Filter Seasonality (jika tersedia) ---
if "Seasonality" in df_raw.columns:
    season_options = sorted(df_raw["Seasonality"].dropna().unique().tolist())
    selected_season = st.sidebar.multiselect(
        "Musim (Seasonality)",
        options=season_options,
        default=season_options
    )
else:
    selected_season = None

# --- Filter Rentang Tanggal (jika tersedia) ---
if "Date" in df_raw.columns and df_raw["Date"].notna().any():
    min_date = df_raw["Date"].min().date()
    max_date = df_raw["Date"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Rentang Tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    selected_date_range = None

st.sidebar.divider()
st.sidebar.markdown(
    "**Legenda Segmen:**\n"
    "- 🔴 High Overstock\n"
    "- 🟢 Moderate / Low Overstock"
)


# ==================================================================
# 6. PENERAPAN FILTER
# ==================================================================

df = df_raw.copy()

df = df[
    df["Region"].isin(selected_region)
    & df["Category"].isin(selected_category)
    & df["Store ID"].isin(selected_store)
]

if selected_season is not None:
    df = df[df["Seasonality"].isin(selected_season)]

if selected_date_range is not None and isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    df = df[
        (df["Date"] >= pd.to_datetime(start_date))
        & (df["Date"] <= pd.to_datetime(end_date))
    ]

if df.empty:
    st.warning("⚠️ Tidak ada data yang sesuai dengan kombinasi filter yang dipilih.")
    st.stop()


# ==================================================================
# 7. RINGKASAN METRIK UTAMA (KPI CARDS)
# ==================================================================

st.subheader("📊 Ringkasan Kondisi Persediaan")

total_high = int((df["Segmen"] == "High Overstock").sum())
total_moderate = int((df["Segmen"] == "Moderate / Low Overstock").sum())
total_all = total_high + total_moderate
pct_high = (total_high / total_all * 100) if total_all > 0 else 0

avg_inventory = df["Inventory Level"].mean()
total_units_sold = df["Units Sold"].sum()
avg_demand_forecast = df["Demand Forecast"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "🔴 Produk High Overstock",
        f"{total_high:,}",
        f"{pct_high:.1f}% dari total data"
    )

with col2:
    st.metric(
        "🟢 Produk Moderate/Low Overstock",
        f"{total_moderate:,}",
        f"{100 - pct_high:.1f}% dari total data"
    )

with col3:
    st.metric(
        "📦 Rata-rata Inventory Level",
        f"{avg_inventory:,.0f} unit"
    )

with col4:
    st.metric(
        "🛒 Total Units Sold",
        f"{total_units_sold:,.0f} unit"
    )

with col5:
    st.metric(
        "📈 Rata-rata Demand Forecast",
        f"{avg_demand_forecast:,.0f} unit"
    )

st.divider()


# ==================================================================
# 8. VISUALISASI UTAMA
# ==================================================================

tab1, tab2, tab3 = st.tabs([
    "📌 Distribusi Overstock",
    "📌 Stok vs Penjualan",
    "📌 Tren Persediaan"
])

# ------------------------------------------------------------------
# CHART 1 — Distribusi Overstock berdasarkan Kategori / Wilayah
# ------------------------------------------------------------------
with tab1:
    st.markdown("#### Distribusi Segmen Overstock")

    dimensi = st.radio(
        "Kelompokkan berdasarkan:",
        options=["Kategori Produk", "Wilayah (Region)"],
        horizontal=True
    )
    kolom_dimensi = "Category" if dimensi == "Kategori Produk" else "Region"

    dist_df = (
        df.groupby([kolom_dimensi, "Segmen"])
        .size()
        .reset_index(name="Jumlah Produk")
    )

    fig1 = px.bar(
        dist_df,
        x=kolom_dimensi,
        y="Jumlah Produk",
        color="Segmen",
        barmode="group",
        text="Jumlah Produk",
        color_discrete_map={
            "High Overstock": "#E4572E",
            "Moderate / Low Overstock": "#29A19C"
        },
        title=f"Jumlah Produk per Segmen Overstock — Berdasarkan {dimensi}"
    )
    fig1.update_layout(
        xaxis_title=dimensi,
        yaxis_title="Jumlah Produk",
        legend_title="Segmen Overstock",
        height=480
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_pie = px.pie(
            df,
            names="Segmen",
            title="Proporsi Segmen Overstock (Keseluruhan)",
            color="Segmen",
            color_discrete_map={
                "High Overstock": "#E4572E",
                "Moderate / Low Overstock": "#29A19C"
            },
            hole=0.45
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("##### 💡 Insight Singkat")
        top_cat = (
            df[df["Segmen"] == "High Overstock"]
            .groupby(kolom_dimensi)
            .size()
            .sort_values(ascending=False)
        )
        if not top_cat.empty:
            st.write(
                f"**{dimensi}** dengan jumlah produk *High Overstock* "
                f"terbanyak adalah **{top_cat.index[0]}** "
                f"dengan **{int(top_cat.iloc[0])} produk**. "
                "Disarankan menjadi prioritas evaluasi strategi harga "
                "dan volume pemesanan berikutnya."
            )

# ------------------------------------------------------------------
# CHART 2 — Inventory Level vs Units Sold (per Cluster)
# ------------------------------------------------------------------
with tab2:
    st.markdown("#### Analisis Persediaan vs Penjualan per Segmen")

    fig2 = px.scatter(
        df,
        x="Inventory Level",
        y="Units Sold",
        color="Segmen",
        size="Demand Forecast",
        hover_data=["Store ID", "Product ID", "Category", "Region"],
        color_discrete_map={
            "High Overstock": "#E4572E",
            "Moderate / Low Overstock": "#29A19C"
        },
        opacity=0.7,
        title="Inventory Level vs Units Sold — Diwarnai Berdasarkan Segmen Overstock"
    )
    fig2.update_layout(
        xaxis_title="Tingkat Persediaan (Inventory Level)",
        yaxis_title="Jumlah Terjual (Units Sold)",
        height=520
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "🔎 **Cara membaca grafik:** Titik berwarna merah (High Overstock) yang berada "
        "di sisi kanan bawah menunjukkan produk dengan **persediaan sangat tinggi namun "
        "penjualan rendah** — inilah kandidat utama untuk diberi diskon atau dikurangi "
        "jumlah pemesanan berikutnya."
    )

# ------------------------------------------------------------------
# CHART 3 — Tren Inventory Level dari Waktu ke Waktu
# ------------------------------------------------------------------
with tab3:
    st.markdown("#### Tren Persediaan dari Waktu ke Waktu")

    if "Date" in df.columns and df["Date"].notna().any():
        granularitas = st.radio(
            "Granularitas waktu:",
            options=["Harian", "Mingguan", "Bulanan"],
            horizontal=True
        )

        freq_map = {"Harian": "D", "Mingguan": "W", "Bulanan": "M"}

        trend_df = (
            df.set_index("Date")
            .groupby([pd.Grouper(freq=freq_map[granularitas]), "Segmen"])["Inventory Level"]
            .mean()
            .reset_index()
        )

        fig3 = px.line(
            trend_df,
            x="Date",
            y="Inventory Level",
            color="Segmen",
            markers=True,
            color_discrete_map={
                "High Overstock": "#E4572E",
                "Moderate / Low Overstock": "#29A19C"
            },
            title=f"Tren Rata-rata Inventory Level ({granularitas}) per Segmen"
        )
        fig3.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Rata-rata Inventory Level",
            height=480
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning(
            "⚠️ Kolom 'Date' tidak tersedia atau tidak valid pada dataset, "
            "sehingga grafik tren waktu tidak dapat ditampilkan."
        )

st.divider()


# ==================================================================
# 9. REKOMENDASI TINGKAT MANAJEMEN
# ==================================================================

st.subheader("📋 Rekomendasi Tindakan Manajerial — Produk High Overstock")

st.markdown(
    "Tabel di bawah ini menampilkan daftar **Produk & Toko** yang teridentifikasi "
    "sebagai **High Overstock**. Manajer disarankan untuk mempertimbangkan "
    "pemberian **potongan harga (discount)** atau **pengurangan kuota Units Ordered** "
    "pada periode pemesanan berikutnya."
)

high_overstock_df = df[df["Segmen"] == "High Overstock"].copy()

if high_overstock_df.empty:
    st.success("✅ Tidak ada produk yang teridentifikasi sebagai High Overstock pada filter saat ini.")
else:
    tampilan_cols = [
        "Store ID", "Product ID", "Category", "Region",
        "Inventory Level", "Units Sold", "Units Ordered",
        "Demand Forecast", "Price", "Discount"
    ]
    tampilan_cols = [c for c in tampilan_cols if c in high_overstock_df.columns]

    tabel_rekomendasi = (
        high_overstock_df[tampilan_cols]
        .drop_duplicates(subset=["Store ID", "Product ID"])
        .sort_values("Inventory Level", ascending=False)
        .reset_index(drop=True)
    )

    tabel_rekomendasi["Rekomendasi"] = np.where(
        tabel_rekomendasi["Units Sold"] < tabel_rekomendasi["Inventory Level"] * 0.3,
        "Beri Diskon Segera & Kurangi Units Ordered",
        "Kurangi Units Ordered Periode Berikutnya"
    )

    st.dataframe(
        tabel_rekomendasi,
        use_container_width=True,
        height=420,
        hide_index=True
    )

    col_dl1, col_dl2 = st.columns([1, 4])
    with col_dl1:
        csv_export = tabel_rekomendasi.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Unduh Daftar (CSV)",
            data=csv_export,
            file_name="rekomendasi_high_overstock.csv",
            mime="text/csv"
        )

    st.caption(
        f"Total **{len(tabel_rekomendasi)}** kombinasi Toko & Produk teridentifikasi "
        "sebagai High Overstock berdasarkan filter yang aktif."
    )

st.divider()
st.caption(
    "Dashboard ini dibangun berdasarkan hasil model Agglomerative Clustering "
    "(2 cluster, linkage: ward) yang telah dilatih sebelumnya. "
    "Label segmen dipetakan secara dinamis berdasarkan rasio Inventory Level "
    "terhadap Units Sold agar tetap konsisten meskipun label numerik cluster berubah."
)