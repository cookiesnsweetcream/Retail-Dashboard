# ==========================================================
# SISTEM INFORMASI ANALITIK RETAIL — SEGMENTASI OVERSTOCK
# Model      : Agglomerative Clustering (Unsupervised Learning)
# Target     : Manajer Retail (Non-Teknis) & Tim Analis
# Versi      : 3.0 — Multi-Page App (Dashboard, Simulasi, Analisis Klaster)
#              Tema Modern Navy/Coral/Teal, Satuan Eksplisit,
#              Dendrogram, Countplot, Boxplot, Heatmap Korelasi
# Catatan    : Membutuhkan streamlit >= 1.33 (st.navigation & st.Page)
# ==========================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

sns.set_theme(style="whitegrid")


# ==========================================================
# 1. KONFIGURASI HALAMAN & KONSTANTA GLOBAL
# ==========================================================

st.set_page_config(
    page_title="Sistem Informasi Analitik Retail — Segmentasi Overstock",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "retail_store_inventory.csv"

# --- Satuan eksplisit untuk setiap metrik/kolom numerik ---
UNIT_LABELS = {
    "Inventory Level": "Inventory Level (Pcs/Unit)",
    "Units Sold": "Units Sold (Pcs/Unit)",
    "Units Ordered": "Units Ordered (Pcs/Unit)",
    "Demand Forecast": "Demand Forecast (Pcs/Unit)",
    "Price": "Price ($)",
    "Discount": "Discount (%)",
    "Competitor Pricing": "Competitor Pricing ($)",
}

# --- Palet warna korporat: kontras & konsisten di Light/Dark Mode ---
COLOR_HIGH = "#D9534F"       # Coral / Soft Red      -> High Overstock
COLOR_MODERATE = "#1ABC9C"   # Teal / Emerald         -> Moderate/Low Overstock
COLOR_ACCENT = "#1E3A8A"     # Navy                   -> aksen UI utama
COLOR_ACCENT_2 = "#34495E"   # Slate Blue             -> aksen UI sekunder
COLOR_MUTED = "#95A5A6"      # Muted Gray             -> elemen sekunder

CLUSTER_COLOR_MAP = {
    "High Overstock": COLOR_HIGH,
    "Moderate/Low Overstock": COLOR_MODERATE,
}

FEATURE_COLS = [
    "Category_enc", "Region_enc", "Inventory Level", "Units Sold",
    "Units Ordered", "Demand Forecast", "Price", "Discount",
    "Weather Condition_enc", "Holiday/Promotion_enc", "Competitor Pricing",
    "Seasonality_enc", "Year", "Month", "Day",
]

CATEGORICAL_COLS = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Holiday/Promotion", "Seasonality",
]

NUMERIC_OUTLIER_COLS = [
    "Inventory Level", "Units Sold", "Units Ordered",
    "Demand Forecast", "Price", "Discount", "Competitor Pricing",
]


# ==========================================================
# 2. TEMA VISUAL — MENYESUAIKAN LIGHT MODE / DARK MODE
# ==========================================================

def get_theme_settings() -> dict:

    base = st.get_option("theme.base")

    if base == "dark":
        return {
            "template": "plotly_dark",

            "font_color": "#ECF0F1",

            # khusus Plotly
            "plotly_grid": "rgba(255,255,255,0.12)",

            # khusus Matplotlib
            "mpl_grid": (1, 1, 1, 0.12),

            "mpl_face": "none",
        }

    return {
        "template": "plotly_white",

        "font_color": COLOR_ACCENT_2,

        # khusus Plotly
        "plotly_grid": "rgba(52,73,94,0.10)",

        # khusus Matplotlib
        "mpl_grid": (52/255, 73/255, 94/255, 0.10),

        "mpl_face": "none",
    }
    
    base = st.get_option("theme.base")
    if base == "dark":
        return {
            "template": "plotly_dark",
            "font_color": "#ECF0F1",
            "grid_color": "rgba(255, 255, 255, 0.12)",
            "mpl_face": "none",
        }
    return {
        "template": "plotly_white",
        "font_color": COLOR_ACCENT_2,
        "grid_color": "rgba(52, 73, 94, 0.10)",
        "mpl_face": "none",
    }


def apply_chart_theme(fig, theme):

    fig.update_layout(
        template=theme["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["font_color"]),
    )

    fig.update_xaxes(
        gridcolor=theme["plotly_grid"],
        zerolinecolor=theme["plotly_grid"]
    )

    fig.update_yaxes(
        gridcolor=theme["plotly_grid"],
        zerolinecolor=theme["plotly_grid"]
    )

    return fig


def apply_matplotlib_theme(fig, ax, theme):

    fig.patch.set_alpha(0)

    ax.set_facecolor(theme["mpl_face"])

    ax.tick_params(colors=theme["font_color"])

    ax.xaxis.label.set_color(theme["font_color"])
    ax.yaxis.label.set_color(theme["font_color"])

    ax.title.set_color(theme["font_color"])

    for spine in ax.spines.values():
        spine.set_color(theme["mpl_grid"])

    ax.grid(
        True,
        color=theme["mpl_grid"],
        linewidth=0.8,
        alpha=0.6,
    )

    return fig, ax


THEME = get_theme_settings()

# --- Injeksi CSS ringan agar aksen warna filter/sidebar/tombol selaras ---
st.markdown(
    f"""
    <style>
    [data-baseweb="tag"] {{
        background-color: {COLOR_ACCENT} !important;
        color: white !important;
    }}
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {{
        border-color: {COLOR_ACCENT} !important;
        box-shadow: 0 0 0 1px {COLOR_ACCENT} !important;
    }}
    button[kind="primary"], .stButton>button {{
        background-color: {COLOR_ACCENT};
        color: white;
        border: none;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_MODERATE};
        color: white;
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2 {{
        color: {COLOR_ACCENT};
    }}
    [data-testid="stMetricValue"] {{
        color: {COLOR_ACCENT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 3. LOAD DATA & PIPELINE CLUSTERING (BACKGROUND, TIDAK TAMPAK KE USER)
# ==========================================================

@st.cache_resource(show_spinner="Memuat dan memproses data...")
def load_and_cluster_data(path: str):
    """
    1. Membaca dataset mentah lokal.
    2. Membersihkan missing value & duplikat.
    3. Melakukan encoding kategorikal (khusus untuk kalkulasi model).
    4. Menangani outlier numerik (IQR clipping).
    5. Mengagregasi data pada level Store ID + Product ID (unit analisis produk).
    6. Menstandardisasi fitur numerik.
    7. Menjalankan AgglomerativeClustering (n_clusters=2, linkage='ward').
    8. Menentukan cluster mana yang merupakan "High Overstock" secara otomatis
       berdasarkan rasio Inventory Level terhadap Units Sold (bukan hardcode index).
    9. Menggabungkan label Cluster & Segment kembali ke data transaksi asli
       agar tetap bisa difilter per baris (Region, Category, Date).
    10. Mengembalikan seluruh objek pendukung (scaler, encoder, centroid, data
        level transaksi & produk) agar bisa dipakai ulang oleh Simulasi &
        Analisis Klaster.
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

    # --- Penanganan outlier (IQR clipping) pada fitur numerik utama ---
    # Dilakukan di sini (level transaksi, sebelum agregasi) sesuai notebook
    # riset "revisi_agglomerative", lalu dipakai bersama untuk dashboard,
    # analisis klaster, dan pipeline model.
    for col in NUMERIC_OUTLIER_COLS:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].clip(lower, upper)

    # --- Pecah tanggal untuk kebutuhan fitur numerik model ---
    df_encoded = df_clean.copy()
    df_encoded["Year"] = df_encoded["Date"].dt.year
    df_encoded["Month"] = df_encoded["Date"].dt.month
    df_encoded["Day"] = df_encoded["Date"].dt.day

    # --- Label Encoding untuk fitur kategorikal (khusus perhitungan model) ---
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col + "_enc"] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    # --- Agregasi ke level Store ID + Product ID (unit analisis produk) ---
    product_df = (
        df_encoded
        .groupby(["Store ID", "Product ID"])
        .agg({
            "Category": "first",
            "Region": "first",
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

    X = product_df[FEATURE_COLS]

    # --- Standardisasi fitur ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    # --- Model Agglomerative Clustering (dijalankan otomatis di latar belakang) ---
    model = AgglomerativeClustering(n_clusters=2, linkage="ward")
    cluster_labels = model.fit_predict(X_scaled)
    product_df["Cluster"] = cluster_labels

    # --- Tentukan cluster mana = "High Overstock" secara otomatis (bukan hardcode) ---
    ratio_per_cluster = (
        product_df.assign(
            overstock_ratio=product_df["Inventory Level"] / (product_df["Units Sold"] + 1)
        )
        .groupby("Cluster")["overstock_ratio"]
        .mean()
    )
    high_cluster_label = ratio_per_cluster.idxmax()
    cluster_name_map = {
        high_cluster_label: "High Overstock",
        1 - high_cluster_label: "Moderate/Low Overstock",
    }
    product_df["Segment"] = product_df["Cluster"].map(cluster_name_map)

    # --- Hitung centroid tiap cluster di ruang fitur yang sudah discale ---
    centroids = {
        label: X_scaled[cluster_labels == label].mean(axis=0)
        for label in np.unique(cluster_labels)
    }

    # --- Gabungkan label Cluster & Segment kembali ke data transaksi asli ---
    df_final = df_clean.merge(
        product_df[["Store ID", "Product ID", "Cluster", "Segment"]],
        on=["Store ID", "Product ID"],
        how="left",
    )

    support_objects = {
        "scaler": scaler,
        "encoders": encoders,
        "centroids": centroids,
        "cluster_name_map": cluster_name_map,
        "product_df": product_df,
        "X_scaled_df": X_scaled_df,
        "df_clean": df_clean,
    }

    return df_final, support_objects


try:
    df, support = load_and_cluster_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"❌ File `{DATA_PATH}` tidak ditemukan. "
        f"Pastikan file dataset berada di folder yang sama dengan `app.py`."
    )
    st.stop()


# ==========================================================
# 4. FUNGSI SIMULASI — MEMETAKAN DATA BARU KE CLUSTER TERDEKAT
# ==========================================================

def predict_cluster_for_new_sample(input_values: dict, support: dict, df_source: pd.DataFrame):
    """
    Memetakan satu baris data simulasi ke segmen cluster terdekat dengan
    memperbaiki efek bias skala akibat perbedaan data agregasi (SUM) vs
    data input tunggal, menggunakan Jarak Euclidean terhadap centroid
    hasil Agglomerative Clustering.
    """
    encoders = support["encoders"]
    scaler = support["scaler"]
    centroids = support["centroids"]
    cluster_name_map = support["cluster_name_map"]

    avg_transaksi_per_produk = df_source.groupby(["Store ID", "Product ID"]).size().mean()
    if pd.isna(avg_transaksi_per_produk) or avg_transaksi_per_produk == 0:
        avg_transaksi_per_produk = 1.0

    scaled_units_sold = input_values["Units Sold"] * avg_transaksi_per_produk
    scaled_units_ordered = input_values["Units Ordered"] * avg_transaksi_per_produk

    outlier_bounds = {
        "Inventory Level": (df_source["Inventory Level"].quantile(0.25), df_source["Inventory Level"].quantile(0.75)),
        "Price": (df_source["Price"].quantile(0.25), df_source["Price"].quantile(0.75)),
        "Discount": (df_source["Discount"].quantile(0.25), df_source["Discount"].quantile(0.75)),
        "Competitor Pricing": (df_source["Competitor Pricing"].quantile(0.25), df_source["Competitor Pricing"].quantile(0.75)),
        "Demand Forecast": (df_source["Demand Forecast"].quantile(0.25), df_source["Demand Forecast"].quantile(0.75)),
    }

    def clip_value(val, col_name):
        q1, q3 = outlier_bounds[col_name]
        iqr = q3 - q1
        return np.clip(val, q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    row = {
        "Category_enc": encoders["Category"].transform([input_values["Category"]])[0],
        "Region_enc": encoders["Region"].transform([input_values["Region"]])[0],
        "Inventory Level": clip_value(input_values["Inventory Level"], "Inventory Level"),
        "Units Sold": scaled_units_sold,
        "Units Ordered": scaled_units_ordered,
        "Demand Forecast": clip_value(input_values["Demand Forecast"], "Demand Forecast"),
        "Price": clip_value(input_values["Price"], "Price"),
        "Discount": clip_value(input_values["Discount"], "Discount"),
        "Weather Condition_enc": encoders["Weather Condition"].transform([input_values["Weather Condition"]])[0],
        "Holiday/Promotion_enc": input_values["Holiday/Promotion"],
        "Competitor Pricing": clip_value(input_values["Competitor Pricing"], "Competitor Pricing"),
        "Seasonality_enc": encoders["Seasonality"].transform([input_values["Seasonality"]])[0],
        "Year": input_values["Year"],
        "Month": input_values["Month"],
        "Day": input_values["Day"],
    }

    X_new = pd.DataFrame([row])[FEATURE_COLS]
    X_new_scaled = scaler.transform(X_new)[0]

    distances = {
        label: float(np.linalg.norm(X_new_scaled - centroid))
        for label, centroid in centroids.items()
    }
    nearest_cluster = min(distances, key=distances.get)
    segment = cluster_name_map[nearest_cluster]

    return segment, distances


# ==========================================================
# 5. FUNGSI TERCACHE UNTUK HALAMAN ANALISIS KLASTER
#    (dendrogram & metrik evaluasi K — cukup dihitung sekali)
# ==========================================================

@st.cache_data(show_spinner="Menyusun struktur dendrogram...")
def compute_linkage_matrix(X_scaled_array: np.ndarray, max_samples: int = 200):
    """
    Menghitung linkage matrix (Ward) untuk dendrogram. Jika jumlah data
    produk cukup besar, diambil sampel acak (random_state tetap) demi
    efisiensi memori & kejelasan visual, sesuai catatan pada notebook riset.
    """
    if len(X_scaled_array) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_scaled_array), max_samples, replace=False)
        X_sample = X_scaled_array[idx]
    else:
        X_sample = X_scaled_array
    Z = linkage(X_sample, method="ward")
    return Z, len(X_sample)


@st.cache_data(show_spinner="Menghitung metrik evaluasi klaster (K, Silhouette, DBI, CHI)...")
def compute_cluster_metrics(X_scaled_array: np.ndarray, k_min: int = 2, k_max: int = 10):
    """Sweep jumlah klaster K dan hitung metrik evaluasi, meniru notebook riset."""
    rows = []
    for k in range(k_min, k_max + 1):
        model = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = model.fit_predict(X_scaled_array)
        sil = silhouette_score(X_scaled_array, labels)
        dbi = davies_bouldin_score(X_scaled_array, labels)
        chi = calinski_harabasz_score(X_scaled_array, labels)
        rows.append({
            "K": k,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin Index": round(dbi, 4),
            "Calinski-Harabasz Index": round(chi, 2),
        })
    return pd.DataFrame(rows)


# ==========================================================
# 6. HALAMAN 1 — DASHBOARD RINGKASAN (EXECUTIVE DASHBOARD)
# ==========================================================

def page_dashboard():
    st.sidebar.title("🔎 Filter Dashboard")
    st.sidebar.markdown("Gunakan filter berikut untuk menyesuaikan tampilan data.")

    region_options = sorted(df["Region"].dropna().unique().tolist())
    selected_regions = st.sidebar.multiselect(
        "Pilih Region", options=region_options, default=region_options
    )

    category_options = sorted(df["Category"].dropna().unique().tolist())
    selected_categories = st.sidebar.multiselect(
        "Pilih Kategori Produk", options=category_options, default=category_options
    )

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    selected_date_range = st.sidebar.date_input(
        "Pilih Rentang Tanggal", value=(min_date, max_date),
        min_value=min_date, max_value=max_date,
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

    mask = (
        df["Region"].isin(selected_regions)
        & df["Category"].isin(selected_categories)
        & (df["Date"].dt.date >= start_date)
        & (df["Date"].dt.date <= end_date)
    )
    df_filtered = df.loc[mask].copy()

    st.title("📊 Dashboard Ringkasan — Segmentasi Overstock Retail")
    st.markdown(
        "Dashboard ini membantu **Manajer Retail** memantau produk yang berpotensi mengalami "
        "*overstock* berdasarkan hasil segmentasi model **Agglomerative Clustering**, "
        "sehingga keputusan operasional (diskon, penghentian order, distribusi ulang stok) "
        "dapat diambil lebih cepat dan berbasis data."
    )
    st.markdown("---")

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data yang sesuai dengan filter yang dipilih. Silakan ubah filter Anda.")
        st.stop()

    # --- KPI Cards ---
    total_high = int((df_filtered["Segment"] == "High Overstock").sum())
    total_moderate = int((df_filtered["Segment"] == "Moderate/Low Overstock").sum())
    avg_inventory = df_filtered["Inventory Level"].mean()
    avg_units_sold = df_filtered["Units Sold"].mean()
    avg_price = df_filtered["Price"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("🔴 High Overstock", f"{total_high:,} baris",
                   help="Jumlah baris data pada segmen High Overstock")
    with col2:
        st.metric("🟢 Moderate/Low Overstock", f"{total_moderate:,} baris",
                   help="Jumlah baris data pada segmen Moderate/Low Overstock")
    with col3:
        st.metric("📊 Rata-rata Inventory", f"{avg_inventory:,.1f} Pcs")
    with col4:
        st.metric("🛒 Rata-rata Units Sold", f"{avg_units_sold:,.1f} Pcs")
    with col5:
        st.metric("💲 Rata-rata Price", f"${avg_price:,.2f}")

    st.markdown("---")

    row1_col1, row1_col2 = st.columns([1, 1.4])

    with row1_col1:
        st.subheader("Proporsi Segmentasi Overstock")
        segment_counts = df_filtered["Segment"].value_counts().reset_index()
        segment_counts.columns = ["Segment", "Jumlah Baris"]
        fig_pie = px.pie(
            segment_counts, names="Segment", values="Jumlah Baris",
            color="Segment", color_discrete_map=CLUSTER_COLOR_MAP, hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        fig_pie = apply_chart_theme(fig_pie, THEME)
        st.plotly_chart(fig_pie, width="stretch")

    with row1_col2:
        st.subheader("Kategori Produk dengan High Overstock Tertinggi")
        high_by_cat = (
            df_filtered[df_filtered["Segment"] == "High Overstock"]
            .groupby("Category").size()
            .reset_index(name="Jumlah Produk (Pcs/Unit)")
            .sort_values("Jumlah Produk (Pcs/Unit)", ascending=False)
        )
        fig_bar_cat = px.bar(
            high_by_cat, x="Category", y="Jumlah Produk (Pcs/Unit)",
            color="Jumlah Produk (Pcs/Unit)",
            color_continuous_scale=[COLOR_MUTED, COLOR_HIGH],
            text="Jumlah Produk (Pcs/Unit)",
        )
        fig_bar_cat.update_layout(
            xaxis_title="Kategori Produk", yaxis_title="Jumlah Produk (Pcs/Unit)",
            margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False,
        )
        fig_bar_cat = apply_chart_theme(fig_bar_cat, THEME)
        st.plotly_chart(fig_bar_cat, width="stretch")

    st.markdown("---")

    # --- Tabel Detail & Rekomendasi Aksi ---
    st.subheader("📋 Detail Produk High Overstock & Rekomendasi Aksi")
    high_overstock_df = df_filtered[df_filtered["Segment"] == "High Overstock"].copy()

    def generate_recommendation(row) -> str:
        if row["Units Sold"] < row["Units Ordered"] * 0.5:
            return "Hentikan sementara Units Ordered & berikan diskon tambahan"
        elif row["Units Ordered"] > 0 and row["Units Sold"] / row["Units Ordered"] < 0.8:
            return "Pertimbangkan diskon tambahan untuk mempercepat penjualan"
        else:
            return "Pindahkan sebagian stok ke Region dengan Units Sold lebih tinggi"

    if not high_overstock_df.empty:
        high_overstock_df["Rekomendasi Aksi Manajer"] = high_overstock_df.apply(generate_recommendation, axis=1)

        display_cols = [
            "Date", "Store ID", "Product ID", "Category", "Region",
            "Inventory Level", "Units Sold", "Units Ordered",
            "Price", "Discount", "Segment", "Rekomendasi Aksi Manajer",
        ]
        display_df = high_overstock_df[display_cols].rename(columns={
            "Inventory Level": UNIT_LABELS["Inventory Level"],
            "Units Sold": UNIT_LABELS["Units Sold"],
            "Units Ordered": UNIT_LABELS["Units Ordered"],
            "Price": UNIT_LABELS["Price"],
            "Discount": UNIT_LABELS["Discount"],
        })
        st.dataframe(
            display_df.sort_values(UNIT_LABELS["Inventory Level"], ascending=False),
           width="stretch", height=420,
        )

        csv_download = high_overstock_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Unduh Data High Overstock (CSV)", data=csv_download,
            file_name="high_overstock_products.csv", mime="text/csv",
        )
    else:
        st.info("Tidak ada produk pada segmen High Overstock untuk kombinasi filter saat ini.")


# ==========================================================
# 7. HALAMAN 2 — SIMULASI & PREDIKSI MODEL (PREDICTIVE SIMULATOR)
# ==========================================================

def page_simulation():
    st.title("🧪 Simulasi & Prediksi Model")
    st.caption(
        "Masukkan data produk hipotetis (misalnya rencana pemesanan stok baru) untuk melihat "
        "ke segmen mana produk tersebut kemungkinan besar akan masuk. Karena Agglomerative "
        "Clustering tidak memiliki fungsi prediksi bawaan, sistem memetakan data baru ke "
        "**cluster dengan centroid (titik pusat) terdekat** menggunakan Jarak Euclidean "
        "terhadap centroid model."
    )
    st.markdown("---")

    category_options = sorted(df["Category"].dropna().unique().tolist())
    region_options = sorted(df["Region"].dropna().unique().tolist())
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    with st.form("simulasi_form"):
        st.markdown("**Informasi Produk**")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            sim_category = st.selectbox("Category", options=category_options)
            sim_region = st.selectbox("Region", options=region_options)
            sim_weather = st.selectbox(
                "Weather Condition",
                options=sorted(df["Weather Condition"].dropna().unique().tolist()),
            )

        with f_col2:
            sim_seasonality = st.selectbox(
                "Seasonality",
                options=sorted(df["Seasonality"].dropna().unique().tolist()),
            )
            sim_holiday = st.selectbox(
                "Holiday/Promotion", options=[0, 1],
                format_func=lambda x: "Ya (Promo/Libur)" if x == 1 else "Tidak",
            )
            sim_date = st.date_input(
                "Tanggal Simulasi", value=max_date, min_value=min_date, max_value=max_date,
            )

        with f_col3:
            st.write("")

        st.markdown("---")
        st.markdown("**Data Kuantitas & Harga**")
        n_col1, n_col2, n_col3, n_col4 = st.columns(4)

        with n_col1:
            sim_inventory = st.number_input(UNIT_LABELS["Inventory Level"], min_value=0.0, value=200.0, step=1.0)
            sim_units_sold = st.number_input(UNIT_LABELS["Units Sold"], min_value=0.0, value=100.0, step=1.0)

        with n_col2:
            sim_units_ordered = st.number_input(UNIT_LABELS["Units Ordered"], min_value=0.0, value=60.0, step=1.0)
            sim_demand_forecast = st.number_input(UNIT_LABELS["Demand Forecast"], min_value=0.0, value=110.0, step=1.0)

        with n_col3:
            sim_price = st.number_input(UNIT_LABELS["Price"], min_value=0.0, value=50.0, step=0.5, format="%.2f")
            sim_competitor_price = st.number_input(UNIT_LABELS["Competitor Pricing"], min_value=0.0, value=48.0, step=0.5, format="%.2f")

        with n_col4:
            sim_discount = st.number_input(UNIT_LABELS["Discount"], min_value=0.0, max_value=100.0, value=10.0, step=1.0)

        submitted = st.form_submit_button("▶️ Jalankan Simulasi")

    if submitted:
        sim_input = {
            "Category": sim_category, "Region": sim_region, "Weather Condition": sim_weather,
            "Seasonality": sim_seasonality, "Holiday/Promotion": sim_holiday,
            "Inventory Level": sim_inventory, "Units Sold": sim_units_sold,
            "Units Ordered": sim_units_ordered, "Demand Forecast": sim_demand_forecast,
            "Price": sim_price, "Competitor Pricing": sim_competitor_price,
            "Discount": sim_discount, "Year": sim_date.year, "Month": sim_date.month, "Day": sim_date.day,
        }

        segment_result, distances = predict_cluster_for_new_sample(sim_input, support, df)
        result_color = CLUSTER_COLOR_MAP.get(segment_result, COLOR_ACCENT)

        st.markdown(
            f"""
            <div style="padding: 18px 22px; border-radius: 10px;
                        background-color: {result_color}22; border-left: 6px solid {result_color};">
                <span style="font-size: 0.95rem; color: {THEME['font_color']};">Hasil Simulasi:</span><br>
                <span style="font-size: 1.4rem; font-weight: 700; color: {result_color};">
                    {segment_result}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Lihat detail jarak ke tiap centroid cluster (semakin kecil = semakin dekat)"):
            dist_df = pd.DataFrame({
                "Segment": [support["cluster_name_map"][c] for c in distances.keys()],
                "Jarak Euclidean": list(distances.values()),
            }).sort_values("Jarak Euclidean")
            st.dataframe(dist_df, width="stretch", hide_index=True)

        if segment_result == "High Overstock":
            st.warning(
                "⚠️ Produk simulasi ini berpotensi **overstock tinggi**. Pertimbangkan menahan "
                "sebagian Units Ordered atau menyiapkan strategi diskon."
            )
        else:
            st.success("✅ Produk simulasi ini berada pada tingkat stok yang **wajar/rendah**.")


# ==========================================================
# 8. HALAMAN 3 — ANALISIS KLASTER (CLUSTER ANALYSIS & DENDROGRAM)
# ==========================================================

def page_cluster_analysis():
    st.title("📌 Analisis Klaster (Cluster Analysis & Dendrogram)")
    st.caption(
        "Halaman ini menjelaskan hasil analisis **Agglomerative Clustering** secara "
        "teknis & akademis, mengikuti alur eksplorasi data pada notebook riset — mulai "
        "dari struktur hierarki klaster, metrik evaluasi model, distribusi data, hingga "
        "korelasi antar fitur numerik."
    )
    st.markdown("---")

    df_clean_raw = support["df_clean"]
    X_scaled_df = support["X_scaled_df"]

    # ------------------------------------------------------
    # 8.1 Dendrogram
    # ------------------------------------------------------
    st.subheader("🌳 Dendrogram Hierarki Klaster (Ward Linkage)")
    Z, n_sample = compute_linkage_matrix(X_scaled_df.values)
    st.caption(
        f"Dendrogram dibentuk menggunakan *Ward linkage* pada {n_sample} sampel data "
        "produk (level Store ID + Product ID) yang telah distandardisasi."
    )
    fig_dendro, ax_dendro = plt.subplots(figsize=(13, 5))
    dendrogram(Z, ax=ax_dendro, color_threshold=None, no_labels=True)
    ax_dendro.set_title("Dendrogram Hierarki Klaster (Ward Linkage)")
    ax_dendro.set_xlabel("Sampel Produk")
    ax_dendro.set_ylabel("Jarak (Ward)")
    apply_matplotlib_theme(fig_dendro, ax_dendro, THEME)
    st.pyplot(fig_dendro, width="stretch")
    plt.close(fig_dendro)

    st.markdown("---")

    # ------------------------------------------------------
    # 8.2 Metrik Evaluasi Model
    # ------------------------------------------------------
    st.subheader("📈 Metrik Evaluasi Model (K, Silhouette, DBI, CHI)")
    st.caption(
        "Perbandingan metrik evaluasi untuk beberapa kandidat jumlah klaster (K), "
        "sebagai dasar pemilihan K=2 pada model final."
    )
    metrics_df = compute_cluster_metrics(X_scaled_df.values, k_min=2, k_max=10)
    st.dataframe(
        metrics_df.style.apply(
            lambda r: ["background-color: rgba(30,58,138,0.15)" if r["K"] == 2 else "" for _ in r],
            axis=1,
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        "Baris **K = 2** ditandai karena merupakan konfigurasi yang digunakan pada model "
        "final (High Overstock vs Moderate/Low Overstock)."
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    row_k2 = metrics_df[metrics_df["K"] == 2].iloc[0]
    with m_col1:
        st.metric("Silhouette Score (K=2)", f"{row_k2['Silhouette Score']:.4f}")
    with m_col2:
        st.metric("Davies-Bouldin Index (K=2)", f"{row_k2['Davies-Bouldin Index']:.4f}")
    with m_col3:
        st.metric("Calinski-Harabasz Index (K=2)", f"{row_k2['Calinski-Harabasz Index']:.2f}")

    st.markdown("---")

    # ------------------------------------------------------
    # 8.3 Distribusi Kategorikal (Countplots)
    # ------------------------------------------------------
    st.subheader("📊 Distribusi Data Kategorikal")

    def make_countplot(col: str, title: str, rotate: bool = False):
        counts = df_clean_raw[col].astype(str).value_counts().reset_index()
        counts.columns = [col, "Jumlah"]
        fig = px.bar(
            counts, x=col, y="Jumlah", color_discrete_sequence=[COLOR_ACCENT], title=title,
        )
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        if rotate:
            fig.update_xaxes(tickangle=-45)
        return apply_chart_theme(fig, THEME)

    cnt_col1, cnt_col2 = st.columns(2)
    with cnt_col1:
        st.plotly_chart(make_countplot("Store ID", "Distribusi Store ID"), width="stretch")
        st.plotly_chart(make_countplot("Region", "Distribusi Region"), width="stretch")
        st.plotly_chart(make_countplot("Seasonality", "Distribusi Seasonality"), width="stretch")
    with cnt_col2:
        st.plotly_chart(make_countplot("Category", "Distribusi Category", rotate=True), width="stretch")
        st.plotly_chart(make_countplot("Weather Condition", "Distribusi Weather Condition"), width="stretch")

    st.markdown("---")

    # ------------------------------------------------------
    # 8.4 Boxplot Karakteristik Klaster
    # ------------------------------------------------------
    st.subheader("📦 Analisis Boxplot Karakteristik Data")

    def make_boxplot(x_col: str, y_col: str, title: str):
        fig = px.box(
            df_clean_raw, x=x_col, y=y_col, color_discrete_sequence=[COLOR_ACCENT],
            title=title, labels={y_col: UNIT_LABELS.get(y_col, y_col)},
        )
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        return apply_chart_theme(fig, THEME)

    box_col1, box_col2 = st.columns(2)
    with box_col1:
        st.plotly_chart(
            make_boxplot("Category", "Inventory Level", "Inventory Level (Pcs) vs Category"),
            width="stretch",
        )
        st.plotly_chart(
            make_boxplot("Store ID", "Inventory Level", "Inventory Level (Pcs) vs Store ID"),
            width="stretch",
        )
    with box_col2:
        st.plotly_chart(
            make_boxplot("Category", "Units Sold", "Units Sold (Pcs) vs Category"),
            width="stretch",
        )
        st.plotly_chart(
            make_boxplot("Store ID", "Units Sold", "Units Sold (Pcs) vs Store ID"),
            width="stretch",
        )

    st.markdown("---")

    # ------------------------------------------------------
    # 8.5 Korelasi Fitur Numerik
    # ------------------------------------------------------
    st.subheader("🔗 Korelasi Antar Fitur Numerik")
    numeric_cols = df_clean_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    corr = df_clean_raw[numeric_cols].corr()

    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig_corr.update_layout(margin=dict(t=10, b=10, l=10, r=10))
    fig_corr = apply_chart_theme(fig_corr, THEME)
    st.plotly_chart(fig_corr, width="stretch")


# ==========================================================
# 9. NAVIGASI MULTI-HALAMAN
# ==========================================================

pg = st.navigation([
    st.Page(page_dashboard, title="Dashboard Ringkasan", icon="📊", default=True),
    st.Page(page_simulation, title="Simulasi & Prediksi Model", icon="🧪"),
    st.Page(page_cluster_analysis, title="Analisis Klaster", icon="📌"),
])
pg.run()