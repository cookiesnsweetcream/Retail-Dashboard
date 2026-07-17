# ==========================================================
# SISTEM INFORMASI ANALITIK RETAIL — SEGMENTASI OVERSTOCK
# Model      : Agglomerative Clustering (Unsupervised Learning)
# Target     : Manajer Retail (Non-Teknis)
# Versi      : 4.0 — Multi-Page App (Dashboard, Simulasi Risiko Overstock,
#              Profil Segmen), Tema Modern Navy/Coral/Teal, Satuan Eksplisit.
#              Seluruh konten teknis/akademis (dendrogram, metrik evaluasi,
#              countplot teknis, heatmap korelasi) telah dihilangkan dan
#              diganti dengan ringkasan bisnis yang intuitif.
# Catatan    : Membutuhkan streamlit >= 1.33 (st.navigation & st.Page)
#              Pipeline Agglomerative Clustering tetap berjalan penuh di
#              latar belakang (background) — tidak ada fungsionalitas model
#              yang dihilangkan, hanya tampilan yang disederhanakan.
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
    """
    Mendeteksi mode tema aktif Streamlit (light/dark) dan mengembalikan
    pengaturan visual (template Plotly, warna font, warna grid) yang
    kontras & konsisten di kedua mode.
    """
    base = st.get_option("theme.base")

    if base == "dark":
        return {
            "template": "plotly_dark",
            "font_color": "#ECF0F1",
            "plotly_grid": "rgba(255,255,255,0.12)",
        }

    return {
        "template": "plotly_white",
        "font_color": COLOR_ACCENT_2,
        "plotly_grid": "rgba(52,73,94,0.10)",
    }


def apply_chart_theme(fig, theme: dict):
    """Menerapkan tema warna & background transparan ke sebuah figur Plotly."""
    fig.update_layout(
        template=theme["template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["font_color"]),
    )
    fig.update_xaxes(gridcolor=theme["plotly_grid"], zerolinecolor=theme["plotly_grid"])
    fig.update_yaxes(gridcolor=theme["plotly_grid"], zerolinecolor=theme["plotly_grid"])
    return fig


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
        halaman Profil Segmen.
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
# 5. FUNGSI TERCACHE — RINGKASAN KARAKTERISTIK SEGMEN
#    (dipakai di Dashboard & halaman Profil Segmen, bukan detail teknis)
# ==========================================================

@st.cache_data(show_spinner="Menyusun profil karakteristik segmen...")
def compute_segment_profile(product_df: pd.DataFrame):
    """
    Menghitung rata-rata metrik bisnis utama per segmen (level produk),
    dipakai untuk narasi profil segmen yang intuitif bagi manajer retail.
    """
    profile = (
        product_df.groupby("Segment")[
            ["Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount"]
        ]
        .mean()
        .round(1)
    )
    return profile


@st.cache_data(show_spinner="Menyusun ringkasan kategori & produk dominan...")
def compute_top_category_and_products(product_df: pd.DataFrame, segment_name: str, top_n: int = 10):
    """
    Menghitung kategori produk & Product ID yang paling mendominasi
    sebuah segmen tertentu (default: High Overstock), berdasarkan jumlah
    produk pada segmen tersebut.
    """
    seg_df = product_df[product_df["Segment"] == segment_name]

    by_category = (
        seg_df.groupby("Category").size()
        .reset_index(name="Jumlah Produk")
        .sort_values("Jumlah Produk", ascending=False)
    )

    by_product = (
        seg_df.groupby(["Product ID", "Category"])
        .agg(
            **{
                "Rata-rata Inventory (Pcs)": ("Inventory Level", "mean"),
                "Total Units Sold (Pcs)": ("Units Sold", "sum"),
            }
        )
        .reset_index()
        .sort_values("Rata-rata Inventory (Pcs)", ascending=False)
        .head(top_n)
    )

    return by_category, by_product


@st.cache_data(show_spinner="Menghitung proporsi segmen per Store ID...")
def compute_segment_proportion_by_store(product_df: pd.DataFrame):
    """
    Menghitung proporsi (%) jumlah produk pada masing-masing segmen
    ('High Overstock' vs 'Moderate/Low Overstock') dikelompokkan
    berdasarkan Store ID. Digunakan untuk visualisasi stacked bar chart
    pada halaman Profil Segmen agar Manajer Retail dapat membandingkan
    komposisi risiko overstock antar toko.
    """
    store_segment_counts = (
        product_df.groupby(["Store ID", "Segment"]).size()
        .reset_index(name="Jumlah Produk")
    )

    store_totals = (
        store_segment_counts.groupby("Store ID")["Jumlah Produk"]
        .transform("sum")
    )

    store_segment_counts["Proporsi (%)"] = (
        store_segment_counts["Jumlah Produk"] / store_totals * 100
    ).round(1)

    # Urutkan Store ID secara alami agar tampilan chart lebih rapi
    store_segment_counts = store_segment_counts.sort_values("Store ID")

    return store_segment_counts


# ==========================================================
# 6. HALAMAN 1 — DASHBOARD (EXECUTIVE DASHBOARD)
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

    st.title("📊 Dashboard — Segmentasi Overstock Retail")
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

    # ------------------------------------------------------
    # Profil Segmen & Karakteristik Kluster (bisnis, bukan teknis)
    # ------------------------------------------------------
    st.subheader("🧭 Profil Segmen")
    st.caption(
        "Ringkasan perilaku rata-rata tiap segmen."
    )

    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.error(
            "**🔴 High Overstock**\n\n"
            "Ditandai dengan tingkat **inventory tinggi**, namun **units sold sangat rendah**. "
            "Pola ini sering terjadi pada produk dengan **harga mahal** atau **diskon rendah**, "
            "sehingga stok menumpuk dan modal tertahan lebih lama."
        )
    with profile_col2:
        st.success(
            "**🟢 Moderate/Low Overstock**\n\n"
            "Perputaran stok tergolong **sehat** — jumlah **units sold sebanding** dengan "
            "**units ordered**, sehingga risiko penumpukan stok relatif rendah."
        )

    st.markdown("")
    profile_table = compute_segment_profile(support["product_df"])
    profile_display = profile_table.rename(columns={
        "Inventory Level": UNIT_LABELS["Inventory Level"],
        "Units Sold": UNIT_LABELS["Units Sold"],
        "Units Ordered": UNIT_LABELS["Units Ordered"],
        "Price": UNIT_LABELS["Price"],
        "Discount": UNIT_LABELS["Discount"],
    })
    st.dataframe(profile_display, use_container_width=True)

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
        st.plotly_chart(fig_pie, use_container_width=True)

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
        st.plotly_chart(fig_bar_cat, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # Produk & Kategori yang mendominasi segmen High Overstock
    # ------------------------------------------------------
    st.subheader("🏷️ Produk & Kategori yang Mendominasi Segmen High Overstock")
    # st.caption(
    #     "Gunakan informasi ini untuk memprioritaskan kategori dan produk mana yang "
    #     "perlu ditinjau ulang strategi pemesanan atau diskonnya."
    # )

    top_cat_df, top_prod_df = compute_top_category_and_products(
        support["product_df"], "High Overstock", top_n=10
    )

    dom_col1, dom_col2 = st.columns([1, 1.4])
    with dom_col1:
        fig_dom_cat = px.bar(
            top_cat_df, x="Category", y="Jumlah Produk",
            color_discrete_sequence=[COLOR_HIGH],
            title="Kategori Produk Paling Dominan (High Overstock)",
        )
        fig_dom_cat.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        fig_dom_cat = apply_chart_theme(fig_dom_cat, THEME)
        st.plotly_chart(fig_dom_cat, use_container_width=True)

    with dom_col2:
        st.markdown("**Top 10 Produk dengan Inventory Rata-rata Tertinggi**")
        top_prod_display = top_prod_df.rename(columns={"Product ID": "ID Produk", "Category": "Kategori"})
        st.dataframe(top_prod_display, use_container_width=True, hide_index=True)

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
            use_container_width=True, height=420,
        )

        csv_download = high_overstock_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Unduh Data High Overstock (CSV)", data=csv_download,
            file_name="high_overstock_products.csv", mime="text/csv",
        )
    else:
        st.info("Tidak ada produk pada segmen High Overstock untuk kombinasi filter saat ini.")


# ==========================================================
# 7. HALAMAN 2 — SIMULASI RISIKO OVERSTOCK (PREDICTIVE SIMULATOR)
# ==========================================================

def page_simulation():
    st.title("🧪 Simulasi Risiko Overstock")
    st.caption(
        "Masukkan data produk hipotetis (misalnya rencana pemesanan stok baru) untuk melihat "
        "ke segmen risiko mana produk tersebut kemungkinan besar akan masuk, berdasarkan pola "
        "historis yang telah dipelajari model."
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

        st.markdown("")
        if segment_result == "High Overstock":
            st.warning(
                "⚠️ Produk simulasi ini berpotensi **overstock tinggi**. Pertimbangkan menahan "
                "sebagian Units Ordered atau menyiapkan strategi diskon."
            )
        else:
            st.success("✅ Produk simulasi ini berada pada tingkat stok yang **wajar/rendah**.")

        with st.expander("Lihat tingkat keyakinan hasil simulasi (semakin dekat ke satu segmen, semakin yakin)"):
            dist_df = pd.DataFrame({
                "Segment": [support["cluster_name_map"][c] for c in distances.keys()],
                "Kedekatan ke Segmen": list(distances.values()),
            }).sort_values("Kedekatan ke Segmen")
            st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ==========================================================
# 8. HALAMAN 3 — PROFIL SEGMEN (RINGKASAN BISNIS)
# ==========================================================

def page_segment_profile():
    st.title("📌 Profil Segmen")
    st.caption(
        "Ringkasan karakteristik tiap segmen hasil model **Agglomerative Clustering**, "
        "disusun dalam bahasa bisnis agar mudah dipahami tanpa latar belakang teknis."
    )
    st.markdown("---")

    product_df = support["product_df"]

    # ------------------------------------------------------
    # 8.1 Karakteristik Umum Segmen
    # ------------------------------------------------------
    st.subheader("🧭 Karakteristik Umum Segmen")

    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        st.error(
            "**🔴 High Overstock**\n\n"
            "Ditandai dengan tingkat **inventory tinggi**, namun **units sold sangat rendah**. "
            "Pola ini sering terjadi pada produk dengan **harga mahal** atau **diskon rendah**, "
            "sehingga stok menumpuk dan modal tertahan lebih lama."
        )
    with profile_col2:
        st.success(
            "**🟢 Moderate/Low Overstock**\n\n"
            "Perputaran stok tergolong **sehat** — jumlah **units sold sebanding** dengan "
            "**units ordered**, sehingga risiko penumpukan stok relatif rendah."
        )

    st.markdown("")
    profile_table = compute_segment_profile(product_df)
    profile_display = profile_table.rename(columns={
        "Inventory Level": UNIT_LABELS["Inventory Level"],
        "Units Sold": UNIT_LABELS["Units Sold"],
        "Units Ordered": UNIT_LABELS["Units Ordered"],
        "Price": UNIT_LABELS["Price"],
        "Discount": UNIT_LABELS["Discount"],
    })
    st.dataframe(profile_display, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------
    # 8.2 Karakteristik Berdasarkan Kategori & Produk
    # ------------------------------------------------------
    st.subheader("🏷️ Karakteristik Berdasarkan Kategori & Produk")
    st.caption(
        "Pilih segmen untuk melihat kategori produk dan Product ID mana saja yang "
        "paling mendominasi segmen tersebut."
    )

    selected_segment = st.selectbox(
        "Pilih Segmen", options=["High Overstock", "Moderate/Low Overstock"]
    )

    top_cat_df, top_prod_df = compute_top_category_and_products(
        product_df, selected_segment, top_n=10
    )

    seg_color = CLUSTER_COLOR_MAP.get(selected_segment, COLOR_ACCENT)

    cat_col1, cat_col2 = st.columns([1, 1.4])
    with cat_col1:
        fig_cat = px.bar(
            top_cat_df, x="Category", y="Jumlah Produk",
            color_discrete_sequence=[seg_color],
            title=f"Kategori Produk Paling Dominan — {selected_segment}",
        )
        fig_cat.update_layout(margin=dict(t=40, b=10, l=10, r=10))
        fig_cat = apply_chart_theme(fig_cat, THEME)
        st.plotly_chart(fig_cat, use_container_width=True)

    with cat_col2:
        st.markdown(f"**Top 10 Produk (ID Produk) — {selected_segment}**")
        top_prod_display = top_prod_df.rename(columns={"Product ID": "ID Produk", "Category": "Kategori"})
        st.dataframe(top_prod_display, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ------------------------------------------------------
    # 8.3 Proporsi Kategori Cluster per Store ID
    # ------------------------------------------------------
    st.subheader("🏬 Proporsi Segmen Overstock per Store ID")
    st.caption(
        "Grafik ini menunjukkan komposisi (%) segmen **High Overstock** vs "
        "**Moderate/Low Overstock** pada masing-masing Store ID, sehingga "
        "dapat mengidentifikasi toko mana yang paling banyak menyimpan "
        "produk berisiko overstock tinggi dan memprioritaskan tindakan (audit stok, "
        "distribusi ulang, atau strategi diskon) pada toko tersebut."
    )

    store_segment_df = compute_segment_proportion_by_store(product_df)

    fig_store_segment = px.bar(
        store_segment_df,
        x="Proporsi (%)",
        y="Store ID",
        color="Segment",
        orientation="h",
        color_discrete_map=CLUSTER_COLOR_MAP,
        text=store_segment_df["Proporsi (%)"].map(lambda v: f"{v:.1f}%"),
        custom_data=["Segment", "Jumlah Produk"],
    )
    fig_store_segment.update_traces(
        hovertemplate=(
            "Store ID: %{y}<br>"
            "Segmen: %{customdata[0]}<br>"
            "Proporsi: %{x:.1f}%<br>"
            "Jumlah Produk: %{customdata[1]}<extra></extra>"
        ),
        textposition="inside",
    )
    fig_store_segment.update_layout(
        barmode="stack",
        xaxis_title="Proporsi Produk (%)",
        yaxis_title="Store ID",
        yaxis=dict(type="category"),
        legend_title_text="Segmen",
        margin=dict(t=20, b=10, l=10, r=10),
        height=max(350, 40 * store_segment_df["Store ID"].nunique()),
    )
    fig_store_segment = apply_chart_theme(fig_store_segment, THEME)
    st.plotly_chart(fig_store_segment, use_container_width=True)


# ==========================================================
# 9. NAVIGASI MULTI-HALAMAN
# ==========================================================

pg = st.navigation([
    st.Page(page_dashboard, title="Dashboard", icon="📊", default=True),
    st.Page(page_simulation, title="Simulasi Risiko Overstock", icon="🧪"),
    st.Page(page_segment_profile, title="Profil Segmen", icon="📌"),
])
pg.run()