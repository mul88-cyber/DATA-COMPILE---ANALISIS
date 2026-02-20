import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Bandarmology Master V3", 
    layout="wide", 
    page_icon="🐋",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header { 
        background: linear-gradient(90deg, #000428, #004e92); 
        padding: 1.5rem; 
        border-radius: 12px; 
        color: white; 
        margin-bottom: 1.5rem; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    .metric-card { 
        background: white; 
        padding: 1rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); 
        border-left: 5px solid #004e92; 
    }
    .kpi-card { 
        background-color: #f8f9fa; 
        border: 1px solid #dee2e6; 
        border-radius: 8px; 
        padding: 15px; 
        text-align: center; 
    }
    .kpi-value { 
        font-size: 24px; 
        font-weight: bold; 
        color: #004e92; 
    }
    .kpi-label { 
        font-size: 14px; 
        color: #6c757d; 
    }
    .stTabs [data-baseweb="tab-list"] { 
        gap: 10px; 
        background-color: white; 
        padding: 10px; 
        border-radius: 10px; 
        border: 1px solid #e2e8f0; 
    }
    .stTabs [data-baseweb="tab"] { 
        border-radius: 5px; 
        padding: 8px 16px; 
        font-weight: 600; 
    }
    .filter-container { 
        background: #f8f9fa; 
        padding: 15px; 
        border-radius: 10px; 
        margin-bottom: 1rem; 
        border: 1px solid #dee2e6;
    }
    .broker-change-positive {
        color: #00c853;
        font-weight: bold;
    }
    .broker-change-negative {
        color: #ff3d00;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 2rem;'>🐋 Bandarmology Master V3</h1>
    <p style='margin:0; opacity:0.8; font-size: 1rem;'>Deep Dive Analytics • Multi-Timeframe • AOVol Tracking • Broker Mutation</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD DATA & PREPROCESSING
# ==========================================
@st.cache_data(ttl=3600, show_spinner="📊 Mengunduh & Memproses Data Market...")
def load_and_preprocess_data():
    try:
        gcp_service_account = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            gcp_service_account, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        
        # Load Transaksi
        req_trans = service.files().get_media(fileId="1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8")
        df_transaksi = pd.read_csv(io.BytesIO(req_trans.execute()))
        
        # Load Kepemilikan
        req_ksei = service.files().get_media(fileId="1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr")
        df_kepemilikan = pd.read_csv(io.BytesIO(req_ksei.execute()))
        
        # --- PREPROCESSING DIMASUKKAN KE DALAM CACHE ---
        # Konversi Tanggal
        df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'].astype(str), errors='coerce')
        df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'].astype(str), errors='coerce')

        # Drop NA Penting
        df_transaksi = df_transaksi.dropna(subset=['Last Trading Date', 'Stock Code'])
        df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data', 'Kode Efek'])

        # Konversi Numerik
        numeric_cols = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                        'Big_Player_Anomaly', 'Close', 'Volume Spike (x)', 'Avg_Order_Volume',
                        'Tradeble Shares', 'Free Float', 'Typical Price', 'TPxV', 'Frequency',
                        'Previous', 'Open Price', 'High', 'Low', 'Change %']

        for col in numeric_cols:
            if col in df_transaksi.columns:
                df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce').fillna(0)

        # Hitung AOVol (Average Order Volume) moving average
        df_transaksi = df_transaksi.sort_values(['Stock Code', 'Last Trading Date'])
        df_transaksi['AOVol_MA20'] = df_transaksi.groupby('Stock Code')['Avg_Order_Volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df_transaksi['AOVol_Ratio'] = df_transaksi['Avg_Order_Volume'] / df_transaksi['AOVol_MA20'].replace(0, np.nan)
        df_transaksi['AOVol_Ratio'] = df_transaksi['AOVol_Ratio'].fillna(1)

        # Metrik Tambahan (Volume % Tradeble)
        if 'Tradeble Shares' in df_transaksi.columns:
            df_transaksi['Volume_Pct_Tradeble'] = np.where(
                df_transaksi['Tradeble Shares'] > 0, 
                (df_transaksi['Volume'] / df_transaksi['Tradeble Shares']) * 100, 
                0
            )
        else:
            df_transaksi['Volume_Pct_Tradeble'] = 0

        return df_transaksi, df_kepemilikan
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Panggil fungsi yang sudah di-cache
df_transaksi, df_kepemilikan = load_and_preprocess_data()

if df_transaksi.empty: 
    st.stop()

unique_stocks = sorted(df_transaksi['Stock Code'].unique())
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")


# ==========================================
# 3. FUNGSI BANTUAN OPTIMASI (V3.3 SUPER FAST!)
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def prepare_chart_data(stock_code, interval, chart_len, max_date, period_map, _df_master):
    """
    ✅ OPTIMIZED: Prepare chart data dengan resampling. 
    Menggunakan parameter stock_code (string) agar hashing cache Streamlit instan.
    """
    # Filter hanya di dalam fungsi yang di-cache
    df_chart = _df_master[_df_master['Stock Code'] == stock_code].copy()
    
    if df_chart.empty:
        return None

    # Potong berdasarkan periode
    if chart_len != "Semua Data":
        days_back = period_map[chart_len]
        start_date_chart = max_date - timedelta(days=days_back)
        df_chart = df_chart[df_chart['Last Trading Date'].dt.date >= start_date_chart]
        
    df_chart = df_chart.sort_values('Last Trading Date')
    
    # ====================================================
    # 🛠️ PERBAIKAN LOGIKA CANDLESTICK (OHLC)
    # ====================================================
    # 1. Ubah nilai 0 menjadi NaN sementara agar mudah ditambal
    for col in ['Close', 'Open Price', 'High', 'Low', 'Previous']:
        if col in df_chart.columns:
            df_chart[col] = df_chart[col].replace(0, np.nan)

    # 2. Jika Close blank/NaN, ambil dari Previous
    if 'Previous' in df_chart.columns:
        df_chart['Close'] = df_chart['Close'].fillna(df_chart['Previous'])
    
    # 3. Jika Close masih kosong juga, isi dengan harga hari sebelumnya (Forward Fill)
    df_chart['Close'] = df_chart['Close'].ffill().bfill()
    
    # 4. Jika Open, High, Low kosong, samakan dengan Close (agar candle jadi garis lurus, tidak anjlok ke 0)
    df_chart['Open Price'] = df_chart['Open Price'].fillna(df_chart['Close'])
    df_chart['High'] = df_chart['High'].fillna(df_chart['Close'])
    df_chart['Low'] = df_chart['Low'].fillna(df_chart['Close'])
    # ====================================================
    
    # Kamus agregasi resampling (DITAMBAH FLOAT & TRADEBLE SHARES)
    agg_dict = {
        'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
        'Avg_Order_Volume': 'mean', 'AOVol_Ratio': 'max', 'Volume Spike (x)': 'max',
        'Change %': 'last'
    }
    
    # Pastikan data float ikut ter-load
    if 'Tradeble Shares' in df_chart.columns:
        agg_dict['Tradeble Shares'] = 'last'
    if 'Free Float' in df_chart.columns:
        agg_dict['Free Float'] = 'last'
    if 'Volume_Pct_Tradeble' in df_chart.columns:
        agg_dict['Volume_Pct_Tradeble'] = 'sum' # Di-sum agar terlihat total % turnover di periode tsb
    if 'VWMA_20D' in df_chart.columns:
        agg_dict['VWMA_20D'] = 'last'
    if 'MA20_vol' in df_chart.columns:
        agg_dict['MA20_vol'] = 'mean'
        
    # Resampling untuk interval Weekly/Monthly
    if interval == "Weekly":
        df_chart = df_chart.set_index('Last Trading Date').resample('W-FRI').agg(agg_dict).dropna(subset=['Close']).reset_index()
    elif interval == "Monthly":
        df_chart = df_chart.set_index('Last Trading Date').resample('M').agg(agg_dict).dropna(subset=['Close']).reset_index()
    
    if len(df_chart) == 0:
        return None
        
    # Date_Label dibuat sekali di sini
    df_chart['Date_Label'] = df_chart['Last Trading Date'].dt.strftime('%d-%b-%Y')
    
    return df_chart

def compute_ksei_mutations_optimized(ksei_stock):
    """
    ✅ OPTIMIZED: Hitung mutasi KSEI dengan vectorized operations (tanpa nested loop)
    """
    if len(ksei_stock) <= 1:
        return pd.DataFrame()
    
    # Kolom Unik Rekening
    ksei_stock = ksei_stock.copy()
    ksei_stock['Rekening_ID'] = ksei_stock['Kode Broker'].fillna('') + ' - ' + ksei_stock['Nama Pemegang Saham'].fillna('')
    
    # Sort untuk shift operation
    ksei_sorted = ksei_stock.sort_values(['Rekening_ID', 'Tanggal_Data']).copy()
    
    # ✅ VECTORISED: Gunakan shift() untuk hitung perubahan
    ksei_sorted['Prev_Holding'] = ksei_sorted.groupby('Rekening_ID')['Jumlah Saham (Curr)'].shift(1)
    ksei_sorted['Change'] = ksei_sorted['Jumlah Saham (Curr)'] - ksei_sorted['Prev_Holding']
    
    # Filter hanya yang ada perubahan
    mutations = ksei_sorted[ksei_sorted['Change'] != 0].copy()
    
    if mutations.empty:
        return pd.DataFrame()
    
    # Format periode ke ISO Week
    mutations['Periode'] = mutations['Tanggal_Data'].dt.strftime('%G-W%V')
    
    # Build output dataframe (Tanpa Perubahan % lama, dihitung dinamis nanti)
    result = pd.DataFrame({
        'Periode': mutations['Periode'],
        'Rekening / Broker': mutations['Rekening_ID'],
        'Sebelum': mutations['Prev_Holding'],
        'Sesudah': mutations['Jumlah Saham (Curr)'],
        'Perubahan': mutations['Change'],
        'Abs_Perubahan': mutations['Change'].abs()
    })
    
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_ksei_timeline(stock_code, interval, chart_len, max_date, period_map, _df_ksei):
    """✅ OPTIMIZED: Pivot KSEI Dinamis (Mengikuti Filter Periode & Interval)"""
    ksei_stock = _df_ksei[_df_ksei['Kode Efek'] == stock_code].copy()
    ksei_stock = ksei_stock.sort_values('Tanggal_Data')
    
    if len(ksei_stock) <= 1:
        return None
        
    # 1. Filter Berdasarkan Periode (Sinkron dengan Chart Atas)
    if chart_len != "Semua Data":
        days_back = period_map[chart_len]
        start_date_chart = max_date - timedelta(days=days_back)
        ksei_stock = ksei_stock[ksei_stock['Tanggal_Data'].dt.date >= start_date_chart]
        
    if ksei_stock.empty:
        return None

    ksei_stock_temp = ksei_stock.copy()
    ksei_stock_temp['Rekening_ID'] = ksei_stock_temp['Kode Broker'].fillna('') + ' - ' + ksei_stock_temp['Nama Pemegang Saham'].fillna('')
    
    # Pivot Table
    ksei_pivot = ksei_stock_temp.pivot_table(
        index='Tanggal_Data', columns='Rekening_ID', 
        values='Jumlah Saham (Curr)', aggfunc='mean'
    )
    
    # 2. Resampling Dinamis Berdasarkan Interval (Daily/Weekly/Monthly)
    if interval == "Daily":
        ksei_pivot = ksei_pivot.resample('D').mean()
        ksei_pivot = ksei_pivot.ffill().fillna(0)
        ksei_pivot.index = ksei_pivot.index.strftime('%d-%b-%Y')
    elif interval == "Weekly":
        ksei_pivot = ksei_pivot.resample('W-FRI').mean() # Akhir pekan
        ksei_pivot = ksei_pivot.ffill().fillna(0)
        ksei_pivot.index = ksei_pivot.index.strftime('%Y-W%V')
    elif interval == "Monthly":
        ksei_pivot = ksei_pivot.resample('M').mean() # Akhir bulan
        ksei_pivot = ksei_pivot.ffill().fillna(0)
        ksei_pivot.index = ksei_pivot.index.strftime('%b %Y')
        
    return ksei_pivot

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_ksei_mutations(stock_code, _df_ksei):
    """✅ OPTIMIZED: Cache untuk tabel mutasi agar tidak lag saat ganti saham"""
    ksei_stock = _df_ksei[_df_ksei['Kode Efek'] == stock_code].copy()
    return compute_ksei_mutations_optimized(ksei_stock)

@st.cache_data(show_spinner=False)
def get_stock_mapping(_df):
    """Membuat kamus/dictionary Nama Perusahaan sekali saja saat awal loading"""
    if 'Company Name' in _df.columns:
        # Ambil nama unik, jadikan dictionary { 'BBCA': 'Bank Central Asia Tbk', ... }
        return _df.drop_duplicates('Stock Code').set_index('Stock Code')['Company Name'].to_dict()
    return {}

@st.cache_data(show_spinner=False)
def get_public_shares_mapping(_df):
    """Membuat kamus total lembar saham Free Float milik publik per emiten"""
    if 'Tradeble Shares' in _df.columns and 'Free Float' in _df.columns:
        # Mengambil data terbaru per saham
        latest = _df.sort_values('Last Trading Date').groupby('Stock Code').last()
        # Jumlah Saham Publik = Total Lembar Tradeble * Persentase Free Float
        return (latest['Tradeble Shares'] * latest['Free Float']).to_dict()
    return {}

# Panggil pembuat kamus
DICT_STOCK_NAME = get_stock_mapping(df_transaksi)
DICT_PUBLIC_SHARES = get_public_shares_mapping(df_transaksi)

def format_stock_label(code):
    """Ambil nama dari kamus. Kecepatan kilat!"""
    name = DICT_STOCK_NAME.get(code, "")
    if name:
        return f"{code} - {name}"
    return code


# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tabs = st.tabs([
    "🎯 SCREENER PRO", 
    "🔍 DEEP DIVE & CHART", 
    "🏦 BROKER MUTASI",
    "🗺️ MARKET MAP"
])

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            min_value = st.number_input("Min Nilai (M)", 0, 10000, 10) * 1e9
        with r1c2:
            min_volume = st.number_input("Min Volume (Juta)", 0, 10000, 100) * 1e6
        with r1c3:
            min_anomali = st.slider("Min Anomali (x)", 0, 20, 3)
        with r1c4:
            min_aoVol = st.slider("Min AOVol Ratio", 0.0, 5.0, 1.5, 0.1)
        
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy", "Net Sell", "Net Buy > 10M", "Net Sell > 10M"])
        with r2c2:
            min_vol_pct = st.slider("Min Volume % Tradeble", 0.0, 10.0, 0.5, 0.1)
        with r2c3:
            min_price = st.number_input("Min Harga", 0, 100000, 50)
        with r2c4:
            date_range = st.date_input("Periode", value=(default_start, max_date))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if not df_filter.empty:
            summary = df_filter.groupby('Stock Code').agg({
                'Close': 'last',
                'Change %': 'mean',
                'Volume': 'sum',
                'Value': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'max',
                'Volume Spike (x)': 'max',
                'Volume_Pct_Tradeble': 'mean',
                'Avg_Order_Volume': 'mean',
                'AOVol_Ratio': 'max',
                'Tradeble Shares': 'last'
            }).reset_index()
            
            summary['Pressure'] = np.where(summary['Value'] > 0, 
                                          (summary['Net Foreign Flow'] / summary['Value'] * 100), 0)
            summary['Inst_Score'] = (
                summary['Volume_Pct_Tradeble'] * 0.3 + 
                summary['Big_Player_Anomaly'] * 0.3 + 
                abs(summary['Pressure']) * 0.2 +
                summary['AOVol_Ratio'] * 0.2
            )
            
            # Apply filters
            summary = summary[summary['Value'] >= min_value]
            summary = summary[summary['Volume'] >= min_volume]
            summary = summary[summary['Big_Player_Anomaly'] >= min_anomali]
            summary = summary[summary['AOVol_Ratio'] >= min_aoVol]
            summary = summary[summary['Volume_Pct_Tradeble'] >= min_vol_pct]
            summary = summary[summary['Close'] >= min_price]
            
            if foreign_filter == "Net Buy":
                summary = summary[summary['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell":
                summary = summary[summary['Net Foreign Flow'] < 0]
            elif foreign_filter == "Net Buy > 10M":
                summary = summary[summary['Net Foreign Flow'] > 10e9]
            elif foreign_filter == "Net Sell > 10M":
                summary = summary[summary['Net Foreign Flow'] < -10e9]
            
            summary = summary.sort_values('Inst_Score', ascending=False).head(100)
            
            st.markdown(f"**🎯 Ditemukan {len(summary)} saham**")
            
            if len(summary) > 0:
                display_df = summary[['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow',
                                     'Volume_Pct_Tradeble', 'Big_Player_Anomaly', 'AOVol_Ratio', 'Inst_Score']].copy()
                
                # Format numbers
                display_df['Close'] = display_df['Close'].apply(lambda x: f"Rp {x:,.0f}")
                display_df['Value'] = display_df['Value'].apply(lambda x: f"Rp {x:,.0f}")
                display_df['Net Foreign Flow'] = display_df['Net Foreign Flow'].apply(lambda x: f"Rp {x:,.0f}")
                display_df['Change %'] = display_df['Change %'].apply(lambda x: f"{x:.2f}%")
                display_df['Volume_Pct_Tradeble'] = display_df['Volume_Pct_Tradeble'].apply(lambda x: f"{x:.2f}%")
                display_df['Big_Player_Anomaly'] = display_df['Big_Player_Anomaly'].apply(lambda x: f"{x:.1f}x")
                display_df['AOVol_Ratio'] = display_df['AOVol_Ratio'].apply(lambda x: f"{x:.1f}x")
                display_df['Inst_Score'] = display_df['Inst_Score'].apply(lambda x: f"{x:.1f}")
                
                display_df.columns = ['Kode', 'Harga', 'Change', 'Nilai', 'Foreign', 
                                     'Vol%Trade', 'Anomali', 'AOVol', 'Score']
                
                st.dataframe(display_df, use_container_width=True, height=500)
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")


# ==================== TAB 2: DEEP DIVE & CHART (V3.3 SUPER FAST!) ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics & VWMA Anchor")
    
    # Control Panel
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, format_func=format_stock_label, key='dd_stock')
    with col2:
        interval = st.selectbox("Interval", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with col3:
        chart_len = st.selectbox("Periode", ["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun", "Semua Data"], 
                                index=1, key='dd_len')
    
    # Map periode ke hari
    period_map = {"3 Bulan": 90, "6 Bulan": 180, "1 Tahun": 365, "2 Tahun": 730, "Semua Data": 9999}
    
    # ✅ OPTIMASI 1: Ambil df_chart via CACHE menggunakan string 'selected_stock'
    with st.spinner("📊 Memproses chart data..."):
        df_chart = prepare_chart_data(selected_stock, interval, chart_len, max_date, period_map, df_transaksi)
    
    if df_chart is None or len(df_chart) == 0:
        st.warning(f"Tidak ada data transaksi untuk {selected_stock} pada interval {interval} dalam periode ini.")
    else:
        # Ambil latest data langsung dari master (Bukan resample)
        df_stock_raw = df_transaksi[df_transaksi['Stock Code'] == selected_stock]
        latest = df_stock_raw.iloc[-1]
        
        total_foreign = df_chart['Net Foreign Flow'].sum()
        max_aoVol = df_chart['AOVol_Ratio'].max()
        
        # Hitung status berdasarkan trend 5 data terakhir
        if len(df_chart) >= 5:
            recent_foreign = df_chart['Net Foreign Flow'].tail(5).sum()
            recent_prices = df_chart['Close'].tail(5)
            price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
            price_change_pct = (price_change / recent_prices.iloc[0] * 100) if recent_prices.iloc[0] > 0 else 0
        else:
            recent_foreign = total_foreign
            price_change = 0
            price_change_pct = 0
        
        # Tentukan status logic
        if recent_foreign > 1e9 and price_change_pct > 3:
            status_text, status_color = "🚀 AKUMULASI KUAT", "darkgreen"
        elif recent_foreign > 0 and price_change_pct > 0:
            status_text, status_color = "📈 AKUMULASI", "green"
        elif recent_foreign < -1e9 and price_change_pct < -3:
            status_text, status_color = "🔻 DISTRIBUSI KUAT", "darkred"
        elif recent_foreign < 0 and price_change_pct < 0:
            status_text, status_color = "📉 DISTRIBUSI", "red"
        elif recent_foreign > 0 and price_change_pct < -2:
            status_text, status_color = "⚠️ DIV. POSITIF", "blue"
        elif recent_foreign < 0 and price_change_pct > 2:
            status_text, status_color = "⚡ MARKUP RITEL", "orange"
        else:
            status_text, status_color = "⏸️ NEUTRAL", "gray"
        
        # KPI Cards Row 1 (Main)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Harga Terkini", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: st.metric("Volume Terkini", f"{latest['Volume']/1e6:,.1f} Jt Lbr") 
        with k3: st.metric("VWMA 20D Anchor", f"Rp {latest['VWMA_20D']:,.0f}" if 'VWMA_20D' in latest else "N/A")
        with k4: st.metric("Max AOVol Ratio", f"{max_aoVol:.1f}x" if max_aoVol > 0 else "N/A")
        with k5: 
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value' style='color:{status_color}; font-size:16px;'>{status_text}</div>
                <div class='kpi-label'>5-Bar Foreign: Rp {recent_foreign/1e9:,.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
            
        # ====================================================
        # 💧 NEW: FLOAT & LIQUIDITY KPI ROW
        # ====================================================
        with st.container():
            st.markdown("<p style='font-size:14px; color:gray; font-weight:bold; margin-top:10px; margin-bottom:5px;'>💧 Float & Liquidity Profile</p>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            
            # Pengolahan angka dari data raw terakhir
            free_float_pct = latest.get('Free Float', 0) * 100 # asumsi raw data desimal 0.15 = 15%
            tradeble_shrs = latest.get('Tradeble Shares', 0)
            public_shares_vol = DICT_PUBLIC_SHARES.get(selected_stock, 0)
            float_mc = tradeble_shrs * latest['Close']
            turnover_pct = latest.get('Volume_Pct_Tradeble', 0)
            
            f1.metric("Free Float (%)", f"{free_float_pct:.2f}%")
            f2.metric("Public Shares (Barang Beredar)", f"{public_shares_vol/1e6:,.1f} Jt Lbr")
            f3.metric("Float Market Cap", f"Rp {float_mc/1e9:,.1f} Miliar")
            f4.metric("Daily Turnover", f"{turnover_pct:.2f}% dari Float")
        # ====================================================
        
        st.divider()
        
        # MAIN CHART - 4 PANEL
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.35, 0.2, 0.2, 0.25],
            subplot_titles=(
                f"<b>Price Action, VWMA Anchor & Big Player Signal</b>",
                "<b>Average Order Volume (AOVol) Tracking</b>",
                "<b>Volume & Participation</b>",
                "<b>Net Foreign Flow</b>"
            )
        )
        
        # PANEL 1: CANDLESTICK
        fig.add_trace(go.Candlestick(
            x=df_chart['Date_Label'], open=df_chart['Open Price'],
            high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'],
            name="Price", showlegend=False,
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350'
        ), row=1, col=1)
        
        # VWMA_20D Overlay Line
        if 'VWMA_20D' in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['VWMA_20D'],
                mode='lines', name='⚓ VWMA 20D (Bandar Avg)',
                line=dict(color='blue', width=2, dash='dot'),
                hoverinfo='name+y'
            ), row=1, col=1)

        # BINTANG UNTUK AOVol SPIKES
        if 'AOVol_Ratio' in df_chart.columns:
            aoVol_spikes = df_chart[df_chart['AOVol_Ratio'] > 1.5].dropna(subset=['Close'])
            if not aoVol_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=aoVol_spikes['Date_Label'], y=aoVol_spikes['High'] * 1.02,
                    mode='markers', name='⭐ AOVol Spike',
                    marker=dict(symbol='star', size=14, color='gold', line=dict(width=2, color='orange')),
                    text=[f"Ratio: {x:.1f}x<br>Volume: {y:,.0f} Lembar" for x, y in zip(aoVol_spikes['AOVol_Ratio'], aoVol_spikes['Avg_Order_Volume'])],
                    hoverinfo='text'
                ), row=1, col=1)
        
        # BINTANG UNTUK BIG PLAYER ANOMALI
        if 'Big_Player_Anomaly' in df_chart.columns:
            anomaly_spikes = df_chart[df_chart['Big_Player_Anomaly'] > 3].dropna(subset=['Close'])
            if not anomaly_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_spikes['Date_Label'], y=anomaly_spikes['Low'] * 0.98,
                    mode='markers', name='💎 BP Anomaly',
                    marker=dict(symbol='diamond', size=12, color='magenta', line=dict(width=2, color='purple')),
                    text=[f"Anomali: {x:.1f}x<br>Harga: Rp {y:,.0f}" for x, y in zip(anomaly_spikes['Big_Player_Anomaly'], anomaly_spikes['Close'])],
                    hoverinfo='text'
                ), row=1, col=1)

        
        # 🐋 MARKER: WHALE SIGNAL & SPLIT SIGNAL
        if 'Whale_Signal' in df_chart.columns:
            # Asumsi Whale_Signal berisi True/1
            ws = df_chart[df_chart['Whale_Signal'] == True].dropna(subset=['High'])
            if not ws.empty:
                fig.add_trace(go.Scatter(
                    x=ws['Date_Label'], 
                    y=ws['High'] * 1.02, # Muncul sedikit di atas harga High
                    mode='markers', 
                    marker=dict(symbol='triangle-down', size=12, color='#00cc00', line=dict(width=1, color='black')), 
                    name='🐋 Whale Buy',
                    hoverinfo='name+y'
                ), row=1, col=1)
        
        if 'Split_Signal' in df_chart.columns:
            # Asumsi Split_Signal berisi True/1
            ss = df_chart[df_chart['Split_Signal'] == True].dropna(subset=['Low'])
            if not ss.empty:
                fig.add_trace(go.Scatter(
                    x=ss['Date_Label'], 
                    y=ss['Low'] * 0.98, # Muncul sedikit di bawah harga Low
                    mode='markers', 
                    marker=dict(symbol='triangle-up', size=12, color='#ff4444', line=dict(width=1, color='black')), 
                    name='✂️ Split/Retail',
                    hoverinfo='name+y'
                ), row=1, col=1)
        
        # PANEL 2: AOVOL ANALYSIS (Sesuai Referensi Baru)
        # Kita gunakan AOVol_MA20 dari tahap preprocessing
        ma_col = 'AOVol_MA20' 
        ma_vals = df_chart[ma_col].fillna(0).values if ma_col in df_chart.columns else np.zeros(len(df_chart))
        
        fig.add_trace(go.Scatter(
            x=df_chart['Date_Label'], y=df_chart['AOVol_Ratio'],
            mode='lines', line=dict(color='#9c88ff', width=2), name='AOV Ratio',
            customdata=np.stack((df_chart['Avg_Order_Volume'], ma_vals), axis=-1),
            hovertemplate='Ratio: %{y:.2f}x<br>Avg: %{customdata[0]:,.0f}<br>MA: %{customdata[1]:,.0f}'
        ), row=2, col=1)
        
        # Garis Batas (Ref Lines)
        fig.add_hline(y=1.5, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", row=2, col=1)
        
        # GARIS BIRU SOLID UNTUK RATIO
        fig.add_trace(go.Scatter(
            x=df_chart['Date_Label'], y=df_chart['AOVol_Ratio'],
            name='AOVol Ratio (x)', mode='lines+markers',
            line=dict(color='blue', width=2),
            yaxis='y3', marker=dict(size=4, color='blue')
        ), row=2, col=1)
        
        # ====================================================
        # 💧 PANEL 3: VOLUME DENGAN TURNOVER INSIGHT
        # ====================================================
        colors_vol = np.where(
            df_chart['Close'].fillna(0) >= df_chart['Open Price'].fillna(0), 
            '#26a69a', '#ef5350'
        ).tolist()
        
        # Siapkan Custom Data (Turnover Float) untuk Hover Tooltip
        vol_customdata = df_chart['Volume_Pct_Tradeble'].fillna(0) if 'Volume_Pct_Tradeble' in df_chart.columns else np.zeros(len(df_chart))
        
        fig.add_trace(go.Bar(
            x=df_chart['Date_Label'], y=df_chart['Volume'] / 1e6,
            name='Volume (Juta Lembar)', marker_color=colors_vol, showlegend=False,
            customdata=vol_customdata,
            hovertemplate='Volume: %{y:,.1f} Jt<br>Turnover Float: %{customdata:.2f}%' # <--- TURNOVER TERLIHAT SAAT HOVER
        ), row=3, col=1)

        # MA20_vol Overlay Line
        if 'MA20_vol' in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart['Date_Label'], y=df_chart['MA20_vol'] / 1e6,
                mode='lines', name='MA20 Vol',
                line=dict(color='black', width=1.5),
                hoverinfo='name+y'
            ), row=3, col=1)
        
        # PANEL 4: FOREIGN FLOW
        colors_ff = np.where(df_chart['Net Foreign Flow'] >= 0, '#26a69a', '#ef5350').tolist()
        fig.add_trace(go.Bar(
            x=df_chart['Date_Label'], y=df_chart['Net Foreign Flow'] / 1e9,
            name='Foreign (Miliar Rp)', marker_color=colors_ff, showlegend=False
        ), row=4, col=1)
        
        # Update layout - PERBAIKAN MARGIN AGAR MENTOK KIRI KANAN
        fig.update_layout(
            height=1000, hovermode='x unified',
            margin=dict(t=50, b=20, l=10, r=10), # <--- Ubah l=10 dan r=10 agar lebar maksimal
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10)),
            title=dict(text=f"<b>Data: {len(df_chart)} periode • {df_chart['Date_Label'].iloc[0]} s/d {df_chart['Date_Label'].iloc[-1]}</b>", font=dict(size=12), y=0.99)
        )
        
        fig.update_xaxes(type='category', categoryorder='trace', tickangle=45, nticks=20)
        
        fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="AOVol (Jt Lbr)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Ratio (x)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Vol (Jt Lbr)", row=3, col=1)
        fig.update_yaxes(title_text="Foreign (M)", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': False,
            'responsive': True,
            'scrollZoom': False,
            'doubleClick': 'reset'
        })
        
        st.divider()
        
        # ✅ OPTIMASI 2: LAZY LOAD EXPANDER DENGAN CACHED FUNCTIONS
        with st.expander("🔄 Klik untuk lihat Analisis Kepemilikan KSEI (Institusi Lokal vs Asing)", expanded=False):
            if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
                
                # Mengambil data KSEI Pivot dari CACHE dengan parameter dinamis
                ksei_pivot = get_cached_ksei_timeline(selected_stock, interval, chart_len, max_date, period_map, df_kepemilikan)
                
                if ksei_pivot is not None and len(ksei_pivot.columns) > 0:
                    st.markdown(f"#### 📅 Timeline Kepemilikan KSEI ({interval})")
                    fig_timeline = go.Figure()
                    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                    x_labels = ksei_pivot.index
                    
                    for i, rekening in enumerate(ksei_pivot.columns):
                        fig_timeline.add_trace(go.Scatter(
                            x=x_labels, 
                            y=ksei_pivot[rekening],
                            name=rekening[:30] + '...' if len(rekening) > 30 else rekening,
                            mode='lines', 
                            line=dict(width=2.5, color=colors[i % len(colors)])
                        ))
                    
                    # Layout dikembalikan ke Tema Terang (White)
                    fig_timeline.update_layout(
                        template='plotly_white', # <--- KEMBALI KE PUTIH
                        height=550, 
                        xaxis_title=f"Waktu ({interval})", 
                        yaxis_title="Average of Jumlah Saham (Curr)",
                        hovermode='x unified', 
                        margin=dict(t=40, b=40, l=20, r=20),
                        legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02, font=dict(size=10))
                    )
                    
                    fig_timeline.update_xaxes(type='category', tickangle=90 if interval != "Daily" else 45)
                    st.plotly_chart(fig_timeline, use_container_width=True, config={'displayModeBar': False})
                    
                    # TABEL MUTASI DETAIL (DENGAN % SERAPAN FLOAT)
                    st.markdown("#### 📋 Detail Mutasi & Float Absorption")
                    
                    with st.spinner("🔍 Memuat data mutasi dari cache..."):
                        # Mengambil hasil kalkulasi mutasi dari CACHE
                        df_mutations = get_cached_ksei_mutations(selected_stock, df_kepemilikan)
                    
                    if not df_mutations.empty:
                        # Hitung % Serap Float
                        df_mutations['Serap_Float_Pct'] = np.where(
                            public_shares_vol > 0, 
                            (df_mutations['Perubahan'] / public_shares_vol) * 100, 
                            0
                        )
                        # Tambahkan Sinyal 🚨 jika serapan >= 2%
                        df_mutations['Sinyal'] = np.where(df_mutations['Serap_Float_Pct'].abs() >= 2.0, '🚨', '')
                        
                        display_mut = df_mutations.sort_values('Abs_Perubahan', ascending=False).head(20).copy()
                        display_mut = display_mut[['Periode', 'Rekening / Broker', 'Sebelum', 'Sesudah', 'Perubahan', 'Serap_Float_Pct', 'Sinyal']]
                        
                        st.dataframe(
                            display_mut, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Periode": st.column_config.TextColumn("Periode"),
                                "Rekening / Broker": st.column_config.TextColumn("Broker - Pemegang Saham"),
                                "Sebelum": st.column_config.NumberColumn("Sebelum (Lbr)", format="%,.0f"),
                                "Sesudah": st.column_config.NumberColumn("Sesudah (Lbr)", format="%,.0f"),
                                "Perubahan": st.column_config.NumberColumn("Mutasi (Lbr)", format="%,.0f"),
                                "Serap_Float_Pct": st.column_config.NumberColumn("% Serap Float", format="%.2f%%"),
                                "Sinyal": st.column_config.TextColumn("Sinyal")
                            }
                        )
                        
                        st.markdown("#### 📊 Summary Mutasi")
                        total_akumulasi = df_mutations[df_mutations['Perubahan'] > 0]['Perubahan'].sum()
                        total_distribusi = abs(df_mutations[df_mutations['Perubahan'] < 0]['Perubahan'].sum())
                        net_change = total_akumulasi - total_distribusi
                        
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        col_sum1.metric("Total Akumulasi", f"{total_akumulasi/1e6:,.1f} Jt Lbr")
                        col_sum2.metric("Total Distribusi", f"{total_distribusi/1e6:,.1f} Jt Lbr")
                        col_sum3.metric("Net Change", f"{net_change/1e6:+,.1f} Jt Lbr")
                        col_sum4.metric("Rekening Aktif", df_mutations['Rekening / Broker'].nunique())
                    else:
                        st.info("Tidak ada perubahan kepemilikan dalam periode ini")
                else:
                    st.info("Data kepemilikan masih terbatas (hanya 1 periode) atau tidak ditemukan.")
            else:
                st.warning("Data KSEI 5% tidak tersedia")


# ==================== TAB 3: BROKER MUTASI ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Mutation Radar - Semua Saham")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        with st.container():
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                mutasi_period = st.selectbox("Periode Analisis", ["1 Minggu", "2 Minggu", "1 Bulan", "3 Bulan", "6 Bulan"], key='m_period')
                days_map = {"1 Minggu": 7, "2 Minggu": 14, "1 Bulan": 30, "3 Bulan": 90, "6 Bulan": 180}
            with col_b2:
                min_mutasi = st.number_input("Min Mutasi (Juta)", 0, 1000, 10) * 1e6
            with col_b3:
                top_n = st.slider("Top N Broker", 5, 30, 15)
        
        start_mutasi = df_kepemilikan['Tanggal_Data'].max() - timedelta(days=days_map[mutasi_period])
        df_ksei_period = df_kepemilikan[df_kepemilikan['Tanggal_Data'] >= start_mutasi].copy()
        
        if not df_ksei_period.empty:
            # Hitung mutasi per broker per saham - ✅ OPTIMIZED: Vectorized
            mutasi = df_ksei_period.sort_values('Tanggal_Data').groupby(['Kode Broker', 'Kode Efek']).agg(
                Awal=('Jumlah Saham (Curr)', 'first'),
                Akhir=('Jumlah Saham (Curr)', 'last'),
                Nama=('Nama Pemegang Saham', 'first')
            ).reset_index()
            
            mutasi['Net_Change'] = mutasi['Akhir'] - mutasi['Awal']
            mutasi = mutasi[abs(mutasi['Net_Change']) >= min_mutasi]
            
            if len(mutasi) > 0:
                # Summary per broker
                broker_summary = mutasi.groupby('Kode Broker').agg({
                    'Net_Change': 'sum',
                    'Kode Efek': lambda x: list(x)
                }).reset_index()
                broker_summary.columns = ['Kode Broker', 'Total Mutasi', 'List Saham']
                
                # Top Accumulator & Distributor
                top_acc = broker_summary.nlargest(top_n, 'Total Mutasi')
                top_dist = broker_summary.nsmallest(top_n, 'Total Mutasi')
                top_dist['Total Mutasi'] = abs(top_dist['Total Mutasi'])
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"#### 🟢 Top {top_n} Accumulator ({mutasi_period})")
                    if not top_acc.empty:
                        display_acc = top_acc[['Kode Broker', 'Total Mutasi']].copy()
                        display_acc['Total Mutasi'] = display_acc['Total Mutasi'].apply(lambda x: f"{x:,.0f}")
                        display_acc.columns = ['Broker', 'Total Akumulasi']
                        st.dataframe(display_acc, use_container_width=True, hide_index=True)
                
                with c2:
                    st.markdown(f"#### 🔴 Top {top_n} Distributor ({mutasi_period})")
                    if not top_dist.empty:
                        display_dist = top_dist[['Kode Broker', 'Total Mutasi']].copy()
                        display_dist['Total Mutasi'] = display_dist['Total Mutasi'].apply(lambda x: f"{x:,.0f}")
                        display_dist.columns = ['Broker', 'Total Distribusi']
                        st.dataframe(display_dist, use_container_width=True, hide_index=True)
                
                # Detail per broker (DENGAN % SERAPAN FLOAT)
                st.divider()
                st.markdown("#### 🔎 Detail Mutasi per Broker & Float Absorption")
                
                all_brokers = sorted(broker_summary['Kode Broker'].unique())
                sel_broker = st.selectbox("Pilih Broker", all_brokers, key='m_broker_detail')
                
                if sel_broker:
                    detail = mutasi[mutasi['Kode Broker'] == sel_broker].copy()
                    detail = detail.sort_values('Net_Change', ascending=False)
                    
                    if not detail.empty:
                        display_detail = detail[['Kode Efek', 'Nama', 'Awal', 'Akhir', 'Net_Change']].copy()
                        
                        # Map Public Shares & Hitung Sinyal
                        display_detail['Public_Shares'] = display_detail['Kode Efek'].map(DICT_PUBLIC_SHARES).fillna(0)
                        display_detail['Serap_Float_Pct'] = np.where(
                            display_detail['Public_Shares'] > 0, 
                            (display_detail['Net_Change'] / display_detail['Public_Shares']) * 100, 
                            0
                        )
                        display_detail['Sinyal'] = np.where(display_detail['Serap_Float_Pct'].abs() >= 2.0, '🚨', '')
                        
                        # Formatting Display
                        display_detail['Awal'] = display_detail['Awal'].apply(lambda x: f"{x:,.0f}")
                        display_detail['Akhir'] = display_detail['Akhir'].apply(lambda x: f"{x:,.0f}")
                        display_detail['Net_Change'] = display_detail['Net_Change'].apply(
                            lambda x: f"+{x:,.0f}" if x > 0 else f"{x:,.0f}"
                        )
                        display_detail['Serap_Float_Pct'] = display_detail['Serap_Float_Pct'].apply(lambda x: f"{x:+.2f}%")
                        
                        display_detail = display_detail[['Kode Efek', 'Nama', 'Awal', 'Akhir', 'Net_Change', 'Serap_Float_Pct', 'Sinyal']]
                        display_detail.columns = ['Saham', 'Pemegang', 'Awal', 'Akhir', 'Mutasi', '% Serap Float', 'Sinyal']
                        st.dataframe(display_detail, use_container_width=True, hide_index=True)
            else:
                st.info(f"Tidak ada mutasi ≥ {min_mutasi/1e6:.0f} juta dalam periode ini")
        else:
            st.warning("Tidak ada data KSEI dalam periode ini")

# ==================== TAB 4: MARKET MAP ====================
with tabs[3]:
    st.markdown("### 🗺️ Market Map & Foreign Radar")
    
    # Foreign Flow Timeframe
    st.markdown("#### 🌍 Top Foreign Flow")
    
    col_period, col_top, col_sort = st.columns(3)
    with col_period:
        ff_period = st.selectbox("Rentang Waktu", 
                                ["Hari Ini", "5 Hari", "10 Hari", "20 Hari", "30 Hari", "60 Hari"], 
                                key='ff_time')
    with col_top:
        top_n_ff = st.slider("Top N", 5, 30, 20, key='top_ff')
    with col_sort:
        sort_ff = st.selectbox("Urut Berdasarkan", ["Net Foreign", "Nilai Transaksi", "Volume"], key='sort_ff')
    
    days_ff_map = {"Hari Ini": 0, "5 Hari": 5, "10 Hari": 10, "20 Hari": 20, 
                   "30 Hari": 30, "60 Hari": 60}
    days_back = days_ff_map[ff_period]
    
    # Filter Data
    if days_back == 0:
        ff_data = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date].copy()
        if not ff_data.empty:
            ff_data = ff_data.groupby('Stock Code').agg({
                'Net Foreign Flow': 'sum',
                'Value': 'sum',
                'Close': 'last',
                'Change %': 'mean',
                'Volume': 'sum'
            }).reset_index()
    else:
        start_date_ff = max_date - timedelta(days=days_back)
        mask_ff = (df_transaksi['Last Trading Date'].dt.date >= start_date_ff) & \
                  (df_transaksi['Last Trading Date'].dt.date <= max_date)
        ff_data = df_transaksi[mask_ff].groupby('Stock Code').agg({
            'Net Foreign Flow': 'sum',
            'Value': 'sum',
            'Close': 'last',
            'Change %': 'mean',
            'Volume': 'sum'
        }).reset_index()
    
    if not ff_data.empty and len(ff_data) > 0:
        # Sort berdasarkan pilihan
        sort_col = {
            "Net Foreign": "Net Foreign Flow",
            "Nilai Transaksi": "Value",
            "Volume": "Volume"
        }[sort_ff]
        
        ff_data = ff_data.sort_values(sort_col, ascending=False)
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f"#### 🟢 Top {top_n_ff} Foreign Buy ({ff_period})")
            top_buy = ff_data.nlargest(top_n_ff, 'Net Foreign Flow')
            if not top_buy.empty:
                display_buy = top_buy[['Stock Code', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_buy['Close'] = display_buy['Close'].apply(lambda x: f"Rp {x:,.0f}")
                display_buy['Net Foreign Flow'] = display_buy['Net Foreign Flow'].apply(lambda x: f"Rp {x:,.0f}")
                display_buy['Change %'] = display_buy['Change %'].apply(lambda x: f"{x:.2f}%")
                display_buy.columns = ['Kode', 'Harga', 'Net Buy', 'Change']
                st.dataframe(display_buy, use_container_width=True, hide_index=True)
            
        with col_f2:
            st.markdown(f"#### 🔴 Top {top_n_ff} Foreign Sell ({ff_period})")
            top_sell = ff_data.nsmallest(top_n_ff, 'Net Foreign Flow')
            if not top_sell.empty:
                display_sell = top_sell[['Stock Code', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_sell['Close'] = display_sell['Close'].apply(lambda x: f"Rp {x:,.0f}")
                display_sell['Net Foreign Flow'] = display_sell['Net Foreign Flow'].apply(lambda x: f"Rp {x:,.0f}")
                display_sell['Change %'] = display_sell['Change %'].apply(lambda x: f"{x:.2f}%")
                display_sell.columns = ['Kode', 'Harga', 'Net Sell', 'Change']
                st.dataframe(display_sell, use_container_width=True, hide_index=True)
        
        # Visualisasi
        st.divider()
        st.markdown("#### 📊 Foreign Flow Distribution")
        
        ff_viz = ff_data.head(50).copy()
        ff_viz['Abs_Net'] = abs(ff_viz['Net Foreign Flow'])
        
        fig = px.scatter(ff_viz, x='Value', y='Net Foreign Flow',
                        text='Stock Code', size='Abs_Net',
                        color='Net Foreign Flow', color_continuous_scale='RdYlGn',
                        title=f"Foreign Flow Map - Size: Nilai Transaksi, Warna: Net Foreign ({ff_period})")
        fig.update_traces(textposition='top center')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"📅 Last Update: {max_date}")
with col2:
    st.caption(f"📊 Total Saham: {len(unique_stocks):,}")
with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
