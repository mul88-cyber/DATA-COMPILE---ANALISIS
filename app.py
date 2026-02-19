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

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Bandarmology Pro Suite", 
    layout="wide", 
    page_icon="🐋",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk styling premium
st.markdown("""
<style>
    /* Premium Styling */
    .main-header {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #2A5298;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .signal-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .signal-negative {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #6c757d, #495057);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    .filter-container {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #dee2e6;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .divider {
        border-top: 2px solid #2A5298;
        margin: 1rem 0;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Header Premium
st.markdown("""
<div class="main-header">
    <h1 style='margin:0;'>🐋 Bandarmology Pro Suite</h1>
    <p style='margin:0; opacity:0.9;'>Advanced Institutional Trading Intelligence Platform</p>
    <p style='margin:0; font-size:0.8rem; margin-top:0.5rem;'>Real-time Big Money Flow Analysis • KSEI 5% Ownership Tracking • Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

# 1. Fungsi Load Data dengan Caching dan Retry
@st.cache_data(ttl=3600, show_spinner=False)
def load_csv_from_gdrive(file_id, retry=3):
    for attempt in range(retry):
        try:
            gcp_service_account = st.secrets["gcp_service_account"]
            credentials = service_account.Credentials.from_service_account_info(
                gcp_service_account,
                scopes=['https://www.googleapis.com/auth/drive.readonly']
            )
            service = build('drive', 'v3', credentials=credentials)
            
            request = service.files().get_media(fileId=file_id)
            downloaded = io.BytesIO(request.execute())
            
            return pd.read_csv(downloaded)
        except Exception as e:
            if attempt == retry - 1:
                st.error(f"Failed to load data after {retry} attempts: {e}")
                return None
            continue

# Load Data dengan Progress
with st.status("📊 Loading Market Data...", expanded=True) as status:
    st.write("Connecting to Google Drive...")
    FILE_ID_TRANSAKSI = "1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8" 
    FILE_ID_KEPEMILIKAN = "1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr"
    
    df_transaksi = load_csv_from_gdrive(FILE_ID_TRANSAKSI)
    st.write("✅ Transaction data loaded")
    
    df_kepemilikan = load_csv_from_gdrive(FILE_ID_KEPEMILIKAN)
    st.write("✅ KSEI ownership data loaded")
    
    status.update(label="✅ Data Ready!", state="complete")

if df_transaksi is None or df_kepemilikan is None:
    st.error("Failed to load data. Please check connection and try again.")
    st.stop()

# Data Preprocessing - ROBUST HANDLING dengan validasi tanggal yang lebih ketat
st.write("🔄 Processing data...")

# Fungsi untuk konversi datetime dengan aman
def safe_convert_to_datetime(series):
    """Konversi ke datetime dengan aman"""
    try:
        # Coba konversi langsung
        return pd.to_datetime(series, errors='coerce')
    except:
        try:
            # Coba konversi dengan format angka
            return pd.to_datetime(series.astype(str), format='%Y%m%d', errors='coerce')
        except:
            # Jika gagal, return NaT
            return pd.Series([pd.NaT] * len(series))

# Konversi tanggal transaksi
if 'Last Trading Date' in df_transaksi.columns:
    # Cek tipe data
    if df_transaksi['Last Trading Date'].dtype == 'object':
        # Jika string, bersihkan
        df_transaksi['Last Trading Date'] = df_transaksi['Last Trading Date'].astype(str)
        df_transaksi['Last Trading Date'] = df_transaksi['Last Trading Date'].str.replace(r'\D', '', regex=True)
    
    df_transaksi['Last Trading Date'] = safe_convert_to_datetime(df_transaksi['Last Trading Date'])
else:
    st.error("Kolom 'Last Trading Date' tidak ditemukan!")
    st.stop()

# Konversi tanggal kepemilikan
if 'Tanggal_Data' in df_kepemilikan.columns:
    if df_kepemilikan['Tanggal_Data'].dtype == 'object':
        df_kepemilikan['Tanggal_Data'] = df_kepemilikan['Tanggal_Data'].astype(str)
        df_kepemilikan['Tanggal_Data'] = df_kepemilikan['Tanggal_Data'].str.replace(r'\D', '', regex=True)
    
    df_kepemilikan['Tanggal_Data'] = safe_convert_to_datetime(df_kepemilikan['Tanggal_Data'])
else:
    st.error("Kolom 'Tanggal_Data' tidak ditemukan!")
    st.stop()

# Drop rows dengan tanggal NaN
df_transaksi = df_transaksi.dropna(subset=['Last Trading Date'])
df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data'])

# Validasi apakah masih ada data setelah drop
if len(df_transaksi) == 0:
    st.error("Tidak ada data transaksi yang valid setelah preprocessing!")
    st.stop()

# Get unique values untuk filter dengan validasi
unique_stocks = sorted(df_transaksi['Stock Code'].dropna().unique().tolist()) if 'Stock Code' in df_transaksi.columns else []
unique_sectors = sorted(df_transaksi['Sector'].dropna().unique().tolist()) if 'Sector' in df_transaksi.columns else []

# Dapatkan min dan max date dengan aman
min_date = df_transaksi['Last Trading Date'].min()
max_date = df_transaksi['Last Trading Date'].max()

# Validasi tanggal
if pd.isna(min_date) or pd.isna(max_date):
    st.error("Data tanggal tidak valid!")
    st.stop()

# Handle kolom Change % dengan aman
if 'Change %' in df_transaksi.columns:
    try:
        if df_transaksi['Change %'].dtype == 'object':
            df_transaksi['Change %'] = df_transaksi['Change %'].astype(str).str.replace('%', '', regex=False)
            df_transaksi['Change %'] = pd.to_numeric(df_transaksi['Change %'], errors='coerce')
        else:
            df_transaksi['Change %'] = pd.to_numeric(df_transaksi['Change %'], errors='coerce')
    except:
        df_transaksi['Change %'] = 0
else:
    if 'Close' in df_transaksi.columns and 'Previous' in df_transaksi.columns:
        df_transaksi['Change %'] = ((df_transaksi['Close'] - df_transaksi['Previous']) / df_transaksi['Previous'] * 100).fillna(0)
    else:
        df_transaksi['Change %'] = 0

# Konversi kolom numerik
numeric_columns = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                   'Big_Player_Anomaly', 'Avg_Order_Volume', 'Volume Spike (x)',
                   'Close', 'Open Price', 'High', 'Low', 'Previous']
for col in numeric_columns:
    if col in df_transaksi.columns:
        df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce').fillna(0)

# Fungsi untuk mendapatkan tanggal dengan aman
def safe_get_date(date_value, default_date=None):
    """Mendapatkan tanggal dengan aman, menghindari NaT"""
    if pd.notna(date_value):
        try:
            return date_value.date()
        except:
            pass
    
    if default_date is None:
        default_date = datetime.now().date()
    return default_date

# Dapatkan tanggal dengan aman
safe_min_date = safe_get_date(min_date)
safe_max_date = safe_get_date(max_date)

# Pastikan min_date tidak lebih besar dari max_date
if safe_min_date > safe_max_date:
    safe_min_date, safe_max_date = safe_max_date, safe_min_date

# Hitung default date range (30 hari terakhir)
default_start = safe_max_date - timedelta(days=30)
if default_start < safe_min_date:
    default_start = safe_min_date

st.success(f"✅ Data siap! {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")
st.info(f"📅 Rentang data: {safe_min_date.strftime('%d-%m-%Y')} s/d {safe_max_date.strftime('%d-%m-%Y')}")

# ==================== MAIN TABS ====================
try:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Market Screener", 
        "🔍 Stock Deep Dive", 
        "👥 KSEI Ownership Tracker",
        "🐋 Big Money Flow",
        "📈 Technical Analysis",
        "⚡ Anomaly Detector"
    ])
    
    # ==================== TAB 1: MARKET SCREENER ====================
    with tab1:
        st.markdown("### 📊 Market Screener - Find Institutional Activity")
        
        # Filter Container untuk Screener
        with st.container():
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if unique_sectors:
                    sector_filter = st.multiselect(
                        "Sektor",
                        options=unique_sectors,
                        default=[]
                    )
                else:
                    sector_filter = []
                    st.info("Kolom sektor tidak tersedia")
            
            with col2:
                min_volume = st.number_input(
                    "Min Volume (Rp Miliar)",
                    min_value=0.0,
                    value=10.0,
                    step=5.0
                )
            
            with col3:
                anomaly_filter = st.selectbox(
                    "Big Player Anomaly",
                    options=["Semua", "Ada Anomali", "Tidak Ada Anomali"],
                    index=0
                )
            
            with col4:
                foreign_filter = st.selectbox(
                    "Net Foreign Flow",
                    options=["Semua", "Net Buy", "Net Sell"],
                    index=0
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Date range untuk screener
        date_range = st.date_input(
            "Periode Analisis",
            value=(default_start, safe_max_date),
            min_value=safe_min_date,
            max_value=safe_max_date
        )
        
        # Pastikan date_range memiliki 2 nilai
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = safe_min_date, safe_max_date
            st.warning("Rentang tanggal tidak valid, menggunakan default")
        
        # Filter dan agregasi data untuk screener
        mask_screener = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
                        (df_transaksi['Last Trading Date'].dt.date <= end_date)
        
        if sector_filter:
            mask_screener &= df_transaksi['Sector'].isin(sector_filter)
        
        df_screener = df_transaksi[mask_screener].copy()
        
        if len(df_screener) > 0:
            # Agregasi per saham
            agg_dict = {
                'Close': 'last',
                'Change %': 'last',
                'Volume': 'sum',
                'Value': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'sum',
                'Volume Spike (x)': 'max'
            }
            
            if 'Company Name' in df_screener.columns:
                agg_dict['Company Name'] = 'first'
            if 'Sector' in df_screener.columns:
                agg_dict['Sector'] = 'first'
            
            screener_result = df_screener.groupby('Stock Code').agg(agg_dict).reset_index()
            
            # Apply filters
            screener_result = screener_result[screener_result['Value'] >= min_volume * 1e9]
            
            if anomaly_filter == "Ada Anomali":
                screener_result = screener_result[screener_result['Big_Player_Anomaly'] > 0]
            elif anomaly_filter == "Tidak Ada Anomali":
                screener_result = screener_result[screener_result['Big_Player_Anomaly'] == 0]
            
            if foreign_filter == "Net Buy":
                screener_result = screener_result[screener_result['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell":
                screener_result = screener_result[screener_result['Net Foreign Flow'] < 0]
            
            # Display screener results
            st.markdown(f"**Hasil Screener: {len(screener_result)} saham ditemukan**")
            
            # Format untuk display
            display_cols = ['Stock Code']
            if 'Company Name' in screener_result.columns:
                display_cols.append('Company Name')
            if 'Sector' in screener_result.columns:
                display_cols.append('Sector')
            display_cols.extend(['Close', 'Change %', 'Volume', 'Value', 'Net Foreign Flow', 'Big_Player_Anomaly'])
            
            screener_display = screener_result[[col for col in display_cols if col in screener_result.columns]].copy()
            
            # Konversi ke unit yang lebih mudah dibaca
            if 'Value' in screener_display.columns:
                screener_display['Value'] = screener_display['Value'] / 1e9
            if 'Volume' in screener_display.columns:
                screener_display['Volume'] = screener_display['Volume'] / 1e6
            if 'Net Foreign Flow' in screener_display.columns:
                screener_display['Net Foreign Flow'] = screener_display['Net Foreign Flow'] / 1e9
            
            # Rename columns
            column_names = {
                'Stock Code': 'Kode',
                'Company Name': 'Nama',
                'Sector': 'Sektor',
                'Close': 'Harga',
                'Change %': 'Change %',
                'Volume': 'Volume (Jt)',
                'Value': 'Nilai (M)',
                'Net Foreign Flow': 'Net Foreign (M)',
                'Big_Player_Anomaly': 'Anomali'
            }
            screener_display = screener_display.rename(columns=column_names)
            
            st.dataframe(screener_display, use_container_width=True, height=500)
            
            # Visualisasi Screener
            if len(screener_result) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'Value' in screener_result.columns and 'Net Foreign Flow' in screener_result.columns:
                        fig = px.scatter(screener_result, x='Value', y='Net Foreign Flow', 
                                        size='Volume' if 'Volume' in screener_result.columns else None, 
                                        color='Change %' if 'Change %' in screener_result.columns else None, 
                                        hover_data=['Stock Code'],
                                        title="Institutional Flow vs Transaction Value")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Big_Player_Anomaly' in screener_result.columns:
                        top_anomaly = screener_result.nlargest(10, 'Big_Player_Anomaly')[['Stock Code', 'Big_Player_Anomaly']]
                        fig = px.bar(top_anomaly, x='Stock Code', y='Big_Player_Anomaly',
                                    title="Top 10 Big Player Anomaly",
                                    color='Big_Player_Anomaly', color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada data untuk periode yang dipilih")
    
    # ==================== TAB 2: STOCK DEEP DIVE ====================
    with tab2:
        st.markdown("### 🔍 Stock Deep Dive Analysis")
        
        if unique_stocks:
            # Filter khusus untuk deep dive
            with st.container():
                st.markdown('<div class="filter-container">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    selected_stock = st.selectbox(
                        "Pilih Saham",
                        options=unique_stocks,
                        index=0
                    )
                
                with col2:
                    dive_date_range = st.date_input(
                        "Periode Analisis",
                        value=(default_start, safe_max_date),
                        min_value=safe_min_date,
                        max_value=safe_max_date,
                        key="dive_date"
                    )
                
                with col3:
                    ma_period = st.selectbox(
                        "Moving Average Period",
                        options=[5, 10, 20, 50, 100],
                        index=2
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Filter data untuk saham terpilih
            if len(dive_date_range) == 2:
                start_dive, end_dive = dive_date_range
            else:
                start_dive, end_dive = default_start, safe_max_date
            
            mask_dive = (df_transaksi['Stock Code'] == selected_stock) & \
                        (df_transaksi['Last Trading Date'].dt.date >= start_dive) & \
                        (df_transaksi['Last Trading Date'].dt.date <= end_dive)
            
            df_dive = df_transaksi[mask_dive].copy().sort_values('Last Trading Date')
            
            if len(df_dive) > 0:
                # Company Overview
                company_info = df_dive.iloc[-1]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Harga Terakhir", f"Rp {company_info['Close']:,.0f}", 
                             f"{company_info['Change %']:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    sector_value = company_info['Sector'] if 'Sector' in company_info else 'N/A'
                    st.metric("Sektor", sector_value)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    free_float = company_info['Free Float'] if 'Free Float' in company_info else 0
                    st.metric("Free Float", f"{free_float:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    tradeble = company_info['Tradeble Shares'] if 'Tradeble Shares' in company_info else 0
                    st.metric("Tradeble Shares", f"{tradeble/1e6:.1f}Jt")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col5:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    avg_order = company_info['Avg_Order_Volume'] if 'Avg_Order_Volume' in company_info else 0
                    st.metric("Avg Order Volume", f"{avg_order:,.0f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.info(f"Menampilkan {len(df_dive)} hari data untuk {selected_stock}")
            else:
                st.warning(f"Tidak ada data untuk {selected_stock} dalam periode ini")
        else:
            st.warning("Tidak ada data saham yang tersedia")
    
    # ==================== TAB 3: KSEI OWNERSHIP TRACKER ====================
    with tab3:
        st.markdown("### 👥 KSEI 5% Ownership Tracker")
        st.info("Fitur KSEI Ownership Tracker akan segera hadir...")
    
    # ==================== TAB 4: BIG MONEY FLOW ====================
    with tab4:
        st.markdown("### 🐋 Big Money Flow Analysis")
        st.info("Fitur Big Money Flow Analysis akan segera hadir...")
    
    # ==================== TAB 5: TECHNICAL ANALYSIS ====================
    with tab5:
        st.markdown("### 📈 Technical Analysis")
        st.info("Fitur Technical Analysis akan segera hadir...")
    
    # ==================== TAB 6: ANOMALY DETECTOR ====================
    with tab6:
        st.markdown("### ⚡ Anomaly Detector")
        st.info("Fitur Anomaly Detector akan segera hadir...")

except Exception as e:
    st.error(f"Error dalam pembuatan tabs: {str(e)}")
    st.exception(e)

# Footer dengan real-time info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📅 Data Update: {safe_max_date.strftime('%d-%m-%Y')}")
with col2:
    st.info(f"📊 Total Saham: {len(unique_stocks)}")
with col3:
    st.info(f"🏭 Total Sektor: {len(unique_sectors)}")

# Auto-refresh button
if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
