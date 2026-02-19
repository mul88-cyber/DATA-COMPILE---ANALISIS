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

# Data Preprocessing - ROBUST HANDLING
st.write("🔄 Processing data...")

# Konversi tanggal
df_transaksi['Last Trading Date'] = pd.to_numeric(df_transaksi['Last Trading Date'], errors='coerce')
df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'], format='%Y%m%d', errors='coerce')

# Handle kolom Change % dengan aman
if 'Change %' in df_transaksi.columns:
    try:
        # Cek tipe data dan konversi dengan aman
        if df_transaksi['Change %'].dtype == 'object':
            # Jika string, bersihkan dan konversi
            df_transaksi['Change %'] = df_transaksi['Change %'].astype(str).str.replace('%', '', regex=False)
            df_transaksi['Change %'] = pd.to_numeric(df_transaksi['Change %'], errors='coerce')
        else:
            # Jika sudah numerik, langsung konversi
            df_transaksi['Change %'] = pd.to_numeric(df_transaksi['Change %'], errors='coerce')
    except Exception as e:
        st.warning(f"Error processing Change %: {e}. Menggunakan metode alternatif.")
        # Fallback: hitung dari Close dan Previous
        if 'Close' in df_transaksi.columns and 'Previous' in df_transaksi.columns:
            df_transaksi['Change %'] = ((df_transaksi['Close'] - df_transaksi['Previous']) / df_transaksi['Previous'] * 100)
else:
    st.warning("Kolom 'Change %' tidak ditemukan. Menggunakan metode alternatif.")
    if 'Close' in df_transaksi.columns and 'Previous' in df_transaksi.columns:
        df_transaksi['Change %'] = ((df_transaksi['Close'] - df_transaksi['Previous']) / df_transaksi['Previous'] * 100)

# Konversi kolom numerik lainnya
numeric_columns = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                   'Big_Player_Anomaly', 'Avg_Order_Volume', 'Volume Spike (x)']
for col in numeric_columns:
    if col in df_transaksi.columns:
        df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce')

# Konversi tanggal untuk df_kepemilikan
df_kepemilikan['Tanggal_Data'] = pd.to_numeric(df_kepemilikan['Tanggal_Data'], errors='coerce')
df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'], format='%Y%m%d', errors='coerce')

# Drop rows dengan tanggal NaN
df_transaksi = df_transaksi.dropna(subset=['Last Trading Date'])
df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data'])

# Get unique values untuk filter
unique_stocks = sorted(df_transaksi['Stock Code'].dropna().unique().tolist())
unique_sectors = sorted(df_transaksi['Sector'].dropna().unique().tolist()) if 'Sector' in df_transaksi.columns else []
min_date = df_transaksi['Last Trading Date'].min()
max_date = df_transaksi['Last Trading Date'].max()

st.success(f"✅ Data siap! {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")

# ==================== MAIN TABS ====================
tabs = st.tabs([
    "📊 Market Screener", 
    "🔍 Stock Deep Dive", 
    "👥 KSEI Ownership Tracker",
    "🐋 Big Money Flow",
    "📈 Technical Analysis",
    "⚡ Anomaly Detector"
])

# ==================== TAB 1: MARKET SCREENER ====================
with tabs[0]:
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
        value=(max_date.date() - timedelta(days=30), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    # Filter dan agregasi data untuk screener
    mask_screener = (df_transaksi['Last Trading Date'].dt.date >= date_range[0]) & \
                    (df_transaksi['Last Trading Date'].dt.date <= date_range[1])
    
    if sector_filter:
        mask_screener &= df_transaksi['Sector'].isin(sector_filter)
    
    df_screener = df_transaksi[mask_screener].copy()
    
    if len(df_screener) > 0:
        # Agregasi per saham
        agg_dict = {
            'Company Name': 'first' if 'Company Name' in df_screener.columns else 'first',
            'Close': 'last',
            'Change %': 'last',
            'Volume': 'sum',
            'Value': 'sum',
            'Net Foreign Flow': 'sum',
            'Big_Player_Anomaly': 'sum',
            'Volume Spike (x)': 'max'
        }
        
        # Tambahkan Sector jika ada
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
        
        # Color coding for change %
        def color_change(val):
            try:
                if pd.notna(val) and val > 0:
                    return 'color: green'
                elif pd.notna(val) and val < 0:
                    return 'color: red'
                else:
                    return 'color: black'
            except:
                return 'color: black'
        
        styled_df = screener_display.style.applymap(color_change, subset=['Change %'] if 'Change %' in screener_display.columns else [])
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        # Visualisasi Screener
        if len(screener_result) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Value' in screener_result.columns and 'Net Foreign Flow' in screener_result.columns:
                    fig = px.scatter(screener_result, x='Value', y='Net Foreign Flow', 
                                    size='Volume' if 'Volume' in screener_result.columns else None, 
                                    color='Change %' if 'Change %' in screener_result.columns else None, 
                                    hover_data=['Stock Code'],
                                    title="Institutional Flow vs Transaction Value",
                                    labels={'Value': 'Transaction Value (Rp)', 'Net Foreign Flow': 'Net Foreign (Rp)'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Big_Player_Anomaly' in screener_result.columns:
                    top_anomaly = screener_result.nlargest(10, 'Big_Player_Anomaly')[['Stock Code', 'Big_Player_Anomaly']]
                    if 'Company Name' in screener_result.columns:
                        top_anomaly['Company Name'] = top_anomaly['Stock Code'].map(
                            screener_result.set_index('Stock Code')['Company Name'].to_dict()
                        )
                    
                    fig = px.bar(top_anomaly, x='Stock Code', y='Big_Player_Anomaly',
                                title="Top 10 Big Player Anomaly",
                                color='Big_Player_Anomaly', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada data untuk periode yang dipilih")


# ==================== TAB 2: STOCK DEEP DIVE ====================
with tabs[1]:
    st.markdown("### 🔍 Stock Deep Dive Analysis")
    
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
                value=(max_date - timedelta(days=90), max_date),
                min_value=min_date,
                max_value=max_date,
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
    mask_dive = (df_transaksi['Stock Code'] == selected_stock) & \
                (df_transaksi['Last Trading Date'].dt.date >= dive_date_range[0]) & \
                (df_transaksi['Last Trading Date'].dt.date <= dive_date_range[1])
    
    df_dive = df_transaksi[mask_dive].copy().sort_values('Last Trading Date')
    
    if len(df_dive) == 0:
        st.warning("Tidak ada data untuk periode ini")
        st.stop()
    
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
        st.metric("Sektor", company_info['Sector'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Free Float", f"{company_info['Free Float']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tradeble Shares", f"{company_info['Tradeble Shares']/1e6:.1f}Jt")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Order Volume", f"{company_info['Avg_Order_Volume']:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced Chart dengan Subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=("Price Action & VWMA", "Volume Profile", "Foreign Flow", "Big Player Anomaly")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_dive['Last Trading Date'],
            open=df_dive['Open Price'],
            high=df_dive['High'],
            low=df_dive['Low'],
            close=df_dive['Close'],
            name="OHLC",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # VWMA
    fig.add_trace(
        go.Scatter(
            x=df_dive['Last Trading Date'],
            y=df_dive['VWMA_20D'],
            name=f"VWMA {ma_period}D",
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    # Volume dengan MA
    colors = ['red' if row['Close'] < row['Open Price'] else 'green' for _, row in df_dive.iterrows()]
    fig.add_trace(
        go.Bar(
            x=df_dive['Last Trading Date'],
            y=df_dive['Volume'],
            name="Volume",
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_dive['Last Trading Date'],
            y=df_dive['MA20_vol'],
            name="MA Volume",
            line=dict(color='purple', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Foreign Flow
    fig.add_trace(
        go.Bar(
            x=df_dive['Last Trading Date'],
            y=df_dive['Foreign Buy'],
            name="Foreign Buy",
            marker_color='green',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df_dive['Last Trading Date'],
            y=-df_dive['Foreign Sell'],
            name="Foreign Sell",
            marker_color='red',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    # Big Player Anomaly
    fig.add_trace(
        go.Scatter(
            x=df_dive['Last Trading Date'],
            y=df_dive['Big_Player_Anomaly'],
            name="Anomaly Score",
            line=dict(color='darkviolet', width=2),
            fill='tozeroy'
        ),
        row=4, col=1
    )
    
    fig.update_layout(
        height=900,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        hovermode='x unified',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical Analysis
    st.markdown("### 📊 Statistical Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation Matrix
        corr_cols = ['Close', 'Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Big_Player_Anomaly']
        corr_matrix = df_dive[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True,
                            aspect="auto",
                            title="Correlation Matrix",
                            color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Distribution Analysis
        fig_dist = make_subplots(rows=2, cols=2, subplot_titles=("Close Price", "Volume", "Foreign Flow", "Anomaly Score"))
        
        fig_dist.add_trace(go.Histogram(x=df_dive['Close'], nbinsx=30, name="Close"), row=1, col=1)
        fig_dist.add_trace(go.Histogram(x=df_dive['Volume'], nbinsx=30, name="Volume"), row=1, col=2)
        fig_dist.add_trace(go.Histogram(x=df_dive['Net Foreign Flow'], nbinsx=30, name="Net Foreign"), row=2, col=1)
        fig_dist.add_trace(go.Histogram(x=df_dive['Big_Player_Anomaly'], nbinsx=30, name="Anomaly"), row=2, col=2)
        
        fig_dist.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

# ==================== TAB 3: KSEI OWNERSHIP TRACKER ====================
with tabs[2]:
    st.markdown("### 👥 KSEI 5% Ownership Tracker")
    
    # Filter untuk KSEI
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ksei_stock = st.selectbox(
                "Pilih Saham untuk Tracking",
                options=unique_stocks,
                index=0,
                key="ksei_stock"
            )
        
        with col2:
            ksei_date_range = st.date_input(
                "Periode Tracking",
                value=(max_date - timedelta(days=180), max_date),
                min_value=min_date,
                max_value=max_date,
                key="ksei_date"
            )
        
        with col3:
            min_ownership = st.number_input(
                "Minimal Kepemilikan (Juta)",
                min_value=0.0,
                value=1.0,
                step=0.5
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data KSEI
    mask_ksei = (df_kepemilikan['Kode Efek'] == ksei_stock) & \
                (df_kepemilikan['Tanggal_Data'].dt.date >= ksei_date_range[0]) & \
                (df_kepemilikan['Tanggal_Data'].dt.date <= ksei_date_range[1])
    
    df_ksei_filter = df_kepemilikan[mask_ksei].copy()
    
    if len(df_ksei_filter) > 0:
        # Ownership Timeline
        ownership_timeline = df_ksei_filter.groupby(['Tanggal_Data', 'Nama Pemegang Saham'])['Jumlah Saham (Curr)'].sum().reset_index()
        
        fig = px.line(ownership_timeline, x='Tanggal_Data', y='Jumlah Saham (Curr)',
                     color='Nama Pemegang Saham',
                     title=f"Ownership Timeline - {ksei_stock}",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Holders Current
        st.markdown("### 🏆 Current Top Holders")
        latest_date = df_ksei_filter['Tanggal_Data'].max()
        current_holders = df_ksei_filter[df_ksei_filter['Tanggal_Data'] == latest_date].copy()
        current_holders = current_holders[current_holders['Jumlah Saham (Curr)'] >= min_ownership * 1e6]
        current_holders = current_holders.nlargest(20, 'Jumlah Saham (Curr)')
        
        display_holders = current_holders[['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)', 'Status']].copy()
        display_holders['Jumlah Saham (Curr)'] = display_holders['Jumlah Saham (Curr)'] / 1e6
        
        st.dataframe(display_holders, use_container_width=True)
        
        # Broker Analysis
        st.markdown("### 🏦 Broker Concentration")
        broker_analysis = df_ksei_filter.groupby('Kode Broker').agg({
            'Jumlah Saham (Curr)': 'sum',
            'Nama Pemegang Saham': 'nunique',
            'Tanggal_Data': 'count'
        }).rename(columns={
            'Jumlah Saham (Curr)': 'Total Saham',
            'Nama Pemegang Saham': 'Unique Holders',
            'Tanggal_Data': 'Transactions'
        }).reset_index()
        
        broker_analysis['Total Saham'] = broker_analysis['Total Saham'] / 1e6
        broker_analysis = broker_analysis.sort_values('Total Saham', ascending=False)
        
        fig = px.sunburst(broker_analysis.head(10), 
                         path=['Kode Broker'], 
                         values='Total Saham',
                         title="Top 10 Broker Concentration")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info(f"Tidak ada data kepemilikan untuk {ksei_stock} dalam periode ini")

# ==================== TAB 4: BIG MONEY FLOW ====================
with tabs[3]:
    st.markdown("### 🐋 Big Money Flow Analysis")
    
    # Filter untuk Big Money Flow
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            flow_date_range = st.date_input(
                "Periode Flow",
                value=(max_date - timedelta(days=30), max_date),
                min_value=min_date,
                max_value=max_date,
                key="flow_date"
            )
        
        with col2:
            flow_type = st.selectbox(
                "Tipe Flow",
                options=["Semua Flow", "Foreign Only", "Local Only", "Big Player Only"],
                index=0
            )
        
        with col3:
            min_flow = st.number_input(
                "Minimal Flow (Rp Miliar)",
                min_value=0.0,
                value=1.0,
                step=0.5
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data
    mask_flow = (df_transaksi['Last Trading Date'].dt.date >= flow_date_range[0]) & \
                (df_transaksi['Last Trading Date'].dt.date <= flow_date_range[1])
    
    df_flow = df_transaksi[mask_flow].copy()
    
    # Flow Analysis by Stock
    flow_by_stock = df_flow.groupby('Stock Code').agg({
        'Company Name': 'first',
        'Sector': 'first',
        'Net Foreign Flow': 'sum',
        'Foreign Buy': 'sum',
        'Foreign Sell': 'sum',
        'Big_Player_Anomaly': 'sum',
        'Value': 'sum'
    }).reset_index()
    
    # Filter by flow type
    if flow_type == "Foreign Only":
        flow_by_stock = flow_by_stock[abs(flow_by_stock['Net Foreign Flow']) >= min_flow * 1e9]
    elif flow_type == "Big Player Only":
        flow_by_stock = flow_by_stock[flow_by_stock['Big_Player_Anomaly'] > 0]
    
    # Top Foreign Flow
    col1, col2 = st.columns(2)
    
    with col1:
        top_buy = flow_by_stock.nlargest(10, 'Net Foreign Flow')[['Stock Code', 'Company Name', 'Net Foreign Flow']]
        top_buy['Net Foreign Flow'] = top_buy['Net Foreign Flow'] / 1e9
        
        fig = px.bar(top_buy, x='Stock Code', y='Net Foreign Flow',
                    title="Top 10 Net Foreign Buy",
                    color='Net Foreign Flow', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_sell = flow_by_stock.nsmallest(10, 'Net Foreign Flow')[['Stock Code', 'Company Name', 'Net Foreign Flow']]
        top_sell['Net Foreign Flow'] = top_sell['Net Foreign Flow'] / 1e9
        
        fig = px.bar(top_sell, x='Stock Code', y='Net Foreign Flow',
                    title="Top 10 Net Foreign Sell",
                    color='Net Foreign Flow', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap Flow by Sector
    flow_by_sector = df_flow.groupby('Sector').agg({
        'Net Foreign Flow': 'sum',
        'Value': 'sum'
    }).reset_index()
    
    fig = px.treemap(flow_by_sector, 
                     path=['Sector'], 
                     values='Value',
                     color='Net Foreign Flow',
                     color_continuous_scale='RdYlGn',
                     title="Sector Flow Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 5: TECHNICAL ANALYSIS ====================
with tabs[4]:
    st.markdown("### 📈 Advanced Technical Analysis")
    
    # Filter untuk Technical Analysis
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ta_stock = st.selectbox(
                "Pilih Saham untuk TA",
                options=unique_stocks,
                index=0,
                key="ta_stock"
            )
        
        with col2:
            ta_date_range = st.date_input(
                "Periode TA",
                value=(max_date - timedelta(days=180), max_date),
                min_value=min_date,
                max_value=max_date,
                key="ta_date"
            )
        
        with col3:
            indicator = st.selectbox(
                "Technical Indicator",
                options=["Bollinger Bands", "RSI", "MACD", "Volume Profile", "All Indicators"],
                index=0
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data untuk TA
    mask_ta = (df_transaksi['Stock Code'] == ta_stock) & \
              (df_transaksi['Last Trading Date'].dt.date >= ta_date_range[0]) & \
              (df_transaksi['Last Trading Date'].dt.date <= ta_date_range[1])
    
    df_ta = df_transaksi[mask_ta].copy().sort_values('Last Trading Date')
    
    if len(df_ta) > 0:
        # Calculate indicators
        df_ta['MA20'] = df_ta['Close'].rolling(window=20).mean()
        df_ta['MA50'] = df_ta['Close'].rolling(window=50).mean()
        df_ta['UpperBB'] = df_ta['MA20'] + (df_ta['Close'].rolling(window=20).std() * 2)
        df_ta['LowerBB'] = df_ta['MA20'] - (df_ta['Close'].rolling(window=20).std() * 2)
        
        # RSI
        delta = df_ta['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ta['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df_ta['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_ta['Close'].ewm(span=26, adjust=False).mean()
        df_ta['MACD'] = exp1 - exp2
        df_ta['Signal'] = df_ta['MACD'].ewm(span=9, adjust=False).mean()
        df_ta['MACD_Hist'] = df_ta['MACD'] - df_ta['Signal']
        
        if indicator == "Bollinger Bands" or indicator == "All Indicators":
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Candlestick(x=df_ta['Last Trading Date'],
                                           open=df_ta['Open Price'],
                                           high=df_ta['High'],
                                           low=df_ta['Low'],
                                           close=df_ta['Close'],
                                           name='Price'))
            fig_bb.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['UpperBB'],
                                       line=dict(color='rgba(250, 0, 0, 0.5)', dash='dash'),
                                       name='Upper BB'))
            fig_bb.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['MA20'],
                                       line=dict(color='orange', width=2),
                                       name='MA20'))
            fig_bb.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['LowerBB'],
                                       line=dict(color='rgba(0, 250, 0, 0.5)', dash='dash'),
                                       name='Lower BB',
                                       fill='tonexty', fillcolor='rgba(128, 128, 128, 0.2)'))
            
            fig_bb.update_layout(title='Bollinger Bands', height=500)
            st.plotly_chart(fig_bb, use_container_width=True)
        
        if indicator == "RSI" or indicator == "All Indicators":
            fig_rsi = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.7, 0.3],
                                   vertical_spacing=0.05)
            
            fig_rsi.add_trace(go.Candlestick(x=df_ta['Last Trading Date'],
                                           open=df_ta['Open Price'],
                                           high=df_ta['High'],
                                           low=df_ta['Low'],
                                           close=df_ta['Close'],
                                           name='Price'), row=1, col=1)
            
            fig_rsi.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['RSI'],
                                       line=dict(color='purple', width=2),
                                       name='RSI'), row=2, col=1)
            
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            fig_rsi.update_layout(title='RSI Indicator', height=600)
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        if indicator == "MACD" or indicator == "All Indicators":
            fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.7, 0.3],
                                   vertical_spacing=0.05)
            
            fig_macd.add_trace(go.Candlestick(x=df_ta['Last Trading Date'],
                                            open=df_ta['Open Price'],
                                            high=df_ta['High'],
                                            low=df_ta['Low'],
                                            close=df_ta['Close'],
                                            name='Price'), row=1, col=1)
            
            fig_macd.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['MACD'],
                                        line=dict(color='blue', width=2),
                                        name='MACD'), row=2, col=1)
            fig_macd.add_trace(go.Scatter(x=df_ta['Last Trading Date'], y=df_ta['Signal'],
                                        line=dict(color='red', width=2),
                                        name='Signal'), row=2, col=1)
            
            colors = ['green' if val >= 0 else 'red' for val in df_ta['MACD_Hist']]
            fig_macd.add_trace(go.Bar(x=df_ta['Last Trading Date'], y=df_ta['MACD_Hist'],
                                    marker_color=colors,
                                    name='Histogram'), row=2, col=1)
            
            fig_macd.update_layout(title='MACD Indicator', height=600)
            st.plotly_chart(fig_macd, use_container_width=True)

# ==================== TAB 6: ANOMALY DETECTOR ====================
with tabs[5]:
    st.markdown("### ⚡ Real-time Anomaly Detection")
    
    # Filter untuk Anomaly Detection
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            anomaly_date = st.date_input(
                "Tanggal Analisis",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="anomaly_date"
            )
        
        with col2:
            volume_threshold = st.slider(
                "Volume Spike Threshold",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )
        
        with col3:
            anomaly_threshold = st.slider(
                "Anomaly Score Threshold",
                min_value=2.0,
                max_value=10.0,
                value=5.0,
                step=1.0
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Deteksi anomaly untuk tanggal terpilih
    mask_anomaly = df_transaksi['Last Trading Date'].dt.date == anomaly_date
    
    if mask_anomaly.any():
        df_anomaly_day = df_transaksi[mask_anomaly].copy()
        
        # Klasifikasi anomaly
        df_anomaly_day['Volume_Spike_Flag'] = df_anomaly_day['Volume Spike (x)'] >= volume_threshold
        df_anomaly_day['Big_Player_Flag'] = df_anomaly_day['Big_Player_Anomaly'] >= anomaly_threshold
        df_anomaly_day['Foreign_Flow_Flag'] = abs(df_anomaly_day['Net Foreign Flow']) > df_anomaly_day['Net Foreign Flow'].std() * 2
        
        # Summary stats
        st.markdown("### 📊 Anomaly Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_volume_spike = df_anomaly_day['Volume_Spike_Flag'].sum()
            st.metric("Volume Spike", f"{total_volume_spike} saham")
        
        with col2:
            total_big_player = df_anomaly_day['Big_Player_Flag'].sum()
            st.metric("Big Player Anomaly", f"{total_big_player} saham")
        
        with col3:
            total_foreign = df_anomaly_day['Foreign_Flow_Flag'].sum()
            st.metric("Abnormal Foreign Flow", f"{total_foreign} saham")
        
        with col4:
            total_anomalies = total_volume_spike + total_big_player + total_foreign
            st.metric("Total Anomalies", f"{total_anomalies} kejadian")
        
        # Volume Spike Stocks
        st.markdown("### 📈 Volume Spike Stocks")
        volume_spike_stocks = df_anomaly_day[df_anomaly_day['Volume_Spike_Flag']].nlargest(20, 'Volume Spike (x)')
        if len(volume_spike_stocks) > 0:
            display_vol = volume_spike_stocks[['Stock Code', 'Company Name', 'Close', 'Volume', 'Volume Spike (x)']].copy()
            st.dataframe(display_vol, use_container_width=True)
        else:
            st.info("Tidak ada volume spike pada tanggal ini")
        
        # Big Player Anomaly Stocks
        st.markdown("### 🐋 Big Player Anomaly")
        big_player_stocks = df_anomaly_day[df_anomaly_day['Big_Player_Flag']].nlargest(20, 'Big_Player_Anomaly')
        if len(big_player_stocks) > 0:
            display_big = big_player_stocks[['Stock Code', 'Company Name', 'Close', 'Big_Player_Anomaly', 'Avg_Order_Volume']].copy()
            st.dataframe(display_big, use_container_width=True)
        else:
            st.info("Tidak ada big player anomaly pada tanggal ini")
        
        # Foreign Flow Anomaly
        st.markdown("### 🌏 Abnormal Foreign Flow")
        foreign_anomaly = df_anomaly_day[df_anomaly_day['Foreign_Flow_Flag']].copy()
        foreign_anomaly = foreign_anomaly.nlargest(20, 'abs(Net Foreign Flow)')
        
        if len(foreign_anomaly) > 0:
            display_foreign = foreign_anomaly[['Stock Code', 'Company Name', 'Close', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow']].copy()
            st.dataframe(display_foreign, use_container_width=True)
        else:
            st.info("Tidak ada abnormal foreign flow pada tanggal ini")
        
        # Visualisasi Anomaly Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Sector distribution of anomalies
            sector_anomaly = df_anomaly_day[df_anomaly_day['Volume_Spike_Flag'] | 
                                           df_anomaly_day['Big_Player_Flag'] | 
                                           df_anomaly_day['Foreign_Flow_Flag']]
            
            sector_count = sector_anomaly['Sector'].value_counts().reset_index()
            sector_count.columns = ['Sector', 'Count']
            
            fig = px.pie(sector_count, values='Count', names='Sector',
                        title="Anomaly Distribution by Sector")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter plot of anomalies
            fig = px.scatter(df_anomaly_day[df_anomaly_day['Volume_Spike_Flag'] | 
                                           df_anomaly_day['Big_Player_Flag']], 
                           x='Volume Spike (x)', y='Big_Player_Anomaly',
                           color='Sector', size='Value',
                           hover_data=['Stock Code'],
                           title="Volume Spike vs Big Player Anomaly")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning(f"Tidak ada data untuk tanggal {anomaly_date.strftime('%d-%m-%Y')}")

# Footer dengan real-time info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"📅 Data Update: {max_date.strftime('%d-%m-%Y')}")
with col2:
    st.info(f"📊 Total Saham: {len(unique_stocks)}")
with col3:
    st.info(f"🏭 Total Sektor: {len(unique_sectors)}")

# Auto-refresh button
if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
