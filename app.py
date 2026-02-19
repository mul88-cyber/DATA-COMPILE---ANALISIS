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
    page_title="Bandarmology Pro", 
    layout="wide", 
    page_icon="🐋",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan compact
st.markdown("""
<style>
    /* Compact Styling */
    .main-header {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 3px solid #2A5298;
        margin: 0.2rem 0;
    }
    
    .filter-container {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    
    .signal-badge {
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #f8f9fa;
        padding: 0.3rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.3rem 0.8rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h2 style='margin:0;'>🐋 Bandarmology Pro - Institutional Money Flow Tracker</h2>
    <p style='margin:0; font-size:0.8rem;'>Real-time Big Money Detection • KSEI Ownership • Anomaly Scanner</p>
</div>
""", unsafe_allow_html=True)

# 1. Fungsi Load Data
@st.cache_data(ttl=3600)
def load_csv_from_gdrive(file_id):
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
        st.error(f"Error loading data: {e}")
        return None

# Load Data
with st.spinner("📊 Loading market data..."):
    FILE_ID_TRANSAKSI = "1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8" 
    FILE_ID_KEPEMILIKAN = "1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr"
    
    df_transaksi = load_csv_from_gdrive(FILE_ID_TRANSAKSI)
    df_kepemilikan = load_csv_from_gdrive(FILE_ID_KEPEMILIKAN)

if df_transaksi is None:
    st.error("Gagal memuat data. Cek koneksi.")
    st.stop()

# PREPROCESSING CEPAT
st.caption("🔄 Processing data...")

# Konversi tanggal
if 'Last Trading Date' in df_transaksi.columns:
    df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'].astype(str), errors='coerce')
if 'Tanggal_Data' in df_kepemilikan.columns:
    df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'].astype(str), errors='coerce')

# Drop NA
df_transaksi = df_transaksi.dropna(subset=['Last Trading Date', 'Stock Code'])
df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data', 'Kode Efek'])

# Kolom numerik
numeric_cols = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                'Big_Player_Anomaly', 'Close', 'Volume Spike (x)']
for col in numeric_cols:
    if col in df_transaksi.columns:
        df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce').fillna(0)

# Hitung Change % jika tidak ada
if 'Change %' not in df_transaksi.columns:
    if 'Close' in df_transaksi.columns and 'Previous' in df_transaksi.columns:
        df_transaksi['Change %'] = ((df_transaksi['Close'] - df_transaksi['Previous']) / df_transaksi['Previous'] * 100).fillna(0)
    else:
        df_transaksi['Change %'] = 0

# Data siap
unique_stocks = sorted(df_transaksi['Stock Code'].unique())
min_date = df_transaksi['Last Trading Date'].min().date()
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")

# ==================== 4 MAIN TABS ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 SCREENER", 
    "🔍 DEEP DIVE", 
    "👥 KSEI TRACKER",
    "📊 MARKET SUMMARY"
])

# ==================== TAB 1: SCREENER ====================
with tab1:
    st.markdown("### 🔍 Institutional Activity Screener")
    
    # FILTER COMPACT
    with st.container():
        cols = st.columns([1, 1, 1, 1.5])
        
        with cols[0]:
            min_value = st.number_input("Min Nilai (M)", 0, 1000, 10) * 1e9
            
        with cols[1]:
            anomaly_filter = st.selectbox("Big Player", ["Semua", "Ada Anomali", "Normal"], index=0)
            
        with cols[2]:
            foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy", "Net Sell"], index=0)
            
        with cols[3]:
            date_range = st.date_input(
                "Periode",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )
    
    # Proses filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Filter data
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if len(df_filter) > 0:
            # Agregasi per saham
            summary = df_filter.groupby('Stock Code').agg({
                'Close': 'last',
                'Change %': 'last',
                'Volume': 'sum',
                'Value': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'sum',
                'Volume Spike (x)': 'max'
            }).reset_index()
            
            # Filter nilai
            summary = summary[summary['Value'] >= min_value]
            
            if anomaly_filter == "Ada Anomali":
                summary = summary[summary['Big_Player_Anomaly'] > 0]
            elif anomaly_filter == "Normal":
                summary = summary[summary['Big_Player_Anomaly'] == 0]
                
            if foreign_filter == "Net Buy":
                summary = summary[summary['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell":
                summary = summary[summary['Net Foreign Flow'] < 0]
            
            # Tampilkan hasil
            st.markdown(f"**{len(summary)} saham ditemukan**")
            
            # Format display
            display = summary.copy()
            display['Value'] = (display['Value'] / 1e9).round(1)
            display['Volume'] = (display['Volume'] / 1e6).round(0)
            display['Net Foreign Flow'] = (display['Net Foreign Flow'] / 1e9).round(1)
            display['Change %'] = display['Change %'].round(2)
            
            # Rename columns
            display.columns = ['Kode', 'Harga', 'Change%', 'Volume(Jt)', 'Nilai(M)', 
                              'Net Foreign(M)', 'Anomali', 'Volume Spike']
            
            # Color coding
            def color_change(val):
                try:
                    if float(val) > 0: return 'color: green'
                    elif float(val) < 0: return 'color: red'
                except: pass
                return ''
            
            st.dataframe(
                display.style.applymap(color_change, subset=['Change%']),
                use_container_width=True,
                height=400
            )
            
            # Visualisasi cepat
            col1, col2 = st.columns(2)
            
            with col1:
                if len(summary) > 0:
                    top_anomaly = summary.nlargest(10, 'Big_Player_Anomaly')[['Stock Code', 'Big_Player_Anomaly']]
                    fig = px.bar(top_anomaly, x='Stock Code', y='Big_Player_Anomaly',
                               title="Top 10 Big Player Activity",
                               color='Big_Player_Anomaly', color_continuous_scale='Reds')
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                top_foreign = summary.nlargest(10, 'Net Foreign Flow')[['Stock Code', 'Net Foreign Flow']]
                if len(top_foreign) > 0:
                    fig = px.bar(top_foreign, x='Stock Code', y='Net Foreign Flow',
                               title="Top 10 Net Foreign Buy",
                               color='Net Foreign Flow', color_continuous_scale='Greens')
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Tidak ada data di periode ini")

# ==================== TAB 2: DEEP DIVE ====================
with tab2:
    st.markdown("### 📈 Stock Deep Dive Analysis")
    
    # Pilihan saham di atas
    col1, col2 = st.columns([2, 3])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dive_stock')
    with col2:
        dive_range = st.select_slider(
            "Periode Analisis",
            options=[7, 14, 30, 60, 90, 180],
            value=30
        )
    
    # Filter data
    start_dive = max_date - timedelta(days=dive_range)
    mask_dive = (df_transaksi['Stock Code'] == selected_stock) & \
                (df_transaksi['Last Trading Date'].dt.date >= start_dive)
    df_dive = df_transaksi[mask_dive].copy().sort_values('Last Trading Date')
    
    if len(df_dive) > 0:
        # METRICS ROW
        latest = df_dive.iloc[-1]
        cols = st.columns(5)
        
        with cols[0]:
            st.metric("Harga", f"Rp{latest['Close']:,.0f}", f"{latest['Change %']:.1f}%")
        with cols[1]:
            st.metric("Volume", f"{latest['Volume']/1e6:.1f}Jt")
        with cols[2]:
            st.metric("Foreign Net", f"Rp{latest['Net Foreign Flow']/1e9:.1f}M")
        with cols[3]:
            st.metric("Anomali", f"{latest['Big_Player_Anomaly']:.1f}x")
        with cols[4]:
            st.metric("Volume Spike", f"{latest['Volume Spike (x)']:.1f}x")
        
        # CHART UTAMA - Compact
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=("Price & Volume", "Foreign Flow", "Big Player Anomaly")
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df_dive['Last Trading Date'],
                open=df_dive['Open Price'],
                high=df_dive['High'],
                low=df_dive['Low'],
                close=df_dive['Close'],
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Volume dengan warna
        colors = ['red' if row['Close'] < row['Open Price'] else 'green' for _, row in df_dive.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df_dive['Last Trading Date'],
                y=df_dive['Volume']/1e6,
                name="Volume (Jt)",
                marker_color=colors,
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Net Foreign Flow
        fig.add_trace(
            go.Bar(
                x=df_dive['Last Trading Date'],
                y=df_dive['Net Foreign Flow']/1e9,
                name="Net Foreign (M)",
                marker_color=['green' if x > 0 else 'red' for x in df_dive['Net Foreign Flow']],
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Big Player Anomaly
        fig.add_trace(
            go.Scatter(
                x=df_dive['Last Trading Date'],
                y=df_dive['Big_Player_Anomaly'],
                name="Anomaly Score",
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # SIGNAL DETECTION
        st.markdown("### 🚨 Trading Signals")
        
        signals = []
        
        # Cek volume spike
        if latest['Volume Spike (x)'] > 2:
            signals.append(("🔴 Volume Spike", f"{latest['Volume Spike (x)']:.1f}x normal"))
        
        # Cek anomaly
        if latest['Big_Player_Anomaly'] > 5:
            signals.append(("🐋 Big Player", f"Anomali {latest['Big_Player_Anomaly']:.1f}x"))
        
        # Cek foreign flow
        if latest['Net Foreign Flow'] > 0:
            signals.append(("🟢 Foreign Buy", f"Rp{latest['Net Foreign Flow']/1e9:.1f}M"))
        elif latest['Net Foreign Flow'] < 0:
            signals.append(("🔴 Foreign Sell", f"Rp{abs(latest['Net Foreign Flow']/1e9):.1f}M"))
        
        # Cek price momentum
        if len(df_dive) > 5:
            ma5 = df_dive['Close'].tail(5).mean()
            if latest['Close'] > ma5 * 1.05:
                signals.append(("📈 Strong Momentum", f"+5% above MA5"))
            elif latest['Close'] < ma5 * 0.95:
                signals.append(("📉 Weak Momentum", f"-5% below MA5"))
        
        if signals:
            for sig in signals:
                if "🔴" in sig[0] or "🔴" in sig[1]:
                    st.warning(f"**{sig[0]}**: {sig[1]}")
                elif "🟢" in sig[0]:
                    st.success(f"**{sig[0]}**: {sig[1]}")
                else:
                    st.info(f"**{sig[0]}**: {sig[1]}")
        else:
            st.info("Tidak ada signal signifikan")
        
        # DATA TABLE
        with st.expander("📋 Detail Data Transaksi"):
            display_cols = ['Last Trading Date', 'Close', 'Volume', 'Value', 
                          'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                          'Big_Player_Anomaly', 'Volume Spike (x)']
            available_cols = [c for c in display_cols if c in df_dive.columns]
            st.dataframe(df_dive[available_cols].tail(20), use_container_width=True)
    
    else:
        st.warning(f"Tidak ada data untuk {selected_stock}")

# ==================== TAB 3: KSEI TRACKER ====================
with tab3:
    st.markdown("### 👥 KSEI 5% Ownership Tracker")
    
    if len(df_kepemilikan) > 0:
        col1, col2 = st.columns([2, 2])
        
        with col1:
            ksei_stock = st.selectbox("Pilih Saham", sorted(df_kepemilikan['Kode Efek'].unique()), key='ksei_stock')
        
        with col2:
            ksei_date = st.date_input(
                "Tanggal",
                value=max_date,
                min_value=df_kepemilikan['Tanggal_Data'].min().date(),
                max_value=df_kepemilikan['Tanggal_Data'].max().date()
            )
        
        # Filter data
        mask_ksei = (df_kepemilikan['Kode Efek'] == ksei_stock) & \
                    (df_kepemilikan['Tanggal_Data'].dt.date == ksei_date)
        
        df_ksei_today = df_kepemilikan[mask_ksei].copy()
        
        if len(df_ksei_today) > 0:
            # Metrics
            total_shares = df_ksei_today['Jumlah Saham (Curr)'].sum()
            unique_holders = df_ksei_today['Nama Pemegang Saham'].nunique()
            unique_brokers = df_ksei_today['Kode Broker'].nunique()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Kepemilikan", f"{total_shares/1e6:.1f}Jt")
            col2.metric("Jumlah Holder", unique_holders)
            col3.metric("Jumlah Broker", unique_brokers)
            
            # Top Holders
            st.markdown("#### 🏆 Top Holders")
            top_holders = df_ksei_today.nlargest(10, 'Jumlah Saham (Curr)')[
                ['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)', 'Status']
            ].copy()
            top_holders['Jumlah Saham (Curr)'] = (top_holders['Jumlah Saham (Curr)'] / 1e6).round(1)
            st.dataframe(top_holders, use_container_width=True)
            
            # Broker Distribution
            st.markdown("#### 🏦 Broker Distribution")
            broker_dist = df_ksei_today.groupby('Kode Broker').agg({
                'Jumlah Saham (Curr)': 'sum',
                'Nama Pemegang Saham': 'count'
            }).rename(columns={
                'Jumlah Saham (Curr)': 'Total Saham',
                'Nama Pemegang Saham': 'Jumlah Holder'
            }).reset_index()
            broker_dist['Total Saham'] = (broker_dist['Total Saham'] / 1e6).round(1)
            broker_dist = broker_dist.sort_values('Total Saham', ascending=False)
            
            fig = px.pie(broker_dist.head(8), values='Total Saham', names='Kode Broker',
                        title="Top 8 Broker Concentration")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info(f"Tidak ada data kepemilikan untuk {ksei_stock} pada {ksei_date.strftime('%d-%m-%Y')}")
    else:
        st.warning("Data KSEI tidak tersedia")

# ==================== TAB 4: MARKET SUMMARY ====================
with tab4:
    st.markdown("### 📊 Market Summary & Heatmap")
    
    # Data hari ini
    today_data = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date].copy()
    
    if len(today_data) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = today_data['Value'].sum() / 1e9
            st.metric("Total Transaksi", f"Rp{total_value:.0f}M")
        
        with col2:
            total_volume = today_data['Volume'].sum() / 1e6
            st.metric("Total Volume", f"{total_volume:.0f}Jt")
        
        with col3:
            net_foreign = today_data['Net Foreign Flow'].sum() / 1e9
            st.metric("Net Foreign", f"Rp{net_foreign:.0f}M", 
                     "Buy" if net_foreign > 0 else "Sell")
        
        with col4:
            anomaly_count = (today_data['Big_Player_Anomaly'] > 5).sum()
            st.metric("Anomali Hari Ini", anomaly_count)
        
        # Top Gainers/Losers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🟢 Top Gainers")
            gainers = today_data.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume']].copy()
            gainers['Change %'] = gainers['Change %'].round(2)
            gainers['Volume'] = (gainers['Volume'] / 1e6).round(1)
            st.dataframe(gainers, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔴 Top Losers")
            losers = today_data.nsmallest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume']].copy()
            losers['Change %'] = losers['Change %'].round(2)
            losers['Volume'] = (losers['Volume'] / 1e6).round(1)
            st.dataframe(losers, use_container_width=True)
        
        # Most Active by Value
        st.markdown("#### 💰 Most Active by Value")
        most_active = today_data.nlargest(15, 'Value')[['Stock Code', 'Close', 'Value', 'Volume', 'Net Foreign Flow']].copy()
        most_active['Value'] = (most_active['Value'] / 1e9).round(1)
        most_active['Volume'] = (most_active['Volume'] / 1e6).round(1)
        most_active['Net Foreign Flow'] = (most_active['Net Foreign Flow'] / 1e9).round(1)
        most_active.columns = ['Kode', 'Harga', 'Nilai(M)', 'Volume(Jt)', 'Net Foreign(M)']
        
        fig = px.bar(most_active.head(10), x='Kode', y='Nilai(M)', 
                    color='Net Foreign(M)', color_continuous_scale='RdYlGn',
                    title="Top 10 by Transaction Value")
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Summary
        st.markdown("#### ⚡ Anomaly Summary")
        anomaly_stocks = today_data[today_data['Big_Player_Anomaly'] > 5].nlargest(10, 'Big_Player_Anomaly')
        if len(anomaly_stocks) > 0:
            anomaly_display = anomaly_stocks[['Stock Code', 'Close', 'Big_Player_Anomaly', 'Volume Spike (x)']].copy()
            anomaly_display.columns = ['Kode', 'Harga', 'Anomali', 'Volume Spike']
            st.dataframe(anomaly_display, use_container_width=True)
        else:
            st.info("Tidak ada anomaly hari ini")
    else:
        st.warning(f"Tidak ada data untuk tanggal {max_date.strftime('%d-%m-%Y')}")

# Footer
st.markdown("---")
st.caption(f"🔄 Last Update: {max_date.strftime('%d-%m-%Y')} | Data: {len(df_transaksi):,} rows")
