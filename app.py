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

# Konfigurasi Halaman
st.set_page_config(
    page_title="Bandarmology Master", 
    layout="wide", 
    page_icon="🐋",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2c5364;
    }
    
    .broker-badge {
        background: #e74c3c;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    .filter-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .signal-box {
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.2rem 0;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #f8fafc;
        padding: 0.4rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.4rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h2 style='margin:0;'>🐋 Bandarmology Master - Institutional Trading Intelligence</h2>
    <p style='margin:0; opacity:0.9;'>Broker Analysis • Ownership Concentration • Big Money Flow • Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data(ttl=3600)
def load_data():
    try:
        gcp_service_account = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            gcp_service_account,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        
        # Load transaksi
        request = service.files().get_media(fileId="1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8")
        df_transaksi = pd.read_csv(io.BytesIO(request.execute()))
        
        # Load kepemilikan
        request = service.files().get_media(fileId="1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr")
        df_kepemilikan = pd.read_csv(io.BytesIO(request.execute()))
        
        return df_transaksi, df_kepemilikan
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

with st.spinner("📊 Loading market intelligence..."):
    df_transaksi, df_kepemilikan = load_data()

if df_transaksi is None:
    st.stop()

# PREPROCESSING
st.caption("🔄 Processing data...")

# Konversi tanggal
df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'].astype(str), errors='coerce')
df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'].astype(str), errors='coerce')

# Drop invalid
df_transaksi = df_transaksi.dropna(subset=['Last Trading Date', 'Stock Code'])
df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data', 'Kode Efek'])

# Konversi numerik
numeric_cols = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                'Big_Player_Anomaly', 'Close', 'Volume Spike (x)', 'Avg_Order_Volume',
                'Tradeble Shares', 'Free Float', 'Typical Price', 'TPxV', 'Frequency',
                'Previous', 'Open Price', 'High', 'Low', 'Change %']
for col in numeric_cols:
    if col in df_transaksi.columns:
        df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce').fillna(0)

# Hitung metrik tambahan dengan aman
df_transaksi['Foreign_Pct'] = 0
df_transaksi['Volume_Pct_Tradeble'] = 0
df_transaksi['Value_Per_Order'] = 0

if 'Tradeble Shares' in df_transaksi.columns:
    mask_tradeble = df_transaksi['Tradeble Shares'] > 0
    df_transaksi.loc[mask_tradeble, 'Foreign_Pct'] = (
        (df_transaksi.loc[mask_tradeble, 'Foreign Buy'] - df_transaksi.loc[mask_tradeble, 'Foreign Sell']) / 
        df_transaksi.loc[mask_tradeble, 'Tradeble Shares'] * 100
    ).fillna(0)
    df_transaksi.loc[mask_tradeble, 'Volume_Pct_Tradeble'] = (
        df_transaksi.loc[mask_tradeble, 'Volume'] / df_transaksi.loc[mask_tradeble, 'Tradeble Shares'] * 100
    ).fillna(0)

if 'Frequency' in df_transaksi.columns:
    mask_freq = df_transaksi['Frequency'] > 0
    df_transaksi.loc[mask_freq, 'Value_Per_Order'] = (
        df_transaksi.loc[mask_freq, 'Value'] / df_transaksi.loc[mask_freq, 'Frequency']
    ).fillna(0)

# Data siap
unique_stocks = sorted(df_transaksi['Stock Code'].unique())
unique_brokers_ksei = sorted(df_kepemilikan['Kode Broker'].dropna().unique()) if 'Kode Broker' in df_kepemilikan.columns else []
min_date = df_transaksi['Last Trading Date'].min().date()
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham, {len(unique_brokers_ksei)} broker terdeteksi")

# ==================== 5 TABS ====================
tabs = st.tabs([
    "🎯 SCREENER PRO", 
    "🔍 DEEP DIVE", 
    "🏦 BROKER INTELLIGENCE",
    "👥 OWNERSHIP CONCENTRATION",
    "📊 MARKET MAP"
])

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    # Advanced Filters
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        row1 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
        with row1[0]:
            min_value = st.number_input("Min Nilai (M)", 0, 10000, 10) * 1e9
        with row1[1]:
            min_volume_pct = st.slider("Min Volume % Tradeble", 0.0, 10.0, 0.5, 0.1)
        with row1[2]:
            anomaly_threshold = st.slider("Min Anomali", 0, 20, 5)
        with row1[3]:
            foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy > 1M", "Net Sell > 1M", "Net Buy Kuat", "Net Sell Kuat"])
        with row1[4]:
            st.markdown("<br>", unsafe_allow_html=True)
            top_only = st.checkbox("Top 50 Only", value=True)
        
        row2 = st.columns(4)
        with row2[0]:
            min_price = st.number_input("Min Harga", 0, 100000, 50)
        with row2[1]:
            min_foreign_pct = st.slider("Min Foreign % Change", -10.0, 10.0, -1.0, 0.1, format="%.1f%%")
        with row2[2]:
            date_range = st.date_input("Periode", value=(default_start, max_date))
        with row2[3]:
            sort_by = st.selectbox("Sort By", ["Anomali", "Nilai", "Volume %", "Foreign Flow", "Volume Spike"])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        
        # Filter data
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if len(df_filter) > 0:
            # Siapkan dictionary untuk agregasi
            agg_dict = {
                'Close': 'last',
                'Change %': 'last',
                'Volume': 'sum',
                'Value': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'max',
                'Volume Spike (x)': 'max',
                'Tradeble Shares': 'last',
                'Free Float': 'last',
                'Avg_Order_Volume': 'mean',
                'Typical Price': 'mean',
                'TPxV': 'sum'
            }
            
            # Tambahkan kolom yang mungkin ada
            if 'Volume_Pct_Tradeble' in df_filter.columns:
                agg_dict['Volume_Pct_Tradeble'] = 'mean'
            if 'Foreign_Pct' in df_filter.columns:
                agg_dict['Foreign_Pct'] = 'mean'
            
            summary = df_filter.groupby('Stock Code').agg(agg_dict).reset_index()
            
            # Hitung metrik turunan dengan aman
            summary['Buying_Pressure'] = 0
            summary['Volume_Concentration'] = 0
            summary['Anomaly_Score'] = 0
            summary['Potential_Score'] = 0
            
            mask_value = summary['Value'] > 0
            summary.loc[mask_value, 'Buying_Pressure'] = (
                summary.loc[mask_value, 'Net Foreign Flow'] / summary.loc[mask_value, 'Value'] * 100
            ).fillna(0)
            
            mask_tradeble = summary['Tradeble Shares'] > 0
            summary.loc[mask_tradeble, 'Volume_Concentration'] = (
                summary.loc[mask_tradeble, 'Volume'] / summary.loc[mask_tradeble, 'Tradeble Shares'] * 100
            ).fillna(0)
            
            summary['Anomaly_Score'] = summary['Big_Player_Anomaly'] * summary['Volume Spike (x)']
            summary['Potential_Score'] = (
                summary['Volume_Concentration'] * 0.3 + 
                abs(summary['Buying_Pressure']) * 0.3 + 
                summary['Anomaly_Score'] * 0.4
            )
            
            # Apply filters
            summary = summary[summary['Value'] >= min_value]
            summary = summary[summary['Volume_Concentration'] >= min_volume_pct]
            summary = summary[summary['Big_Player_Anomaly'] >= anomaly_threshold]
            summary = summary[summary['Close'] >= min_price]
            
            if foreign_filter == "Net Buy > 1M":
                summary = summary[summary['Net Foreign Flow'] > 1e9]
            elif foreign_filter == "Net Sell > 1M":
                summary = summary[summary['Net Foreign Flow'] < -1e9]
            elif foreign_filter == "Net Buy Kuat" and 'Foreign_Pct' in summary.columns:
                summary = summary[summary['Foreign_Pct'] > 2]
            elif foreign_filter == "Net Sell Kuat" and 'Foreign_Pct' in summary.columns:
                summary = summary[summary['Foreign_Pct'] < -2]
            
            # Sort
            sort_map = {
                "Anomali": "Anomaly_Score",
                "Nilai": "Value",
                "Volume %": "Volume_Concentration",
                "Foreign Flow": "Net Foreign Flow",
                "Volume Spike": "Volume Spike (x)"
            }
            if sort_by in sort_map and sort_map[sort_by] in summary.columns:
                summary = summary.sort_values(sort_map[sort_by], ascending=False)
            
            if top_only:
                summary = summary.head(50)
            
            # Display
            st.markdown(f"**🎯 {len(summary)} saham ditemukan dengan potensi tinggi**")
            
            if len(summary) > 0:
                # Format untuk display
                display = summary.copy()
                display['Value'] = (display['Value'] / 1e9).round(1)
                display['Net Foreign Flow'] = (display['Net Foreign Flow'] / 1e9).round(1)
                display['Volume_Concentration'] = display['Volume_Concentration'].round(2)
                display['Buying_Pressure'] = display['Buying_Pressure'].round(1)
                display['Potential_Score'] = display['Potential_Score'].round(1)
                if 'Change %' in display.columns:
                    display['Change %'] = display['Change %'].round(2)
                
                # Select columns
                cols_display = ['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow', 
                               'Volume_Concentration', 'Buying_Pressure', 'Anomaly_Score', 
                               'Volume Spike (x)', 'Potential_Score']
                
                available_cols = [c for c in cols_display if c in display.columns]
                display = display[available_cols]
                
                # Rename columns
                col_names = {
                    'Stock Code': 'Kode',
                    'Close': 'Harga',
                    'Change %': 'Change%',
                    'Value': 'Nilai(M)',
                    'Net Foreign Flow': 'Foreign(M)',
                    'Volume_Concentration': 'Vol%Tradeble',
                    'Buying_Pressure': 'Pressure%',
                    'Anomaly_Score': 'Anomali',
                    'Volume Spike (x)': 'Spike',
                    'Potential_Score': 'Potensi'
                }
                display = display.rename(columns={k: v for k, v in col_names.items() if k in display.columns})
                
                st.dataframe(display, use_container_width=True, height=500)
                
                # Visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(summary) > 0 and 'Volume_Concentration' in summary.columns and 'Net Foreign Flow' in summary.columns:
                        fig = px.scatter(summary.head(20), x='Volume_Concentration', y='Net Foreign Flow',
                                       size='Potential_Score' if 'Potential_Score' in summary.columns else None, 
                                       color='Change %' if 'Change %' in summary.columns else None,
                                       hover_data=['Stock Code'],
                                       title="Volume Concentration vs Foreign Flow")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if 'Potential_Score' in summary.columns:
                        top_potential = summary.nlargest(15, 'Potential_Score')[['Stock Code', 'Potential_Score']]
                        fig = px.bar(top_potential, x='Stock Code', y='Potential_Score',
                                   title="Top 15 Highest Potential Stocks",
                                   color='Potential_Score', color_continuous_scale='Viridis')
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")
        else:
            st.warning("Tidak ada data untuk periode yang dipilih")

# ==================== TAB 2: DEEP DIVE ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dive_stock')
    with col2:
        period = st.selectbox("Periode", [7, 14, 30, 60, 90, 180], index=2)
    with col3:
        vs_tradeble = st.checkbox("Normalize to Tradeble", value=True)
    
    # Filter data
    start_dive = max_date - timedelta(days=period)
    mask_dive = (df_transaksi['Stock Code'] == selected_stock) & \
                (df_transaksi['Last Trading Date'].dt.date >= start_dive)
    df_dive = df_transaksi[mask_dive].copy().sort_values('Last Trading Date')
    
    if len(df_dive) > 0:
        latest = df_dive.iloc[-1]
        tradeble = latest['Tradeble Shares'] if vs_tradeble and latest['Tradeble Shares'] > 0 else 1
        
        # METRICS
        cols = st.columns(6)
        
        with cols[0]:
            st.metric("Harga", f"Rp{latest['Close']:,.0f}", f"{latest['Change %']:.1f}%")
        with cols[1]:
            vol_pct = (latest['Volume'] / tradeble * 100) if tradeble > 0 else 0
            st.metric("Volume % Tradeble", f"{vol_pct:.2f}%")
        with cols[2]:
            st.metric("Foreign Net", f"Rp{latest['Net Foreign Flow']/1e9:.1f}M")
        with cols[3]:
            st.metric("Anomali", f"{latest['Big_Player_Anomaly']:.1f}x")
        with cols[4]:
            st.metric("Avg Order", f"{latest['Avg_Order_Volume']:,.0f}")
        with cols[5]:
            foreign_pct = (latest['Net Foreign Flow'] / tradeble * 100) if tradeble > 0 else 0
            st.metric("Foreign % Tradeble", f"{foreign_pct:.3f}%")
        
        # CHART
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.35, 0.2, 0.2, 0.25],
            subplot_titles=("Price Action", "Volume Analysis", "Foreign Flow", "Big Player Activity")
        )
        
        # Price
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
        
        # Volume
        if vs_tradeble and tradeble > 0:
            volume_data = df_dive['Volume'] / tradeble * 100
            volume_title = "Volume % Tradeble"
        else:
            volume_data = df_dive['Volume'] / 1e6
            volume_title = "Volume (Jt)"
        
        colors = ['red' if row['Close'] < row['Open Price'] else 'green' for _, row in df_dive.iterrows()]
        fig.add_trace(
            go.Bar(
                x=df_dive['Last Trading Date'],
                y=volume_data,
                name="Volume",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Foreign Flow
        fig.add_trace(
            go.Bar(
                x=df_dive['Last Trading Date'],
                y=df_dive['Net Foreign Flow']/1e9,
                name="Net Foreign (M)",
                marker_color=['green' if x > 0 else 'red' for x in df_dive['Net Foreign Flow']],
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Big Player Anomaly
        fig.add_trace(
            go.Scatter(
                x=df_dive['Last Trading Date'],
                y=df_dive['Big_Player_Anomaly'],
                name="Anomaly",
                line=dict(color='purple', width=2),
                fill='tozeroy'
            ),
            row=4, col=1
        )
        
        if 'Volume Spike (x)' in df_dive.columns:
            fig.add_trace(
                go.Scatter(
                    x=df_dive['Last Trading Date'],
                    y=df_dive['Volume Spike (x)'],
                    name="Volume Spike",
                    line=dict(color='orange', width=2, dash='dash')
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            height=700,
            xaxis_rangeslider_visible=False,
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Harga", row=1, col=1)
        fig.update_yaxes(title_text=volume_title, row=2, col=1)
        fig.update_yaxes(title_text="Foreign (M)", row=3, col=1)
        fig.update_yaxes(title_text="Score", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # KSEI Ownership
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            st.markdown("### 👥 Institutional Ownership")
            
            ksei_data = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            if len(ksei_data) > 0:
                latest_ksei = ksei_data['Tanggal_Data'].max()
                ksei_latest = ksei_data[ksei_data['Tanggal_Data'] == latest_ksei]
                
                total_owned = ksei_latest['Jumlah Saham (Curr)'].sum()
                pct_owned = (total_owned / tradeble * 100) if tradeble > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total 5% Holders", f"{total_owned/1e6:.1f}Jt")
                col2.metric("% dari Tradeble", f"{pct_owned:.2f}%")
                col3.metric("Jumlah Holder", len(ksei_latest))
                
                if 'Kode Broker' in ksei_latest.columns:
                    broker_summary = ksei_latest.groupby('Kode Broker').agg({
                        'Jumlah Saham (Curr)': 'sum',
                        'Nama Pemegang Saham': 'count'
                    }).rename(columns={
                        'Jumlah Saham (Curr)': 'Total Saham',
                        'Nama Pemegang Saham': 'Jumlah Holder'
                    }).reset_index()
                    
                    broker_summary['Total Saham'] = broker_summary['Total Saham'] / 1e6
                    if tradeble > 0:
                        broker_summary['% Tradeble'] = (broker_summary['Total Saham'] * 1e6 / tradeble * 100).round(2)
                    broker_summary = broker_summary.sort_values('Total Saham', ascending=False)
                    
                    st.dataframe(broker_summary, use_container_width=True)
        
        # SIGNALS
        st.markdown("### 🚨 Trading Signals")
        
        signals = []
        
        if latest['Volume Spike (x)'] > 2:
            signals.append(("🔴 Volume Spike", f"{latest['Volume Spike (x)']:.1f}x normal"))
        
        if vs_tradeble and vol_pct > 5:
            signals.append(("📊 High Volume", f"{vol_pct:.2f}% of tradeble"))
        
        if latest['Big_Player_Anomaly'] > 5:
            signals.append(("🐋 Strong Anomaly", f"{latest['Big_Player_Anomaly']:.1f}x avg order"))
        
        if latest['Net Foreign Flow'] > 1e9:
            signals.append(("🟢 Big Foreign Buy", f"Rp{latest['Net Foreign Flow']/1e9:.1f}M"))
        elif latest['Net Foreign Flow'] < -1e9:
            signals.append(("🔴 Big Foreign Sell", f"Rp{abs(latest['Net Foreign Flow']/1e9):.1f}M"))
        
        if vs_tradeble and tradeble > 0 and 'pct_owned' in locals() and pct_owned > 20:
            signals.append(("👑 High Concentration", f"{pct_owned:.1f}% owned by 5% holders"))
        
        if len(df_dive) > 5:
            ma5 = df_dive['Close'].tail(5).mean()
            if latest['Close'] > ma5 * 1.07:
                signals.append(("🚀 Strong Momentum", f"+7% above MA5"))
            elif latest['Close'] < ma5 * 0.93:
                signals.append(("📉 Weak Momentum", f"-7% below MA5"))
        
        if signals:
            for sig in signals:
                if "🔴" in sig[0] or "🔴" in sig[1]:
                    st.warning(f"**{sig[0]}**: {sig[1]}")
                elif "🟢" in sig[0] or "🚀" in sig[0]:
                    st.success(f"**{sig[0]}**: {sig[1]}")
                else:
                    st.info(f"**{sig[0]}**: {sig[1]}")
        else:
            st.info("Tidak ada signal signifikan")

# ==================== TAB 3: BROKER INTELLIGENCE ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Intelligence - Institutional Footprint")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Top brokers
            broker_power = df_kepemilikan.groupby('Kode Broker').agg({
                'Jumlah Saham (Curr)': 'sum',
                'Nama Pemegang Saham': 'nunique',
                'Kode Efek': 'nunique'
            }).rename(columns={
                'Jumlah Saham (Curr)': 'Total Saham',
                'Nama Pemegang Saham': 'Unique Holders',
                'Kode Efek': 'Unique Stocks'
            }).reset_index()
            
            broker_power['Total Saham'] = broker_power['Total Saham'] / 1e9
            broker_power = broker_power.sort_values('Total Saham', ascending=False).head(20)
            
            st.markdown("#### 💪 Top Brokers by Assets")
            st.dataframe(broker_power, use_container_width=True)
        
        with col2:
            if len(broker_power) > 0:
                selected_broker = st.selectbox("Analisis Broker", broker_power['Kode Broker'].head(10).tolist())
                
                if selected_broker:
                    broker_stocks = df_kepemilikan[df_kepemilikan['Kode Broker'] == selected_broker].copy()
                    broker_stocks = broker_stocks.groupby('Kode Efek').agg({
                        'Jumlah Saham (Curr)': 'sum',
                        'Nama Pemegang Saham': 'count'
                    }).rename(columns={
                        'Jumlah Saham (Curr)': 'Total Saham',
                        'Nama Pemegang Saham': 'Jumlah Holder'
                    }).reset_index()
                    
                    broker_stocks['Total Saham'] = broker_stocks['Total Saham'] / 1e6
                    broker_stocks = broker_stocks.sort_values('Total Saham', ascending=False).head(15)
                    
                    if len(broker_stocks) > 0:
                        fig = px.bar(broker_stocks, x='Kode Efek', y='Total Saham',
                                   title=f"Top Holdings - Broker {selected_broker}",
                                   color='Total Saham', color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)
        
        # Timeline
        st.markdown("#### 📈 Broker Activity Timeline")
        
        if len(broker_power) > 0:
            top_brokers = broker_power.head(5)['Kode Broker'].tolist()
            timeline_data = df_kepemilikan[df_kepemilikan['Kode Broker'].isin(top_brokers)].copy()
            if len(timeline_data) > 0:
                timeline_data = timeline_data.groupby(['Tanggal_Data', 'Kode Broker'])['Jumlah Saham (Curr)'].sum().reset_index()
                timeline_data['Jumlah Saham (Curr)'] = timeline_data['Jumlah Saham (Curr)'] / 1e9
                
                fig = px.line(timeline_data, x='Tanggal_Data', y='Jumlah Saham (Curr)',
                             color='Kode Broker', title="Top Brokers Holdings Timeline (Rp Miliar)",
                             markers=True)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data broker tidak tersedia")

# ==================== TAB 4: OWNERSHIP CONCENTRATION ====================
with tabs[3]:
    st.markdown("### 👑 Ownership Concentration Analysis")
    
    if len(df_kepemilikan) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stocks with highest concentration
            latest_date = df_kepemilikan['Tanggal_Data'].max()
            latest_ownership = df_kepemilikan[df_kepemilikan['Tanggal_Data'] == latest_date].copy()
            
            if len(latest_ownership) > 0:
                # Gabung dengan tradeble shares
                tradeble_info = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date][['Stock Code', 'Tradeble Shares']].drop_duplicates('Stock Code')
                
                if len(tradeble_info) > 0:
                    concentration = latest_ownership.groupby('Kode Efek')['Jumlah Saham (Curr)'].sum().reset_index()
                    concentration = concentration.merge(tradeble_info, left_on='Kode Efek', right_on='Stock Code', how='left')
                    
                    # Hitung persentase dengan aman
                    concentration['Pct_of_Tradeble'] = 0
                    mask_tradeble = concentration['Tradeble Shares'] > 0
                    concentration.loc[mask_tradeble, 'Pct_of_Tradeble'] = (
                        concentration.loc[mask_tradeble, 'Jumlah Saham (Curr)'] / 
                        concentration.loc[mask_tradeble, 'Tradeble Shares'] * 100
                    ).round(2)
                    
                    concentration = concentration.sort_values('Pct_of_Tradeble', ascending=False).head(20)
                    
                    st.markdown("#### 🎯 Highest Concentration Stocks")
                    
                    fig = px.bar(concentration, x='Kode Efek', y='Pct_of_Tradeble',
                               title="% of Tradeble Shares Held by 5% Owners",
                               color='Pct_of_Tradeble', color_continuous_scale='Reds')
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution
            st.markdown("#### 📊 Ownership Distribution")
            
            if len(latest_ownership) > 0:
                # Hitung dalam juta dengan benar
                ownership_values = latest_ownership['Jumlah Saham (Curr)'] / 1e6
                
                fig = px.histogram(
                    x=ownership_values,
                    nbins=50, 
                    title="Distribution of Holdings Size (Juta Saham)",
                    labels={'x': 'Jumlah Saham (Juta)', 'y': 'Frequency'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Stock picker
        st.markdown("#### 🔍 Stock Ownership Details")
        
        if len(latest_ownership) > 0:
            own_stock = st.selectbox("Pilih Saham", sorted(latest_ownership['Kode Efek'].unique()), key='own_stock')
            
            if own_stock:
                stock_owners = latest_ownership[latest_ownership['Kode Efek'] == own_stock].copy()
                
                # Dapatkan tradeble value
                tradeble_value = 1
                if len(tradeble_info) > 0:
                    tradeble_match = tradeble_info[tradeble_info['Stock Code'] == own_stock]['Tradeble Shares'].values
                    tradeble_value = tradeble_match[0] if len(tradeble_match) > 0 and tradeble_match[0] > 0 else 1
                
                # Hitung metrik
                stock_owners['% Tradeble'] = 0
                if tradeble_value > 0:
                    stock_owners['% Tradeble'] = (stock_owners['Jumlah Saham (Curr)'] / tradeble_value * 100).round(3)
                
                stock_owners['Jumlah Saham (Curr)'] = stock_owners['Jumlah Saham (Curr)'] / 1e6
                
                display_cols = ['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)', '% Tradeble', 'Status']
                available_cols = [c for c in display_cols if c in stock_owners.columns]
                display_owners = stock_owners[available_cols].copy()
                
                if 'Jumlah Saham (Curr)' in display_owners.columns:
                    display_owners = display_owners.sort_values('Jumlah Saham (Curr)', ascending=False)
                
                display_owners.columns = [c.replace('_', ' ') for c in display_owners.columns]
                
                st.dataframe(display_owners, use_container_width=True)
                
                # Analisis konsentrasi
                total_concentration = stock_owners['Jumlah Saham (Curr)'].sum() * 1e6
                pct_concentration = (total_concentration / tradeble_value * 100) if tradeble_value > 0 else 0
                
                st.markdown(f"""
                <div class="signal-box" style="background: {'#ffeb3b' if pct_concentration > 30 else '#e3f2fd'}">
                    <strong>📊 Ownership Analysis:</strong><br>
                    - Total 5% Holdings: {total_concentration/1e6:.1f}Jt ({pct_concentration:.1f}% of tradeble)<br>
                    - Jumlah Major Holders: {len(stock_owners)}<br>
                    - Unique Brokers: {stock_owners['Kode Broker'].nunique() if 'Kode Broker' in stock_owners.columns else 'N/A'}<br>
                    - <strong>Potensi Terbang: {'TINGGI' if pct_concentration > 40 else 'SEDANG' if pct_concentration > 20 else 'RENDAH'}</strong>
                </div>
                """, unsafe_allow_html=True)

# ==================== TAB 5: MARKET MAP ====================
with tabs[4]:
    st.markdown("### 🗺️ Market Heatmap & Summary")
    
    # Data hari ini
    today_data = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date].copy()
    
    if len(today_data) > 0:
        # Market Overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = today_data['Value'].sum() / 1e9
            st.metric("Total Nilai", f"Rp{total_value:,.0f}M")
        
        with col2:
            total_volume = today_data['Volume'].sum() / 1e6
            st.metric("Total Volume", f"{total_volume:,.0f}Jt")
        
        with col3:
            net_foreign = today_data['Net Foreign Flow'].sum() / 1e9
            st.metric("Net Foreign", f"Rp{net_foreign:,.0f}M", 
                     "🟢 Buy" if net_foreign > 0 else "🔴 Sell")
        
        with col4:
            anomaly_count = (today_data['Big_Player_Anomaly'] > 5).sum()
            st.metric("Anomali", f"{anomaly_count} saham")
        
        # Market Heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Sector' in today_data.columns:
                sector_perf = today_data.groupby('Sector').agg({
                    'Change %': 'mean',
                    'Value': 'sum'
                }).reset_index()
                
                fig = px.treemap(sector_perf, path=['Sector'], values='Value',
                               color='Change %', color_continuous_scale='RdYlGn',
                               title="Sector Performance Heatmap")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Foreign flow map - PERBAIKAN: Buat kolom absolute value terlebih dahulu
            foreign_map = today_data.groupby('Stock Code').agg({
                'Net Foreign Flow': 'sum',
                'Value': 'sum'
            }).reset_index()
            
            # Buat kolom absolute value
            foreign_map['Abs_Net_Foreign'] = abs(foreign_map['Net Foreign Flow'])
            foreign_map = foreign_map.nlargest(20, 'Abs_Net_Foreign')
            
            fig = px.scatter(foreign_map, x='Value', y='Net Foreign Flow',
                           text='Stock Code', size='Abs_Net_Foreign',
                           color='Net Foreign Flow', color_continuous_scale='RdYlGn',
                           title="Foreign Flow Map")
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top Movers
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🚀 Top Gainers")
            gainers = today_data.nlargest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume', 'Net Foreign Flow']].copy()
            if len(gainers) > 0:
                gainers['Change %'] = gainers['Change %'].round(2)
                gainers['Volume'] = (gainers['Volume'] / 1e6).round(1)
                gainers['Net Foreign Flow'] = (gainers['Net Foreign Flow'] / 1e9).round(1)
                gainers.columns = ['Kode', 'Harga', 'Change%', 'Volume(Jt)', 'Foreign(M)']
                st.dataframe(gainers, use_container_width=True)
        
        with col2:
            st.markdown("#### 📉 Top Losers")
            losers = today_data.nsmallest(10, 'Change %')[['Stock Code', 'Close', 'Change %', 'Volume', 'Net Foreign Flow']].copy()
            if len(losers) > 0:
                losers['Change %'] = losers['Change %'].round(2)
                losers['Volume'] = (losers['Volume'] / 1e6).round(1)
                losers['Net Foreign Flow'] = (losers['Net Foreign Flow'] / 1e9).round(1)
                losers.columns = ['Kode', 'Harga', 'Change%', 'Volume(Jt)', 'Foreign(M)']
                st.dataframe(losers, use_container_width=True)
        
        # Most Active & Anomaly
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💰 Most Active by Value")
            active = today_data.nlargest(10, 'Value')[['Stock Code', 'Close', 'Value', 'Volume']].copy()
            if len(active) > 0:
                active['Value'] = (active['Value'] / 1e9).round(1)
                active['Volume'] = (active['Volume'] / 1e6).round(1)
                active.columns = ['Kode', 'Harga', 'Nilai(M)', 'Volume(Jt)']
                st.dataframe(active, use_container_width=True)
        
        with col2:
            st.markdown("#### ⚡ Top Anomalies")
            anomalies = today_data[today_data['Big_Player_Anomaly'] > 3].nlargest(10, 'Big_Player_Anomaly')
            if len(anomalies) > 0:
                anomalies_display = anomalies[['Stock Code', 'Close', 'Big_Player_Anomaly', 'Volume Spike (x)']].copy()
                anomalies_display.columns = ['Kode', 'Harga', 'Anomali', 'Spike']
                st.dataframe(anomalies_display, use_container_width=True)
            else:
                st.info("Tidak ada anomaly signifikan")
        
        # Market Breadth
        st.markdown("#### 📊 Market Breadth")
        
        advance = (today_data['Change %'] > 0).sum()
        decline = (today_data['Change %'] < 0).sum()
        unchanged = (today_data['Change %'] == 0).sum()
        total = len(today_data)
        
        if total > 0:
            col1, col2, col3 = st.columns(3)
            col1.metric("Advancing", advance, f"{(advance/total*100):.0f}%")
            col2.metric("Declining", decline, f"{(decline/total*100):.0f}%")
            col3.metric("Unchanged", unchanged, f"{(unchanged/total*100):.0f}%")
            
            fig = go.Figure(data=[go.Pie(labels=['Advance', 'Decline', 'Unchanged'],
                                        values=[advance, decline, unchanged],
                                        marker_colors=['green', 'red', 'gray'])])
            fig.update_layout(height=300, title="Market Breadth")
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(f"Tidak ada data untuk {max_date.strftime('%d-%m-%Y')}")

# Footer
st.markdown("---")
st.caption(f"🔄 Last Update: {max_date.strftime('%d-%m-%Y')} | Total Data: {len(df_transaksi):,} rows | Broker Tracked: {len(unique_brokers_ksei)}")

# Auto-refresh button
if st.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()
