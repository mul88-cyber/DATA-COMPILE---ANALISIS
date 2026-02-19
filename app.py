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
    .big-number {
        font-size: 1.2rem;
        font-weight: 600;
        color: #004e92;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 2rem;'>🐋 Bandarmology Master V3</h1>
    <p style='margin:0; opacity:0.8; font-size: 1rem;'>Deep Dive Analytics • Multi-Timeframe • Foreign Flow Radar • Spike Detection</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOAD DATA & PREPROCESSING
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
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
        
        return df_transaksi, df_kepemilikan
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

with st.spinner("📊 Memuat Data Bandarmology..."):
    df_transaksi, df_kepemilikan = load_data()

if df_transaksi is None: 
    st.stop()

# --- PREPROCESSING ---
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

# Metrik Tambahan
if 'Tradeble Shares' in df_transaksi.columns:
    df_transaksi['Volume_Pct_Tradeble'] = np.where(
        df_transaksi['Tradeble Shares'] > 0, 
        (df_transaksi['Volume'] / df_transaksi['Tradeble Shares']) * 100, 
        0
    )
else:
    df_transaksi['Volume_Pct_Tradeble'] = 0

# Hitung AOVol Spike (jika belum ada)
if 'AOVol_Spike' not in df_transaksi.columns and 'Avg_Order_Volume' in df_transaksi.columns:
    df_transaksi['AOVol_Spike'] = df_transaksi['Avg_Order_Volume'] / df_transaksi['Avg_Order_Volume'].rolling(20, min_periods=1).mean()
    df_transaksi['AOVol_Spike'] = df_transaksi['AOVol_Spike'].fillna(1)

unique_stocks = sorted(df_transaksi['Stock Code'].unique())
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
tabs = st.tabs([
    "🎯 SCREENER PRO", 
    "🔍 DEEP DIVE & CHART", 
    "🏦 BROKER MUTASI",
    "🗺️ MARKET MAP"
])

# ==================== TAB 1: SCREENER PRO (LENGKAP) ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        # Row 1 - Filter Utama
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            min_value = st.number_input("Min Nilai Transaksi (M)", 0, 10000, 10) * 1e9
        with r1c2:
            min_volume = st.number_input("Min Volume (Juta)", 0, 10000, 100) * 1e6
        with r1c3:
            min_anomali = st.slider("Min Big Player Anomali (x)", 0, 20, 3)
        with r1c4:
            min_spike = st.slider("Min Volume Spike (x)", 0.0, 5.0, 1.5, 0.1)
        
        # Row 2 - Filter Foreign & Periode
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy", "Net Sell", "Net Buy > 10M", "Net Sell > 10M"])
        with r2c2:
            min_vol_pct = st.slider("Min Volume % Tradeble", 0.0, 10.0, 0.5, 0.1)
        with r2c3:
            min_price = st.number_input("Min Harga", 0, 100000, 50)
        with r2c4:
            date_range = st.date_input("Periode Analisis", value=(default_start, max_date))
        
        # Row 3 - Filter Tambahan
        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        with r3c1:
            min_aoVol = st.slider("Min AOVol Spike", 0.0, 5.0, 1.5, 0.1)
        with r3c2:
            top_only = st.checkbox("Hanya Top 100", value=True)
        with r3c3:
            sort_by = st.selectbox("Urutkan Berdasarkan", 
                                  ["Inst Score", "Anomali", "Volume %", "Foreign Flow", "Nilai"])
        with r3c4:
            st.markdown("<br>", unsafe_allow_html=True)
            show_details = st.checkbox("Tampilkan Semua Kolom", value=False)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if not df_filter.empty:
            # Agregasi lengkap
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
                'Tradeble Shares': 'last',
                'Free Float': 'last'
            }).reset_index()
            
            # Hitung metrik tambahan
            summary['Pressure'] = np.where(summary['Value'] > 0, 
                                          (summary['Net Foreign Flow'] / summary['Value'] * 100), 0)
            summary['AOVol_Spike'] = summary['Avg_Order_Volume'] / summary['Avg_Order_Volume'].mean()
            summary['Inst_Score'] = (
                summary['Volume_Pct_Tradeble'] * 0.3 + 
                summary['Big_Player_Anomaly'] * 0.3 + 
                abs(summary['Pressure']) * 0.2 +
                summary['AOVol_Spike'] * 0.2
            )
            
            # Apply filters
            summary = summary[summary['Value'] >= min_value]
            summary = summary[summary['Volume'] >= min_volume]
            summary = summary[summary['Big_Player_Anomaly'] >= min_anomali]
            summary = summary[summary['Volume Spike (x)'] >= min_spike]
            summary = summary[summary['Volume_Pct_Tradeble'] >= min_vol_pct]
            summary = summary[summary['Close'] >= min_price]
            
            if 'AOVol_Spike' in summary.columns:
                summary = summary[summary['AOVol_Spike'] >= min_aoVol]
            
            # Foreign filters
            if foreign_filter == "Net Buy":
                summary = summary[summary['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell":
                summary = summary[summary['Net Foreign Flow'] < 0]
            elif foreign_filter == "Net Buy > 10M":
                summary = summary[summary['Net Foreign Flow'] > 10e9]
            elif foreign_filter == "Net Sell > 10M":
                summary = summary[summary['Net Foreign Flow'] < -10e9]
            
            # Sorting
            sort_map = {
                "Inst Score": "Inst_Score",
                "Anomali": "Big_Player_Anomaly",
                "Volume %": "Volume_Pct_Tradeble",
                "Foreign Flow": "Net Foreign Flow",
                "Nilai": "Value"
            }
            summary = summary.sort_values(sort_map[sort_by], ascending=False)
            
            if top_only:
                summary = summary.head(100)
            
            st.markdown(f"**🎯 Ditemukan {len(summary)} saham dengan kriteria terpilih**")
            
            if len(summary) > 0:
                # Format display
                if show_details:
                    display_cols = ['Stock Code', 'Close', 'Change %', 'Value', 'Volume', 
                                   'Net Foreign Flow', 'Volume_Pct_Tradeble', 'Big_Player_Anomaly',
                                   'Volume Spike (x)', 'AOVol_Spike', 'Pressure', 'Inst_Score']
                else:
                    display_cols = ['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow',
                                   'Volume_Pct_Tradeble', 'Big_Player_Anomaly', 'Inst_Score']
                
                available_cols = [c for c in display_cols if c in summary.columns]
                display_df = summary[available_cols].copy()
                
                # Format numbers with thousand separators
                for col in display_df.columns:
                    if col in ['Value', 'Net Foreign Flow', 'Volume']:
                        display_df[col] = display_df[col].apply(lambda x: f"Rp {x:,.0f}")
                    elif col == 'Close':
                        display_df[col] = display_df[col].apply(lambda x: f"Rp {x:,.0f}")
                    elif col == 'Change %':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                    elif col in ['Volume_Pct_Tradeble', 'Pressure']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
                    elif col in ['Big_Player_Anomaly', 'Volume Spike (x)', 'AOVol_Spike', 'Inst_Score']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x")
                
                # Rename columns
                col_names = {
                    'Stock Code': 'Kode',
                    'Close': 'Harga',
                    'Change %': 'Change',
                    'Value': 'Nilai',
                    'Volume': 'Volume',
                    'Net Foreign Flow': 'Foreign',
                    'Volume_Pct_Tradeble': 'Vol%Trade',
                    'Big_Player_Anomaly': 'Anomali',
                    'Volume Spike (x)': 'Spike',
                    'AOVol_Spike': 'AOVol',
                    'Pressure': 'Tekanan',
                    'Inst_Score': 'Score'
                }
                display_df = display_df.rename(columns={k: v for k, v in col_names.items() if k in display_df.columns})
                
                st.dataframe(display_df, use_container_width=True, height=500)
                
                # Visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.scatter(summary.head(30), x='Volume_Pct_Tradeble', y='Net Foreign Flow',
                                   size='Inst_Score', color='Change %',
                                   hover_data=['Stock Code'],
                                   title="Volume Concentration vs Foreign Flow",
                                   color_continuous_scale='RdYlGn')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    top_potential = summary.nlargest(15, 'Inst_Score')[['Stock Code', 'Inst_Score']]
                    fig = px.bar(top_potential, x='Stock Code', y='Inst_Score',
                               title="Top 15 Institutional Score",
                               color='Inst_Score', color_continuous_scale='Viridis')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")
        else:
            st.info("Tidak ada data untuk periode ini")

# ==================== TAB 2: DEEP DIVE & CHART (DENGAN BINTANG) ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics")
    
    # Control Panel
    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([2, 1, 1, 1])
    with c_sel1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dd_stock_select')
    with c_sel2:
        interval = st.selectbox("Interval", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with c_sel3:
        chart_len = st.selectbox("Jumlah Data", [60, 90, 120, 180, 365], index=2, key='dd_len')
    with c_sel4:
        spike_threshold = st.slider("Threshold Anomali", 1.0, 5.0, 2.0, 0.5, key='spike_th')

    # Data Filter
    df_dive = df_transaksi[df_transaksi['Stock Code'] == selected_stock].copy().sort_values('Last Trading Date')
    
    if not df_dive.empty:
        # Resampling logic
        if interval == "Weekly":
            df_resampled = df_dive.set_index('Last Trading Date').resample('W').agg({
                'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
                'Volume Spike (x)': 'max', 'Avg_Order_Volume': 'mean'
            }).dropna().reset_index()
        elif interval == "Monthly":
            df_resampled = df_dive.set_index('Last Trading Date').resample('M').agg({
                'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
                'Volume Spike (x)': 'max', 'Avg_Order_Volume': 'mean'
            }).dropna().reset_index()
        else:
            df_resampled = df_dive.copy()

        df_chart = df_resampled.tail(chart_len)
        latest = df_dive.iloc[-1]

        # KPI Cards
        total_foreign = df_chart['Net Foreign Flow'].sum()
        price_change = df_chart['Close'].iloc[-1] - df_chart['Close'].iloc[0] if len(df_chart) > 1 else 0
        
        status_text = "NEUTRAL"
        status_color = "gray"
        if total_foreign > 0 and price_change >= 0: 
            status_text = "AKUMULASI"; status_color = "green"
        elif total_foreign < 0 and price_change < 0: 
            status_text = "DISTRIBUSI"; status_color = "red"
        elif total_foreign > 0 and price_change < 0: 
            status_text = "DIV. POSITIF"; status_color = "blue"
        elif total_foreign < 0 and price_change > 0: 
            status_text = "MARKUP RITEL"; status_color = "orange"

        spike_vol_count = len(df_chart[df_chart['Volume Spike (x)'] > spike_threshold])
        spike_anom_count = len(df_chart[df_chart['Big_Player_Anomaly'] > spike_threshold])

        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: 
            st.metric("Harga", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{status_color}'>{status_text}</div><div class='kpi-label'>Status</div></div>", unsafe_allow_html=True)
        with k3: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{spike_anom_count}</div><div class='kpi-label'>Anomali >{spike_threshold}x</div></div>", unsafe_allow_html=True)
        with k4: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{spike_vol_count}</div><div class='kpi-label'>Spike >{spike_threshold}x</div></div>", unsafe_allow_html=True)
        with k5: 
            st.metric("Total Foreign", f"Rp {total_foreign/1e9:,.1f} M")

        st.divider()

        # ADVANCED CHART DENGAN BINTANG CERAH
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05, 
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f"Price Action dengan Big Player Signal ({interval})", 
                "Volume Analysis", 
                "Net Foreign Flow"
            )
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_chart['Last Trading Date'],
            open=df_chart['Open Price'], 
            high=df_chart['High'],
            low=df_chart['Low'], 
            close=df_chart['Close'],
            name="OHLC",
            showlegend=False,
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)

        # BINTANG UNTUK AOVol SPIKING (WARNA CERAH)
        if 'Avg_Order_Volume' in df_chart.columns:
            # Hitung AOVol spike jika belum ada
            if 'AOVol_Spike' not in df_chart.columns:
                df_chart['AOVol_Spike'] = df_chart['Avg_Order_Volume'] / df_chart['Avg_Order_Volume'].rolling(5, min_periods=1).mean()
            
            # Titik dengan AOVol spike tinggi
            aovol_spikes = df_chart[df_chart['AOVol_Spike'] > spike_threshold]
            if not aovol_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=aovol_spikes['Last Trading Date'],
                    y=aovol_spikes['High'] * 1.03,  # Naikkan posisi bintang
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        size=14,
                        color='gold',
                        line=dict(width=2, color='orange')
                    ),
                    name=f'AOVol Spike >{spike_threshold}x',
                    text=[f"AOVol: {x:.1f}x<br>Harga: Rp {y:,.0f}" 
                          for x, y in zip(aovol_spikes['AOVol_Spike'], aovol_spikes['Close'])],
                    hoverinfo='text'
                ), row=1, col=1)

        # BINTANG UNTUK BIG PLAYER ANOMALI (WARNA CERAH)
        if 'Big_Player_Anomaly' in df_chart.columns:
            anomaly_points = df_chart[df_chart['Big_Player_Anomaly'] > spike_threshold]
            if not anomaly_points.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_points['Last Trading Date'],
                    y=anomaly_points['Low'] * 0.97,  # Turunkan posisi bintang
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=12,
                        color='magenta',
                        line=dict(width=2, color='purple')
                    ),
                    name=f'Big Player >{spike_threshold}x',
                    text=[f"Anomali: {x:.1f}x<br>Harga: Rp {y:,.0f}" 
                          for x, y in zip(anomaly_points['Big_Player_Anomaly'], anomaly_points['Close'])],
                    hoverinfo='text'
                ), row=1, col=1)

        # Volume bars
        colors_vol = ['red' if row['Close'] < row['Open Price'] else 'green' for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'],
            y=df_chart['Volume'] / 1e6,
            name="Volume (Jt)",
            marker_color=colors_vol,
            showlegend=False
        ), row=2, col=1)

        # Foreign Flow
        colors_ff = ['green' if val >= 0 else 'red' for val in df_chart['Net Foreign Flow']]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'],
            y=df_chart['Net Foreign Flow'] / 1e9,
            name="Foreign (M)",
            marker_color=colors_ff,
            showlegend=False
        ), row=3, col=1)

        # Layout dengan rangebreaks untuk skip weekend
        fig.update_layout(
            height=800,
            hovermode="x unified",
            margin=dict(t=50, b=40, l=40, r=40),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            title_text="Tanggal",
            row=3, col=1
        )
        
        fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (Juta)", row=2, col=1)
        fig.update_yaxes(title_text="Foreign (Miliar)", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

        # Legenda sinyal
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("⭐ **Bintang Kuning**: AOVol Spike (Average Order Volume melonjak)")
        with col2:
            st.markdown("💎 **Diamond Pink**: Big Player Anomali")
        with col3:
            st.markdown(f"📊 Threshold: {spike_threshold}x dari normal")

        # Broker Activity
        st.subheader("🕵️ Aktivitas Broker (KSEI 5%)")
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            ksei_stock = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            if not ksei_stock.empty:
                ksei_latest = ksei_stock[ksei_stock['Tanggal_Data'] == ksei_stock['Tanggal_Data'].max()]
                
                if not ksei_latest.empty:
                    display_ksei = ksei_latest[['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)', 'Status']].copy()
                    display_ksei['Jumlah Saham (Curr)'] = display_ksei['Jumlah Saham (Curr)'].apply(lambda x: f"{x:,.0f}")
                    display_ksei.columns = ['Pemegang Saham', 'Broker', 'Jumlah Lembar', 'Status']
                    st.dataframe(display_ksei, use_container_width=True, hide_index=True)

# ==================== TAB 3: BROKER MUTASI ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Mutation Radar")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        with st.container():
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                mutasi_period = st.selectbox("Periode Mutasi", ["1 Minggu", "2 Minggu", "1 Bulan", "3 Bulan"], key='m_period')
                days_map = {"1 Minggu": 7, "2 Minggu": 14, "1 Bulan": 30, "3 Bulan": 90}
            with col_b2:
                min_mutasi = st.number_input("Min Mutasi (Juta)", 0, 1000, 10) * 1e6
            with col_b3:
                top_n = st.slider("Top N Broker", 5, 30, 15)
        
        start_mutasi = df_kepemilikan['Tanggal_Data'].max() - timedelta(days=days_map[mutasi_period])
        df_ksei_period = df_kepemilikan[df_kepemilikan['Tanggal_Data'] >= start_mutasi].copy()
        
        if not df_ksei_period.empty:
            mutasi = df_ksei_period.sort_values('Tanggal_Data').groupby(['Kode Broker', 'Kode Efek']).agg(
                Awal=('Jumlah Saham (Curr)', 'first'),
                Akhir=('Jumlah Saham (Curr)', 'last')
            ).reset_index()
            
            mutasi['Net_Change'] = mutasi['Akhir'] - mutasi['Awal']
            mutasi = mutasi[abs(mutasi['Net_Change']) >= min_mutasi]
            
            broker_summary = mutasi.groupby('Kode Broker')['Net_Change'].sum().reset_index()
            broker_summary = broker_summary[abs(broker_summary['Net_Change']) >= min_mutasi]
            
            top_acc = broker_summary.nlargest(top_n, 'Net_Change')
            top_dist = broker_summary.nsmallest(top_n, 'Net_Change')
            top_dist['Net_Change'] = abs(top_dist['Net_Change'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟢 Top Accumulator")
                if not top_acc.empty:
                    fig_acc = px.bar(top_acc, x='Net_Change', y='Kode Broker', orientation='h',
                                   title=f"Top {top_n} Accumulator ({mutasi_period})",
                                   color='Net_Change', color_continuous_scale='Greens')
                    fig_acc.update_layout(xaxis=dict(tickformat=",.0f"))
                    fig_acc.update_xaxes(title_text="Net Buy (Lembar)")
                    st.plotly_chart(fig_acc, use_container_width=True)
            
            with c2:
                st.markdown("#### 🔴 Top Distributor")
                if not top_dist.empty:
                    fig_dist = px.bar(top_dist, x='Net_Change', y='Kode Broker', orientation='h',
                                    title=f"Top {top_n} Distributor ({mutasi_period})",
                                    color='Net_Change', color_continuous_scale='Reds')
                    fig_dist.update_layout(xaxis=dict(tickformat=",.0f"))
                    fig_dist.update_xaxes(title_text="Net Sell (Lembar)")
                    st.plotly_chart(fig_dist, use_container_width=True)
                
            st.divider()
            
            if len(broker_summary) > 0:
                sel_broker = st.selectbox("🔍 Detail Mutasi per Broker", 
                                         sorted(broker_summary['Kode Broker'].unique()), 
                                         key='m_broker')
                detail = mutasi[mutasi['Kode Broker'] == sel_broker].sort_values('Net_Change', ascending=False)
                
                if not detail.empty:
                    detail_display = detail[['Kode Efek', 'Awal', 'Akhir', 'Net_Change']].copy()
                    detail_display['Awal'] = detail_display['Awal'].apply(lambda x: f"{x:,.0f}")
                    detail_display['Akhir'] = detail_display['Akhir'].apply(lambda x: f"{x:,.0f}")
                    detail_display['Net_Change'] = detail_display['Net_Change'].apply(lambda x: f"{x:+,.0f}")
                    detail_display.columns = ['Saham', 'Awal (lembar)', 'Akhir (lembar)', 'Net Mutasi']
                    st.dataframe(detail_display, use_container_width=True, hide_index=True)
        else:
            st.warning("Tidak ada data KSEI dalam periode ini")
    else:
        st.warning("Data broker tidak tersedia")

# ==================== TAB 4: MARKET MAP ====================
with tabs[3]:
    st.markdown("### 🗺️ Market Map & Foreign Radar")
    
    # Foreign Flow Timeframe
    st.markdown("#### 🌍 Top Foreign Flow")
    
    col_period, col_top = st.columns(2)
    with col_period:
        ff_period = st.selectbox("Rentang Waktu", 
                                ["Hari Ini", "5 Hari", "10 Hari", "20 Hari", "30 Hari", "60 Hari"], 
                                key='ff_time')
    with col_top:
        top_n_ff = st.slider("Top N", 5, 30, 20, key='top_ff')
    
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
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f"#### 🟢 Top {top_n_ff} Foreign Buy ({ff_period})")
            top_buy = ff_data.nlargest(top_n_ff, 'Net Foreign Flow')
            if not top_buy.empty:
                display_buy = top_buy[['Stock Code', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_buy['Close'] = display_buy['Close'].apply(lambda x: f"Rp {x:,.0f}")
                display_buy['Net Foreign Flow'] = display_buy['Net Foreign Flow'].apply(lambda x: f"Rp {x:,.0f}")
                display_buy['Change %'] = display_buy['Change %'].apply(lambda x: f"{x:.2f}%")
                display_buy.columns = ['Kode', 'Harga', 'Net Buy', 'Change %']
                st.dataframe(display_buy, use_container_width=True, hide_index=True)
            
        with col_f2:
            st.markdown(f"#### 🔴 Top {top_n_ff} Foreign Sell ({ff_period})")
            top_sell = ff_data.nsmallest(top_n_ff, 'Net Foreign Flow')
            if not top_sell.empty:
                display_sell = top_sell[['Stock Code', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_sell['Close'] = display_sell['Close'].apply(lambda x: f"Rp {x:,.0f}")
                display_sell['Net Foreign Flow'] = display_sell['Net Foreign Flow'].apply(lambda x: f"Rp {x:,.0f}")
                display_sell['Change %'] = display_sell['Change %'].apply(lambda x: f"{x:.2f}%")
                display_sell.columns = ['Kode', 'Harga', 'Net Sell', 'Change %']
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
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("Tidak ada data transaksi untuk rentang waktu ini.")

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
