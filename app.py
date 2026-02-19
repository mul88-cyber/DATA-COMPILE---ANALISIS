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
    .main-header { background: linear-gradient(90deg, #000428, #004e92); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); border-left: 5px solid #004e92; }
    .kpi-card { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 15px; text-align: center; }
    .kpi-value { font-size: 24px; font-weight: bold; color: #004e92; }
    .kpi-label { font-size: 14px; color: #6c757d; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: white; padding: 10px; border-radius: 10px; border: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { border-radius: 5px; padding: 8px 16px; font-weight: 600; }
    .filter-container { background: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 1rem; }
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
        # Ganti dengan st.secrets Anda di production
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

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: 
            min_value = st.number_input("Min Transaksi (Miliar)", 0, 1000, 10, key='s_val') * 1e9
        with c2: 
            min_anomali = st.slider("Min Anomali AOV (x)", 0, 20, 2, key='s_anom')
        with c3: 
            foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy", "Net Sell"], key='s_ff')
        with c4: 
            date_range = st.date_input("Periode Agregasi", value=(default_start, max_date), key='s_date')
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if not df_filter.empty:
            # Agregasi
            summary = df_filter.groupby('Stock Code').agg({
                'Close': 'last', 
                'Change %': 'mean', 
                'Value': 'sum', 
                'Net Foreign Flow': 'sum', 
                'Big_Player_Anomaly': 'max', 
                'Volume Spike (x)': 'max', 
                'Volume_Pct_Tradeble': 'mean'
            }).reset_index()
            
            # Hitung metrik tambahan
            summary['Pressure'] = np.where(summary['Value'] > 0, (summary['Net Foreign Flow'] / summary['Value'] * 100), 0)
            summary['Inst_Score'] = (summary['Volume_Pct_Tradeble'] * 0.4) + (summary['Big_Player_Anomaly'] * 0.3) + (abs(summary['Pressure']) * 0.3)
            
            # Filter
            summary = summary[summary['Value'] >= min_value]
            summary = summary[summary['Big_Player_Anomaly'] >= min_anomali]
            
            if foreign_filter == "Net Buy": 
                summary = summary[summary['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell": 
                summary = summary[summary['Net Foreign Flow'] < 0]
            
            summary = summary.sort_values('Inst_Score', ascending=False).head(100)
            
            st.markdown(f"**Menampilkan Top {len(summary)} Saham (Diurutkan berdasarkan Bandar Score)**")
            
            if len(summary) > 0:
                # Konversi numpy values ke Python native untuk ProgressColumn
                max_vol_pct = float(summary['Volume_Pct_Tradeble'].max()) if len(summary) > 0 else 1.0
                if max_vol_pct <= 0:
                    max_vol_pct = 1.0
                
                # Siapkan dataframe untuk display
                display_df = summary[['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow', 
                                      'Volume_Pct_Tradeble', 'Big_Player_Anomaly', 'Inst_Score']].copy()
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=600,
                    column_config={
                        "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                        "Close": st.column_config.NumberColumn("Harga", format="Rp %,.0f"),
                        "Change %": st.column_config.NumberColumn("Avg Chg%", format="%.2f%%"),
                        "Value": st.column_config.NumberColumn("Total Value", format="Rp %,.0f"),
                        "Net Foreign Flow": st.column_config.NumberColumn("Net Foreign", format="Rp %,.0f"),
                        "Volume_Pct_Tradeble": st.column_config.ProgressColumn(
                            "Vol % Tradeble", 
                            format="%.2f%%", 
                            min_value=0.0, 
                            max_value=max_vol_pct
                        ),
                        "Big_Player_Anomaly": st.column_config.NumberColumn("Max Anomali (x)", format="%.1f"),
                        "Inst_Score": st.column_config.NumberColumn("Score", format="%.1f")
                    },
                    hide_index=True
                )
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")
        else:
            st.info("Data tidak ditemukan untuk periode ini.")

# ==================== TAB 2: DEEP DIVE & CHART ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics")
    
    # Control Panel
    c_sel1, c_sel2, c_sel3 = st.columns([1, 1, 2])
    with c_sel1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dd_stock_select')
    with c_sel2:
        interval = st.selectbox("Interval Chart", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with c_sel3:
        chart_len = st.slider("Jumlah Data (Candle)", 30, 365, 120, key='dd_len')

    # Data Filter
    df_dive = df_transaksi[df_transaksi['Stock Code'] == selected_stock].copy().sort_values('Last Trading Date')
    
    if not df_dive.empty:
        # --- LOGIC RESAMPLING ---
        if interval == "Weekly":
            df_resampled = df_dive.set_index('Last Trading Date').resample('W').agg({
                'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max', 
                'Volume Spike (x)': 'max'
            }).dropna().reset_index()
        elif interval == "Monthly":
            df_resampled = df_dive.set_index('Last Trading Date').resample('M').agg({
                'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max', 
                'Volume Spike (x)': 'max'
            }).dropna().reset_index()
        else:
            df_resampled = df_dive.copy()

        # Potong sesuai panjang chart
        df_chart = df_resampled.tail(chart_len)
        latest = df_dive.iloc[-1]

        # --- KPI CARDS ---
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

        spike_vol_count = len(df_chart[df_chart['Volume Spike (x)'] > 1.5]) if 'Volume Spike (x)' in df_chart.columns else 0
        spike_anom_count = len(df_chart[df_chart['Big_Player_Anomaly'] > 3.0]) if 'Big_Player_Anomaly' in df_chart.columns else 0

        # TAMPILAN KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: 
            st.metric("Harga Terakhir", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{status_color}'>{status_text}</div><div class='kpi-label'>Status Trend</div></div>", unsafe_allow_html=True)
        with k3: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{spike_anom_count}x</div><div class='kpi-label'>Freq Anomali (>3x)</div></div>", unsafe_allow_html=True)
        with k4: 
            st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{spike_vol_count}x</div><div class='kpi-label'>Freq Vol Spike (>1.5x)</div></div>", unsafe_allow_html=True)
        with k5: 
            st.metric("Total Foreign (Periode)", f"Rp {total_foreign/1e9:,.1f} M")

        st.divider()

        # --- ADVANCED CHART ---
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3],
            subplot_titles=(f"Price Action ({interval})", "Net Foreign Flow")
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_chart['Last Trading Date'],
            open=df_chart['Open Price'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'],
            name="OHLC"
        ), row=1, col=1)

        # Marker untuk Anomali
        if 'Big_Player_Anomaly' in df_chart.columns:
            anomali_points = df_chart[df_chart['Big_Player_Anomaly'] > 3.0]
            if not anomali_points.empty:
                fig.add_trace(go.Scatter(
                    x=anomali_points['Last Trading Date'],
                    y=anomali_points['High'] * 1.02,
                    mode='markers',
                    marker=dict(symbol='star', size=12, color='orange', line=dict(width=1, color='black')),
                    name='High Anomaly'
                ), row=1, col=1)

        # Foreign Flow Bar
        colors_ff = ['green' if val >= 0 else 'red' for val in df_chart['Net Foreign Flow']]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'],
            y=df_chart['Net Foreign Flow'] / 1e9,  # Convert ke Miliar
            name="Net Foreign (M)",
            marker_color=colors_ff
        ), row=2, col=1)

        fig.update_layout(
            height=700, 
            hovermode="x unified", 
            margin=dict(t=30, b=30, l=30, r=30),
            xaxis_rangeslider_visible=False,
            showlegend=True
        )
        fig.update_yaxes(title_text="Harga", row=1, col=1)
        fig.update_yaxes(title_text="Foreign (M)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

        # --- RINGKASAN BROKER (KSEI) ---
        st.subheader("🕵️ Ringkasan Aktivitas Broker Owner (KSEI 5%)")
        
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            ksei_stock = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            
            if not ksei_stock.empty:
                ksei_grouped = ksei_stock.sort_values('Tanggal_Data').groupby(['Kode Broker', 'Nama Pemegang Saham']).agg(
                    Awal=('Jumlah Saham (Curr)', 'first'),
                    Akhir=('Jumlah Saham (Curr)', 'last'),
                    Tgl_Awal=('Tanggal_Data', 'first'),
                    Tgl_Akhir=('Tanggal_Data', 'last')
                ).reset_index()
                
                ksei_grouped['Net Change'] = ksei_grouped['Akhir'] - ksei_grouped['Awal']
                ksei_grouped['Status'] = np.where(
                    ksei_grouped['Net Change'] > 0, "Accumulation", 
                    np.where(ksei_grouped['Net Change'] < 0, "Distribution", "Hold")
                )
                
                active_ksei = ksei_grouped[ksei_grouped['Net Change'] != 0].sort_values('Net Change', ascending=False)
                
                if not active_ksei.empty:
                    st.dataframe(
                        active_ksei[['Kode Broker', 'Nama Pemegang Saham', 'Awal', 'Akhir', 'Net Change', 'Status']],
                        column_config={
                            "Awal": st.column_config.NumberColumn("Awal (Lembar)", format="%d"),
                            "Akhir": st.column_config.NumberColumn("Akhir (Lembar)", format="%d"),
                            "Net Change": st.column_config.NumberColumn("Net Change", format="%+d"),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Tidak ada perubahan kepemilikan >5% pada data yang tersedia.")
            else:
                st.warning("Data KSEI tidak tersedia untuk saham ini.")
    else:
        st.error("Data saham tidak ditemukan.")

# ==================== TAB 3: BROKER MUTASI ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Mutation Radar")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        with st.container():
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                mutasi_period = st.selectbox("Pilih Periode Mutasi", ["1 Minggu", "2 Minggu", "1 Bulan", "3 Bulan"], key='m_period')
                days_map = {"1 Minggu": 7, "2 Minggu": 14, "1 Bulan": 30, "3 Bulan": 90}
        
        start_mutasi = df_kepemilikan['Tanggal_Data'].max() - timedelta(days=days_map[mutasi_period])
        df_ksei_period = df_kepemilikan[df_kepemilikan['Tanggal_Data'] >= start_mutasi].copy()
        
        if not df_ksei_period.empty:
            mutasi = df_ksei_period.sort_values('Tanggal_Data').groupby(['Kode Broker', 'Kode Efek']).agg(
                Awal=('Jumlah Saham (Curr)', 'first'),
                Akhir=('Jumlah Saham (Curr)', 'last')
            ).reset_index()
            
            mutasi['Net_Change'] = mutasi['Akhir'] - mutasi['Awal']
            broker_summary = mutasi.groupby('Kode Broker')['Net_Change'].sum().reset_index()
            
            top_acc = broker_summary.nlargest(10, 'Net_Change')
            top_dist = broker_summary.nsmallest(10, 'Net_Change')
            top_dist['Net_Change'] = abs(top_dist['Net_Change'])
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🟢 Top Accumulator")
                if not top_acc.empty:
                    fig_acc = px.bar(top_acc, x='Net_Change', y='Kode Broker', orientation='h', 
                                   title="Broker dengan Akumulasi Terbesar",
                                   color_discrete_sequence=['#00c853'])
                    fig_acc.update_layout(xaxis=dict(tickformat=",.0f"))
                    st.plotly_chart(fig_acc, use_container_width=True)
            
            with c2:
                st.markdown("#### 🔴 Top Distributor")
                if not top_dist.empty:
                    fig_dist = px.bar(top_dist, x='Net_Change', y='Kode Broker', orientation='h',
                                    title="Broker dengan Distribusi Terbesar",
                                    color_discrete_sequence=['#ff3d00'])
                    fig_dist.update_layout(xaxis=dict(tickformat=",.0f"))
                    st.plotly_chart(fig_dist, use_container_width=True)
                
            st.divider()
            st.markdown("#### 🔎 Detail Mutasi per Broker")
            
            if len(broker_summary) > 0:
                sel_broker = st.selectbox("Pilih Broker", sorted(broker_summary['Kode Broker'].unique()), key='m_broker')
                detail = mutasi[(mutasi['Kode Broker'] == sel_broker) & (mutasi['Net_Change'] != 0)].sort_values('Net_Change', ascending=False)
                
                if not detail.empty:
                    st.dataframe(
                        detail[['Kode Efek', 'Awal', 'Akhir', 'Net_Change']],
                        column_config={
                            "Awal": st.column_config.NumberColumn("Awal (Lembar)", format="%d"),
                            "Akhir": st.column_config.NumberColumn("Akhir (Lembar)", format="%d"),
                            "Net_Change": st.column_config.NumberColumn("Net Mutasi", format="%+d")
                        },
                        use_container_width=True, 
                        hide_index=True
                    )
                else:
                    st.info("Tidak ada mutasi untuk broker ini")
        else:
            st.warning("Tidak ada data KSEI dalam periode ini")
    else:
        st.warning("Data broker tidak tersedia")

# ==================== TAB 4: MARKET MAP & FOREIGN ====================
with tabs[3]:
    st.markdown("### 🗺️ Market Flow & Foreign Radar")
    
    # Foreign Flow Timeframe
    st.markdown("#### 🌍 Top Foreign Flow (Multi-Timeframe)")
    ff_period = st.selectbox("Rentang Waktu Foreign Flow", 
                            ["Hari Ini", "5 Hari", "10 Hari", "20 Hari", "30 Hari", "60 Hari"], 
                            key='ff_time')
    
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
                'Change %': 'mean'
            }).reset_index()
    else:
        start_date_ff = max_date - timedelta(days=days_back)
        mask_ff = (df_transaksi['Last Trading Date'].dt.date >= start_date_ff) & \
                  (df_transaksi['Last Trading Date'].dt.date <= max_date)
        ff_data = df_transaksi[mask_ff].groupby('Stock Code').agg({
            'Net Foreign Flow': 'sum',
            'Value': 'sum',
            'Close': 'last',
            'Change %': 'mean'
        }).reset_index()
    
    if not ff_data.empty and len(ff_data) > 0:
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f"#### 🟢 Top Foreign Buy ({ff_period})")
            top_buy = ff_data.nlargest(20, 'Net Foreign Flow')
            if not top_buy.empty:
                st.dataframe(
                    top_buy[['Stock Code', 'Close', 'Net Foreign Flow']],
                    column_config={
                        "Stock Code": "Kode",
                        "Close": st.column_config.NumberColumn("Harga", format="Rp %d"),
                        "Net Foreign Flow": st.column_config.NumberColumn("Net Buy", format="Rp %,.0f")
                    },
                    hide_index=True, 
                    use_container_width=True, 
                    height=500
                )
            
        with col_f2:
            st.markdown(f"#### 🔴 Top Foreign Sell ({ff_period})")
            top_sell = ff_data.nsmallest(20, 'Net Foreign Flow')
            if not top_sell.empty:
                st.dataframe(
                    top_sell[['Stock Code', 'Close', 'Net Foreign Flow']],
                    column_config={
                        "Stock Code": "Kode",
                        "Close": st.column_config.NumberColumn("Harga", format="Rp %d"),
                        "Net Foreign Flow": st.column_config.NumberColumn("Net Sell", format="Rp %,.0f")
                    },
                    hide_index=True, 
                    use_container_width=True, 
                    height=500
                )
        
        # Visualisasi tambahan - PERBAIKAN: Buat kolom absolute value
        st.divider()
        st.markdown("#### 📊 Foreign Flow Distribution")
        
        # Buat copy dataframe untuk visualisasi
        ff_viz = ff_data.head(50).copy()
        
        # Buat kolom absolute value
        ff_viz['Abs_Net_Foreign'] = abs(ff_viz['Net Foreign Flow'])
        
        # Buat scatter plot dengan kolom yang sudah dibuat
        fig = px.scatter(ff_viz, x='Value', y='Net Foreign Flow',
                        text='Stock Code', size='Abs_Net_Foreign',
                        color='Net Foreign Flow', color_continuous_scale='RdYlGn',
                        title=f"Foreign Flow Map ({ff_period})")
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
    st.caption(f"📊 Total Saham: {len(unique_stocks)}")
with col3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
