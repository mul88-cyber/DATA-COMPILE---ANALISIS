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

# Hitung AOVol (Average Order Volume) moving average
df_transaksi = df_transaksi.sort_values(['Stock Code', 'Last Trading Date'])
df_transaksi['AOVol_MA20'] = df_transaksi.groupby('Stock Code')['Avg_Order_Volume'].transform(
    lambda x: x.rolling(20, min_periods=1).mean()
)
df_transaksi['AOVol_Ratio'] = df_transaksi['Avg_Order_Volume'] / df_transaksi['AOVol_MA20'].replace(0, np.nan)
df_transaksi['AOVol_Ratio'] = df_transaksi['AOVol_Ratio'].fillna(1)

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

# ==================== TAB 2: DEEP DIVE & CHART (IMPROVED) ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics")
    
    # Control Panel
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dd_stock')
    with col2:
        interval = st.selectbox("Interval", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with col3:
        chart_len = st.selectbox("Periode", ["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun"], 
                                index=1, key='dd_len')
    
    # Map periode ke hari
    period_map = {"3 Bulan": 90, "6 Bulan": 180, "1 Tahun": 365, "2 Tahun": 730}
    days_back = period_map[chart_len]
    
    # Filter data untuk saham terpilih
    df_stock = df_transaksi[df_transaksi['Stock Code'] == selected_stock].copy().sort_values('Last Trading Date')
    
    if not df_stock.empty:
        # Potong berdasarkan periode
        start_date = max_date - timedelta(days=days_back)
        df_stock = df_stock[df_stock['Last Trading Date'].dt.date >= start_date]
        
        # RESAMPLING UNTUK INTERVAL WEEKLY/MONTHLY
        if interval == "Weekly":
            df_chart = df_stock.set_index('Last Trading Date').resample('W-FRI').agg({
                'Open Price': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'max',
                'Avg_Order_Volume': 'mean',
                'AOVol_Ratio': 'max',
                'Volume Spike (x)': 'max'
            }).dropna(subset=['Close']).reset_index()
            df_chart.columns = ['Last Trading Date', 'Open Price', 'High', 'Low', 'Close', 
                               'Volume', 'Net Foreign Flow', 'Big_Player_Anomaly', 
                               'Avg_Order_Volume', 'AOVol_Ratio', 'Volume Spike (x)']
            
        elif interval == "Monthly":
            df_chart = df_stock.set_index('Last Trading Date').resample('M').agg({
                'Open Price': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'Net Foreign Flow': 'sum',
                'Big_Player_Anomaly': 'max',
                'Avg_Order_Volume': 'mean',
                'AOVol_Ratio': 'max',
                'Volume Spike (x)': 'max'
            }).dropna(subset=['Close']).reset_index()
            df_chart.columns = ['Last Trading Date', 'Open Price', 'High', 'Low', 'Close',
                               'Volume', 'Net Foreign Flow', 'Big_Player_Anomaly',
                               'Avg_Order_Volume', 'AOVol_Ratio', 'Volume Spike (x)']
        else:
            df_chart = df_stock.copy()
        
        if len(df_chart) == 0:
            st.warning(f"Tidak ada data untuk interval {interval} dalam periode ini")
            st.stop()
        
        # Latest data untuk KPI
        latest = df_stock.iloc[-1]
        
        # KPI Cards
        total_foreign = df_chart['Net Foreign Flow'].sum()
        avg_aoVol = df_chart['Avg_Order_Volume'].mean()
        max_aoVol = df_chart['AOVol_Ratio'].max()
        
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: 
            st.metric("Harga", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: 
            st.metric("Rata2 AOVol", f"Rp {avg_aoVol:,.0f}")
        with k3: 
            st.metric("Max AOVol Ratio", f"{max_aoVol:.1f}x")
        with k4: 
            st.metric("Total Foreign", f"Rp {total_foreign/1e9:,.1f}M")
        with k5: 
            spike_count = len(df_chart[df_chart['Volume Spike (x)'] > 1.5])
            st.metric("Volume Spike", f"{spike_count}x")
        
        st.divider()
        
        # MAIN CHART - COMBO CHART (3 panel)
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=(
                f"Price Action & AOVol Signal - {selected_stock} ({interval})",
                "Volume & Foreign Flow (Combo)",
                "AOVol Analysis"
            )
        )
        
        # PANEL 1: CANDLESTICK dengan BINTANG AOVol
        fig.add_trace(go.Candlestick(
            x=df_chart['Last Trading Date'],
            open=df_chart['Open Price'],
            high=df_chart['High'],
            low=df_chart['Low'],
            close=df_chart['Close'],
            name="Price",
            showlegend=False,
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ), row=1, col=1)
        
        # BINTANG UNTUK AOVol SPIKES
        aoVol_spikes = df_chart[df_chart['AOVol_Ratio'] > 1.5]
        if not aoVol_spikes.empty:
            fig.add_trace(go.Scatter(
                x=aoVol_spikes['Last Trading Date'],
                y=aoVol_spikes['High'] * 1.02,
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=14,
                    color='gold',
                    line=dict(width=2, color='orange')
                ),
                name='AOVol Spike',
                text=[f"AOVol: {x:.1f}x<br>Value: Rp {y:,.0f}" 
                      for x, y in zip(aoVol_spikes['AOVol_Ratio'], aoVol_spikes['Avg_Order_Volume'])],
                hoverinfo='text'
            ), row=1, col=1)
        
        # PANEL 2: COMBO CHART - Volume (bar) + Foreign Flow (line)
        # Volume bars
        colors_vol = ['#26a69a' if row['Close'] >= row['Open Price'] else '#ef5350' 
                     for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'],
            y=df_chart['Volume'] / 1e6,
            name='Volume (Juta)',
            marker_color=colors_vol,
            yaxis='y2'
        ), row=2, col=1)
        
        # Foreign Flow sebagai line
        fig.add_trace(go.Scatter(
            x=df_chart['Last Trading Date'],
            y=df_chart['Net Foreign Flow'] / 1e9,
            name='Net Foreign (Miliar)',
            line=dict(color='blue', width=2),
            mode='lines+markers',
            marker=dict(size=6),
            yaxis='y3'
        ), row=2, col=1)
        
        # PANEL 3: AOVol Analysis
        fig.add_trace(go.Scatter(
            x=df_chart['Last Trading Date'],
            y=df_chart['Avg_Order_Volume'] / 1e6,
            name='AOVol (Juta)',
            line=dict(color='purple', width=2),
            mode='lines+markers',
            marker=dict(size=4),
            fill='tozeroy'
        ), row=3, col=1)
        
        # Add AOVol Ratio sebagai line kedua (di secondary axis)
        fig.add_trace(go.Scatter(
            x=df_chart['Last Trading Date'],
            y=df_chart['AOVol_Ratio'],
            name='AOVol Ratio',
            line=dict(color='orange', width=2, dash='dash'),
            yaxis='y4'
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=900,
            hovermode='x unified',
            margin=dict(t=50, b=40, l=40, r=40),
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1,
                bgcolor='rgba(255,255,255,0.8)'
            )
        )
        
        # Configure axes
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])] if interval == "Daily" else [],
            title_text="Tanggal",
            row=3, col=1
        )
        
        # Y-axis titles
        fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (Juta)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Foreign (M)", row=2, col=1, secondary_y=True, overlaying='y2')
        fig.update_yaxes(title_text="AOVol (Juta)", row=3, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Ratio (x)", row=3, col=1, secondary_y=True, overlaying='y4')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # BROKER MUTATION ANALYSIS (IMPROVED)
        st.subheader("🔄 Broker Mutation Analysis (Perubahan Kepemilikan)")
        
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            ksei_stock = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            ksei_stock = ksei_stock.sort_values('Tanggal_Data')
            
            if len(ksei_stock) > 1:
                # Deteksi perubahan per broker
                mutations = []
                
                for broker in ksei_stock['Kode Broker'].unique():
                    broker_data = ksei_stock[ksei_stock['Kode Broker'] == broker].sort_values('Tanggal_Data')
                    
                    if len(broker_data) > 1:
                        for i in range(1, len(broker_data)):
                            prev = broker_data.iloc[i-1]
                            curr = broker_data.iloc[i]
                            
                            change = curr['Jumlah Saham (Curr)'] - prev['Jumlah Saham (Curr)']
                            if change != 0:
                                mutations.append({
                                    'Periode': f"{prev['Tanggal_Data'].strftime('%d/%m/%y')} - {curr['Tanggal_Data'].strftime('%d/%m/%y')}",
                                    'Broker': broker,
                                    'Pemegang': prev['Nama Pemegang Saham'],
                                    'Sebelum': prev['Jumlah Saham (Curr)'],
                                    'Sesudah': curr['Jumlah Saham (Curr)'],
                                    'Perubahan': change,
                                    '% Perubahan': (change / prev['Jumlah Saham (Curr)'] * 100) if prev['Jumlah Saham (Curr)'] > 0 else 0,
                                    'Status': 'Akumulasi' if change > 0 else 'Distribusi'
                                })
                
                if mutations:
                    df_mutations = pd.DataFrame(mutations)
                    df_mutations = df_mutations.sort_values('Perubahan', ascending=False)
                    
                    # Tampilkan mutations
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown("#### 📊 Mutasi Terbesar")
                        display_mut = df_mutations.head(10)[['Periode', 'Broker', 'Pemegang', 
                                                             'Sebelum', 'Sesudah', 'Perubahan', '% Perubahan']].copy()
                        
                        # Format numbers
                        display_mut['Sebelum'] = display_mut['Sebelum'].apply(lambda x: f"{x:,.0f}")
                        display_mut['Sesudah'] = display_mut['Sesudah'].apply(lambda x: f"{x:,.0f}")
                        display_mut['Perubahan'] = display_mut['Perubahan'].apply(
                            lambda x: f"<span class='broker-change-positive'>{x:+,.0f}</span>" if x > 0 
                            else f"<span class='broker-change-negative'>{x:+,.0f}</span>"
                        )
                        display_mut['% Perubahan'] = display_mut['% Perubahan'].apply(lambda x: f"{x:+.1f}%")
                        
                        display_mut.columns = ['Periode', 'Broker', 'Pemegang', 'Sebelum', 'Sesudah', 'Mutasi', '%']
                        
                        st.write(display_mut.to_html(escape=False, index=False), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("#### 📈 Summary Mutasi")
                        total_akumulasi = df_mutations[df_mutations['Perubahan'] > 0]['Perubahan'].sum()
                        total_distribusi = abs(df_mutations[df_mutations['Perubahan'] < 0]['Perubahan'].sum())
                        
                        st.metric("Total Akumulasi", f"{total_akumulasi:,.0f} lembar")
                        st.metric("Total Distribusi", f"{total_distribusi:,.0f} lembar")
                        st.metric("Net Change", f"{total_akumulasi - total_distribusi:+,.0f} lembar")
                        
                        # Broker teraktif
                        active_brokers = df_mutations.groupby('Broker')['Perubahan'].sum().abs().nlargest(5)
                        st.markdown("#### 🏦 Broker Teraktif")
                        for broker, val in active_brokers.items():
                            st.write(f"• {broker}: {val:,.0f} lembar")
                else:
                    st.info("Tidak ada perubahan kepemilikan dalam periode data yang tersedia")
            else:
                st.info("Data kepemilikan masih terbatas, belum bisa mendeteksi perubahan")
        else:
            st.warning("Data KSEI tidak tersedia")

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
            # Hitung mutasi per broker per saham
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
                
                # Detail per broker
                st.divider()
                st.markdown("#### 🔎 Detail Mutasi per Broker")
                
                all_brokers = sorted(broker_summary['Kode Broker'].unique())
                sel_broker = st.selectbox("Pilih Broker", all_brokers, key='m_broker_detail')
                
                if sel_broker:
                    detail = mutasi[mutasi['Kode Broker'] == sel_broker].copy()
                    detail = detail.sort_values('Net_Change', ascending=False)
                    
                    if not detail.empty:
                        display_detail = detail[['Kode Efek', 'Nama', 'Awal', 'Akhir', 'Net_Change']].copy()
                        display_detail['Awal'] = display_detail['Awal'].apply(lambda x: f"{x:,.0f}")
                        display_detail['Akhir'] = display_detail['Akhir'].apply(lambda x: f"{x:,.0f}")
                        display_detail['Net_Change'] = display_detail['Net_Change'].apply(
                            lambda x: f"+{x:,.0f}" if x > 0 else f"{x:,.0f}"
                        )
                        display_detail.columns = ['Saham', 'Pemegang', 'Awal', 'Akhir', 'Mutasi']
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
        st.plotly_chart(fig, use_container_width=True)

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
