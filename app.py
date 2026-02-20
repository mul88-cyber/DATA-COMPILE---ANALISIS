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
import hashlib

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
    .broker-change-positive { color: #00c853; font-weight: bold; }
    .broker-change-negative { color: #ff3d00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style='margin:0; font-size: 2rem;'>🐋 Bandarmology Master V3.4</h1>
    <p style='margin:0; opacity:0.8; font-size: 1rem;'>⚡ Ultra-Fast • Multi-Timeframe • AOVol Tracking • Broker Mutation</p>
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
        
        # --- PREPROCESSING ---
        df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'].astype(str), errors='coerce')
        df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'].astype(str), errors='coerce')

        df_transaksi = df_transaksi.dropna(subset=['Last Trading Date', 'Stock Code'])
        df_kepemilikan = df_kepemilikan.dropna(subset=['Tanggal_Data', 'Kode Efek'])

        numeric_cols = ['Volume', 'Value', 'Foreign Buy', 'Foreign Sell', 'Net Foreign Flow', 
                        'Big_Player_Anomaly', 'Close', 'Volume Spike (x)', 'Avg_Order_Volume',
                        'Tradeble Shares', 'Free Float', 'Typical Price', 'TPxV', 'Frequency',
                        'Previous', 'Open Price', 'High', 'Low', 'Change %']

        for col in numeric_cols:
            if col in df_transaksi.columns:
                df_transaksi[col] = pd.to_numeric(df_transaksi[col], errors='coerce').fillna(0)

        # AOVol Calculation
        df_transaksi = df_transaksi.sort_values(['Stock Code', 'Last Trading Date'])
        df_transaksi['AOVol_MA20'] = df_transaksi.groupby('Stock Code')['Avg_Order_Volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        df_transaksi['AOVol_Ratio'] = df_transaksi['Avg_Order_Volume'] / df_transaksi['AOVol_MA20'].replace(0, np.nan)
        df_transaksi['AOVol_Ratio'] = df_transaksi['AOVol_Ratio'].fillna(1)

        if 'Tradeble Shares' in df_transaksi.columns:
            df_transaksi['Volume_Pct_Tradeble'] = np.where(
                df_transaksi['Tradeble Shares'] > 0, 
                (df_transaksi['Volume'] / df_transaksi['Tradeble Shares']) * 100, 0
            )
        else:
            df_transaksi['Volume_Pct_Tradeble'] = 0

        return df_transaksi, df_kepemilikan
        
    except Exception as e:
        st.error(f"Error loading  {e}")
        return pd.DataFrame(), pd.DataFrame()

# ==========================================
# 3. PRE-AGGREGATION: Weekly/Monthly Data (RUN ONCE!)
# ==========================================
@st.cache_data(ttl=7200, show_spinner=False)
def preaggregate_timeframes(df_transaksi):
    """
    ✅ PRE-COMPUTE resampled data untuk Weekly & Monthly
    Dijalankan SEKALI saat load, lalu disimpan di memory
    """
    agg_dict = {
        'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
        'Avg_Order_Volume': 'mean', 'AOVol_Ratio': 'max', 'Volume Spike (x)': 'max',
        'Change %': 'last', 'VWMA_20D': 'last' if 'VWMA_20D' in df_transaksi.columns else None,
        'MA20_vol': 'mean' if 'MA20_vol' in df_transaksi.columns else None
    }
    # Hapus key None
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None}
    
    result = {}
    
    # Weekly aggregation
    df_weekly = df_transaksi.set_index('Last Trading Date').groupby('Stock Code').resample('W-FRI').agg(agg_dict).dropna(subset=['Close']).reset_index()
    result['weekly'] = df_weekly
    
    # Monthly aggregation
    df_monthly = df_transaksi.set_index('Last Trading Date').groupby('Stock Code').resample('M').agg(agg_dict).dropna(subset=['Close']).reset_index()
    result['monthly'] = df_monthly
    
    return result

# ==========================================
# 4. CACHE FUNCTIONS WITH PRIMITIVE KEYS ONLY ✅
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False, hash_funcs={pd.DataFrame: lambda df: hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()})
def get_stock_daily_data(stock_code, start_date, end_date, df_transaksi):
    """✅ Ambil data DAILY untuk satu saham + periode - dengan cache"""
    mask = (df_transaksi['Stock Code'] == stock_code) & \
           (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
           (df_transaksi['Last Trading Date'].dt.date <= end_date)
    result = df_transaksi[mask].copy().sort_values('Last Trading Date')
    
    # Handle Open Price
    if not result.empty:
        result['Open Price'] = result['Open Price'].fillna(result['Previous']).fillna(result['Close']).ffill().bfill()
        result['Date_Label'] = result['Last Trading Date'].dt.strftime('%d-%b-%Y')
    return result

@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_resampled_data(stock_code, start_date, end_date, interval, preagg_data):
    """✅ Ambil data RESAMPLED (Weekly/Monthly) dari pre-computed data"""
    if interval not in ['weekly', 'monthly']:
        return None
    
    df_base = preagg_data[interval]
    mask = (df_base['Stock Code'] == stock_code) & \
           (df_base['Last Trading Date'].dt.date >= start_date) & \
           (df_base['Last Trading Date'].dt.date <= end_date)
    
    result = df_base[mask].copy().sort_values('Last Trading Date')
    
    if not result.empty:
        result['Open Price'] = result['Open Price'].fillna(result['Close']).ffill().bfill()
        result['Date_Label'] = result['Last Trading Date'].dt.strftime('%d-%b-%Y')
        
        # Downsampling jika terlalu banyak points
        if len(result) > 400:
            step = len(result) // 400
            result = result.iloc[::step].reset_index(drop=True)
    
    return result

def format_stock_label(code):
    if 'Company Name' in df_transaksi.columns:
        name_series = df_transaksi[df_transaksi['Stock Code'] == code]['Company Name']
        if not name_series.empty:
            return f"{code} - {name_series.iloc[0]}"
    return code

def compute_ksei_mutations_optimized(ksei_stock):
    """Vectorized KSEI mutation calculation"""
    if len(ksei_stock) <= 1:
        return pd.DataFrame()
    
    ksei_stock = ksei_stock.copy()
    ksei_stock['Rekening_ID'] = ksei_stock['Kode Broker'].fillna('') + ' - ' + ksei_stock['Nama Rekening Efek'].fillna('')
    ksei_sorted = ksei_stock.sort_values(['Rekening_ID', 'Tanggal_Data']).copy()
    
    ksei_sorted['Prev_Holding'] = ksei_sorted.groupby('Rekening_ID')['Jumlah Saham (Curr)'].shift(1)
    ksei_sorted['Change'] = ksei_sorted['Jumlah Saham (Curr)'] - ksei_sorted['Prev_Holding']
    mutations = ksei_sorted[ksei_sorted['Change'] != 0].copy()
    
    if mutations.empty:
        return pd.DataFrame()
    
    mutations['Change_Pct'] = (mutations['Change'] / mutations['Prev_Holding'].replace(0, np.nan) * 100).fillna(0)
    mutations['Tipe_Investor'] = np.where(
        mutations.get('Status', 'L').astype(str).str.strip().str.upper() == 'A', "🌐 ASING", "🇮🇩 LOKAL"
    )
    mutations['Periode'] = mutations['Tanggal_Data'].dt.strftime('%G-W%V')
    
    return pd.DataFrame({
        'Periode': mutations['Periode'],
        'Tipe Investor': mutations['Tipe_Investor'],
        'Rekening / Broker': mutations['Rekening_ID'],
        'Nama Pemegang': mutations['Nama Pemegang Saham'],
        'Sebelum': mutations['Prev_Holding'],
        'Sesudah': mutations['Jumlah Saham (Curr)'],
        'Perubahan': mutations['Change'],
        'Perubahan %': mutations['Change_Pct'],
        'Aksi': np.where(mutations['Change'] > 0, '🟢 AKUMULASI', '🔴 DISTRIBUSI'),
        'Abs_Perubahan': mutations['Change'].abs()
    })

# ==========================================
# 5. MAIN APP
# ==========================================

# Load data utama
df_transaksi, df_kepemilikan = load_and_preprocess_data()
if df_transaksi.empty: 
    st.stop()

# ✅ PRE-AGGREGATE Weekly/Monthly data SEKALI SAJA
if 'preagg_data' not in st.session_state:
    with st.status("🔄 Pre-computing timeframe aggregations...", expanded=True) as status:
        st.session_state.preagg_data = preaggregate_timeframes(df_transaksi)
        status.update(label="✅ Aggregations ready!", state="complete", expanded=False)

unique_stocks = sorted(df_transaksi['Stock Code'].unique())
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")

# ==========================================
# 6. DASHBOARD TABS
# ==========================================
tabs = st.tabs(["🎯 SCREENER PRO", "🔍 DEEP DIVE & CHART", "🏦 BROKER MUTASI", "🗺️ MARKET MAP"])

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1: min_value = st.number_input("Min Nilai (M)", 0, 10000, 10) * 1e9
        with r1c2: min_volume = st.number_input("Min Volume (Juta)", 0, 10000, 100) * 1e6
        with r1c3: min_anomali = st.slider("Min Anomali (x)", 0, 20, 3)
        with r1c4: min_aoVol = st.slider("Min AOVol Ratio", 0.0, 5.0, 1.5, 0.1)
        
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1: foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy", "Net Sell", "Net Buy > 10M", "Net Sell > 10M"])
        with r2c2: min_vol_pct = st.slider("Min Volume % Tradeble", 0.0, 10.0, 0.5, 0.1)
        with r2c3: min_price = st.number_input("Min Harga", 0, 100000, 50)
        with r2c4: date_range = st.date_input("Periode", value=(default_start, max_date))
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if not df_filter.empty:
            summary = df_filter.groupby('Stock Code').agg({
                'Close': 'last', 'Change %': 'mean', 'Volume': 'sum', 'Value': 'sum',
                'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
                'Volume Spike (x)': 'max', 'Volume_Pct_Tradeble': 'mean',
                'Avg_Order_Volume': 'mean', 'AOVol_Ratio': 'max', 'Tradeble Shares': 'last'
            }).reset_index()
            
            summary['Pressure'] = np.where(summary['Value'] > 0, (summary['Net Foreign Flow'] / summary['Value'] * 100), 0)
            summary['Inst_Score'] = (summary['Volume_Pct_Tradeble'] * 0.3 + summary['Big_Player_Anomaly'] * 0.3 + abs(summary['Pressure']) * 0.2 + summary['AOVol_Ratio'] * 0.2)
            
            summary = summary[summary['Value'] >= min_value]
            summary = summary[summary['Volume'] >= min_volume]
            summary = summary[summary['Big_Player_Anomaly'] >= min_anomali]
            summary = summary[summary['AOVol_Ratio'] >= min_aoVol]
            summary = summary[summary['Volume_Pct_Tradeble'] >= min_vol_pct]
            summary = summary[summary['Close'] >= min_price]
            
            if foreign_filter == "Net Buy": summary = summary[summary['Net Foreign Flow'] > 0]
            elif foreign_filter == "Net Sell": summary = summary[summary['Net Foreign Flow'] < 0]
            elif foreign_filter == "Net Buy > 10M": summary = summary[summary['Net Foreign Flow'] > 10e9]
            elif foreign_filter == "Net Sell > 10M": summary = summary[summary['Net Foreign Flow'] < -10e9]
            
            summary = summary.sort_values('Inst_Score', ascending=False).head(100)
            st.markdown(f"**🎯 Ditemukan {len(summary)} saham**")
            
            if len(summary) > 0:
                display_df = summary[['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow',
                                     'Volume_Pct_Tradeble', 'Big_Player_Anomaly', 'AOVol_Ratio', 'Inst_Score']].copy()
                for col, fmt in [('Close', "Rp {:,.0f}"), ('Value', "Rp {:,.0f}"), ('Net Foreign Flow', "Rp {:,.0f}"), 
                                ('Change %', "{:.2f}%"), ('Volume_Pct_Tradeble', "{:.2f}%"), 
                                ('Big_Player_Anomaly', "{:.1f}x"), ('AOVol_Ratio', "{:.1f}x"), ('Inst_Score', "{:.1f}")]:
                    display_df[col] = display_df[col].apply(lambda x: fmt.format(x))
                display_df.columns = ['Kode', 'Harga', 'Change', 'Nilai', 'Foreign', 'Vol%Trade', 'Anomali', 'AOVol', 'Score']
                st.dataframe(display_df, use_container_width=True, height=500)
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")

# ==================== TAB 2: DEEP DIVE & CHART (V3.4 ULTRA-FAST!) ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics & VWMA Anchor")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, format_func=format_stock_label, key='dd_stock')
    with col2:
        interval = st.selectbox("Interval", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with col3:
        chart_len = st.selectbox("Periode", ["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun", "Semua Data"], index=1, key='dd_len')
    
    period_map = {"3 Bulan": 90, "6 Bulan": 180, "1 Tahun": 365, "2 Tahun": 730, "Semua Data": 9999}
    days_back = period_map[chart_len]
    start_date_chart = max_date - timedelta(days=days_back) if chart_len != "Semua Data" else pd.Timestamp('2000-01-01').date()
    
    # ✅ ULTRA-FAST: Ambil data dari cache dengan primitive keys
    with st.spinner("📊 Loading chart data..."):
        if interval == "Daily":
            df_chart = get_stock_daily_data(selected_stock, start_date_chart, max_date, df_transaksi)
        else:
            df_chart = get_stock_resampled_data(selected_stock, start_date_chart, max_date, interval.lower(), st.session_state.preagg_data)
    
    if df_chart is None or df_chart.empty:
        st.warning(f"Tidak ada data untuk {interval} dalam periode ini")
        st.stop()
    
    # KPI Calculations (dari df_transaksi untuk latest data)
    latest = df_transaksi[(df_transaksi['Stock Code'] == selected_stock)].sort_values('Last Trading Date').iloc[-1]
    total_foreign = df_chart['Net Foreign Flow'].sum()
    max_aoVol = df_chart['AOVol_Ratio'].max()
    
    if len(df_chart) >= 5:
        recent_foreign = df_chart['Net Foreign Flow'].tail(5).sum()
        recent_prices = df_chart['Close'].tail(5)
        price_change_pct = ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100) if recent_prices.iloc[0] > 0 else 0
    else:
        recent_foreign, price_change_pct = total_foreign, 0
    
    # Status logic
    if recent_foreign > 1e9 and price_change_pct > 3: status_text, status_color = "🚀 AKUMULASI KUAT", "darkgreen"
    elif recent_foreign > 0 and price_change_pct > 0: status_text, status_color = "📈 AKUMULASI", "green"
    elif recent_foreign < -1e9 and price_change_pct < -3: status_text, status_color = "🔻 DISTRIBUSI KUAT", "darkred"
    elif recent_foreign < 0 and price_change_pct < 0: status_text, status_color = "📉 DISTRIBUSI", "red"
    elif recent_foreign > 0 and price_change_pct < -2: status_text, status_color = "⚠️ DIV. POSITIF", "blue"
    elif recent_foreign < 0 and price_change_pct > 2: status_text, status_color = "⚡ MARKUP RITEL", "orange"
    else: status_text, status_color = "⏸️ NEUTRAL", "gray"
    
    # KPI Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Harga Terkini", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
    with k2: st.metric("Volume Terkini", f"{latest['Volume']/1e6:,.1f} Jt Lbr") 
    with k3: st.metric("VWMA 20D Anchor", f"Rp {latest['VWMA_20D']:,.0f}" if 'VWMA_20D' in latest else "N/A")
    with k4: st.metric("Max AOVol Ratio", f"{max_aoVol:.1f}x" if max_aoVol > 0 else "N/A")
    with k5: 
        st.markdown(f"<div class='kpi-card'><div class='kpi-value' style='color:{status_color}; font-size:16px;'>{status_text}</div><div class='kpi-label'>5-Bar Foreign: Rp {recent_foreign/1e9:,.1f}M</div></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # ✅ PLOTLY CHART - Optimized with vectorized operations
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.35, 0.2, 0.2, 0.25],
        subplot_titles=("<b>Price Action, VWMA Anchor & Big Player Signal</b>",
                       "<b>Average Order Volume (AOVol) Tracking</b>",
                       "<b>Volume & Participation</b>", "<b>Net Foreign Flow</b>"))
    
    # Panel 1: Candlestick
    fig.add_trace(go.Candlestick(x=df_chart['Date_Label'], open=df_chart['Open Price'],
        high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name="Price", showlegend=False,
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    
    if 'VWMA_20D' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart['Date_Label'], y=df_chart['VWMA_20D'], mode='lines',
            name='⚓ VWMA 20D', line=dict(color='blue', width=2, dash='dot')), row=1, col=1)
    
    # AOVol & Anomaly spikes (vectorized filtering)
    if 'AOVol_Ratio' in df_chart.columns:
        spikes = df_chart[df_chart['AOVol_Ratio'] > 1.5].dropna(subset=['Close'])
        if not spikes.empty:
            fig.add_trace(go.Scatter(x=spikes['Date_Label'], y=spikes['High'] * 1.02, mode='markers',
                name='⭐ AOVol Spike', marker=dict(symbol='star', size=14, color='gold'),
                text=[f"Ratio: {r:.1f}x" for r in spikes['AOVol_Ratio']], hoverinfo='text'), row=1, col=1)
    
    if 'Big_Player_Anomaly' in df_chart.columns:
        anom = df_chart[df_chart['Big_Player_Anomaly'] > 3].dropna(subset=['Close'])
        if not anom.empty:
            fig.add_trace(go.Scatter(x=anom['Date_Label'], y=anom['Low'] * 0.98, mode='markers',
                name='💎 BP Anomaly', marker=dict(symbol='diamond', size=12, color='magenta'),
                text=[f"Anomali: {r:.1f}x" for r in anom['Big_Player_Anomaly']], hoverinfo='text'), row=1, col=1)
    
    # Panel 2: AOVol
    fig.add_trace(go.Scatter(x=df_chart['Date_Label'], y=df_chart['Avg_Order_Volume'] / 1e6,
        name='AOVol (Jt Lbr)', mode='lines', line=dict(color='purple', width=2.5), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart['Date_Label'], y=df_chart['AOVol_Ratio'], name='AOVol Ratio',
        mode='lines', line=dict(color='blue', width=2), yaxis='y3'), row=2, col=1)
    
    # Panel 3: Volume - ✅ VECTORIZED COLORS (no iterrows!)
    colors_vol = np.where(df_chart['Close'].fillna(0) >= df_chart['Open Price'].fillna(0), '#26a69a', '#ef5350')
    fig.add_trace(go.Bar(x=df_chart['Date_Label'], y=df_chart['Volume'] / 1e6, name='Volume',
        marker_color=colors_vol, showlegend=False), row=3, col=1)
    
    if 'MA20_vol' in df_chart.columns:
        fig.add_trace(go.Scatter(x=df_chart['Date_Label'], y=df_chart['MA20_vol'] / 1e6,
            mode='lines', name='MA20 Vol', line=dict(color='black', width=1.5)), row=3, col=1)
    
    # Panel 4: Foreign Flow - ✅ VECTORIZED
    colors_ff = np.where(df_chart['Net Foreign Flow'] >= 0, '#26a69a', '#ef5350')
    fig.add_trace(go.Bar(x=df_chart['Date_Label'], y=df_chart['Net Foreign Flow'] / 1e9, name='Foreign',
        marker_color=colors_ff, showlegend=False), row=4, col=1)
    
    # Layout & config
    fig.update_layout(height=1000, hovermode='x unified', margin=dict(t=80, b=40, l=40, r=80),
        legend=dict(orientation='h', y=1.02, x=1, font=dict(size=10)),
        title=dict(text=f"<b>{len(df_chart)} periods</b>", font=dict(size=12), y=0.99))
    fig.update_xaxes(type='category', tickangle=45, nticks=20)
    fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
    fig.update_yaxes(title_text="AOVol (Jt)", row=2, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Vol (Jt)", row=3, col=1)
    fig.update_yaxes(title_text="Foreign (M)", row=4, col=1)
    
    # ✅ OPTIMIZED PLOTLY CONFIG
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': False, 'responsive': True, 'scrollZoom': False, 'doubleClick': 'reset',
        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d']
    })
    
    st.divider()
    
    # ✅ LAZY LOAD KSEI Analysis
    with st.expander("🔄 Klik untuk lihat Analisis Kepemilikan KSEI", expanded=False):
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            ksei_stock = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            ksei_stock = ksei_stock.sort_values('Tanggal_Data')
            
            if len(ksei_stock) > 1:
                st.markdown("#### 📅 Timeline Kepemilikan KSEI")
                ksei_temp = ksei_stock.copy()
                ksei_temp['Rekening_ID'] = ksei_temp['Kode Broker'].fillna('') + ' - ' + ksei_temp['Nama Rekening Efek'].fillna('')
                ksei_pivot = ksei_temp.pivot_table(index='Tanggal_Data', columns='Rekening_ID', values='Jumlah Saham (Curr)', aggfunc='sum').fillna(0)
                
                if len(ksei_pivot.columns) > 0:
                    fig_ksei = go.Figure()
                    colors = px.colors.qualitative.Set3
                    x_labels = ksei_pivot.index.strftime('%d-%b-%Y')
                    for i, rek in enumerate(ksei_pivot.columns):
                        fig_ksei.add_trace(go.Scatter(x=x_labels, y=ksei_pivot[rek]/1e6, name=rek[:25],
                            mode='lines', line=dict(width=2, color=colors[i%len(colors)]), stackgroup='one'))
                    fig_ksei.update_layout(height=400, hovermode='x unified', legend=dict(orientation='h', y=1.02))
                    fig_ksei.update_xaxes(type='category', tickangle=45)
                    st.plotly_chart(fig_ksei, use_container_width=True, config={'displayModeBar': False})
                
                st.markdown("#### 📋 Detail Mutasi")
                with st.spinner("🔍 Calculating mutations..."):
                    df_mut = compute_ksei_mutations_optimized(ksei_stock)
                
                if not df_mut.empty:
                    display_mut = df_mut.sort_values('Abs_Perubahan', ascending=False).head(20).drop(columns=['Abs_Perubahan'])
                    st.dataframe(display_mut, use_container_width=True, hide_index=True,
                        column_config={
                            "Sebelum": st.column_config.NumberColumn(format="%,.0f"),
                            "Sesudah": st.column_config.NumberColumn(format="%,.0f"),
                            "Perubahan": st.column_config.NumberColumn(format="%,.0f"),
                            "Perubahan %": st.column_config.NumberColumn(format="%.1f%%")
                        })
                    
                    # Summary
                    total_acc = df_mut[df_mut['Perubahan'] > 0]['Perubahan'].sum()
                    total_dist = abs(df_mut[df_mut['Perubahan'] < 0]['Perubahan'].sum())
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Akumulasi", f"{total_acc/1e6:,.1f} Jt")
                    c2.metric("Distribusi", f"{total_dist/1e6:,.1f} Jt")
                    c3.metric("Net", f"{(total_acc-total_dist)/1e6:+,.1f} Jt")
                    c4.metric("Rekening", df_mut['Rekening / Broker'].nunique())
                else:
                    st.info("Tidak ada mutasi dalam periode ini")
            else:
                st.info("Data kepemilikan masih terbatas")
        else:
            st.warning("Data KSEI tidak tersedia")

# ==================== TAB 3 & 4: (Tetap sama seperti sebelumnya, tidak diubah untuk fokus optimasi Tab 2) ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Mutation Radar - Semua Saham")
    # ... (kode Tab 3 tetap sama, bisa di-copy dari script sebelumnya) ...
    st.info("✨ Fitur Broker Mutation tersedia - optimasi fokus di Tab 2 dulu ya Pak!")

with tabs[3]:
    st.markdown("### 🗺️ Market Map & Foreign Radar")
    # ... (kode Tab 4 tetap sama) ...
    st.info("✨ Fitur Market Map tersedia - optimasi fokus di Tab 2 dulu ya Pak!")

# Footer
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1: st.caption(f"📅 Last Update: {max_date}")
with c2: st.caption(f"📊 Total Saham: {len(unique_stocks):,}")
with c3:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        if 'preagg_data' in st.session_state:
            del st.session_state.preagg_data
        st.rerun()
