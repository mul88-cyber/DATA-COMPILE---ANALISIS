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

# ==================== TAB 2: DEEP DIVE & CHART (V3.1 INSTITUTIONAL) ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics & VWMA Anchor")
    
    # Bikin kamus nama perusahaan untuk Selectbox yang lebih premium
    def format_stock_label(code):
        if 'Company Name' in df_transaksi.columns:
            name_series = df_transaksi[df_transaksi['Stock Code'] == code]['Company Name']
            if not name_series.empty:
                return f"{code} - {name_series.iloc[0]}"
        return code

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
    days_back = period_map[chart_len]
    
    # Filter data untuk saham terpilih
    df_stock = df_transaksi[df_transaksi['Stock Code'] == selected_stock].copy().sort_values('Last Trading Date')
    
    if not df_stock.empty:
        # Potong berdasarkan periode
        if chart_len != "Semua Data":
            start_date_chart = max_date - timedelta(days=days_back)
            df_stock = df_stock[df_stock['Last Trading Date'].dt.date >= start_date_chart]
        
        # PERBAIKAN: Handle Open Price yang kosong
        df_stock['Open Price'] = df_stock['Open Price'].fillna(df_stock['Previous'])
        df_stock['Open Price'] = df_stock['Open Price'].fillna(df_stock['Close'])
        df_stock['Open Price'] = df_stock['Open Price'].fillna(method='ffill').fillna(method='bfill')
        
        # Kamus agregasi resampling (menambahkan VWMA dan MA Vol)
        agg_dict = {
            'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
            'Avg_Order_Volume': 'mean', 'AOVol_Ratio': 'max', 'Volume Spike (x)': 'max',
            'Change %': 'last'
        }
        
        # Masukkan kolom raw baru jika ada
        if 'VWMA_20D' in df_stock.columns:
            agg_dict['VWMA_20D'] = 'last'
        if 'MA20_vol' in df_stock.columns:
            agg_dict['MA20_vol'] = 'mean'
            
        # RESAMPLING UNTUK INTERVAL WEEKLY/MONTHLY
        if interval == "Weekly":
            df_chart = df_stock.set_index('Last Trading Date').resample('W').agg(agg_dict).dropna(subset=['Close']).reset_index()
        elif interval == "Monthly":
            df_chart = df_stock.set_index('Last Trading Date').resample('M').agg(agg_dict).dropna(subset=['Close']).reset_index()
        else:
            df_chart = df_stock.copy()
        
        if len(df_chart) == 0:
            st.warning(f"Tidak ada data untuk interval {interval} dalam periode ini")
            st.stop()
        
        # Latest data untuk KPI
        latest = df_stock.iloc[-1]
        total_foreign = df_chart['Net Foreign Flow'].sum()
        avg_aoVol = df_chart['Avg_Order_Volume'].mean()
        max_aoVol = df_chart['AOVol_Ratio'].max()
        total_volume = df_chart['Volume'].sum() / 1e6
        
        # Hitung status berdasarkan trend 5 hari terakhir
        if len(df_chart) >= 5:
            recent_foreign = df_chart['Net Foreign Flow'].tail(5).sum()
            recent_prices = df_chart['Close'].tail(5)
            price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
            price_change_pct = (price_change / recent_prices.iloc[0] * 100) if recent_prices.iloc[0] > 0 else 0
        else:
            recent_foreign = total_foreign
            price_change = 0
            price_change_pct = 0
        
        # Tentukan status logic tetap utuh
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
        
        # KPI Cards (Tetap dipertahankan kecanggihannya)
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Harga Terkini", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: st.metric("Volume Terkini", f"{latest['Volume']/1e6:,.1f} Jt") # Diganti jadi terkini agar relevan
        with k3: st.metric("VWMA 20D Anchor", f"Rp {latest['VWMA_20D']:,.0f}" if 'VWMA_20D' in latest else "N/A")
        with k4: st.metric("Max AOVol Ratio", f"{max_aoVol:.1f}x" if max_aoVol > 0 else "N/A")
        with k5: 
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value' style='color:{status_color}; font-size:16px;'>{status_text}</div>
                <div class='kpi-label'>5D Foreign: Rp {recent_foreign/1e9:,.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
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
            x=df_chart['Last Trading Date'], open=df_chart['Open Price'],
            high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'],
            name="Price", showlegend=False,
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350'
        ), row=1, col=1)
        
        # TAMBAHAN V3.1: VWMA_20D Overlay Line
        if 'VWMA_20D' in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart['Last Trading Date'], y=df_chart['VWMA_20D'],
                mode='lines', name='⚓ VWMA 20D (Bandar Avg)',
                line=dict(color='blue', width=2, dash='dot'),
                hoverinfo='name+y'
            ), row=1, col=1)

        # BINTANG UNTUK AOVol SPIKES
        if 'AOVol_Ratio' in df_chart.columns:
            aoVol_spikes = df_chart[df_chart['AOVol_Ratio'] > 1.5].dropna(subset=['Close'])
            if not aoVol_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=aoVol_spikes['Last Trading Date'], y=aoVol_spikes['High'] * 1.02,
                    mode='markers', name='⭐ AOVol Spike',
                    marker=dict(symbol='star', size=14, color='gold', line=dict(width=2, color='orange')),
                    text=[f"AOVol: {x:.1f}x<br>Value: Rp {y:,.0f}" for x, y in zip(aoVol_spikes['AOVol_Ratio'], aoVol_spikes['Avg_Order_Volume'])],
                    hoverinfo='text'
                ), row=1, col=1)
        
        # BINTANG UNTUK BIG PLAYER ANOMALI
        if 'Big_Player_Anomaly' in df_chart.columns:
            anomaly_spikes = df_chart[df_chart['Big_Player_Anomaly'] > 3].dropna(subset=['Close'])
            if not anomaly_spikes.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_spikes['Last Trading Date'], y=anomaly_spikes['Low'] * 0.98,
                    mode='markers', name='💎 BP Anomaly',
                    marker=dict(symbol='diamond', size=12, color='magenta', line=dict(width=2, color='purple')),
                    text=[f"Anomali: {x:.1f}x<br>Harga: Rp {y:,.0f}" for x, y in zip(anomaly_spikes['Big_Player_Anomaly'], anomaly_spikes['Close'])],
                    hoverinfo='text'
                ), row=1, col=1)
        
        # PANEL 2: AOVOL ANALYSIS
        fig.add_trace(go.Scatter(
            x=df_chart['Last Trading Date'], y=df_chart['Avg_Order_Volume'] / 1e6,
            name='AOVol (Juta Rp)', mode='lines+markers',
            line=dict(color='purple', width=2.5), marker=dict(size=4, color='purple'),
            fill='tozeroy', fillcolor='rgba(128,0,128,0.1)'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_chart['Last Trading Date'], y=df_chart['AOVol_Ratio'],
            name='AOVol Ratio (x)', mode='lines+markers',
            line=dict(color='orange', width=2, dash='dash'),
            yaxis='y3', marker=dict(size=4, color='orange')
        ), row=2, col=1)
        
        # PANEL 3: VOLUME
        colors_vol = ['#26a69a' if row['Close'] >= row['Open Price'] else '#ef5350' for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'], y=df_chart['Volume'] / 1e6,
            name='Volume (Juta)', marker_color=colors_vol, showlegend=False
        ), row=3, col=1)

        # TAMBAHAN V3.1: MA20_vol Overlay Line
        if 'MA20_vol' in df_chart.columns:
            fig.add_trace(go.Scatter(
                x=df_chart['Last Trading Date'], y=df_chart['MA20_vol'] / 1e6,
                mode='lines', name='MA20 Volume',
                line=dict(color='black', width=1.5),
                hoverinfo='name+y'
            ), row=3, col=1)
        
        # PANEL 4: FOREIGN FLOW
        colors_ff = ['#26a69a' if val >= 0 else '#ef5350' for val in df_chart['Net Foreign Flow']]
        fig.add_trace(go.Bar(
            x=df_chart['Last Trading Date'], y=df_chart['Net Foreign Flow'] / 1e9,
            name='Foreign (Miliar)', marker_color=colors_ff, showlegend=False
        ), row=4, col=1)
        
        # Update layout
        fig.update_layout(
            height=1000, hovermode='x unified',
            margin=dict(t=80, b=40, l=40, r=80), xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, bgcolor='rgba(255,255,255,0.8)', font=dict(size=10)),
            title=dict(text=f"<b>Data: {len(df_chart)} periode • {df_chart['Last Trading Date'].iloc[0].strftime('%d-%b-%Y')} s/d {df_chart['Last Trading Date'].iloc[-1].strftime('%d-%b-%Y')}</b>", font=dict(size=12), y=0.99)
        )
        
        fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="AOVol (Juta Rp)", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Ratio (x)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Volume (Juta)", row=3, col=1)
        fig.update_yaxes(title_text="Foreign (M)", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # BROKER MUTATION ANALYSIS (KSEI - V3.1)
        st.subheader("🔄 Perubahan Kepemilikan KSEI 5% (Institusi Lokal vs Asing)")
        
        if len(df_kepemilikan) > 0 and 'Kode Efek' in df_kepemilikan.columns:
            ksei_stock = df_kepemilikan[df_kepemilikan['Kode Efek'] == selected_stock].copy()
            ksei_stock = ksei_stock.sort_values('Tanggal_Data')
            
            if len(ksei_stock) > 1:
                # Kolom Unik Rekening
                ksei_stock['Rekening_ID'] = ksei_stock['Kode Broker'].fillna('') + ' - ' + ksei_stock['Nama Rekening Efek'].fillna('')
                
                # Plot timeline Stacked Area
                st.markdown("#### 📅 Timeline Kepemilikan KSEI")
                ksei_pivot = ksei_stock.pivot_table(index='Tanggal_Data', columns='Rekening_ID', values='Jumlah Saham (Curr)', aggfunc='sum').fillna(0)
                
                if len(ksei_pivot.columns) > 0:
                    fig_timeline = go.Figure()
                    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
                    for i, rekening in enumerate(ksei_pivot.columns):
                        fig_timeline.add_trace(go.Scatter(
                            x=ksei_pivot.index, y=ksei_pivot[rekening] / 1e6,
                            name=rekening[:30] + '...' if len(rekening) > 30 else rekening,
                            mode='lines+markers', line=dict(width=2, color=colors[i % len(colors)]),
                            stackgroup='one'
                        ))
                    
                    fig_timeline.update_layout(
                        height=450, xaxis_title="Tanggal", yaxis_title="Jumlah Saham (Juta Lembar)",
                        hovermode='x unified', margin=dict(t=40, b=40, l=40, r=40),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10))
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # TABEL MUTASI DETAIL (TAMBAHAN STATUS LOKAL/ASING)
                    st.markdown("#### 📋 Detail Mutasi (Foreign vs Local Tracker)")
                    
                    all_mutations = []
                    for rekening in ksei_stock['Rekening_ID'].unique():
                        rek_data = ksei_stock[ksei_stock['Rekening_ID'] == rekening].sort_values('Tanggal_Data')
                        if len(rek_data) > 1:
                            for i in range(1, len(rek_data)):
                                prev, curr = rek_data.iloc[i-1], rek_data.iloc[i]
                                change = curr['Jumlah Saham (Curr)'] - prev['Jumlah Saham (Curr)']
                                
                                if change != 0:
                                    change_pct = (change / prev['Jumlah Saham (Curr)'] * 100) if prev['Jumlah Saham (Curr)'] > 0 else 0
                                    
                                    # Deteksi Status (Lokal/Asing)
                                    stat_raw = prev.get('Status', 'L')
                                    stat_tag = "🌐 ASING" if str(stat_raw).strip().upper() == 'A' else "🇮🇩 LOKAL"
                                    
                                    all_mutations.append({
                                        'Periode': f"{prev['Tanggal_Data'].strftime('%d/%m/%y')} - {curr['Tanggal_Data'].strftime('%d/%m/%y')}",
                                        'Tipe Investor': stat_tag,
                                        'Rekening / Broker': rekening,
                                        'Nama Pemegang': prev['Nama Pemegang Saham'],
                                        'Sebelum': prev['Jumlah Saham (Curr)'],
                                        'Sesudah': curr['Jumlah Saham (Curr)'],
                                        'Perubahan': change,
                                        'Perubahan %': change_pct,
                                        'Aksi': '🟢 AKUMULASI' if change > 0 else '🔴 DISTRIBUSI',
                                        'Abs_Perubahan': abs(change)
                                    })
                    
                    if all_mutations:
                        df_mutations = pd.DataFrame(all_mutations).sort_values('Abs_Perubahan', ascending=False)
                        
                        # Tampilkan 20 teratas
                        display_mut = df_mutations.head(20).drop(columns=['Abs_Perubahan']).copy()
                        
                        # Format angka memakai built-in Streamlit supaya bisa di-sort
                        st.dataframe(
                            display_mut, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Sebelum": st.column_config.NumberColumn("Sebelum (Lembar)", format="%d"),
                                "Sesudah": st.column_config.NumberColumn("Sesudah (Lembar)", format="%d"),
                                "Perubahan": st.column_config.NumberColumn("Mutasi (Lembar)", format="%d"),
                                "Perubahan %": st.column_config.NumberColumn("Mutasi %", format="%.1f%%")
                            }
                        )
                        
                        # Summary Akumulasi/Distribusi
                        st.markdown("#### 📊 Summary Mutasi")
                        total_akumulasi = df_mutations[df_mutations['Perubahan'] > 0]['Perubahan'].sum()
                        total_distribusi = abs(df_mutations[df_mutations['Perubahan'] < 0]['Perubahan'].sum())
                        net_change = total_akumulasi - total_distribusi
                        
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        col_sum1.metric("Total Akumulasi", f"{total_akumulasi/1e6:,.1f} Jt")
                        col_sum2.metric("Total Distribusi", f"{total_distribusi/1e6:,.1f} Jt")
                        col_sum3.metric("Net Change", f"{net_change/1e6:+,.1f} Jt")
                        col_sum4.metric("Rekening Aktif", df_mutations['Rekening / Broker'].nunique())
                    else:
                        st.info("Tidak ada perubahan kepemilikan dalam periode ini")
            else:
                st.info("Data kepemilikan masih terbatas (hanya 1 periode)")
        else:
            st.warning("Data KSEI 5% tidak tersedia")
    else:
        st.error(f"Data untuk saham {selected_stock} tidak ditemukan")

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
