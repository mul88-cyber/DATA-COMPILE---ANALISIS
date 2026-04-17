# ==========================================
# BANDARMOLOGY MASTER V4
# Lightweight Charts + ECharts Edition
# ==========================================
from pyecharts.commons.utils import JsCode
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
import json
from streamlit_lightweight_charts import render_streamlit_lightweight_charts
from streamlit_echarts import st_echarts

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="Bandarmology Master V4", 
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
    <h1 style='margin:0; font-size: 2rem;'>🐋 Bandarmology Master V4</h1>
    <p style='margin:0; opacity:0.8; font-size: 1rem;'>Lightweight Charts • ECharts • Volume Profile • Pattern Recognition</p>
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

        # Koreksi Data Asing (Ubah Lembar Jadi Rupiah)
        df_transaksi['Daily_VWAP'] = np.where(
            df_transaksi['Volume'] > 0, 
            df_transaksi['Value'] / df_transaksi['Volume'], 
            df_transaksi['Close']
        )
        
        if 'Net Foreign Flow' in df_transaksi.columns:
            df_transaksi['Net_Foreign_Volume'] = df_transaksi['Net Foreign Flow']
            df_transaksi['Net Foreign Flow'] = df_transaksi['Net Foreign Flow'] * df_transaksi['Daily_VWAP']
            df_transaksi['Foreign Buy'] = df_transaksi['Foreign Buy'] * df_transaksi['Daily_VWAP']
            df_transaksi['Foreign Sell'] = df_transaksi['Foreign Sell'] * df_transaksi['Daily_VWAP']

        # Hitung AOVol
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

        # Hitung VWMA 20D
        df_transaksi['TPxV'] = df_transaksi['Typical Price'] * df_transaksi['Volume']
        df_transaksi['Cum_TPxV'] = df_transaksi.groupby('Stock Code')['TPxV'].transform(
            lambda x: x.rolling(20, min_periods=1).sum()
        )
        df_transaksi['Cum_Vol'] = df_transaksi.groupby('Stock Code')['Volume'].transform(
            lambda x: x.rolling(20, min_periods=1).sum()
        )
        df_transaksi['VWMA_20D'] = df_transaksi['Cum_TPxV'] / df_transaksi['Cum_Vol'].replace(0, np.nan)
        
        # MA20 Volume
        df_transaksi['MA20_vol'] = df_transaksi.groupby('Stock Code')['Volume'].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )

        return df_transaksi, df_kepemilikan
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

df_transaksi, df_kepemilikan = load_and_preprocess_data()

if df_transaksi.empty: 
    st.stop()

unique_stocks = sorted(df_transaksi['Stock Code'].unique())
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

st.success(f"✅ Data siap: {len(df_transaksi):,} transaksi, {len(unique_stocks)} saham")

# ==========================================
# 3. FUNGSI BANTUAN & CACHE
# ==========================================

@st.cache_data(ttl=1800, show_spinner=False)
def prepare_chart_data(stock_code, interval, chart_len, max_date, period_map, _df_master):
    df_chart = _df_master[_df_master['Stock Code'] == stock_code].copy()
    
    if df_chart.empty:
        return None

    if chart_len != "Semua Data":
        days_back = period_map[chart_len]
        start_date_chart = max_date - timedelta(days=days_back)
        df_chart = df_chart[df_chart['Last Trading Date'].dt.date >= start_date_chart]
        
    df_chart = df_chart.sort_values('Last Trading Date')
    
    for col in ['Close', 'Open Price', 'High', 'Low', 'Previous']:
        if col in df_chart.columns:
            df_chart[col] = df_chart[col].replace(0, np.nan)

    if 'Previous' in df_chart.columns:
        df_chart['Close'] = df_chart['Close'].fillna(df_chart['Previous'])
    
    df_chart['Close'] = df_chart['Close'].ffill().bfill()
    df_chart['Open Price'] = df_chart['Open Price'].fillna(df_chart['Close'])
    df_chart['High'] = df_chart['High'].fillna(df_chart['Close'])
    df_chart['Low'] = df_chart['Low'].fillna(df_chart['Close'])
    
    agg_dict = {
        'Open Price': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum', 'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max',
        'Avg_Order_Volume': 'mean', 'AOVol_Ratio': 'max', 'Volume Spike (x)': 'max',
        'Change %': 'last'
    }
    
    if 'Tradeble Shares' in df_chart.columns:
        agg_dict['Tradeble Shares'] = 'last'
    if 'Free Float' in df_chart.columns:
        agg_dict['Free Float'] = 'last'
    if 'Volume_Pct_Tradeble' in df_chart.columns:
        agg_dict['Volume_Pct_Tradeble'] = 'sum'
    if 'VWMA_20D' in df_chart.columns:
        agg_dict['VWMA_20D'] = 'last'
    if 'MA20_vol' in df_chart.columns:
        agg_dict['MA20_vol'] = 'mean'
        
    if interval == "Weekly":
        df_chart = df_chart.set_index('Last Trading Date').resample('W-FRI').agg(agg_dict).dropna(subset=['Close']).reset_index()
    elif interval == "Monthly":
        df_chart = df_chart.set_index('Last Trading Date').resample('M').agg(agg_dict).dropna(subset=['Close']).reset_index()
    
    if len(df_chart) == 0:
        return None
        
    df_chart['Date_Label'] = df_chart['Last Trading Date'].dt.strftime('%d-%b-%Y')
    
    return df_chart

@st.cache_data(show_spinner=False)
def get_stock_mapping(_df):
    if 'Company Name' in _df.columns:
        return _df.drop_duplicates('Stock Code').set_index('Stock Code')['Company Name'].to_dict()
    return {}

@st.cache_data(show_spinner=False)
def get_public_shares_mapping(_df):
    if 'Tradeble Shares' in _df.columns and 'Free Float' in _df.columns:
        latest = _df.sort_values('Last Trading Date').groupby('Stock Code').last()
        return (latest['Tradeble Shares'] * (latest['Free Float'] / 100)).to_dict()
    return {}

@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_ksei_mutations(stock_code, _df_ksei):
    ksei_stock = _df_ksei[_df_ksei['Kode Efek'] == stock_code].copy()
    if len(ksei_stock) <= 1:
        return pd.DataFrame()
    
    ksei_stock['Rekening_ID'] = ksei_stock['Kode Broker'].fillna('') + ' - ' + ksei_stock['Nama Pemegang Saham'].fillna('')
    ksei_sorted = ksei_stock.sort_values(['Rekening_ID', 'Tanggal_Data']).copy()
    ksei_sorted['Prev_Holding'] = ksei_sorted.groupby('Rekening_ID')['Jumlah Saham (Curr)'].shift(1)
    ksei_sorted['Change'] = ksei_sorted['Jumlah Saham (Curr)'] - ksei_sorted['Prev_Holding']
    mutations = ksei_sorted[ksei_sorted['Change'] != 0].copy()
    
    if mutations.empty:
        return pd.DataFrame()
    
    mutations['Periode'] = mutations['Tanggal_Data'].dt.strftime('%G-W%V')
    result = pd.DataFrame({
        'Periode': mutations['Periode'],
        'Rekening / Broker': mutations['Rekening_ID'],
        'Sebelum': mutations['Prev_Holding'],
        'Sesudah': mutations['Jumlah Saham (Curr)'],
        'Perubahan': mutations['Change'],
        'Abs_Perubahan': mutations['Change'].abs()
    })
    return result

DICT_STOCK_NAME = get_stock_mapping(df_transaksi)
DICT_PUBLIC_SHARES = get_public_shares_mapping(df_transaksi)

def format_stock_label(code):
    name = DICT_STOCK_NAME.get(code, "")
    if name:
        return f"{code} - {name}"
    return code

# ==========================================
# 4. FUNGSI LIGHTWEIGHT CHARTS
# ==========================================

def create_lightweight_chart_data(df_chart):
    """Siapkan data untuk Lightweight Charts"""
    candles = []
    volumes = []
    foreigns = []
    
    for _, row in df_chart.iterrows():
        time_val = row['Last Trading Date'].strftime('%Y-%m-%d')
        
        candles.append({
            'time': time_val,
            'open': float(row['Open Price']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close'])
        })
        
        volumes.append({
            'time': time_val,
            'value': float(row['Volume']) / 1e6,
            'color': '#26a69a' if row['Close'] >= row['Open Price'] else '#ef5350'
        })
        
        foreigns.append({
            'time': time_val,
            'value': float(row['Net Foreign Flow']) / 1e9,
            'color': '#26a69a' if row['Net Foreign Flow'] >= 0 else '#ef5350'
        })
    
    return candles, volumes, foreigns

def render_lw_chart(candles, volumes, foreigns):
    """Render Lightweight Charts dengan HTML/JS"""
    
    chart_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #chart-container {{ width: 100%; height: 600px; }}
        </style>
    </head>
    <body>
        <div id="chart-container"></div>
        <script>
            const chart = LightweightCharts.createChart(document.getElementById('chart-container'), {{
                width: document.getElementById('chart-container').clientWidth,
                height: 600,
                layout: {{
                    background: {{ type: 'solid', color: 'white' }},
                    textColor: 'black',
                }},
                grid: {{
                    vertLines: {{ color: '#e6e6e6' }},
                    horzLines: {{ color: '#e6e6e6' }},
                }},
                crosshair: {{ mode: 1 }},
                rightPriceScale: {{ borderColor: '#D1D5DB' }},
                timeScale: {{ borderColor: '#D1D5DB', timeVisible: true }},
            }});
            
            // Candlestick Series
            const candleSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderUpColor: '#26a69a',
                borderDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                wickDownColor: '#ef5350',
            }});
            candleSeries.setData({json.dumps(candles)});
            
            // Volume Series
            const volumeSeries = chart.addHistogramSeries({{
                priceScaleId: 'volume',
                priceFormat: {{ type: 'volume' }},
                scaleMargins: {{ top: 0.8, bottom: 0 }},
            }});
            volumeSeries.setData({json.dumps(volumes)});
            
            // Foreign Flow Series
            const foreignSeries = chart.addHistogramSeries({{
                priceScaleId: 'foreign',
                priceFormat: {{ type: 'custom', formatter: (price) => price.toFixed(1) + 'B' }},
                scaleMargins: {{ top: 0.9, bottom: 0 }},
            }});
            foreignSeries.setData({json.dumps(foreigns)});
            
            // Handle resize
            window.addEventListener('resize', () => {{
                chart.applyOptions({{
                    width: document.getElementById('chart-container').clientWidth,
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    return chart_html

# ==========================================
# 5. FUNGSI ECHARTS & ANALISIS TAMBAHAN
# ==========================================

def create_echarts_bubble(ff_data):
    """Bubble chart interaktif dengan ECharts (FIXED)"""
    
    data_points = []
    for _, row in ff_data.iterrows():
        # Hitung symbol size langsung di Python agar tidak error di JS
        val_abs = float(abs(row['Net Foreign Flow']) / 1e9)
        symbol_size = np.sqrt(val_abs) * 15 + 10 
        
        data_points.append({
            "name": row['Stock Code'],
            "value": [
                float(row['Value'] / 1e9),
                float(row['Net Foreign Flow'] / 1e9),
                val_abs
            ],
            "symbolSize": symbol_size, # Menggunakan angka langsung
            "itemStyle": {
                "color": "#26a69a" if row['Net Foreign Flow'] >= 0 else "#ef5350"
            }
        })
    
    options = {
        "title": {
            "text": "Foreign Flow Bubble Map",
            "left": "center",
            "textStyle": {"fontSize": 16, "fontWeight": "bold"}
        },
        "tooltip": {
            "trigger": "item",
            # Gunakan JsCode agar fungsi JS tereksekusi dengan benar
            "formatter": JsCode("""
                function(params) {
                    return `<b>${params.name}</b><br/>
                            Nilai Transaksi: Rp ${params.value[0].toFixed(1)} M<br/>
                            Net Foreign: Rp ${params.value[1].toFixed(1)} M`;
                }
            """)
        },
        "xAxis": {
            "name": "Total Nilai Transaksi (Miliar Rp)",
            "nameLocation": "middle",
            "nameGap": 30,
            "type": "value",
            "splitLine": {"lineStyle": {"color": "#e0e0e0", "type": "dashed"}}
        },
        "yAxis": {
            "name": "Net Foreign Flow (Miliar Rp)",
            "nameLocation": "middle",
            "nameGap": 40,
            "type": "value",
            "splitLine": {"lineStyle": {"color": "#e0e0e0", "type": "dashed"}},
            "axisLabel": {"formatter": "{value} M"}
        },
        "series": [{
            "type": "scatter",
            "data": data_points,
            "label": {
                "show": True,
                "position": "top",
                "formatter": "{b}",
                "fontSize": 9,
                "fontWeight": "bold"
            },
            "emphasis": {
                "label": {"show": True, "fontWeight": "bold"},
                "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}
            }
        }],
        "grid": {
            "left": "12%",
            "right": "8%",
            "bottom": "15%",
            "containLabel": True
        }
    }
    
    return options

def create_volume_profile(df_stock, bins=50):
    """Membuat Volume Profile"""
    
    df_recent = df_stock.tail(60).copy()
    price_min, price_max = df_recent['Low'].min(), df_recent['High'].max()
    price_bins = np.linspace(price_min, price_max, bins)
    
    volume_profile = []
    for i in range(len(price_bins)-1):
        bin_low, bin_high = price_bins[i], price_bins[i+1]
        mask = (df_recent['Low'] <= bin_high) & (df_recent['High'] >= bin_low)
        volume_in_bin = df_recent.loc[mask, 'Volume'].sum()
        volume_profile.append({
            'price': (bin_low + bin_high) / 2,
            'volume': volume_in_bin
        })
    
    df_vp = pd.DataFrame(volume_profile)
    
    poc_idx = df_vp['volume'].idxmax()
    poc_price = df_vp.loc[poc_idx, 'price'] if not df_vp.empty else 0
    
    if not df_vp.empty:
        df_vp_sorted = df_vp.sort_values('volume', ascending=False)
        df_vp_sorted['cum_volume'] = df_vp_sorted['volume'].cumsum()
        df_vp_sorted['cum_pct'] = df_vp_sorted['cum_volume'] / df_vp_sorted['volume'].sum()
        va_70 = df_vp_sorted[df_vp_sorted['cum_pct'] <= 0.7]
        vah = va_70['price'].max() if not va_70.empty else price_max
        val = va_70['price'].min() if not va_70.empty else price_min
    else:
        vah, val = price_max, price_min
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_vp['price'],
        x=df_vp['volume'] / 1e6,
        orientation='h',
        name='Volume Profile',
        marker_color='rgba(100, 108, 255, 0.7)',
        hovertemplate='Harga: Rp %{y:,.0f}<br>Volume: %{x:,.1f} Juta<extra></extra>'
    ))
    
    fig.add_hline(y=poc_price, line_dash="dash", line_color="red", 
                  annotation_text=f"POC: Rp {poc_price:,.0f}")
    fig.add_hline(y=vah, line_dash="dot", line_color="orange", 
                  annotation_text=f"VAH: Rp {vah:,.0f}")
    fig.add_hline(y=val, line_dash="dot", line_color="orange", 
                  annotation_text=f"VAL: Rp {val:,.0f}")
    
    fig.update_layout(
        title="Volume Profile (60 Hari Terakhir)",
        xaxis_title="Volume (Juta Lembar)",
        yaxis_title="Harga (Rp)",
        height=500,
        hovermode='y unified',
        showlegend=False
    )
    
    return fig, {'poc': poc_price, 'vah': vah, 'val': val}

def detect_candlestick_patterns(df_chart):
    """Deteksi pola candlestick"""
    
    patterns = []
    df = df_chart.copy()
    
    df['body'] = abs(df['Close'] - df['Open Price'])
    df['upper_shadow'] = df['High'] - df[['Close', 'Open Price']].max(axis=1)
    df['lower_shadow'] = df[['Close', 'Open Price']].min(axis=1) - df['Low']
    df['body_ratio'] = df['body'] / (df['High'] - df['Low']).replace(0, 1)
    
    for i in range(1, len(df)):
        if df.iloc[i]['body_ratio'] < 0.1:
            patterns.append({'index': i, 'type': 'Doji', 'signal': 'Reversal'})
        elif (df.iloc[i]['lower_shadow'] > 2 * df.iloc[i]['body']) and \
             (df.iloc[i]['upper_shadow'] < 0.1 * df.iloc[i]['body']):
            patterns.append({'index': i, 'type': 'Hammer', 'signal': 'Bullish Reversal'})
        elif (df.iloc[i]['upper_shadow'] > 2 * df.iloc[i]['body']) and \
             (df.iloc[i]['lower_shadow'] < 0.1 * df.iloc[i]['body']):
            patterns.append({'index': i, 'type': 'Shooting Star', 'signal': 'Bearish Reversal'})
        
        if i > 0:
            if (df.iloc[i]['Close'] > df.iloc[i]['Open Price']) and \
               (df.iloc[i-1]['Close'] < df.iloc[i-1]['Open Price']) and \
               (df.iloc[i]['Open Price'] < df.iloc[i-1]['Close']) and \
               (df.iloc[i]['Close'] > df.iloc[i-1]['Open Price']):
                patterns.append({'index': i, 'type': 'Bullish Engulfing', 'signal': 'Strong Bullish'})
            elif (df.iloc[i]['Close'] < df.iloc[i]['Open Price']) and \
                 (df.iloc[i-1]['Close'] > df.iloc[i-1]['Open Price']) and \
                 (df.iloc[i]['Open Price'] > df.iloc[i-1]['Close']) and \
                 (df.iloc[i]['Close'] < df.iloc[i-1]['Open Price']):
                patterns.append({'index': i, 'type': 'Bearish Engulfing', 'signal': 'Strong Bearish'})
    
    return patterns

def calculate_market_breadth(df_transaksi):
    """Hitung Advance/Decline"""
    
    latest_date = df_transaksi['Last Trading Date'].max()
    df_latest = df_transaksi[df_transaksi['Last Trading Date'] == latest_date]
    
    advancing = len(df_latest[df_latest['Change %'] > 0])
    declining = len(df_latest[df_latest['Change %'] < 0])
    unchanged = len(df_latest[df_latest['Change %'] == 0])
    
    ad_ratio = advancing / declining if declining > 0 else advancing
    ad_line = advancing - declining
    
    net_advances = []
    for date in sorted(df_transaksi['Last Trading Date'].unique())[-20:]:
        df_date = df_transaksi[df_transaksi['Last Trading Date'] == date]
        net = len(df_date[df_date['Change %'] > 0]) - len(df_date[df_date['Change %'] < 0])
        net_advances.append(net)
    
    return {
        'advancing': advancing,
        'declining': declining,
        'unchanged': unchanged,
        'ad_ratio': ad_ratio,
        'ad_line': ad_line,
        'net_advances': net_advances,
        'breadth_status': 'Bullish' if ad_ratio > 1.5 else 'Bearish' if ad_ratio < 0.67 else 'Neutral'
    }

def create_market_breadth_chart(breadth_data):
    """Visualisasi Market Breadth dengan ECharts"""
    
    options = {
        "title": {
            "text": "Market Breadth Indicator (20 Hari)",
            "left": "center",
            "textStyle": {"fontSize": 14}
        },
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": [f"-{i}" for i in range(len(breadth_data['net_advances']), 0, -1)]
        },
        "yAxis": {
            "type": "value",
            "name": "Net Advances"
        },
        "series": [{
            "data": breadth_data['net_advances'],
            "type": "line",
            "smooth": True,
            "areaStyle": {
                "color": {
                    "type": "linear",
                    "x": 0, "y": 0, "x2": 0, "y2": 1,
                    "colorStops": [
                        {"offset": 0, "color": "rgba(38, 166, 154, 0.5)"},
                        {"offset": 1, "color": "rgba(239, 83, 80, 0.1)"}
                    ]
                }
            },
            "lineStyle": {"color": "#26a69a", "width": 2}
        }],
        "grid": {"left": "10%", "right": "5%", "bottom": "10%"}
    }
    
    return options

# ==========================================
# 6. DASHBOARD TABS
# ==========================================
tabs = st.tabs([
    "🎯 SCREENER PRO", 
    "🔍 DEEP DIVE & CHART", 
    "🏦 BROKER MUTASI",
    "🗺️ MARKET MAP"
])

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Whale & Float Absorption Radar 🚀")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            target_deteksi = st.radio("Target Deteksi:", ("🐋 Whale Accumulation (High AOV)", "🩸 Retail Panic / Mark-down (Low AOV)"))
        with col_m2:
            price_condition = st.selectbox(
                "Fase Pergerakan Harga (Price Context):",
                [
                    "🔍 SEMUA FASE (Tampilkan Semua)",
                    "💎 HIDDEN GEM (Sideways/Datar: -2% s/d +2%)", 
                    "⚓ BOTTOM FISHING (Lagi Turun: < VWMA atau Minus)",
                    "🚀 EARLY MOVE (Baru Mulai Naik: 0% s/d +4%)"
                ]
            )
        with col_m3:
            date_range = st.date_input("Periode Screener", value=(default_start, max_date))

        st.divider()

        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            min_avg_val = st.number_input("Min Rata-rata Nilai/Hari (M)", 0, 5000, 5) * 1e9
        with r1c2:
            min_price = st.number_input("Min Harga (Rp)", 0, 50000, 50)
        with r1c3:
            if "Whale" in target_deteksi:
                min_spikes = st.number_input("Min Whale Spikes (>1.5x)", 0, 50, 2)
            else:
                min_spikes = st.number_input("Min Retail Drops (<0.6x)", 0, 50, 2)
        with r1c4:
            min_turnover = st.slider("Min Serapan Float (%)", 0.0, 50.0, 1.0, 0.5)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & \
               (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if not df_filter.empty:
            with st.spinner("🔍 Memindai jejak Whale & pergerakan Ritel..."):
                df_filter['Is_Whale_Spike'] = (df_filter['AOVol_Ratio'] >= 1.5).astype(int)
                df_filter['Is_Retail_Drop'] = ((df_filter['AOVol_Ratio'] <= 0.6) & (df_filter['AOVol_Ratio'] > 0)).astype(int)
                
                summary = df_filter.groupby('Stock Code').agg(
                    Close=('Close', 'last'),
                    Last_Change=('Change %', 'last'), 
                    VWMA_20D=('VWMA_20D', 'last'),
                    Total_Value=('Value', 'sum'),
                    Total_Volume=('Volume', 'sum'),
                    Net_Foreign=('Net Foreign Flow', 'sum'),
                    Whale_Spikes=('Is_Whale_Spike', 'sum'),
                    Retail_Drops=('Is_Retail_Drop', 'sum'),
                    Max_Anomaly=('Big_Player_Anomaly', 'max'),
                    Max_AOV_Ratio=('AOVol_Ratio', 'max'), 
                    Trading_Days=('Last Trading Date', 'nunique')
                ).reset_index()
                
                summary['Avg_Daily_Value'] = summary['Total_Value'] / summary['Trading_Days'].replace(0, 1)
                summary['Avg_Daily_Volume'] = summary['Total_Volume'] / summary['Trading_Days'].replace(0, 1)
                summary['Public_Shares'] = summary['Stock Code'].map(DICT_PUBLIC_SHARES).fillna(0)
                summary['Turnover_Float_Pct'] = np.where(
                    summary['Public_Shares'] > 0, 
                    (summary['Avg_Daily_Volume'] / summary['Public_Shares']) * 100, 
                    0
                )
                
                summary = summary[summary['Avg_Daily_Value'] >= min_avg_val]
                summary = summary[summary['Close'] >= min_price]
                summary = summary[summary['Turnover_Float_Pct'] >= min_turnover]
                
                if "Whale" in target_deteksi:
                    summary = summary[summary['Whale_Spikes'] >= min_spikes]
                    target_col = 'Whale_Spikes'
                else:
                    summary = summary[summary['Retail_Drops'] >= min_spikes]
                    target_col = 'Retail_Drops'

                if price_condition == "💎 HIDDEN GEM (Sideways/Datar: -2% s/d +2%)":
                    summary = summary[(summary['Last_Change'] >= -2.0) & (summary['Last_Change'] <= 2.0)]
                elif price_condition == "⚓ BOTTOM FISHING (Lagi Turun: < VWMA atau Minus)":
                    summary = summary[(summary['Close'] < summary['VWMA_20D']) | (summary['Last_Change'] < 0)]
                elif price_condition == "🚀 EARLY MOVE (Baru Mulai Naik: 0% s/d +4%)":
                    summary = summary[(summary['Last_Change'] > 0) & (summary['Last_Change'] <= 4.0)]
                
                if "Whale" in target_deteksi:
                    summary['Conviction_Score'] = (
                        (summary['Whale_Spikes'] * 5) + 
                        (summary['Turnover_Float_Pct'] * 1.5) + 
                        (np.where(summary['Net_Foreign'] > 0, 10, 0)) + 
                        (summary['Max_Anomaly'] * 2) +
                        (summary['Max_AOV_Ratio'] * 2) 
                    )
                else:
                    summary['Conviction_Score'] = (
                        (summary['Retail_Drops'] * 5) + 
                        (summary['Turnover_Float_Pct'] * 1.0) + 
                        (np.where(summary['Net_Foreign'] < 0, 10, 0)) 
                    )
                
                summary = summary.sort_values('Conviction_Score', ascending=False).head(100)
                
                st.markdown(f"**🎯 Ditemukan {len(summary)} saham potensial**")
                
                if len(summary) > 0:
                    display_df = summary[['Stock Code', 'Close', 'Last_Change', 'Avg_Daily_Value', 'Turnover_Float_Pct', 
                                         target_col, 'Max_AOV_Ratio', 'Net_Foreign', 'Max_Anomaly', 'Conviction_Score']].copy()
                    
                    display_df['Avg_Daily_Value'] = display_df['Avg_Daily_Value'] / 1e9
                    display_df['Net_Foreign'] = display_df['Net_Foreign'] / 1e9
                    
                    col_target_name = "Whale Spikes" if "Whale" in target_deteksi else "Retail Drops"
                    
                    display_df.columns = ['Kode', 'Harga', 'Change Terakhir', 'Avg Value/Hari (M)', '% Serap Float/Hari', 
                                         col_target_name, 'Max AOV Ratio', 'Net Foreign (M)', 'Max Anomali', 'Conviction Score']
                    
                    styled_df = display_df.style
                    
                    if "Whale" in target_deteksi:
                        styled_df = styled_df.background_gradient(subset=['Max AOV Ratio'], cmap='Greens', vmin=1.5, vmax=5.0)
                        styled_df = styled_df.background_gradient(subset=['Conviction Score'], cmap='Greens')
                        styled_df = styled_df.background_gradient(subset=[col_target_name], cmap='Greens')
                    else:
                        styled_df = styled_df.background_gradient(subset=['Max AOV Ratio'], cmap='Reds_r', vmin=0.0, vmax=0.6)
                        styled_df = styled_df.background_gradient(subset=['Conviction Score'], cmap='Reds')
                        styled_df = styled_df.background_gradient(subset=[col_target_name], cmap='Reds')
                    
                    styled_df = styled_df.background_gradient(subset=['% Serap Float/Hari'], cmap='Blues')
                    styled_df = styled_df.background_gradient(subset=['Max Anomali'], cmap='Purples')
                    styled_df = styled_df.background_gradient(subset=['Avg Value/Hari (M)'], cmap='Oranges')
                    
                    def color_pos_neg(val):
                        if val > 0: return 'color: #10b981; font-weight: bold;'
                        if val < 0: return 'color: #ef4444; font-weight: bold;'
                        return ''
                    
                    styled_df = styled_df.map(color_pos_neg, subset=['Change Terakhir', 'Net Foreign (M)'])
                    
                    st.dataframe(
                        styled_df, 
                        use_container_width=True, 
                        hide_index=True,
                        height=550,
                        column_config={
                            "Harga": st.column_config.NumberColumn("Harga", format="Rp %d"),
                            "Change Terakhir": st.column_config.NumberColumn("Change", format="%+.2f %%"),
                            "Avg Value/Hari (M)": st.column_config.NumberColumn("Avg Value/Hari (M)", format="Rp %.1f M"),
                            "% Serap Float/Hari": st.column_config.NumberColumn("% Serap/Hari", format="%.2f %%"),
                            col_target_name: st.column_config.NumberColumn(col_target_name, format="%d Kali"),
                            "Max AOV Ratio": st.column_config.NumberColumn("Max AOV Ratio", format="%.2f x"),
                            "Net Foreign (M)": st.column_config.NumberColumn("Net Foreign (M)", format="Rp %.1f M"),
                            "Max Anomali": st.column_config.NumberColumn("Max Anomali", format="%.1f x"),
                            "Conviction Score": st.column_config.NumberColumn("Conviction Score", format="%.1f")
                        }
                    )
                else:
                    st.info("Tidak ada saham yang memenuhi kriteria fase ini.")

# ==================== TAB 2: DEEP DIVE & CHART ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive: Multi-Timeframe Analytics & VWMA Anchor")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_stock = st.selectbox("Pilih Saham", unique_stocks, format_func=format_stock_label, key='dd_stock')
    with col2:
        interval = st.selectbox("Interval", ["Daily", "Weekly", "Monthly"], key='dd_interval')
    with col3:
        chart_len = st.selectbox("Periode", ["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun", "Semua Data"], 
                                index=1, key='dd_len')
    
    period_map = {"3 Bulan": 90, "6 Bulan": 180, "1 Tahun": 365, "2 Tahun": 730, "Semua Data": 9999}
    
    with st.spinner("📊 Memproses chart data..."):
        df_chart = prepare_chart_data(selected_stock, interval, chart_len, max_date, period_map, df_transaksi)
    
    if df_chart is None or len(df_chart) == 0:
        st.warning(f"Tidak ada data transaksi untuk {selected_stock} pada interval {interval} dalam periode ini.")
    else:
        df_stock_raw = df_transaksi[df_transaksi['Stock Code'] == selected_stock]
        latest = df_stock_raw.iloc[-1]
        
        total_foreign = df_chart['Net Foreign Flow'].sum()
        aovol_spike_count = len(df_chart[df_chart['AOVol_Ratio'] > 1.5])
        
        if len(df_chart) >= 5:
            recent_foreign = df_chart['Net Foreign Flow'].tail(5).sum()
            recent_prices = df_chart['Close'].tail(5)
            price_change = recent_prices.iloc[-1] - recent_prices.iloc[0]
            price_change_pct = (price_change / recent_prices.iloc[0] * 100) if recent_prices.iloc[0] > 0 else 0
        else:
            recent_foreign = df_chart['Net Foreign Flow'].sum()
            price_change = 0
            price_change_pct = 0
        
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
        
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: st.metric("Harga Terkini", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.2f}%")
        with k2: st.metric("Volume Terkini", f"{latest['Volume']/1e6:,.1f} Jt Lbr") 
        with k3: st.metric("VWMA 20D Anchor", f"Rp {latest['VWMA_20D']:,.0f}" if 'VWMA_20D' in latest else "N/A")
        with k4: st.metric("AOVol Spikes (>1.5x)", f"{aovol_spike_count} Kali")
        with k5: 
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value' style='color:{status_color}; font-size:16px;'>{status_text}</div>
                <div class='kpi-label'>5-Bar Foreign: Rp {recent_foreign/1e9:,.1f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<p style='font-size:14px; color:gray; font-weight:bold; margin-top:10px; margin-bottom:5px;'>💧 Float & Liquidity Profile</p>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns(4)
            
            free_float_pct = latest.get('Free Float', 0)
            tradeble_shrs = latest.get('Tradeble Shares', 0)
            public_shares_vol = DICT_PUBLIC_SHARES.get(selected_stock, 0)
            float_mc = tradeble_shrs * latest['Close']
            turnover_pct = latest.get('Volume_Pct_Tradeble', 0)
            
            f1.metric("Free Float (%)", f"{free_float_pct:.2f}%")
            f2.metric("Public Shares", f"{public_shares_vol/1e6:,.1f} Jt Lbr")
            f3.metric("Float Market Cap", f"Rp {float_mc/1e9:,.1f} Miliar")
            f4.metric("Daily Turnover", f"{turnover_pct:.2f}% dari Float")
        
        st.divider()
        
        # Sub-tabs untuk chart dan analisis tambahan
        subtab1, subtab2, subtab3 = st.tabs(["📊 Chart Utama", "🎯 Volume Profile", "🧠 Pattern Recognition"])
        
        with subtab1:
            st.markdown("#### Candlestick Chart")
            
            chart_type = st.radio("Chart Library", ["Lightweight Charts (Fast)", "Plotly (Rich Features)"], horizontal=True)
            
            if chart_type == "Lightweight Charts (Fast)":
                try:
                    candles, volumes, foreigns = create_lightweight_chart_data(df_chart)
                    lw_html = render_lw_chart(candles, volumes, foreigns)
                    st.components.v1.html(lw_html, height=650)
                except Exception as e:
                    st.warning(f"Lightweight Charts gagal load, fallback ke Plotly")
                    chart_type = "Plotly (Rich Features)"
            
            if chart_type == "Plotly (Rich Features)":
                fig = make_subplots(
                    rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.35, 0.2, 0.2, 0.25],
                    subplot_titles=(
                        "<b>Price Action, VWMA Anchor & Big Player Signal</b>",
                        "<b>Average Order Volume (AOVol) Tracking</b>",
                        "<b>Volume & Participation</b>",
                        "<b>Net Foreign Flow</b>"
                    )
                )
                
                fig.add_trace(go.Candlestick(
                    x=df_chart['Date_Label'], open=df_chart['Open Price'],
                    high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'],
                    name="Price", showlegend=False,
                    increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                ), row=1, col=1)
                
                if 'VWMA_20D' in df_chart.columns:
                    fig.add_trace(go.Scatter(
                        x=df_chart['Date_Label'], y=df_chart['VWMA_20D'],
                        mode='lines', name='⚓ VWMA 20D',
                        line=dict(color='blue', width=2, dash='dot')
                    ), row=1, col=1)

                if 'AOVol_Ratio' in df_chart.columns:
                    aoVol_spikes = df_chart[df_chart['AOVol_Ratio'] > 1.5].dropna(subset=['Close'])
                    if not aoVol_spikes.empty:
                        fig.add_trace(go.Scatter(
                            x=aoVol_spikes['Date_Label'], y=aoVol_spikes['High'] * 1.02,
                            mode='markers', name='⭐ AOVol Spike',
                            marker=dict(symbol='star', size=14, color='gold', line=dict(width=2, color='orange'))
                        ), row=1, col=1)
                
                if 'Big_Player_Anomaly' in df_chart.columns:
                    anomaly_spikes = df_chart[df_chart['Big_Player_Anomaly'] > 3].dropna(subset=['Close'])
                    if not anomaly_spikes.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_spikes['Date_Label'], y=anomaly_spikes['Low'] * 0.98,
                            mode='markers', name='💎 BP Anomaly',
                            marker=dict(symbol='diamond', size=12, color='magenta', line=dict(width=2, color='purple'))
                        ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_chart['Date_Label'], y=df_chart['AOVol_Ratio'],
                    mode='lines', line=dict(color='#9c88ff', width=2), name='AOV Ratio'
                ), row=2, col=1)
                
                fig.add_hline(y=1.5, line_dash="dash", line_color="green", row=2, col=1)
                fig.add_hline(y=0.6, line_dash="dash", line_color="red", row=2, col=1)
                
                colors_vol = np.where(
                    df_chart['Close'].fillna(0) >= df_chart['Open Price'].fillna(0), 
                    '#26a69a', '#ef5350'
                ).tolist()
                
                fig.add_trace(go.Bar(
                    x=df_chart['Date_Label'], y=df_chart['Volume'] / 1e6,
                    name='Volume (Juta Lembar)', marker_color=colors_vol, showlegend=False
                ), row=3, col=1)

                if 'MA20_vol' in df_chart.columns:
                    fig.add_trace(go.Scatter(
                        x=df_chart['Date_Label'], y=df_chart['MA20_vol'] / 1e6,
                        mode='lines', name='MA20 Vol',
                        line=dict(color='black', width=1.5)
                    ), row=3, col=1)
                
                colors_ff = np.where(df_chart['Net Foreign Flow'] >= 0, '#26a69a', '#ef5350').tolist()
                fig.add_trace(go.Bar(
                    x=df_chart['Date_Label'], y=df_chart['Net Foreign Flow'] / 1e9,
                    name='Foreign (Miliar Rp)', marker_color=colors_ff, showlegend=False
                ), row=4, col=1)
                
                fig.update_layout(
                    height=1000, hovermode='x unified',
                    margin=dict(t=50, b=20, l=10, r=10),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                fig.update_xaxes(type='category', categoryorder='trace', tickangle=45, nticks=20)
                fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
                fig.update_yaxes(title_text="AOVol Ratio", row=2, col=1)
                fig.update_yaxes(title_text="Vol (Jt Lbr)", row=3, col=1)
                fig.update_yaxes(title_text="Foreign (M)", row=4, col=1)
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.divider()
            st.markdown("#### 📊 Market Breadth Context")
            breadth_data = calculate_market_breadth(df_transaksi)
            
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("Advancing", breadth_data['advancing'], delta="Bullish" if breadth_data['advancing'] > breadth_data['declining'] else "")
            bc2.metric("Declining", breadth_data['declining'], delta="Bearish" if breadth_data['declining'] > breadth_data['advancing'] else "")
            bc3.metric("A/D Ratio", f"{breadth_data['ad_ratio']:.2f}")
            bc4.metric("Market Status", breadth_data['breadth_status'])
            
            st_echarts(options=create_market_breadth_chart(breadth_data), height="300px")
        
        with subtab2:
            st.markdown("#### 🎯 Volume Profile Analysis")
            
            vp_fig, vp_levels = create_volume_profile(df_stock_raw)
            st.plotly_chart(vp_fig, use_container_width=True)
            
            col_vp1, col_vp2, col_vp3 = st.columns(3)
            col_vp1.metric("Point of Control (POC)", f"Rp {vp_levels['poc']:,.0f}")
            col_vp2.metric("Value Area High (VAH)", f"Rp {vp_levels['vah']:,.0f}")
            col_vp3.metric("Value Area Low (VAL)", f"Rp {vp_levels['val']:,.0f}")
            
            current_price = df_stock_raw.iloc[-1]['Close']
            if current_price > vp_levels['vah']:
                st.info(f"💡 Harga di ATAS Value Area - Potensi overbought atau breakout")
            elif current_price < vp_levels['val']:
                st.info(f"💡 Harga di BAWAH Value Area - Potensi oversold atau breakdown")
            elif vp_levels['poc'] > 0 and abs(current_price - vp_levels['poc']) / vp_levels['poc'] < 0.02:
                st.info(f"💡 Harga di sekitar POC - Area equilibrium/fair value")
        
        with subtab3:
            st.markdown("#### 🧠 AI Pattern Recognition")
            
            patterns = detect_candlestick_patterns(df_chart)
            
            if patterns:
                df_patterns = pd.DataFrame(patterns)
                df_patterns['Date'] = df_patterns['index'].map(lambda x: df_chart.iloc[x]['Date_Label'])
                df_patterns['Price'] = df_patterns['index'].map(lambda x: df_chart.iloc[x]['Close'])
                
                display_pat = df_patterns[['Date', 'type', 'signal', 'Price']].copy()
                display_pat.columns = ['Tanggal', 'Pola', 'Sinyal', 'Harga']
                
                def color_signal(val):
                    if 'Bullish' in val: return 'color: #10b981; font-weight: bold'
                    elif 'Bearish' in val: return 'color: #ef4444; font-weight: bold'
                    return ''
                
                styled_pat = display_pat.style.map(color_signal, subset=['Sinyal'])
                
                st.dataframe(
                    styled_pat,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Tanggal": st.column_config.TextColumn("Tanggal"),
                        "Pola": st.column_config.TextColumn("Pola Candlestick"),
                        "Sinyal": st.column_config.TextColumn("Interpretasi"),
                        "Harga": st.column_config.NumberColumn("Harga Saat Itu", format="Rp %.0f")
                    }
                )
                
                success_rate = []
                for i, pat in enumerate(patterns):
                    if pat['index'] < len(df_chart) - 5:
                        future_return = (df_chart.iloc[pat['index'] + 5]['Close'] - df_chart.iloc[pat['index']]['Close']) / df_chart.iloc[pat['index']]['Close']
                        if ('Bullish' in pat['signal'] and future_return > 0) or ('Bearish' in pat['signal'] and future_return < 0):
                            success_rate.append(1)
                        else:
                            success_rate.append(0)
                
                if success_rate:
                    accuracy = sum(success_rate) / len(success_rate) * 100
                    st.metric("Historical Accuracy (5-bar forward)", f"{accuracy:.1f}%", 
                             delta="Reliable" if accuracy > 60 else "Caution")
            else:
                st.info("Tidak ada pola candlestick signifikan terdeteksi dalam periode ini")

# ==================== TAB 3: KSEI SCREENER & BROKER RADAR ====================
with tabs[2]:
    st.markdown("### 🏦 KSEI Master: Stock Screener & Broker Radar")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        with st.container():
            st.markdown('<div class="filter-container">', unsafe_allow_html=True)
            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
            with col_b1:
                mutasi_period = st.selectbox("Periode Analisis", ["1 Minggu", "2 Minggu", "1 Bulan", "3 Bulan", "6 Bulan"], key='m_period')
                days_map = {"1 Minggu": 7, "2 Minggu": 14, "1 Bulan": 30, "3 Bulan": 90, "6 Bulan": 180}
            with col_b2:
                min_mutasi = st.number_input("Min Mutasi/Rekening (Juta Lbr)", 0, 1000, 5) * 1e6
            with col_b3:
                mode_ksei = st.selectbox("Tampilkan Mode:", ["🎯 KSEI Stock Screener", "🕵️ Top Broker Radar"])
            with col_b4:
                if mode_ksei == "🎯 KSEI Stock Screener":
                    sort_by = st.selectbox("Urutkan Berdasarkan:", ["🔥 Highest % Serap Float", "🟢 Top Net Accumulation (Lbr)", "🔴 Top Net Distribution (Lbr)"])
                else:
                    top_n = st.slider("Top N Broker", 5, 50, 15)
            st.markdown('</div>', unsafe_allow_html=True)
        
        max_date_ksei = df_kepemilikan['Tanggal_Data'].max()
        start_mutasi = max_date_ksei - timedelta(days=days_map[mutasi_period])
        df_ksei_period = df_kepemilikan[df_kepemilikan['Tanggal_Data'] >= start_mutasi].copy()
        
        if not df_ksei_period.empty:
            with st.spinner("⏳ Menghitung mutasi KSEI..."):
                mutasi = df_ksei_period.sort_values('Tanggal_Data').groupby(['Kode Broker', 'Kode Efek']).agg(
                    Awal=('Jumlah Saham (Curr)', 'first'),
                    Akhir=('Jumlah Saham (Curr)', 'last'),
                    Nama=('Nama Pemegang Saham', 'first')
                ).reset_index()
                
                mutasi['Net_Change'] = mutasi['Akhir'] - mutasi['Awal']
                mutasi = mutasi[abs(mutasi['Net_Change']) >= min_mutasi]
            
            if len(mutasi) > 0:
                if mode_ksei == "🎯 KSEI Stock Screener":
                    st.markdown(f"#### 🎯 Saham dengan Perubahan KSEI Paling Ekstrem ({mutasi_period})")
                    
                    mutasi_sorted = mutasi.sort_values('Net_Change', key=abs, ascending=False)
                    mutasi_sorted['Aktor'] = mutasi_sorted['Kode Broker'].fillna(mutasi_sorted['Nama'].str[:15])
                    
                    stock_ksei = mutasi_sorted.groupby('Kode Efek').agg(
                        Akumulasi=('Net_Change', lambda x: x[x > 0].sum()),
                        Distribusi=('Net_Change', lambda x: x[x < 0].sum()),
                        Total_Net_Change=('Net_Change', 'sum'),
                        Aktor_Utama=('Aktor', lambda x: ", ".join(x.dropna().unique()[:3]))
                    ).reset_index()
                    
                    stock_ksei['Saham'] = stock_ksei['Kode Efek'] + " - " + stock_ksei['Kode Efek'].map(DICT_STOCK_NAME).fillna('')
                    stock_ksei['Public_Shares'] = stock_ksei['Kode Efek'].map(DICT_PUBLIC_SHARES).fillna(0)
                    stock_ksei['Net_Float_Absorbed'] = np.where(
                        stock_ksei['Public_Shares'] > 0, 
                        (stock_ksei['Total_Net_Change'] / stock_ksei['Public_Shares']) * 100, 
                        0
                    )
                    
                    if sort_by == "🔥 Highest % Serap Float":
                        stock_ksei = stock_ksei.sort_values('Net_Float_Absorbed', ascending=False)
                    elif sort_by == "🟢 Top Net Accumulation (Lbr)":
                        stock_ksei = stock_ksei.sort_values('Total_Net_Change', ascending=False)
                    else:
                        stock_ksei = stock_ksei.sort_values('Total_Net_Change', ascending=True)
                        
                    display_ksei = stock_ksei[['Saham', 'Aktor_Utama', 'Akumulasi', 'Distribusi', 'Total_Net_Change', 'Net_Float_Absorbed']].head(100).copy()
                    
                    styled_ksei = display_ksei.style
                    styled_ksei = styled_ksei.background_gradient(subset=['Net_Float_Absorbed'], cmap='RdYlGn', vmin=-5, vmax=5)
                    
                    def color_ksei_net(val):
                        if val > 0: return 'color: #10b981; font-weight: bold;'
                        if val < 0: return 'color: #ef4444; font-weight: bold;'
                        return ''
                    styled_ksei = styled_ksei.map(color_ksei_net, subset=['Total_Net_Change'])
                    
                    styled_ksei = styled_ksei.format({
                        'Akumulasi': '{:,.0f}',
                        'Distribusi': '{:,.0f}',
                        'Total_Net_Change': '{:+,.0f}',
                        'Net_Float_Absorbed': '{:+.2f}%'
                    })
                    
                    st.dataframe(
                        styled_ksei,
                        use_container_width=True, hide_index=True, height=600,
                        column_config={
                            "Saham": st.column_config.TextColumn("Saham", width="medium"),
                            "Aktor_Utama": st.column_config.TextColumn("Top 3 Aktor", width="medium"),
                            "Akumulasi": st.column_config.Column("Total Serokan (Lbr)"),
                            "Distribusi": st.column_config.Column("Total Buangan (Lbr)"),
                            "Total_Net_Change": st.column_config.Column("Net Mutasi (Lbr)"),
                            "Net_Float_Absorbed": st.column_config.Column("Net % Serap Float")
                        }
                    )
                    
                else:
                    broker_summary = mutasi.groupby('Kode Broker').agg({
                        'Net_Change': 'sum',
                        'Kode Efek': lambda x: list(x)
                    }).reset_index()
                    broker_summary.columns = ['Kode Broker', 'Total Mutasi', 'List Saham']
                    
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
            else:
                st.info(f"Tidak ada mutasi ≥ {min_mutasi/1e6:.0f} juta dalam periode ini")
        else:
            st.warning("Tidak ada data KSEI dalam periode ini")

# ==================== TAB 4: MARKET MAP ====================
with tabs[3]:
    st.markdown("### 🗺️ Market Map & Foreign Radar")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        col_period, col_top, col_sort = st.columns(3)
        with col_period:
            ff_period = st.selectbox("Rentang Waktu", 
                                    ["Hari Ini", "5 Hari", "10 Hari", "20 Hari", "30 Hari", "60 Hari"], 
                                    key='ff_time')
        with col_top:
            top_n_ff = st.slider("Tampilkan Top N Saham", 5, 50, 20, key='top_ff')
        with col_sort:
            sort_ff = st.selectbox("Urutkan Bubble Chart Berdasarkan", ["Net Foreign", "Nilai Transaksi", "Volume"], key='sort_ff')
        st.markdown('</div>', unsafe_allow_html=True)
        
    days_ff_map = {"Hari Ini": 0, "5 Hari": 5, "10 Hari": 10, "20 Hari": 20, "30 Hari": 30, "60 Hari": 60}
    days_back = days_ff_map[ff_period]
    
    if days_back == 0:
        mask_ff = (df_transaksi['Last Trading Date'].dt.date == max_date)
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
        
        total_inflow = ff_data[ff_data['Net Foreign Flow'] > 0]['Net Foreign Flow'].sum()
        total_outflow = ff_data[ff_data['Net Foreign Flow'] < 0]['Net Foreign Flow'].sum()
        net_market_flow = total_inflow + total_outflow
        
        st.markdown("#### 🌊 Market Foreign Flow Summary")
        mk1, mk2, mk3 = st.columns(3)
        mk1.metric("Gross Foreign Inflow", f"Rp {total_inflow/1e9:,.1f} Miliar")
        mk2.metric("Gross Foreign Outflow", f"Rp {abs(total_outflow)/1e9:,.1f} Miliar")
        mk3.metric("Net Market Foreign Flow", f"Rp {net_market_flow/1e9:+,.1f} Miliar", 
                   "Bullish" if net_market_flow > 0 else "Bearish")
        
        st.divider()
        
        ff_data['Nama'] = ff_data['Stock Code'].map(DICT_STOCK_NAME).fillna('-')
        ff_data['Saham'] = ff_data['Stock Code'] + " - " + ff_data['Nama']
        
        sort_col = {"Net Foreign": "Net Foreign Flow", "Nilai Transaksi": "Value", "Volume": "Volume"}[sort_ff]
        ff_viz = ff_data.sort_values(sort_col, ascending=False).head(75).copy()
        ff_viz['Abs_Net'] = abs(ff_viz['Net Foreign Flow'])
        
        st.markdown("#### 📊 Foreign Flow Scatter Map (Interactive ECharts)")
        st_echarts(options=create_echarts_bubble(ff_viz), height="600px")
        
        st.divider()
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            st.markdown(f"#### 🟢 Top {top_n_ff} Foreign Buy ({ff_period})")
            top_buy = ff_data.nlargest(top_n_ff, 'Net Foreign Flow')
            if not top_buy.empty:
                display_buy = top_buy[['Saham', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_buy['Net Foreign Flow (M)'] = display_buy['Net Foreign Flow'] / 1e9
                display_buy = display_buy.drop(columns=['Net Foreign Flow'])
                
                styled_buy = display_buy.style
                styled_buy = styled_buy.background_gradient(subset=['Net Foreign Flow (M)'], cmap='Greens')
                
                def color_pos_neg(val):
                    if val > 0: return 'color: #10b981; font-weight: bold;'
                    if val < 0: return 'color: #ef4444; font-weight: bold;'
                    return ''
                styled_buy = styled_buy.map(color_pos_neg, subset=['Change %'])
                
                st.dataframe(
                    styled_buy, 
                    use_container_width=True, hide_index=True, height=500,
                    column_config={
                        "Saham": st.column_config.TextColumn("Saham", width="medium"),
                        "Close": st.column_config.NumberColumn("Harga", format="Rp %d"),
                        "Net Foreign Flow (M)": st.column_config.NumberColumn("Net Buy (M)", format="Rp %.1f M"),
                        "Change %": st.column_config.NumberColumn("Change", format="%+.2f%%")
                    }
                )
            
        with col_f2:
            st.markdown(f"#### 🔴 Top {top_n_ff} Foreign Sell ({ff_period})")
            top_sell = ff_data.nsmallest(top_n_ff, 'Net Foreign Flow')
            if not top_sell.empty:
                display_sell = top_sell[['Saham', 'Close', 'Net Foreign Flow', 'Change %']].copy()
                display_sell['Net Foreign Flow (M)'] = display_sell['Net Foreign Flow'] / 1e9
                display_sell = display_sell.drop(columns=['Net Foreign Flow'])
                
                styled_sell = display_sell.style
                styled_sell = styled_sell.background_gradient(subset=['Net Foreign Flow (M)'], cmap='Reds_r')
                styled_sell = styled_sell.map(color_pos_neg, subset=['Change %'])
                
                st.dataframe(
                    styled_sell, 
                    use_container_width=True, hide_index=True, height=500,
                    column_config={
                        "Saham": st.column_config.TextColumn("Saham", width="medium"),
                        "Close": st.column_config.NumberColumn("Harga", format="Rp %d"),
                        "Net Foreign Flow (M)": st.column_config.NumberColumn("Net Sell (M)", format="Rp %.1f M"),
                        "Change %": st.column_config.NumberColumn("Change", format="%+.2f%%")
                    }
                )

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

st.markdown("---")
st.caption("🐋 Bandarmology Master V4 • Lightweight Charts + ECharts Edition • Developed for Professional Analysis")
