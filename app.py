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

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Bandarmology Dashboard", 
    layout="wide", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .warning-text {
        color: #ff4b4b;
        font-weight: 600;
    }
    .success-text {
        color: #00c853;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">📈 Dashboard Analisa Bandarmology Pro</p>', unsafe_allow_html=True)

# 1. Fungsi Load Data CSV dari GDrive
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

# === FILE ID GOOGLE DRIVE ===
FILE_ID_TRANSAKSI = "1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8" 
FILE_ID_KEPEMILIKAN = "1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr"

# 2. Proses Load Data dengan Progress Bar
with st.spinner('📊 Memuat jutaan baris data dari Google Drive...'):
    progress_bar = st.progress(0)
    
    df_transaksi = load_csv_from_gdrive(FILE_ID_TRANSAKSI)
    progress_bar.progress(50)
    
    df_kepemilikan = load_csv_from_gdrive(FILE_ID_KEPEMILIKAN)
    progress_bar.progress(100)
    
    progress_bar.empty()

if df_transaksi is None or df_kepemilikan is None:
    st.error("Gagal memuat data. Silakan cek koneksi dan file ID.")
    st.stop()

# Konversi kolom tanggal
df_transaksi['Last Trading Date'] = pd.to_datetime(df_transaksi['Last Trading Date'])
df_kepemilikan['Tanggal_Data'] = pd.to_datetime(df_kepemilikan['Tanggal_Data'])

# 3. Sidebar untuk Filter dan Parameter
st.sidebar.header("🔍 Filter & Parameter Analisa")

# Daftar saham
daftar_saham = sorted(df_transaksi['Stock Code'].dropna().unique().tolist())
saham_pilihan = st.sidebar.selectbox("Pilih Kode Saham:", daftar_saham)

# Rentang tanggal
min_date = df_transaksi['Last Trading Date'].min()
max_date = df_transaksi['Last Trading Date'].max()
date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(max_date - timedelta(days=30), max_date),
    min_value=min_date,
    max_value=max_date
)

# Parameter analisis
st.sidebar.subheader("⚙️ Parameter Analisis")
volume_spike_threshold = st.sidebar.slider(
    "Threshold Volume Spike (x dari rata-rata)",
    min_value=1.0, max_value=5.0, value=2.0, step=0.5
)

anomaly_threshold = st.sidebar.slider(
    "Threshold Big Player Anomaly (x dari avg order)",
    min_value=2.0, max_value=10.0, value=5.0, step=1.0
)

# 4. Filter Data berdasarkan Pilihan
mask_transaksi = (
    (df_transaksi['Stock Code'] == saham_pilihan) &
    (df_transaksi['Last Trading Date'].dt.date >= date_range[0]) &
    (df_transaksi['Last Trading Date'].dt.date <= date_range[1])
)
df_saham = df_transaksi[mask_transaksi].copy()

mask_kepemilikan = (
    (df_kepemilikan['Kode Efek'] == saham_pilihan) &
    (df_kepemilikan['Tanggal_Data'].dt.date >= date_range[0]) &
    (df_kepemilikan['Tanggal_Data'].dt.date <= date_range[1])
)
df_kepemilikan_filter = df_kepemilikan[mask_kepemilikan].copy()

if len(df_saham) == 0:
    st.warning(f"Tidak ada data untuk {saham_pilihan} pada rentang tanggal yang dipilih")
    st.stop()

# 5. METRICS UTAMA (Top Row)
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    last_close = df_saham['Close'].iloc[-1]
    prev_close = df_saham['Close'].iloc[-2] if len(df_saham) > 1 else last_close
    change_pct = ((last_close - prev_close) / prev_close * 100)
    st.metric(
        "Harga Terakhir", 
        f"Rp {last_close:,.0f}",
        f"{change_pct:.2f}%"
    )

with col2:
    total_volume = df_saham['Volume'].sum()
    st.metric("Total Volume", f"{total_volume:,.0f}")

with col3:
    avg_volume = df_saham['Volume'].mean()
    st.metric("Rata-rata Volume", f"{avg_volume:,.0f}")

with col4:
    net_foreign = df_saham['Net Foreign Flow'].sum()
    color = "success-text" if net_foreign > 0 else "warning-text"
    st.markdown(f'<div class="metric-card"><h3>Net Foreign Flow</h3><p class="{color}">Rp {net_foreign:,.0f}</p></div>', unsafe_allow_html=True)

with col5:
    total_anomaly = df_saham['Big_Player_Anomaly'].sum()
    st.metric("Total Anomaly", f"{total_anomaly:,.0f}")

# 6. CHART UTAMA - Price Action dengan Volume
st.subheader(f"📊 Price Action & Volume Analysis - {saham_pilihan}")

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.5, 0.25, 0.25],
    subplot_titles=("Harga & Indikator", "Volume", "Net Foreign Flow")
)

# Candlestick chart
fig.add_trace(
    go.Candlestick(
        x=df_saham['Last Trading Date'],
        open=df_saham['Open Price'],
        high=df_saham['High'],
        low=df_saham['Low'],
        close=df_saham['Close'],
        name="Harga",
        showlegend=False
    ),
    row=1, col=1
)

# Add VWMA
fig.add_trace(
    go.Scatter(
        x=df_saham['Last Trading Date'],
        y=df_saham['VWMA_20D'],
        name="VWMA 20D",
        line=dict(color='orange', width=2)
    ),
    row=1, col=1
)

# Volume bars dengan warna berdasarkan harga
colors = ['red' if close < open else 'green' for close, open in zip(df_saham['Close'], df_saham['Open Price'])]
fig.add_trace(
    go.Bar(
        x=df_saham['Last Trading Date'],
        y=df_saham['Volume'],
        name="Volume",
        marker_color=colors,
        showlegend=False
    ),
    row=2, col=1
)

# Volume MA
fig.add_trace(
    go.Scatter(
        x=df_saham['Last Trading Date'],
        y=df_saham['MA20_vol'],
        name="MA20 Volume",
        line=dict(color='purple', width=2, dash='dash')
    ),
    row=2, col=1
)

# Net Foreign Flow
fig.add_trace(
    go.Bar(
        x=df_saham['Last Trading Date'],
        y=df_saham['Net Foreign Flow'],
        name="Net Foreign",
        marker_color=['green' if x > 0 else 'red' for x in df_saham['Net Foreign Flow']],
        showlegend=False
    ),
    row=3, col=1
)

# Highlight volume spike
spike_dates = df_saham[df_saham['Volume Spike (x)'] > volume_spike_threshold]['Last Trading Date']
for date in spike_dates:
    fig.add_vline(x=date, line_width=1, line_dash="dash", line_color="red", opacity=0.3, row=2, col=1)

fig.update_layout(
    height=800,
    xaxis_rangeslider_visible=False,
    template='plotly_white',
    hovermode='x unified'
)

fig.update_yaxes(title_text="Harga", row=1, col=1)
fig.update_yaxes(title_text="Volume", row=2, col=1)
fig.update_yaxes(title_text="Net Foreign", row=3, col=1)

st.plotly_chart(fig, use_container_width=True)

# 7. ANALISIS BANDARMOLOGY
st.subheader("🎯 Analisis Big Player & Anomali")

col1, col2 = st.columns(2)

with col1:
    # Volume Spike Analysis
    st.markdown("### 📈 Volume Spike Detection")
    volume_spike = df_saham[df_saham['Volume Spike (x)'] > volume_spike_threshold].copy()
    
    if len(volume_spike) > 0:
        volume_spike_display = volume_spike[['Last Trading Date', 'Close', 'Volume', 'Volume Spike (x)', 'Net Foreign Flow']].sort_values('Volume Spike (x)', ascending=False)
        st.dataframe(volume_spike_display, use_container_width=True)
    else:
        st.info("Tidak ada volume spike dalam periode ini")
    
    # Typical Price Analysis
    st.markdown("### 📊 Typical Price Analysis")
    fig_tp = px.scatter(
        df_saham,
        x='Last Trading Date',
        y='Typical Price',
        size='TPxV',
        color='Change %',
        title="Typical Price vs Volume Weighted",
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_tp, use_container_width=True)

with col2:
    # Big Player Anomaly
    st.markdown("### 🐋 Big Player Anomaly")
    anomaly = df_saham[df_saham['Big_Player_Anomaly'] > anomaly_threshold].copy()
    
    if len(anomaly) > 0:
        anomaly_display = anomaly[['Last Trading Date', 'Close', 'Big_Player_Anomaly', 'Avg_Order_Volume', 'Volume', 'Net Foreign Flow']]
        st.dataframe(anomaly_display, use_container_width=True)
    else:
        st.info("Tidak ada anomali big player dalam periode ini")
    
    # Foreign Flow Analysis
    st.markdown("### 🌏 Foreign Flow Analysis")
    
    # Pie chart for foreign activity
    foreign_buy = df_saham['Foreign Buy'].sum()
    foreign_sell = df_saham['Foreign Sell'].sum()
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Foreign Buy', 'Foreign Sell'],
        values=[foreign_buy, foreign_sell],
        hole=.3,
        marker_colors=['green', 'red']
    )])
    fig_pie.update_layout(title="Komposisi Transaksi Asing")
    st.plotly_chart(fig_pie, use_container_width=True)

# 8. KEPEMILIKAN ANALISIS
st.subheader("👥 Analisis Perubahan Kepemilikan (KSEI 5%)")

if len(df_kepemilikan_filter) > 0:
    # Summary kepemilikan
    total_investors = df_kepemilikan_filter['Nama Pemegang Saham'].nunique()
    total_brokers = df_kepemilikan_filter['Kode Broker'].nunique()
    total_shares = df_kepemilikan_filter['Jumlah Saham (Curr)'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Pemegang Saham", total_investors)
    with col2:
        st.metric("Total Broker", total_brokers)
    with col3:
        st.metric("Total Saham (Juta)", f"{total_shares/1e6:.2f}M")
    with col4:
        # Persentase terhadap tradeble shares
        tradeble_shares = df_saham['Tradeble Shares'].iloc[-1] if len(df_saham) > 0 else 0
        if tradeble_shares > 0:
            pct = (total_shares / tradeble_shares) * 100
            st.metric("% dari Tradeble Shares", f"{pct:.2f}%")
    
    # Timeline perubahan kepemilikan
    st.markdown("### Timeline Perubahan Kepemilikan")
    
    # Group by tanggal untuk melihat total saham
    timeline = df_kepemilikan_filter.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
    
    fig_timeline = px.line(
        timeline,
        x='Tanggal_Data',
        y='Jumlah Saham (Curr)',
        title="Total Saham yang Dilaporkan (5% holders)",
        markers=True
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Top holders saat ini
    st.markdown("### 🏆 Top 10 Pemegang Saham Terbesar")
    latest_date = df_kepemilikan_filter['Tanggal_Data'].max()
    latest_holders = df_kepemilikan_filter[df_kepemilikan_filter['Tanggal_Data'] == latest_date].copy()
    latest_holders = latest_holders.nlargest(10, 'Jumlah Saham (Curr)')[['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)', 'Status']]
    
    st.dataframe(latest_holders, use_container_width=True)
    
    # Broker analysis
    st.markdown("### 🏦 Analisis Broker")
    broker_activity = df_kepemilikan_filter.groupby('Kode Broker').agg({
        'Jumlah Saham (Curr)': 'sum',
        'Nama Pemegang Saham': 'nunique'
    }).rename(columns={'Jumlah Saham (Curr)': 'Total Saham', 'Nama Pemegang Saham': 'Unique Holders'}).reset_index()
    broker_activity = broker_activity.sort_values('Total Saham', ascending=False)
    
    st.dataframe(broker_activity, use_container_width=True)
    
else:
    st.info(f"Tidak ada data kepemilikan untuk {saham_pilihan} dalam periode ini")

# 9. DATA TABLE DETAIL
st.subheader("📋 Detail Data Transaksi")

# Tabs untuk berbagai tampilan data
tab_detail, tab_summary, tab_export = st.tabs(["📊 Data Detail", "📈 Ringkasan Statistik", "💾 Export Data"])

with tab_detail:
    st.dataframe(df_saham, use_container_width=True)

with tab_summary:
    st.markdown("### Statistik Deskriptif")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Ringkasan Harga & Volume**")
        stats_price = df_saham[['Close', 'Volume', 'Value', 'Typical Price']].describe()
        st.dataframe(stats_price)
    
    with col2:
        st.markdown("**Ringkasan Foreign Flow**")
        stats_foreign = df_saham[['Foreign Buy', 'Foreign Sell', 'Net Foreign Flow']].describe()
        st.dataframe(stats_foreign)

with tab_export:
    st.markdown("### Export Data ke CSV")
    
    # Convert data to CSV
    csv_transaksi = df_saham.to_csv(index=False).encode('utf-8')
    csv_kepemilikan = df_kepemilikan_filter.to_csv(index=False).encode('utf-8')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download Data Transaksi",
            data=csv_transaksi,
            file_name=f"{saham_pilihan}_transaksi_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        st.download_button(
            label="📥 Download Data Kepemilikan",
            data=csv_kepemilikan,
            file_name=f"{saham_pilihan}_kepemilikan_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# 10. SIGNAL & INSIGHTS
st.subheader("💡 Signal & Insights")

# Generate signals berdasarkan data
signals = []

# Cek volume spike terbaru
latest_volume_spike = df_saham.iloc[-1]['Volume Spike (x)'] if len(df_saham) > 0 else 0
if latest_volume_spike > volume_spike_threshold:
    signals.append(("🔴 Volume Spike", f"Volume {latest_volume_spike:.2f}x dari rata-rata", "warning"))

# Cek net foreign
latest_net_foreign = df_saham.iloc[-1]['Net Foreign Flow'] if len(df_saham) > 0 else 0
if latest_net_foreign > 0:
    signals.append(("🟢 Net Foreign Beli", f"Net foreign beli Rp {latest_net_foreign:,.0f}", "success"))
elif latest_net_foreign < 0:
    signals.append(("🔴 Net Foreign Jual", f"Net foreign jual Rp {abs(latest_net_foreign):,.0f}", "warning"))

# Cek big player anomaly terbaru
latest_anomaly = df_saham.iloc[-1]['Big_Player_Anomaly'] if len(df_saham) > 0 else 0
if latest_anomaly > anomaly_threshold:
    signals.append(("🐋 Big Player Activity", f"Anomali big player: {latest_anomaly:.2f}x", "info"))

# Cek perubahan kepemilikan
if len(df_kepemilikan_filter) > 0:
    earliest_date = df_kepemilikan_filter['Tanggal_Data'].min()
    latest_date = df_kepemilikan_filter['Tanggal_Data'].max()
    
    if earliest_date != latest_date:
        earliest_shares = df_kepemilikan_filter[df_kepemilikan_filter['Tanggal_Data'] == earliest_date]['Jumlah Saham (Curr)'].sum()
        latest_shares = df_kepemilikan_filter[df_kepemilikan_filter['Tanggal_Data'] == latest_date]['Jumlah Saham (Curr)'].sum()
        
        pct_change = ((latest_shares - earliest_shares) / earliest_shares * 100) if earliest_shares > 0 else 0
        if pct_change > 5:
            signals.append(("📈 Kepemilikan Naik", f"Kepemilikan 5% holders naik {pct_change:.1f}%", "success"))
        elif pct_change < -5:
            signals.append(("📉 Kepemilikan Turun", f"Kepemilikan 5% holders turun {abs(pct_change):.1f}%", "warning"))

# Tampilkan signals
if signals:
    for signal in signals:
        if signal[2] == "success":
            st.success(f"**{signal[0]}**: {signal[1]}")
        elif signal[2] == "warning":
            st.warning(f"**{signal[0]}**: {signal[1]}")
        else:
            st.info(f"**{signal[0]}**: {signal[1]}")
else:
    st.info("Tidak ada signal khusus dalam periode ini")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Dashboard Bandarmology Pro v2.0 | Data diperbarui setiap jam</p>",
    unsafe_allow_html=True
)
