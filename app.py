import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io
from datetime import datetime, timedelta

# ==============================================================================
# 1. KONFIGURASI HALAMAN & CSS
# ==============================================================================
st.set_page_config(
    page_title="Frequency Analyzer Intelligence Dashboard",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="collapsed" # Sengaja disembunyikan sesuai request
)

# Custom CSS
st.markdown("""
<style>
    .whale-card { background: linear-gradient(135deg, #e6fffa 0%, #b2f5ea 100%); border-left: 5px solid #00cc00; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    .split-card { background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%); border-left: 5px solid #ff4444; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    .neutral-card { background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%); border-left: 5px solid #718096; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    .metric-card { background: white; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); }
    .big-text { font-size: 24px; font-weight: 800; margin-bottom: 5px; }
    .small-text { font-size: 12px; color: #718096; }
    .value-text { font-size: 20px; font-weight: 700; color: #2d3748; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display: flex; align-items: center; gap: 15px; margin-bottom: 20px;'>
    <div style='font-size: 48px;'>🐋</div>
    <div>
        <h1 style='margin: 0; color: #2d3748;'>Bandarmology Intelligence Dashboard</h1>
        <p style='margin: 0; color: #718096; font-size: 16px;'>MA50 Standard | Whale Detection | KSEI 5% Tracker</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Helper Format
def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    if angka >= 1e9: return f"Rp {angka/1e9:.2f} M"
    if angka >= 1e6: return f"Rp {angka/1e6:.2f} Jt"
    return f"Rp {angka:,.0f}"

def format_lembar(angka):
    if pd.isna(angka): return "0"
    if abs(angka) >= 1e6: return f"{angka/1e6:.2f} Jt"
    return f"{angka:,.0f}"

# ==============================================================================
# 2. LOAD DATA DARI GOOGLE DRIVE CSV
# ==============================================================================
FILE_ID_TRANSAKSI = "1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8" 
FILE_ID_KEPEMILIKAN = "1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr"

@st.cache_data(ttl=3600)
def load_csv_from_gdrive(file_id):
    gcp_service_account = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(
        gcp_service_account, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=credentials)
    request = service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO(request.execute())
    return pd.read_csv(downloaded)

with st.spinner('Sedang menyiapkan data pasar (Daily & KSEI)...'):
    try:
        df_raw = load_csv_from_gdrive(FILE_ID_TRANSAKSI)
        df_ksei = load_csv_from_gdrive(FILE_ID_KEPEMILIKAN)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# ==============================================================================
# 3. GLOBAL CALCULATION (MA50 LOGIC & PREPROCESSING)
# ==============================================================================
# PREP KSEI
df_ksei['Tanggal_Data'] = pd.to_datetime(df_ksei['Tanggal_Data'])

# PREP TRANSAKSI
df_raw['Last Trading Date'] = pd.to_datetime(df_raw['Last Trading Date'])
df = df_raw.sort_values(by=['Stock Code', 'Last Trading Date']).copy()

# A. Pastikan MA50 Ada
if 'MA50_AOVol' not in df.columns:
    df['MA50_AOVol'] = df.groupby('Stock Code')['Avg_Order_Volume'].transform(lambda x: x.rolling(50, min_periods=1).mean())

# B. Hitung Ratio Anomali
df['AOV_Ratio'] = np.where(df['MA50_AOVol'] > 0, df['Avg_Order_Volume'] / df['MA50_AOVol'], 0)

# C. Kolom Signal
df['Whale_Signal'] = df['AOV_Ratio'] >= 1.5
df['Split_Signal'] = (df['AOV_Ratio'] <= 0.6) & (df['AOV_Ratio'] > 0)

# D. Net Foreign Calc
if 'Foreign Buy' in df.columns and 'Foreign Sell' in df.columns:
    df['Net Foreign'] = df['Foreign Buy'] - df['Foreign Sell']
else:
    df['Net Foreign'] = 0

# E. Value Spike (Money Flow)
df['MA20_Value'] = df.groupby('Stock Code')['Value'].transform(lambda x: x.rolling(20, min_periods=1).mean())
df['Value_Ratio'] = np.where(df['MA20_Value'] > 0, df['Value'] / df['MA20_Value'], 0)

max_date = df['Last Trading Date'].max()

# ==============================================================================
# 4. DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Deep Dive & KSEI Chart", 
    "🐋 Whale Screener", 
    "💎 Bluechip Radar",
    "🐳 KSEI 5% Whale Tracker",
    "🧪 Research Lab"
])

# ==============================================================================
# TAB 1: DEEP DIVE ANALYSIS (GABUNGAN KONSEP 1 & 2)
# ==============================================================================
with tab1:
    st.markdown("### 📈 Deep Dive Stock Analysis & KSEI 5% Activity")
    
    # --- A. FILTER SECTION (DI DALAM TAB) ---
    c_sel1, c_sel2, c_sel3 = st.columns([2, 1, 1])
    with c_sel1:
        all_stocks = sorted(df['Stock Code'].dropna().unique().tolist())
        selected_stock = st.selectbox("🔍 Pilih Saham", all_stocks, key="deepdive_stock")
    with c_sel2:
        chart_days = st.selectbox("Rentang Chart", [30, 60, 90, 120, 200, 365], index=3, format_func=lambda x: f"{x} Hari")
    with c_sel3:
        chart_type = st.radio("Tipe Chart Harga", ["Candle", "Line"], horizontal=True, label_visibility="collapsed")
    
    # --- B. DATA PROCESSING ---
    stock_data = df[df['Stock Code'] == selected_stock].tail(chart_days).copy()
    stock_ksei = df_ksei[df_ksei['Kode Efek'] == selected_stock].copy()
    
    if not stock_data.empty:
        last_row = stock_data.iloc[-1]
        
        # --- C. STATUS CARD (VERDICT WHALE) ---
        aov_ratio = last_row.get('AOV_Ratio', 1)
        if aov_ratio >= 1.5:
            conviction_score = min(99, ((aov_ratio - 1.5) / (5 - 1.5)) * 80 + 20)
            st.markdown(f"""
            <div class="whale-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><div class="big-text">🐋 WHALE DETECTED</div><div class="small-text">Indikasi Akumulasi Besar (Lot Gede)</div></div>
                    <div style="text-align: right;"><div class="value-text">Score: {conviction_score:.0f}%</div><div class="small-text">AOV Ratio: <b>{aov_ratio:.2f}x</b></div></div>
                </div>
            </div>""", unsafe_allow_html=True)
        elif aov_ratio <= 0.6 and aov_ratio > 0:
            conviction_score = min(99, ((0.6 - aov_ratio) / 0.6) * 80 + 20)
            st.markdown(f"""
            <div class="split-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><div class="big-text">⚡ SPLIT / RETAIL</div><div class="small-text">Indikasi Distribusi atau Akumulasi Pecah Order</div></div>
                    <div style="text-align: right;"><div class="value-text">Score: {conviction_score:.0f}%</div><div class="small-text">AOV Ratio: <b>{aov_ratio:.2f}x</b></div></div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="neutral-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div><div class="big-text">⚖️ NORMAL ACTIVITY</div><div class="small-text">Pergerakan Volume Wajar (Sesuai Rata-rata)</div></div>
                    <div style="text-align: right;"><div class="value-text">Neutral</div><div class="small-text">AOV Ratio: <b>{aov_ratio:.2f}x</b></div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # --- D. CHARTING 1: PRICE & VOLUME AOV ---
        st.markdown(f"#### A. Struktur Harga & Volume Anomali ({selected_stock})")
        fig1 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2])
        
        if chart_type == "Candle":
            fig1.add_trace(go.Candlestick(x=stock_data['Last Trading Date'], open=stock_data['Open Price'], high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], name='OHLC'), row=1, col=1)
        else:
            fig1.add_trace(go.Scatter(x=stock_data['Last Trading Date'], y=stock_data['Close'], mode='lines', line=dict(color='#2962ff', width=2), name='Close'), row=1, col=1)
        
        # Markers
        ws = stock_data[stock_data['Whale_Signal']]
        if not ws.empty: fig1.add_trace(go.Scatter(x=ws['Last Trading Date'], y=ws['High']*1.02, mode='markers', marker=dict(symbol='triangle-down', size=12, color='#00cc00'), name='Whale'), row=1, col=1)
        
        # Vol & AOV
        colors = ['#00cc00' if r >= 1.5 else '#ff4444' if (r <= 0.6 and r > 0) else '#cfd8dc' for r in stock_data['AOV_Ratio']]
        fig1.add_trace(go.Bar(x=stock_data['Last Trading Date'], y=stock_data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        fig1.add_trace(go.Scatter(x=stock_data['Last Trading Date'], y=stock_data['AOV_Ratio'], mode='lines', line=dict(color='#9c88ff', width=2), name='AOV Ratio'), row=3, col=1)
        fig1.add_hline(y=1.5, line_dash="dash", line_color="green", row=3, col=1)
        
        fig1.update_layout(height=600, showlegend=False, hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        # --- E. CHARTING 2: KONSEP 2 (DUAL AXIS KSEI VS PRICE) ---
        st.markdown(f"#### B. Analisa Kepemilikan Institusi/Paus 5% KSEI ({selected_stock})")
        
        if not stock_ksei.empty:
            # Aggregate KSEI data per date to get total owned by >5% holders
            ksei_agg = stock_ksei.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            
            # Merge with price data to match dates
            merged_ksei_price = pd.merge(ksei_agg, stock_data[['Last Trading Date', 'Close']], left_on='Tanggal_Data', right_on='Last Trading Date', how='inner')
            
            if not merged_ksei_price.empty:
                fig_deep = go.Figure()

                # Bar Chart: Total Kepemilikan (Primary Y Axis)
                fig_deep.add_trace(go.Bar(
                    x=merged_ksei_price['Tanggal_Data'], 
                    y=merged_ksei_price['Jumlah Saham (Curr)'],
                    name="Total Lembar Saham (UBO 5%)",
                    marker_color='#3498DB',
                    opacity=0.6,
                    yaxis="y"
                ))

                # Line Chart: Harga (Secondary Y Axis)
                fig_deep.add_trace(go.Scatter(
                    x=merged_ksei_price['Tanggal_Data'], 
                    y=merged_ksei_price['Close'],
                    name="Harga Close",
                    mode='lines+markers',
                    line=dict(color='#E74C3C', width=3),
                    marker=dict(size=6),
                    yaxis="y2"
                ))

                # UPDATE LAYOUT DUAL AXIS (Dari Konsep 2)
                fig_deep.update_layout(
                    title=f"<b>AKUMULASI 5% vs HARGA</b> - {selected_stock}",
                    height=500,
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    yaxis=dict(
                        title=dict(text="Jumlah Saham (Lembar)", font=dict(color='#2C3E50')),
                        tickformat=",.0f", gridcolor='lightgray', showgrid=True
                    ),
                    yaxis2=dict(
                        title=dict(text="Harga (Rp)", font=dict(color='#E74C3C')),
                        tickformat=",.0f", overlaying='y', side='right', showgrid=False
                    ),
                    xaxis=dict(title="Tanggal", tickformat="%d-%b-%Y", gridcolor='lightgray'),
                    plot_bgcolor='white', paper_bgcolor='white', margin=dict(l=50, r=50, t=50, b=50)
                )
                fig_deep.update_xaxes(rangeslider_visible=True, rangeslider_thickness=0.05)
                
                st.plotly_chart(fig_deep, use_container_width=True)

                # METRIK RINGKASAN KONSEP 2
                all_ubos = stock_ksei['Nama Pemegang Saham'].unique()
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    last_price = last_row.get('Close', 0)
                    chg_pct = last_row.get('Change %', 0)
                    st.metric("Harga Terkini", format_rupiah(last_price), f"{chg_pct:+.2f}%")
                with col_m2:
                    st.metric("Whale/Institusi Aktif (>5%)", len(all_ubos))
                with col_m3:
                    spike_30d = stock_data[(stock_data['AOV_Ratio'] > 1.5)].shape[0]
                    st.metric("Sinyal Whale (Periode Chart)", f"{spike_30d} Hari")
                with col_m4:
                    ma20 = stock_data['Close'].tail(20).mean()
                    st.metric("MA20 Harga", format_rupiah(ma20))

                # TABEL RINGKASAN PER UBO
                with st.expander("📋 Lihat Ringkasan Aktivitas per Ultimate Beneficial Owner (UBO)"):
                    summary_data = []
                    for ubo in all_ubos:
                        df_ubo_sum = stock_ksei[stock_ksei['Nama Pemegang Saham'] == ubo].sort_values('Tanggal_Data')
                        if not df_ubo_sum.empty:
                            first_date = df_ubo_sum['Tanggal_Data'].iloc[0]
                            last_date = df_ubo_sum['Tanggal_Data'].iloc[-1]
                            first_hold = df_ubo_sum['Jumlah Saham (Curr)'].iloc[0]
                            last_hold = df_ubo_sum['Jumlah Saham (Curr)'].iloc[-1]
                            change = last_hold - first_hold
                            change_pct = (change / first_hold * 100) if first_hold > 0 else 0
                            
                            summary_data.append({
                                'UBO': ubo[:30] + '...' if len(ubo) > 30 else ubo,
                                'Periode Terekam': f"{first_date.strftime('%d/%m/%y')} - {last_date.strftime('%d/%m/%y')}",
                                'Kepemilikan Awal': format_lembar(first_hold),
                                'Kepemilikan Akhir': format_lembar(last_hold),
                                'Perubahan (Lembar)': change,
                                'Perubahan String': format_lembar(change),
                                'Δ %': f"{change_pct:+.1f}%"
                            })
                    
                    df_summary = pd.DataFrame(summary_data).sort_values(by='Perubahan (Lembar)', ascending=False)
                    # Hapus kolom bantuan sorting
                    df_summary = df_summary.drop(columns=['Perubahan (Lembar)'])
                    st.dataframe(df_summary, use_container_width=True, hide_index=True)

            else:
                st.warning("Data KSEI 5% dan Transaksi tidak memiliki irisan tanggal untuk dirender.")
        else:
            st.warning(f"Belum ada data kepemilikan >5% KSEI untuk saham {selected_stock}.")
    else:
        st.warning("Data tidak tersedia.")

# ==============================================================================
# TAB 2: WHALE SCREENER (ANOMALI HARIAN/PERIOD)
# ==============================================================================
with tab2:
    st.markdown("### 🐋 Whale & Retail Detection Screener")
    
    with st.container(border=True):
        col_set1, col_set2, col_set3 = st.columns(3)
        with col_set1:
            scan_mode = st.radio("Metode Scanning:", ("📸 Daily Snapshot", "🗓️ Period Scanner"))
            anomaly_type = st.radio("Target:", ("🐋 Whale Signal (High AOV)", "⚡ Split/Retail Signal (Low AOV)"))
            
        with col_set2:
            if scan_mode == "📸 Daily Snapshot":
                selected_date = pd.to_datetime(st.date_input("Pilih Tanggal", max_date))
            else:
                period_days = st.selectbox("Analisa Data Terakhir:", [5, 10, 20, 60], index=1, format_func=lambda x: f"{x} Hari Kerja")
                start_date_scan = max_date - timedelta(days=period_days * 1.5)
            
            price_condition = st.selectbox("Kondisi Harga:", [
                "🔍 SEMUA FASE", "💎 HIDDEN GEM (Sideways)", "⚓ BOTTOM FISHING (Downtrend)", "🚀 EARLY MOVE (Awal Naik)"
            ])
            
        with col_set3:
            st.markdown("#### 💰 Filter Nilai Transaksi")
            # Filter Minimum Value Transaksi (Sesuai Request)
            min_value = st.number_input("Minimum Value (Rp)", value=1_000_000_000, step=500_000_000, format="%d")

    # Data Prep
    if scan_mode == "📸 Daily Snapshot": target_df = df[df['Last Trading Date'] == selected_date].copy()
    else: target_df = df[df['Last Trading Date'] >= start_date_scan].copy()

    # Filtering
    if anomaly_type == "🐋 Whale Signal (High AOV)":
        suspects = target_df[(target_df['AOV_Ratio'] >= 2.0) & (target_df['Value'] >= min_value)]
    else:
        suspects = target_df[(target_df['AOV_Ratio'] <= 0.6) & (target_df['AOV_Ratio'] > 0) & (target_df['Value'] >= min_value)]

    if not suspects.empty and price_condition != "🔍 SEMUA FASE":
        if 'VWMA_20D' not in suspects.columns:
            suspects['TP'] = (suspects['High'] + suspects['Low'] + suspects['Close']) / 3
            suspects['VP'] = suspects['TP'] * suspects['Volume']
            suspects['VWMA_20D'] = suspects.groupby('Stock Code')['VP'].transform(lambda x: x.rolling(20).sum() / x.rolling(20).sum())

        if "HIDDEN GEM" in price_condition: suspects = suspects[(suspects['Change %'] >= -2.0) & (suspects['Change %'] <= 2.0)]
        elif "BOTTOM FISHING" in price_condition: suspects = suspects[(suspects['Close'] < suspects['VWMA_20D']) | (suspects['Change %'] < 0)]
        elif "EARLY MOVE" in price_condition: suspects = suspects[(suspects['Change %'] > 0) & (suspects['Change %'] <= 4.0)]

    if suspects.empty:
        st.warning("Tidak ditemukan saham dengan kriteria tersebut.")
    else:
        if scan_mode == "📸 Daily Snapshot":
            suspects = suspects.sort_values(by='AOV_Ratio', ascending=False)
            st.metric("Saham Ditemukan", len(suspects))
            
            display_cols = ['Stock Code', 'Company Name', 'Close', 'Change %', 'Volume', 'Value', 'Avg_Order_Volume', 'AOV_Ratio']
            display_df = suspects[[c for c in display_cols if c in suspects.columns]]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            summary = suspects.groupby(['Stock Code']).agg(
                Freq_Muncul=('Last Trading Date', 'count'),
                Avg_AOV_Ratio=('AOV_Ratio', 'mean'),
                Total_Value=('Value', 'sum'),
            ).reset_index().sort_values(by='Freq_Muncul', ascending=False).head(50)
            st.metric("Emiten Terdeteksi", len(summary))
            st.dataframe(summary, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: BLUECHIP RADAR
# ==============================================================================
with tab3:
    st.markdown("### 💎 Bluechip Radar (Big Caps Only)")
    
    with st.container(border=True):
        col_bc1, col_bc2, col_bc3 = st.columns(3)
        with col_bc1:
            bc_scan_mode = st.radio("Metode Scan:", ("📸 Harian", "🗓️ Periode"), key="bc_mode")
            if bc_scan_mode == "📸 Harian": bc_date = pd.to_datetime(st.date_input("Tanggal", max_date, key="bc_date"))
            else: bc_period = st.selectbox("Periode:", [5, 10, 20], key="bc_period"); bc_start = max_date - timedelta(days=bc_period * 1.5)
        with col_bc2:
            min_bc_value = st.number_input("Min Transaksi (Rp)", value=20_000_000_000, step=5_000_000_000, key="bc_val")
        with col_bc3:
            bc_aov_threshold = st.slider("Min AOV Ratio", 1.1, 2.0, 1.25, 0.05, key="bc_aov")

    if bc_scan_mode == "📸 Harian": df_bc = df[df['Last Trading Date'] == bc_date].copy()
    else: df_bc = df[df['Last Trading Date'] >= bc_start].copy()

    bc_suspects = df_bc[(df_bc['Value'] >= min_bc_value) & (df_bc['AOV_Ratio'] >= bc_aov_threshold)]

    if not bc_suspects.empty:
        if bc_scan_mode == "📸 Harian":
            st.dataframe(bc_suspects[['Stock Code', 'Close', 'Change %', 'Net Foreign', 'Value', 'AOV_Ratio']].sort_values(by='Value', ascending=False), hide_index=True)
        else:
            summary_bc = bc_suspects.groupby('Stock Code').agg(
                Freq_Anomali=('Last Trading Date', 'count'), Total_Net_Foreign=('Net Foreign', 'sum')
            ).reset_index().sort_values(by='Total_Net_Foreign', ascending=False)
            st.dataframe(summary_bc, hide_index=True)
    else:
        st.warning("Tidak ditemukan bluechip.")

# ==============================================================================
# TAB 4: KSEI 5% WHALE TRACKER (IMPROVISASI KONSEP 3)
# ==============================================================================
with tab4:
    st.markdown("### 🐳 KSEI 5% Whale Tracker (Top Accumulator & Distributor)")
    st.markdown("Melacak pergerakan bersih (Net Flow) pemegang saham mayoritas dari seluruh pasar.")
    
    with st.container(border=True):
        col_k1, col_k2 = st.columns(2)
        with col_k1:
            ksei_period = st.selectbox("Pilih Rentang Analisa KSEI:", ["1 Minggu Terakhir", "1 Bulan Terakhir", "3 Bulan Terakhir", "Year to Date"])
        with col_k2:
            st.markdown("<br>", unsafe_allow_html=True)
            ksei_btn = st.button("🔍 Scan Pergerakan Paus", use_container_width=True, type="primary")
            
    if ksei_btn:
        with st.spinner("Menghitung mutasi saham seluruh entitas KSEI 5%..."):
            max_ksei_date = df_ksei['Tanggal_Data'].max()
            if ksei_period == "1 Minggu Terakhir": start_ksei = max_ksei_date - timedelta(days=7)
            elif ksei_period == "1 Bulan Terakhir": start_ksei = max_ksei_date - timedelta(days=30)
            elif ksei_period == "3 Bulan Terakhir": start_ksei = max_ksei_date - timedelta(days=90)
            else: start_ksei = pd.to_datetime(f"{max_ksei_date.year}-01-01")
            
            # Filter Data sesuai tanggal
            df_ksei_filtered = df_ksei[(df_ksei['Tanggal_Data'] >= start_ksei) & (df_ksei['Tanggal_Data'] <= max_ksei_date)].copy()
            
            # Grouping untuk mencari nilai awal dan akhir per UBO per Saham
            ksei_mutasi = df_ksei_filtered.sort_values('Tanggal_Data').groupby(['Kode Efek', 'Nama Pemegang Saham']).agg(
                Awal=('Jumlah Saham (Curr)', 'first'),
                Akhir=('Jumlah Saham (Curr)', 'last')
            ).reset_index()
            
            ksei_mutasi['Net_Perubahan'] = ksei_mutasi['Akhir'] - ksei_mutasi['Awal']
            
            # Pisahkan Akumulasi dan Distribusi
            akumulasi = ksei_mutasi[ksei_mutasi['Net_Perubahan'] > 0].sort_values('Net_Perubahan', ascending=False).head(20)
            distribusi = ksei_mutasi[ksei_mutasi['Net_Perubahan'] < 0].sort_values('Net_Perubahan', ascending=True).head(20)
            
            c_aku, c_dist = st.columns(2)
            
            with c_aku:
                st.markdown(f"#### 🟩 Top 20 Akumulasi ({ksei_period})")
                akumulasi['Label'] = akumulasi['Nama Pemegang Saham'].str[:20] + " (" + akumulasi['Kode Efek'] + ")"
                fig_aku = px.bar(akumulasi, x='Net_Perubahan', y='Label', orientation='h', color_discrete_sequence=['#00cc00'])
                fig_aku.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
                st.plotly_chart(fig_aku, use_container_width=True)
                
            with c_dist:
                st.markdown(f"#### 🟥 Top 20 Distribusi ({ksei_period})")
                # Mutlakkan nilai distribusi agar bar arahnya positif untuk plotting ranking
                distribusi['Abs_Perubahan'] = abs(distribusi['Net_Perubahan'])
                distribusi['Label'] = distribusi['Nama Pemegang Saham'].str[:20] + " (" + distribusi['Kode Efek'] + ")"
                fig_dist = px.bar(distribusi, x='Abs_Perubahan', y='Label', orientation='h', color_discrete_sequence=['#ff4444'])
                fig_dist.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
                st.plotly_chart(fig_dist, use_container_width=True)

# ==============================================================================
# TAB 5: RESEARCH LAB
# ==============================================================================
with tab5:
    st.markdown("### 🧪 Research Lab: Uji Hipotesis")
    with st.container(border=True):
        col_res1, col_res2 = st.columns(2)
        with col_res1: test_mode = st.selectbox("Sinyal:", ["Whale (AOV Tinggi)", "Split (AOV Rendah)"])
        with col_res2: min_tx_test = st.number_input("Filter Saham Liquid (Min Rp):", value=500_000_000, key="res_val")

        if st.button("🚀 JALANKAN BACKTEST", type="primary", use_container_width=True):
            st.info("Fitur Backtesting berhasil diinisiasi. Silakan modifikasi algoritma logic return di block script Tab 5.")
