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
    page_title="Bandarmology Master V2", 
    layout="wide", 
    page_icon="🐋",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #0f2027, #203a43, #2c5364); padding: 1.2rem; border-radius: 12px; color: white; margin-bottom: 1.2rem; }
    .metric-card { background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #2c5364; }
    .filter-container { background: #f8fafc; padding: 1rem; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
    .signal-box { padding: 0.5rem 1rem; border-radius: 8px; margin: 0.2rem 0; font-weight: 600; border: 1px solid #e0e0e0; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; background-color: #f8fafc; padding: 0.4rem; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.4rem 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h2 style='margin:0;'>🐋 Bandarmology Master V2 - Institutional Intelligence</h2>
    <p style='margin:0; opacity:0.9;'>Advanced Screener • KSEI 5% Overlay • Broker Mutasi • Foreign Flow Radar</p>
</div>
""", unsafe_allow_html=True)

# Helper Function
def format_rupiah(angka):
    if pd.isna(angka): return "Rp 0"
    if abs(angka) >= 1e9: return f"Rp {angka/1e9:,.1f} M"
    if abs(angka) >= 1e6: return f"Rp {angka/1e6:,.1f} Jt"
    return f"Rp {angka:,.0f}"

# ==========================================
# 2. LOAD & PREPROCESS DATA
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    try:
        gcp_service_account = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            gcp_service_account, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=credentials)
        
        req_trans = service.files().get_media(fileId="1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8")
        df_transaksi = pd.read_csv(io.BytesIO(req_trans.execute()))
        
        req_ksei = service.files().get_media(fileId="1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr")
        df_kepemilikan = pd.read_csv(io.BytesIO(req_ksei.execute()))
        
        return df_transaksi, df_kepemilikan
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

with st.spinner("📊 Extracting market intelligence from Drive..."):
    df_transaksi, df_kepemilikan = load_data()

if df_transaksi is None: st.stop()

# Konversi tanggal
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

df_transaksi['Foreign_Pct'] = np.where(df_transaksi['Tradeble Shares'] > 0, (df_transaksi['Net Foreign Flow'] / df_transaksi['Tradeble Shares']) * 100, 0)
df_transaksi['Volume_Pct_Tradeble'] = np.where(df_transaksi['Tradeble Shares'] > 0, (df_transaksi['Volume'] / df_transaksi['Tradeble Shares']) * 100, 0)

unique_stocks = sorted(df_transaksi['Stock Code'].unique())
max_date = df_transaksi['Last Trading Date'].max().date()
default_start = max_date - timedelta(days=30)

# ==========================================
# 3. DASHBOARD TABS
# ==========================================
tabs = st.tabs([
    "🎯 SCREENER PRO", 
    "🔍 DEEP DIVE & KSEI", 
    "🏦 BROKER MUTASI",
    "👥 OWNERSHIP RADAR",
    "🗺️ MARKET MAP"
])

# ==================== TAB 1: SCREENER PRO ====================
with tabs[0]:
    st.markdown("### 🎯 Screener Pro - Institutional Activity")
    
    with st.container():
        st.markdown('<div class="filter-container">', unsafe_allow_html=True)
        row1 = st.columns([1.5, 1.5, 1.5, 1.5, 1])
        with row1[0]: min_value = st.number_input("Min Nilai (M)", 0, 10000, 10) * 1e9
        with row1[1]: min_volume_pct = st.slider("Min Vol % Tradeble", 0.0, 10.0, 0.5, 0.1)
        with row1[2]: anomaly_threshold = st.slider("Min Anomali AOV (x)", 0, 20, 2)
        with row1[3]: foreign_filter = st.selectbox("Foreign Flow", ["Semua", "Net Buy > 1M", "Net Sell > 1M", "Net Buy Kuat", "Net Sell Kuat"])
        with row1[4]: st.markdown("<br>", unsafe_allow_html=True); top_only = st.checkbox("Top 50 Only", value=True)
        
        row2 = st.columns(4)
        with row2[0]: min_price = st.number_input("Min Harga", 0, 100000, 50)
        with row2[1]: min_foreign_pct = st.slider("Min Foreign % Change", -10.0, 10.0, -1.0, 0.1, format="%.1f%%")
        with row2[2]: date_range = st.date_input("Periode Agregasi", value=(default_start, max_date))
        with row2[3]: sort_by = st.selectbox("Sort By", ["Potential Score", "Anomali", "Nilai", "Volume %", "Foreign Flow"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df_transaksi['Last Trading Date'].dt.date >= start_date) & (df_transaksi['Last Trading Date'].dt.date <= end_date)
        df_filter = df_transaksi[mask].copy()
        
        if len(df_filter) > 0:
            agg_dict = {
                'Close': 'last', 'Change %': 'mean', 'Volume': 'sum', 'Value': 'sum', 
                'Net Foreign Flow': 'sum', 'Big_Player_Anomaly': 'max', 'Volume Spike (x)': 'max',
                'Tradeble Shares': 'last', 'Avg_Order_Volume': 'mean'
            }
            summary = df_filter.groupby('Stock Code').agg(agg_dict).reset_index()
            
            summary['Buying_Pressure'] = np.where(summary['Value'] > 0, (summary['Net Foreign Flow'] / summary['Value'] * 100), 0)
            summary['Volume_Concentration'] = np.where(summary['Tradeble Shares'] > 0, (summary['Volume'] / summary['Tradeble Shares'] * 100), 0)
            summary['Anomaly_Score'] = summary['Big_Player_Anomaly'] * summary['Volume Spike (x)']
            summary['Potential_Score'] = (summary['Volume_Concentration'] * 0.3) + (abs(summary['Buying_Pressure']) * 0.3) + (summary['Anomaly_Score'] * 0.4)
            
            # Apply filters
            summary = summary[(summary['Value'] >= min_value) & (summary['Volume_Concentration'] >= min_volume_pct) & 
                              (summary['Big_Player_Anomaly'] >= anomaly_threshold) & (summary['Close'] >= min_price)]
            
            if foreign_filter == "Net Buy > 1M": summary = summary[summary['Net Foreign Flow'] > 1e9]
            elif foreign_filter == "Net Sell > 1M": summary = summary[summary['Net Foreign Flow'] < -1e9]
            
            sort_map = {"Potential Score": "Potential_Score", "Anomali": "Anomaly_Score", "Nilai": "Value", "Volume %": "Volume_Concentration", "Foreign Flow": "Net Foreign Flow"}
            summary = summary.sort_values(sort_map.get(sort_by, "Potential_Score"), ascending=False)
            if top_only: summary = summary.head(50)
            
            st.markdown(f"**🎯 {len(summary)} Saham Terdeteksi (Mode Agregasi)**")
            
            if len(summary) > 0:
                # Stylish Dataframe
                display_df = summary[['Stock Code', 'Close', 'Change %', 'Value', 'Net Foreign Flow', 'Volume_Concentration', 'Big_Player_Anomaly', 'Volume Spike (x)', 'Potential_Score']].copy()
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500,
                    column_config={
                        "Stock Code": st.column_config.TextColumn("Kode", width="small"),
                        "Close": st.column_config.NumberColumn("Harga (Rp)", format="%d"),
                        "Change %": st.column_config.NumberColumn("Avg Chg %", format="%.2f%%"),
                        "Value": st.column_config.NumberColumn("Total Transaksi", format="%d"),
                        "Net Foreign Flow": st.column_config.NumberColumn("Net Foreign", format="%d"),
                        "Volume_Concentration": st.column_config.ProgressColumn("Vol % Tradeble", format="%.1f%%", min_value=0, max_value=20),
                        "Big_Player_Anomaly": st.column_config.NumberColumn("Max Anomali (x)", format="%.1f"),
                        "Potential_Score": st.column_config.NumberColumn("Score", format="%.1f")
                    },
                    hide_index=True
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.scatter(summary.head(20), x='Volume_Concentration', y='Net Foreign Flow', size='Potential_Score', color='Change %', hover_data=['Stock Code'], title="Volume Concentration vs Foreign Flow (Top 20)")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    top_potential = summary.nlargest(15, 'Potential_Score')
                    fig = px.bar(top_potential, x='Stock Code', y='Potential_Score', title="Top 15 Highest Potential Stocks", color='Potential_Score', color_continuous_scale='Viridis')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Tidak ada saham yang memenuhi kriteria")

# ==================== TAB 2: DEEP DIVE & KSEI ====================
with tabs[1]:
    st.markdown("### 🔍 Deep Dive & KSEI 5% Analysis")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: selected_stock = st.selectbox("Pilih Saham", unique_stocks, key='dive_stock')
    with col2: period = st.selectbox("Rentang Hari", [14, 30, 60, 90, 180, 365], index=3)
    with col3: vs_tradeble = st.checkbox("Normalize to Tradeble", value=True)
    
    start_dive = max_date - timedelta(days=period)
    df_dive = df_transaksi[(df_transaksi['Stock Code'] == selected_stock) & (df_transaksi['Last Trading Date'].dt.date >= start_dive)].copy().sort_values('Last Trading Date')
    ksei_dive = df_kepemilikan[(df_kepemilikan['Kode Efek'] == selected_stock) & (df_kepemilikan['Tanggal_Data'].dt.date >= start_dive)].copy()
    
    if len(df_dive) > 0:
        latest = df_dive.iloc[-1]
        tradeble = latest['Tradeble Shares'] if vs_tradeble and latest['Tradeble Shares'] > 0 else 1
        
        # METRICS
        cols = st.columns(6)
        cols[0].metric("Harga Terakhir", f"Rp {latest['Close']:,.0f}", f"{latest['Change %']:.1f}%")
        cols[1].metric("Vol % Tradeble", f"{(latest['Volume']/tradeble*100):.2f}%" if tradeble>0 else "0%")
        cols[2].metric("Net Foreign Flow", format_rupiah(latest['Net Foreign Flow']))
        cols[3].metric("AOV Anomali", f"{latest['Big_Player_Anomaly']:.1f}x")
        cols[4].metric("Avg Order Vol", f"{latest['Avg_Order_Volume']:,.0f}")
        cols[5].metric("Value Spike", f"{latest['Volume Spike (x)']:.1f}x")
        
        # IMPROVEMENT: DUAL AXIS KSEI VS PRICE CHART
        st.markdown(f"#### 📊 Overlay Kepemilikan 5% KSEI vs Harga ({selected_stock})")
        
        if not ksei_dive.empty:
            ksei_agg = ksei_dive.groupby('Tanggal_Data')['Jumlah Saham (Curr)'].sum().reset_index()
            merged_ksei = pd.merge(ksei_agg, df_dive[['Last Trading Date', 'Close']], left_on='Tanggal_Data', right_on='Last Trading Date', how='right')
            # Forward fill untuk data KSEI yang kosong di hari tertentu
            merged_ksei['Jumlah Saham (Curr)'] = merged_ksei['Jumlah Saham (Curr)'].ffill()
            
            fig_ksei = go.Figure()
            fig_ksei.add_trace(go.Bar(x=merged_ksei['Last Trading Date'], y=merged_ksei['Jumlah Saham (Curr)'], name="Total Saham UBO 5%", marker_color='#3498DB', opacity=0.6, yaxis="y"))
            fig_ksei.add_trace(go.Scatter(x=merged_ksei['Last Trading Date'], y=merged_ksei['Close'], name="Harga Close", mode='lines+markers', line=dict(color='#E74C3C', width=3), yaxis="y2"))
            
            fig_ksei.update_layout(
                height=450, hovermode='x unified', legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                yaxis=dict(title="Jumlah Saham (Lembar)", gridcolor='lightgray'),
                yaxis2=dict(title="Harga (Rp)", overlaying='y', side='right', showgrid=False),
                margin=dict(l=40, r=40, t=20, b=20), plot_bgcolor='white'
            )
            st.plotly_chart(fig_ksei, use_container_width=True)
        else:
            st.warning(f"Belum ada entitas yang memiliki >5% saham di {selected_stock} pada periode ini.")
        
        # Technical Subplots
        st.markdown("#### 📈 Price Action & Anomaly Indicators")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df_dive['Last Trading Date'], open=df_dive['Open Price'], high=df_dive['High'], low=df_dive['Low'], close=df_dive['Close'], name="Price"), row=1, col=1)
        # Foreign Flow
        fig.add_trace(go.Bar(x=df_dive['Last Trading Date'], y=df_dive['Net Foreign Flow']/1e9, name="Net Foreign (M)", marker_color=['green' if x > 0 else 'red' for x in df_dive['Net Foreign Flow']]), row=2, col=1)
        # Anomaly
        fig.add_trace(go.Scatter(x=df_dive['Last Trading Date'], y=df_dive['Big_Player_Anomaly'], name="Anomaly AOV", line=dict(color='purple', width=2), fill='tozeroy'), row=3, col=1)
        
        fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=20, b=20), hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

# ==================== TAB 3: BROKER MUTASI ====================
with tabs[2]:
    st.markdown("### 🏦 Broker Mutasi & Intelligence")
    st.markdown("Melacak broker mana yang sedang **Akumulasi** dan **Distribusi** dalam periode tertentu.")
    
    if len(df_kepemilikan) > 0 and 'Kode Broker' in df_kepemilikan.columns:
        
        with st.container(border=True):
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                broker_period = st.selectbox("Periode Mutasi KSEI:", ["1 Minggu Terakhir", "1 Bulan Terakhir", "3 Bulan Terakhir"])
                days_map = {"1 Minggu Terakhir": 7, "1 Bulan Terakhir": 30, "3 Bulan Terakhir": 90}
                start_ksei = df_kepemilikan['Tanggal_Data'].max() - timedelta(days=days_map[broker_period])
            with col_b2:
                st.markdown("<br>", unsafe_allow_html=True)
                st.caption("Menghitung delta (perubahan) jumlah saham yang dikuasai broker pada awal periode vs akhir periode.")

        # Filter Data Mutasi
        df_ksei_mutasi = df_kepemilikan[df_kepemilikan['Tanggal_Data'] >= start_ksei].copy()
        
        if not df_ksei_mutasi.empty:
            # Cari kepemilikan awal dan akhir per broker
            mutasi = df_ksei_mutasi.sort_values('Tanggal_Data').groupby(['Kode Efek', 'Kode Broker']).agg(
                Awal=('Jumlah Saham (Curr)', 'first'),
                Akhir=('Jumlah Saham (Curr)', 'last')
            ).reset_index()
            
            mutasi['Net_Mutasi'] = mutasi['Akhir'] - mutasi['Awal']
            
            # Agregasi Total Mutasi per Broker (Lintas Saham)
            broker_delta = mutasi.groupby('Kode Broker')['Net_Mutasi'].sum().reset_index()
            
            top_acc = broker_delta.nlargest(10, 'Net_Mutasi')
            top_dist = broker_delta.nsmallest(10, 'Net_Mutasi')
            top_dist['Net_Mutasi'] = abs(top_dist['Net_Mutasi']) # Dimutlakkan untuk visual chart
            
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown(f"#### 🟢 Top 10 Broker Accumulator ({broker_period})")
                fig_acc = px.bar(top_acc, x='Net_Mutasi', y='Kode Broker', orientation='h', color_discrete_sequence=['#10B981'])
                fig_acc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_acc, use_container_width=True)
                
            with col_chart2:
                st.markdown(f"#### 🔴 Top 10 Broker Distributor ({broker_period})")
                fig_dist = px.bar(top_dist, x='Net_Mutasi', y='Kode Broker', orientation='h', color_discrete_sequence=['#EF4444'])
                fig_dist.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_dist, use_container_width=True)
                
            st.divider()
            
            # Cek Mutasi Detail per Broker
            st.markdown("#### 🔎 Detail Mutasi per Broker")
            selected_broker_mutasi = st.selectbox("Pilih Broker untuk melihat saham apa saja yang diakumulasi/distribusi:", sorted(mutasi['Kode Broker'].unique()))
            
            detail_broker = mutasi[(mutasi['Kode Broker'] == selected_broker_mutasi) & (mutasi['Net_Mutasi'] != 0)].copy()
            if not detail_broker.empty:
                detail_broker = detail_broker.sort_values('Net_Mutasi', ascending=False)
                
                def color_mutasi(val):
                    return 'color: green; font-weight:bold;' if val > 0 else 'color: red; font-weight:bold;'
                
                st.dataframe(
                    detail_broker[['Kode Efek', 'Awal', 'Akhir', 'Net_Mutasi']].style.map(color_mutasi, subset=['Net_Mutasi']).format("{:,.0f}", subset=['Awal', 'Akhir', 'Net_Mutasi']),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("Tidak ada pergerakan (mutasi) >5% untuk broker ini pada periode yang dipilih.")
                
# ==================== TAB 4: OWNERSHIP RADAR ====================
with tabs[3]:
    st.markdown("### 👑 Ownership Concentration Radar")
    
    if len(df_kepemilikan) > 0:
        latest_date = df_kepemilikan['Tanggal_Data'].max()
        latest_ownership = df_kepemilikan[df_kepemilikan['Tanggal_Data'] == latest_date].copy()
        
        col_o1, col_o2 = st.columns([1.5, 1])
        with col_o1:
            st.markdown("#### 🎯 Saham Paling Terkonsentrasi (Dikuasai Paus)")
            tradeble_info = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date][['Stock Code', 'Tradeble Shares']].drop_duplicates()
            
            if len(tradeble_info) > 0:
                concentration = latest_ownership.groupby('Kode Efek')['Jumlah Saham (Curr)'].sum().reset_index()
                concentration = concentration.merge(tradeble_info, left_on='Kode Efek', right_on='Stock Code', how='left')
                
                concentration['Pct_of_Tradeble'] = np.where(concentration['Tradeble Shares'] > 0, (concentration['Jumlah Saham (Curr)'] / concentration['Tradeble Shares'] * 100), 0)
                concentration = concentration.sort_values('Pct_of_Tradeble', ascending=False).head(20)
                
                fig = px.bar(concentration, x='Kode Efek', y='Pct_of_Tradeble', color='Pct_of_Tradeble', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
                
        with col_o2:
            st.markdown("#### 🔍 Lacak Pemegang Saham")
            own_stock = st.selectbox("Cek Siapa Pemilik Saham:", sorted(latest_ownership['Kode Efek'].unique()), key='own_stock')
            if own_stock:
                stock_owners = latest_ownership[latest_ownership['Kode Efek'] == own_stock].copy()
                stock_owners = stock_owners.sort_values('Jumlah Saham (Curr)', ascending=False)
                st.dataframe(
                    stock_owners[['Nama Pemegang Saham', 'Kode Broker', 'Jumlah Saham (Curr)']],
                    column_config={"Jumlah Saham (Curr)": st.column_config.NumberColumn("Lembar Saham", format="%d")},
                    hide_index=True, use_container_width=True
                )

# ==================== TAB 5: MARKET MAP ====================
with tabs[4]:
    st.markdown("### 🗺️ Sectoral Market Heatmap & Foreign Flow")
    
    today_data = df_transaksi[df_transaksi['Last Trading Date'].dt.date == max_date].copy()
    if len(today_data) > 0:
        
        # IMPROVEMENT: TREEMAP FOREIGN FLOW PER SECTOR
        if 'Sector' in today_data.columns:
            st.markdown("#### 🏢 Foreign Flow per Sector (Treemap)")
            sector_perf = today_data.groupby(['Sector', 'Stock Code']).agg({
                'Net Foreign Flow': 'sum',
                'Value': 'sum',
                'Change %': 'mean'
            }).reset_index()
            
            # Filter saham yang ada transaksinya saja
            sector_perf = sector_perf[sector_perf['Value'] > 0]
            
            # Gunakan Absolute Value untuk ukuran box, warna untuk arah Foreign Flow
            sector_perf['Abs_Value'] = sector_perf['Value']
            
            fig_tree = px.treemap(
                sector_perf, 
                path=[px.Constant("IHSG"), 'Sector', 'Stock Code'], 
                values='Abs_Value',
                color='Net Foreign Flow', 
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                hover_data=['Change %']
            )
            fig_tree.update_layout(height=600, margin=dict(t=20, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)
        
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🟢 Top Foreign Buy (Hari Ini)")
            top_fb = today_data.nlargest(10, 'Net Foreign Flow')[['Stock Code', 'Close', 'Change %', 'Net Foreign Flow']]
            st.dataframe(top_fb.style.format({'Close': '{:,.0f}', 'Change %': '{:+.2f}%', 'Net Foreign Flow': lambda x: f"Rp {x/1e9:.1f} M"}), hide_index=True, use_container_width=True)
            
        with col_m2:
            st.markdown("#### 🔴 Top Foreign Sell (Hari Ini)")
            top_fs = today_data.nsmallest(10, 'Net Foreign Flow')[['Stock Code', 'Close', 'Change %', 'Net Foreign Flow']]
            st.dataframe(top_fs.style.format({'Close': '{:,.0f}', 'Change %': '{:+.2f}%', 'Net Foreign Flow': lambda x: f"Rp {x/1e9:.1f} M"}), hide_index=True, use_container_width=True)

    else:
        st.warning(f"Belum ada data transaksi untuk tanggal {max_date.strftime('%d-%m-%Y')}")

st.markdown("---")
st.caption(f"🔄 Last Update: {max_date.strftime('%d %b %Y')} | Data Loaded Successfully.")
