import streamlit as st
import pandas as pd
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Bandarmology Dashboard", layout="wide", page_icon="📈")
st.title("📈 Dashboard Analisa Bandarmology")

# 1. Fungsi Load Data CSV dari GDrive
@st.cache_data(ttl=3600) # Cache data selama 1 jam agar tidak loading terus
def load_csv_from_gdrive(file_id):
    # Mengambil rahasia Service Account dari Streamlit Secrets
    gcp_service_account = st.secrets["gcp_service_account"]
    
    credentials = service_account.Credentials.from_service_account_info(
        gcp_service_account,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=credentials)
    
    request = service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO(request.execute())
    
    # Baca sebagai CSV
    return pd.read_csv(downloaded)

# === FILE ID GOOGLE DRIVE BAPAK (SUDAH OTOMATIS DIMASUKKAN) ===
FILE_ID_TRANSAKSI = "1GvDd3NDh6A2y9Dm6bCzXO057-RjSKbT8" 
FILE_ID_KEPEMILIKAN = "1PTr6XmBp6on-RNyaHC4mWpn6Y3vsR8xr"

# 2. Proses Load Data 
with st.spinner('Membaca jutaan baris data dari Google Drive...'):
    try:
        df_transaksi = load_csv_from_gdrive(FILE_ID_TRANSAKSI)
        df_kepemilikan = load_csv_from_gdrive(FILE_ID_KEPEMILIKAN)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# 3. Fitur Filter Saham di Sidebar
st.sidebar.header("🔍 Filter Analisa")

# Mengambil daftar kode saham unik dari data transaksi
daftar_saham = sorted(df_transaksi['Stock Code'].dropna().unique().tolist())
saham_pilihan = st.sidebar.selectbox("Pilih Kode Saham:", ["Semua Saham"] + daftar_saham)

# 4. Logika Filter DataFrame
if saham_pilihan != "Semua Saham":
    # Filter transaksi (kolomnya bernama 'Stock Code')
    df_transaksi_tampil = df_transaksi[df_transaksi['Stock Code'] == saham_pilihan]
    # Filter kepemilikan (kolomnya bernama 'Kode Efek')
    df_kepemilikan_tampil = df_kepemilikan[df_kepemilikan['Kode Efek'] == saham_pilihan]
else:
    df_transaksi_tampil = df_transaksi
    df_kepemilikan_tampil = df_kepemilikan

# 5. Tampilan Dashboard (Menggunakan Tabs agar rapi)
tab1, tab2 = st.tabs(["📊 Data Daily Transaksi", "💼 Perubahan Kepemilikan KSEI 5%"])

with tab1:
    st.markdown(f"### Rekap Transaksi: **{saham_pilihan}**")
    st.caption(f"Menampilkan {len(df_transaksi_tampil):,} baris data")
    st.dataframe(df_transaksi_tampil, use_container_width=True)

with tab2:
    st.markdown(f"### Pergerakan Pemegang Saham Besar: **{saham_pilihan}**")
    st.caption(f"Menampilkan {len(df_kepemilikan_tampil):,} baris data")
    st.dataframe(df_kepemilikan_tampil, use_container_width=True)
