import streamlit as st
import pandas as pd
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Bandarmology Dashboard", layout="wide")
st.title("📈 Dashboard Analisa Bandarmology")

# 1. Fungsi Authentikasi & Load Data dari GDrive
@st.cache_data(ttl=3600) # Cache data selama 1 jam agar tidak loading terus menerus
def load_data_from_gdrive(file_id, is_excel=False):
    # Mengambil rahasia Service Account dari Streamlit Secrets
    gcp_service_account = st.secrets["gcp_service_account"]
    
    # Authentikasi
    credentials = service_account.Credentials.from_service_account_info(
        gcp_service_account,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=credentials)
    
    # Download file
    request = service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO(request.execute())
    
    # Baca ke Pandas DataFrame
    if is_excel:
        return pd.read_excel(downloaded, engine='openpyxl')
    else:
        return pd.read_csv(downloaded)

# === GANTI DENGAN FILE ID GOOGLE DRIVE BAPAK ===
# Cara dapatkan File ID: Buka file di GDrive, copy teks acak di antara /d/ dan /view di URL
FILE_ID_TRANSAKSI = "MASUKKAN_FILE_ID_CSV_TRANSAKSI_DISINI" 
FILE_ID_KEPEMILIKAN = "MASUKKAN_FILE_ID_CSV_KEPEMILIKAN_DISINI"

# 2. Proses Load Data (Akan ada indikator loading di dashboard)
with st.spinner('Mengambil data dari Google Drive...'):
    try:
        df_transaksi = load_data_from_gdrive(FILE_ID_TRANSAKSI, is_excel=False)
        df_kepemilikan = load_data_from_gdrive(FILE_ID_KEPEMILIKAN, is_excel=False)
        st.success("Data berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# 3. Tampilan Dashboard Awal
st.write("### Preview Data Daily Transaksi")
st.dataframe(df_transaksi.head(5))

st.write("### Preview Data Perubahan Kepemilikan 5%")
st.dataframe(df_kepemilikan.head(5))
