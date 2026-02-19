import streamlit as st
import pandas as pd
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Bandarmology Dashboard", layout="wide")
st.title("📈 Dashboard Analisa Bandarmology")

# 1. Fungsi Authentikasi & Load Data dari GDrive
@st.cache_data(ttl=3600) # Cache data selama 1 jam
def load_excel_from_gdrive(file_id):
    # Mengambil rahasia Service Account dari Streamlit Secrets
    gcp_service_account = st.secrets["gcp_service_account"]
    
    # Authentikasi
    credentials = service_account.Credentials.from_service_account_info(
        gcp_service_account,
        scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=credentials)
    
    # Download 1 file Excel
    request = service.files().get_media(fileId=file_id)
    downloaded = io.BytesIO(request.execute())
    
    # Membaca struktur file Excel
    xls = pd.ExcelFile(downloaded, engine='openpyxl')
    
    # Memisahkan datanya berdasarkan nama sheet
    df_trans = pd.read_excel(xls, sheet_name='Daily Transaksi')
    df_kepemilikan = pd.read_excel(xls, sheet_name='Daily Perubahan Kepemilikan')
    
    return df_trans, df_kepemilikan

# === GANTI DENGAN 1 FILE ID GOOGLE DRIVE BAPAK SAJA ===
# Cara dapatkan File ID: Buka file Excel di GDrive, copy teks acak di antara /d/ dan /view di URL
FILE_ID_EXCEL = "1dT9GMsA_WJpHzoP8-B4hnV-3jDqfVQEp" 

# 2. Proses Load Data (Akan ada indikator loading di dashboard)
with st.spinner('Membaca data 63MB dari Google Drive (mungkin butuh waktu beberapa saat)...'):
    try:
        df_transaksi, df_kepemilikan = load_excel_from_gdrive(FILE_ID_EXCEL)
        st.success("Data berhasil dimuat!")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

# 3. Tampilan Dashboard Awal
st.write("### Preview Data Daily Transaksi")
st.dataframe(df_transaksi.head(5))

st.write("### Preview Data Perubahan Kepemilikan 5%")
st.dataframe(df_kepemilikan.head(5))
