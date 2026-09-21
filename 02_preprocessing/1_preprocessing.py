# Preprocessing Data Berita Detik.com
# Membaca data_berita_detik.xlsx, membersihkan teks, lalu menyimpan hasilnya

import re
import pandas as pd

# Baca data hasil crawling
df = pd.read_excel("data_berita_detik.xlsx")
print(f"Total data: {len(df)}")

# Daftar kata tidak baku / singkatan yang akan dibuang
kata_tidak_baku = {
    'tdk', 'gak', 'ga', 'gk', 'yg', 'dgn', 'utk', 'krn', 'dg', 'dr',
    'pd', 'jg', 'lg', 'sdh', 'blm', 'tp', 'kl', 'klo', 'bs', 'bgt',
    'byk', 'smua', 'sm', 'sy', 'ak', 'gw', 'gue', 'lu', 'lo', 'aja',
    'udh', 'udah', 'emg', 'emang', 'org', 'dpt', 'stlh', 'sblm',
    'krna', 'brg', 'dri', 'dll', 'dsb', 'dst', 'tsb', 'spt', 'dmn',
    'kmn', 'gmn', 'bgmn', 'hrs', 'trs', 'thd', 'ttg', 'tgl', 'thn',
    'bln', 'jd', 'jdi', 'sdg', 'shg', 'stl', 'skrg', 'trhdp', 'dlm',
    'dkk', 'sblmnya', 'tsbt', 'sbg', 'krg', 'brp', 'dsbnya', 'msh',
    'bkn', 'blh', 'thn', 'mgkn', 'bnyk', 'smpe', 'smpai', 'kpd',
    'trmsuk', 'thp', 'dng', 'kt', 'mrk', 'sorg', 'ckp', 'bbrp'
}


def preprocess_teks(teks):
    # Hapus angka
    teks = re.sub(r'\d+', '', teks)
    # Hapus tanda baca dan karakter khusus, sisakan huruf dan spasi
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    # Hapus spasi berlebih
    teks = re.sub(r'\s+', ' ', teks).strip()
    # Hapus kata tidak baku dan kata 1 huruf
    kata_list = teks.split()
    kata_bersih = [k for k in kata_list if k not in kata_tidak_baku and len(k) > 1]
    return ' '.join(kata_bersih)


# Terapkan preprocessing
df['isi_berita_clean'] = df['isi_berita'].astype(str).apply(preprocess_teks)

print(f"\nContoh sebelum: {df['isi_berita'].iloc[0][:100]}...")
print(f"Contoh setelah: {df['isi_berita_clean'].iloc[0][:100]}...")

# Simpan hasil preprocessing
df.to_csv("data_berita_preprocessed.csv", index=False, encoding="utf-8")
df.to_excel("data_berita_preprocessed.xlsx", index=False)
print(f"\nTersimpan: data_berita_preprocessed.csv / .xlsx")
