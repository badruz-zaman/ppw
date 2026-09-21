"""
Preprocessing, TF-IDF, dan Reduksi Dimensi Data Berita Detik.com
Mata Kuliah: Pencarian dan Penambangan Web (PPW)

Alur:
1. Baca data hasil crawling (data_berita_detik.xlsx)
2. Preprocessing teks (lowercase, hapus angka, hapus tanda baca, hapus kata tidak baku)
3. Ekstraksi fitur TF-IDF
4. Split data: 160 training + 40 testing
5. Simpan versi tanpa reduksi dimensi
6. Reduksi dimensi dengan PCA (200 dimensi) dan SVD (200 dimensi)
7. Simpan versi dengan reduksi dimensi

Install (jika belum):
    pip install pandas openpyxl scikit-learn
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split

# ================================================================
# 1. BACA DATA
# ================================================================
print("=" * 60)
print("1. MEMBACA DATA")
print("=" * 60)

df = pd.read_excel("data_berita_detik.xlsx")
print(f"   Total data  : {len(df)}")
print(f"   Kolom       : {list(df.columns)}")
print(f"   Sport       : {len(df[df['label'] == 'sport'])}")
print(f"   Finance     : {len(df[df['label'] == 'finance'])}")

# ================================================================
# 2. PREPROCESSING TEKS
# ================================================================
print("\n" + "=" * 60)
print("2. PREPROCESSING TEKS")
print("=" * 60)

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
    """
    Preprocessing teks:
    1. Ubah ke lowercase
    2. Hapus angka
    3. Hapus tanda baca dan karakter khusus
    4. Hapus kata tidak baku / singkatan
    5. Hapus kata dengan panjang <= 1 huruf
    """
    # Lowercase
    teks = teks.lower()

    # Hapus angka
    teks = re.sub(r'\d+', '', teks)

    # Hapus tanda baca dan karakter khusus (sisakan huruf dan spasi)
    teks = re.sub(r'[^a-z\s]', '', teks)

    # Hapus spasi berlebih
    teks = re.sub(r'\s+', ' ', teks).strip()

    # Hapus kata tidak baku dan kata 1 huruf
    kata_list = teks.split()
    kata_bersih = [k for k in kata_list if k not in kata_tidak_baku and len(k) > 1]

    return ' '.join(kata_bersih)


# Terapkan preprocessing ke semua isi berita
df['isi_berita_clean'] = df['isi_berita'].astype(str).apply(preprocess_teks)

print("   Preprocessing selesai!")
print(f"\n   Contoh SEBELUM preprocessing:")
print(f"   {df['isi_berita'].iloc[0][:100]}...")
print(f"\n   Contoh SETELAH preprocessing:")
print(f"   {df['isi_berita_clean'].iloc[0][:100]}...")

# ================================================================
# 3. EKSTRAKSI FITUR TF-IDF
# ================================================================
print("\n" + "=" * 60)
print("3. EKSTRAKSI FITUR TF-IDF")
print("=" * 60)

# Buat TF-IDF dari teks yang sudah dipreprocessing
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['isi_berita_clean'])

# Ambil nama-nama kata unik (fitur)
kata_unik = vectorizer.get_feature_names_out()

print(f"   Jumlah dokumen     : {tfidf_matrix.shape[0]}")
print(f"   Jumlah kata unik   : {tfidf_matrix.shape[1]}")

# Konversi ke DataFrame
# Format: ID | kata1 | kata2 | ... | kataN | Label
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=kata_unik
)

# Tambah kolom ID di awal
df_tfidf.insert(0, 'ID', df['id'].values)

# Tambah kolom Label di akhir (1 = sport, 2 = finance)
df_tfidf['Label'] = df['label'].map({'sport': 1, 'finance': 2}).values

print(f"   Bentuk matriks TF-IDF: {df_tfidf.shape}")
print(f"   Kolom: ID + {len(kata_unik)} kata unik + Label")

# ================================================================
# 4. SPLIT DATA: 160 TRAINING + 40 TESTING
# ================================================================
print("\n" + "=" * 60)
print("4. SPLIT DATA (160 TRAINING + 40 TESTING)")
print("=" * 60)

# Stratified split: proporsi sport/finance sama di train dan test
df_train, df_test = train_test_split(
    df_tfidf,
    test_size=40,
    train_size=160,
    random_state=42,
    stratify=df_tfidf['Label']  # 80 sport + 80 finance di training
)

# Reset index
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"   Training : {len(df_train)} data (Sport: {len(df_train[df_train['Label']==1])}, Finance: {len(df_train[df_train['Label']==2])})")
print(f"   Testing  : {len(df_test)} data (Sport: {len(df_test[df_test['Label']==1])}, Finance: {len(df_test[df_test['Label']==2])})")

# ================================================================
# 5. SIMPAN VERSI TANPA REDUKSI DIMENSI
# ================================================================
print("\n" + "=" * 60)
print("5. SIMPAN TF-IDF TANPA REDUKSI DIMENSI")
print("=" * 60)

df_train.to_csv("tfidf_training.csv", index=False)
df_train.to_excel("tfidf_training.xlsx", index=False)
print(f"   Tersimpan: tfidf_training.csv / .xlsx ({df_train.shape})")

df_test.to_csv("tfidf_testing.csv", index=False)
df_test.to_excel("tfidf_testing.xlsx", index=False)
print(f"   Tersimpan: tfidf_testing.csv / .xlsx ({df_test.shape})")

# ================================================================
# 6. REDUKSI DIMENSI
# ================================================================
print("\n" + "=" * 60)
print("6. REDUKSI DIMENSI (200 DIMENSI)")
print("=" * 60)

N_KOMPONEN = 200

# Ambil hanya kolom fitur (tanpa ID dan Label)
fitur_train = df_train.drop(columns=['ID', 'Label']).values
fitur_test = df_test.drop(columns=['ID', 'Label']).values

# --- 6a. PCA ---
print("\n   [PCA] Mereduksi dimensi...")
pca = PCA(n_components=N_KOMPONEN, random_state=42)
pca_train = pca.fit_transform(fitur_train)
pca_test = pca.transform(fitur_test)

# Buat DataFrame hasil PCA
kolom_pca = [f'PC{i+1}' for i in range(N_KOMPONEN)]

df_pca_train = pd.DataFrame(pca_train, columns=kolom_pca)
df_pca_train.insert(0, 'ID', df_train['ID'].values)
df_pca_train['Label'] = df_train['Label'].values

df_pca_test = pd.DataFrame(pca_test, columns=kolom_pca)
df_pca_test.insert(0, 'ID', df_test['ID'].values)
df_pca_test['Label'] = df_test['Label'].values

variance_pca = sum(pca.explained_variance_ratio_) * 100
print(f"   [PCA] Variance explained: {variance_pca:.2f}%")
print(f"   [PCA] Bentuk training: {df_pca_train.shape}")
print(f"   [PCA] Bentuk testing : {df_pca_test.shape}")

# Simpan PCA
df_pca_train.to_csv("tfidf_pca_training.csv", index=False)
df_pca_train.to_excel("tfidf_pca_training.xlsx", index=False)
print(f"   Tersimpan: tfidf_pca_training.csv / .xlsx")

df_pca_test.to_csv("tfidf_pca_testing.csv", index=False)
df_pca_test.to_excel("tfidf_pca_testing.xlsx", index=False)
print(f"   Tersimpan: tfidf_pca_testing.csv / .xlsx")

# --- 6b. SVD (TruncatedSVD) ---
print(f"\n   [SVD] Mereduksi dimensi...")
svd = TruncatedSVD(n_components=N_KOMPONEN, random_state=42)
svd_train = svd.fit_transform(fitur_train)
svd_test = svd.transform(fitur_test)

# Buat DataFrame hasil SVD
kolom_svd = [f'SVD{i+1}' for i in range(N_KOMPONEN)]

df_svd_train = pd.DataFrame(svd_train, columns=kolom_svd)
df_svd_train.insert(0, 'ID', df_train['ID'].values)
df_svd_train['Label'] = df_train['Label'].values

df_svd_test = pd.DataFrame(svd_test, columns=kolom_svd)
df_svd_test.insert(0, 'ID', df_test['ID'].values)
df_svd_test['Label'] = df_test['Label'].values

variance_svd = sum(svd.explained_variance_ratio_) * 100
print(f"   [SVD] Variance explained: {variance_svd:.2f}%")
print(f"   [SVD] Bentuk training: {df_svd_train.shape}")
print(f"   [SVD] Bentuk testing : {df_svd_test.shape}")

# Simpan SVD
df_svd_train.to_csv("tfidf_svd_training.csv", index=False)
df_svd_train.to_excel("tfidf_svd_training.xlsx", index=False)
print(f"   Tersimpan: tfidf_svd_training.csv / .xlsx")

df_svd_test.to_csv("tfidf_svd_testing.csv", index=False)
df_svd_test.to_excel("tfidf_svd_testing.xlsx", index=False)
print(f"   Tersimpan: tfidf_svd_testing.csv / .xlsx")

# ================================================================
# 7. RINGKASAN
# ================================================================
print("\n" + "=" * 60)
print("RINGKASAN FILE OUTPUT")
print("=" * 60)
print("""
   TANPA REDUKSI DIMENSI (TF-IDF):
   ├── tfidf_training.csv / .xlsx   (160 x {kolom_tfidf})
   └── tfidf_testing.csv / .xlsx    (40 x {kolom_tfidf})

   DENGAN REDUKSI PCA (200 dimensi):
   ├── tfidf_pca_training.csv / .xlsx   (160 x 202)
   └── tfidf_pca_testing.csv / .xlsx    (40 x 202)

   DENGAN REDUKSI SVD (200 dimensi):
   ├── tfidf_svd_training.csv / .xlsx   (160 x 202)
   └── tfidf_svd_testing.csv / .xlsx    (40 x 202)

   Format kolom:
   - Tanpa reduksi : ID | kata1 | kata2 | ... | kata{n_kata} | Label
   - Dengan reduksi: ID | dim1 | dim2 | ... | dim200 | Label
   - Label: 1 = Sport, 2 = Finance
""".format(
    kolom_tfidf=len(kata_unik) + 2,
    n_kata=len(kata_unik)
))

print("Selesai!")
