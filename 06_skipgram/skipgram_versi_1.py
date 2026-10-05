# =============================================================================
# SKIP-GRAM VERSI 1 (DENGAN ANGKA)
# Preprocessing: Membuang tanda baca dan kata tidak baku (angka dipertahankan)
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


# =============================================================================
# 1. LOAD DATA AWAL (HASIL CRAWLING BERITA DETIK.COM)
# =============================================================================

dataset_path = os.path.join("..", "01_crawling", "data_berita_detik.csv")
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {dataset_path}")

df = pd.read_csv(dataset_path)
print(f"Total Dokumen Berita : {len(df)} berita")
print(f"Distribusi Kelas     :\n{df['label'].value_counts().to_string()}\n")


# =============================================================================
# 2. PREPROCESSING (VERSI 1)
#    - Membuang tanda baca
#    - Membuang kata tidak baku / slang
#    - Angka TETAP dipertahankan
#    - TANPA lowercase (huruf kapital tetap dipertahankan)
# =============================================================================

# Kumpulan kata tidak baku / slang / singkatan yang umum di berita online
daftar_kata_tidak_baku = {
    "gak", "gk", "ga", "nggak", "ngga", "enggak", "kagak",
    "udah", "udh", "dah",
    "emang", "emg", "mmg",
    "banget", "bgt", "bngt",
    "aja", "aj",
    "gimana", "gmn",
    "gitu", "gtu",
    "kayak", "kyk",
    "biar", "byr",
    "doang", "dong", "dng",
    "nih", "neh",
    "tuh",
    "sih", "si",
    "yg", "yng",
    "dgn", "dg",
    "utk",
    "dlm",
    "tdk", "gk",
    "blm",
    "krn", "krna",
    "bs", "bsa",
    "jg", "jga",
    "sm", "sama2",
    "org",
    "trs", "trus", "terus2an",
    "lg", "lgi",
    "dr",
    "pd",
    "klo", "kalo",
    "ama",
    "bkn",
    "tp", "tpi",
    "dll",
    "dsb",
    "dst",
    "dkk",
    "wkwk", "wkwkwk", "haha", "hihi", "hehe",
    "btw", "fyi", "omg", "lol",
    "nyokap", "bokap",
    "cuy", "cuk", "bro", "sis", "gan",
    "gue", "gw", "gwa", "lo", "lu",
    "ngapain", "napa", "kenapa2",
    "mah", "atuh", "teh",
}

# Fungsi preprocessing Versi 1
# Ambil semua kata dan angka (\w+), buang tanda baca, buang kata tidak baku
def preprocess_versi_1(teks):
    list_token = re.findall(r'\b\w+\b', teks)
    clean_token = [kata for kata in list_token if kata.lower() not in daftar_kata_tidak_baku]
    return clean_token

# Terapkan preprocessing ke seluruh dokumen
data_token = [preprocess_versi_1(isi) for isi in df['isi_berita']]

total_kata = sum(len(kalimat) for kalimat in data_token)
kosakata_unik = len(set(kata for kalimat in data_token for kata in kalimat))
print(f"Hasil Preprocessing Versi 1 (Dengan Angka):")
print(f"    - Total Kata Keseluruhan : {total_kata:,} kata")
print(f"    - Total Kosakata Unik    : {kosakata_unik:,} kata\n")


# =============================================================================
# 3. SPLIT DATA (DATA TRAINING & DATA TESTING) = 80:20
# =============================================================================

# Bagi data menjadi 80% training dan 20% testing (stratified agar proporsi label seimbang)
token_training, token_testing, label_training, label_testing = train_test_split(
    data_token,
    df['label'].values,
    train_size=0.8,
    test_size=0.2,
    random_state=42,
    stratify=df['label'].values
)

print(f"Split Data 80:20 (Stratified):")
print(f"    - Data Training : {len(token_training)} dokumen")
print(f"    - Data Testing  : {len(token_testing)} dokumen")

# Simpan hasil split data ke CSV (teks hasil preprocessing + label)
df_split_training = pd.DataFrame({
    'teks_preprocessed': [' '.join(tokens) for tokens in token_training],
    'label': label_training
})
df_split_training.to_csv("split_v1_training.csv", index=False)

df_split_testing = pd.DataFrame({
    'teks_preprocessed': [' '.join(tokens) for tokens in token_testing],
    'label': label_testing
})
df_split_testing.to_csv("split_v1_testing.csv", index=False)

print(f"    - Disimpan: split_v1_training.csv ({len(df_split_training)} baris)")
print(f"    - Disimpan: split_v1_testing.csv ({len(df_split_testing)} baris)\n")


# =============================================================================
# 4. EMBEDDING SKIP-GRAM (REPRESENTASI KATA MENJADI VEKTOR)
#    - Model dilatih HANYA menggunakan Data Training
#    - vector_size = 100 (dimensi vektor)
#    - window = 5 (jendela konteks: 5 kata kiri & 5 kata kanan)
#    - sg = 1 (arsitektur Skip-Gram: Target -> Context)
# =============================================================================

dimensi_vektor = 100
ukuran_jendela = 5

print(f"Training Model Skip-Gram (pada Data Training saja):")
print(f"    - vector_size (Dimensi Vektor)  : {dimensi_vektor}")
print(f"    - window (Jendela Konteks)       : {ukuran_jendela} (kiri & kanan)")
print(f"    - sg = 1 (Skip-Gram: Target -> Context)")
print(f"    - min_count = 1 (semua kata diikutkan)")
print(f"    - epochs = 30 (iterasi pelatihan)")

# Training Skip-Gram hanya pada data training
model_skipgram = Word2Vec(
    sentences=token_training,
    vector_size=dimensi_vektor,
    window=ukuran_jendela,
    min_count=1,
    sg=1,              # 1 = arsitektur SKIP-GRAM (Target -> Context)
    epochs=30,         # Jumlah iterasi pelatihan
    seed=42,
    workers=1
)

jumlah_kosakata_model = len(model_skipgram.wv)
print(f"    - Kosakata yang dipelajari model : {jumlah_kosakata_model:,} kata\n")


# Fungsi Mean Pooling: mengubah vektor kata menjadi vektor dokumen
# Menghitung rata-rata vektor dari seluruh kata dalam satu dokumen
def hitung_vektor_dokumen(list_token, model_skipgram, dimensi_vektor):
    vektor_kata_valid = [model_skipgram.wv[kata] for kata in list_token if kata in model_skipgram.wv]
    if len(vektor_kata_valid) > 0:
        return np.mean(vektor_kata_valid, axis=0)
    else:
        return np.zeros(dimensi_vektor)


# Hitung vektor dokumen untuk data training dan testing
fitur_training = np.array([hitung_vektor_dokumen(token, model_skipgram, dimensi_vektor) for token in token_training])
fitur_testing  = np.array([hitung_vektor_dokumen(token, model_skipgram, dimensi_vektor) for token in token_testing])

print(f"Hasil Embedding (Mean Pooling):")
print(f"    - Matriks Training : {fitur_training.shape} ({len(token_training)} dokumen x {dimensi_vektor} dimensi)")
print(f"    - Matriks Testing  : {fitur_testing.shape} ({len(token_testing)} dokumen x {dimensi_vektor} dimensi)\n")


# =============================================================================
# 5. EXPORT CSV (UNTUK KLASIFIKASI DI ORANGE DATA MINING)
# =============================================================================

nama_kolom = [f"dim_{i+1}" for i in range(dimensi_vektor)]

# Export CSV Data Training
df_training = pd.DataFrame(fitur_training, columns=nama_kolom)
df_training['label'] = label_training
df_training.to_csv("skipgram_v1_training.csv", index=False)

# Export CSV Data Testing
df_testing = pd.DataFrame(fitur_testing, columns=nama_kolom)
df_testing['label'] = label_testing
df_testing.to_csv("skipgram_v1_testing.csv", index=False)

print(f"Export CSV Berhasil:")
print(f"    - skipgram_v1_training.csv : {len(df_training)} baris x {len(df_training.columns)} kolom")
print(f"    - skipgram_v1_testing.csv  : {len(df_testing)} baris x {len(df_testing.columns)} kolom")
print(f"\n>>> Selanjutnya, lakukan klasifikasi Naive Bayes & kNN di Orange Data Mining <<<")
