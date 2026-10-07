import os
import re
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset (Crawling)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "01_crawling", "data_berita_detik.csv")
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

print(f"Dataset : {dataset_path}")
print(f"Target  : {models_dir}\n")

df = pd.read_csv(dataset_path)
print(f"Total Dokumen Berita: {len(df)}")
print(f"Distribusi Kelas:\n{df['label'].value_counts().to_string()}\n")

# 2. Kata Tidak Baku
daftar_kata_tidak_baku = {
    "gak", "gk", "ga", "nggak", "ngga", "enggak", "kagak",
    "udah", "udh", "dah", "emang", "emg", "mmg", "banget", "bgt", "bngt",
    "aja", "aj", "gimana", "gmn", "gitu", "gtu", "kayak", "kyk", "biar", "byr",
    "doang", "dong", "dng", "nih", "neh", "tuh", "sih", "si", "yg", "yng",
    "dgn", "dg", "utk", "dlm", "tdk", "blm", "krn", "krna", "bs", "bsa",
    "jg", "jga", "sm", "sama2", "org", "trs", "trus", "terus2an", "lg", "lgi",
    "dr", "pd", "klo", "kalo", "ama", "bkn", "tp", "tpi", "dll", "dsb",
    "dst", "dkk", "wkwk", "wkwkwk", "haha", "hihi", "hehe", "btw", "fyi",
    "omg", "lol", "nyokap", "bokap", "cuy", "cuk", "bro", "sis", "gan",
    "gue", "gw", "gwa", "lo", "lu", "ngapain", "napa", "kenapa2", "mah",
    "atuh", "teh"
}

# 3. Preprocessing
def preprocess_v1(teks):
    # Versi 1: angka tidak dibuang #
    tokens = re.findall(r'\b\w+\b', str(teks))
    return [k for k in tokens if k.lower() not in daftar_kata_tidak_baku]

def preprocess_v2(teks):
    # Versi 2: buang angka #
    tokens = re.findall(r'\b[a-zA-Z]+\b', str(teks))
    return [k for k in tokens if k.lower() not in daftar_kata_tidak_baku]

# 4. Mean Pooling : Rata-rata vektor kata dalam satu dokumen

def hitung_vektor_dokumen(list_token, model_sg, dim=100):
    vektor_valid = [model_sg.wv[kata] for kata in list_token if kata in model_sg.wv]
    if len(vektor_valid) > 0:
        return np.mean(vektor_valid, axis=0)
    else:
        return np.zeros(dim)

# 5. Train dan Save Model

def train_save_model(nama_versi, fungsi_preprocess, prefix_file):
    # 1. Preprocessing seluruh teks
    data_token = [fungsi_preprocess(isi) for isi in df['isi_berita']]
    labels = df['label'].values
    
    # 2. Split Data 80:20 
    token_train, token_test, y_train, y_test = train_test_split(
        data_token,
        labels,
        train_size=0.8,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    print(f"Data Training : {len(token_train)} dokumen (80%)")
    print(f"Data Testing  : {len(token_test)} dokumen (20%)")
    
    # 3. Latih Word2Vec Skip-Gram pada Data Training
    model_sg = Word2Vec(
        sentences=token_train,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,              # 1 = Skip-Gram
        epochs=30,
        seed=42,
        workers=1
    )
    kosakata = len(model_sg.wv)
    print(f"Jumlah kosakata yang dipelajari: {kosakata:,} kata")
    
    # Simpan Model Skip-Gram
    sg_path = os.path.join(models_dir, f"skipgram_{prefix_file}.model")
    model_sg.save(sg_path)
    
    # 4. Mean Pooling untuk Data Training dan Testing
    X_train = np.array([hitung_vektor_dokumen(t, model_sg, dim=100) for t in token_train])
    X_test  = np.array([hitung_vektor_dokumen(t, model_sg, dim=100) for t in token_test])
    
    # 5. Latih & Simpan Model k-Nearest Neighbors (kNN)
    model_knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    model_knn.fit(X_train, y_train)
    
    y_pred_knn = model_knn.predict(X_test)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Akurasi kNN pada Data Testing: {acc_knn * 100:.2f}%")
    
    knn_path = os.path.join(models_dir, f"knn_{prefix_file}.joblib")
    joblib.dump(model_knn, knn_path)
    
    # 6. Latih & Simpan Model Naive Bayes
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    
    y_pred_nb = model_nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Akurasi Naive Bayes pada Data Testing: {acc_nb * 100:.2f}%")
    
    nb_path = os.path.join(models_dir, f"nb_{prefix_file}.joblib")
    joblib.dump(model_nb, nb_path)
    
    return {
        "versi": nama_versi,
        "acc_knn": acc_knn,
        "acc_nb": acc_nb
    }

hasil_v1 = train_save_model("Skip-Gram Versi 1 (Dengan Angka)", preprocess_v1, "v1")
hasil_v2 = train_save_model("Skip-Gram Versi 2 (Tanpa Angka)", preprocess_v2, "v2")


# 6. Ringkasan

print(f"1. Versi 1 (Dengan Angka) : kNN = {hasil_v1['acc_knn']*100:.1f}%, Naive Bayes = {hasil_v1['acc_nb']*100:.1f}%")
print(f"2. Versi 2 (Tanpa Angka)  : kNN = {hasil_v2['acc_knn']*100:.1f}%, Naive Bayes = {hasil_v2['acc_nb']*100:.1f}%")