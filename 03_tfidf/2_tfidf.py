# TF-IDF dan Split Training/Testing
# Membaca data_berita_preprocessed.xlsx, membuat matriks TF-IDF, lalu split 160 train + 40 test

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Baca data hasil preprocessing
df = pd.read_excel("data_berita_preprocessed.xlsx")
print(f"Total data: {len(df)}")

# Buat TF-IDF dari teks yang sudah bersih
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['isi_berita_clean'])

# Ambil nama kata unik
kata_unik = vectorizer.get_feature_names_out()
print(f"Jumlah kata unik: {len(kata_unik)}")

# Buat DataFrame TF-IDF dengan format: ID | kata1 | kata2 | ... | Label
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=kata_unik)
df_tfidf.insert(0, 'ID', df['id'].values)
df_tfidf['Label'] = df['label'].map({'sport': 1, 'finance': 2}).values

print(f"Bentuk matriks: {df_tfidf.shape}")

# Split: 160 training + 40 testing (stratified)
df_train, df_test = train_test_split(
    df_tfidf,
    test_size=40,
    train_size=160,
    random_state=42,
    stratify=df_tfidf['Label']
)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

sport_train = len(df_train[df_train['Label'] == 1])
finance_train = len(df_train[df_train['Label'] == 2])
sport_test = len(df_test[df_test['Label'] == 1])
finance_test = len(df_test[df_test['Label'] == 2])

print(f"\nTraining: {len(df_train)} data (Sport: {sport_train}, Finance: {finance_train})")
print(f"Testing : {len(df_test)} data (Sport: {sport_test}, Finance: {finance_test})")

# Simpan
df_train.to_csv("tfidf_training.csv", index=False)
df_train.to_excel("tfidf_training.xlsx", index=False)
print(f"\nTersimpan: tfidf_training.csv / .xlsx ({df_train.shape})")

df_test.to_csv("tfidf_testing.csv", index=False)
df_test.to_excel("tfidf_testing.xlsx", index=False)
print(f"Tersimpan: tfidf_testing.csv / .xlsx ({df_test.shape})")
