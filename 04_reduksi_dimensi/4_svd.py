# Reduksi Dimensi dengan SVD (TruncatedSVD)
# Membaca tfidf_training/testing, mereduksi kolom kata unik dari ribuan menjadi 200 dimensi

import pandas as pd
from sklearn.decomposition import TruncatedSVD

N_KOMPONEN = 160

# Baca data TF-IDF
df_train = pd.read_csv("tfidf_training.csv")
df_test = pd.read_csv("tfidf_testing.csv")
print(f"Training sebelum reduksi: {df_train.shape}")
print(f"Testing sebelum reduksi : {df_test.shape}")

# Pisahkan kolom fitur (tanpa ID dan Label)
fitur_train = df_train.drop(columns=['ID', 'Label']).values
fitur_test = df_test.drop(columns=['ID', 'Label']).values

# Terapkan SVD
svd = TruncatedSVD(n_components=N_KOMPONEN, random_state=42)
svd_train = svd.fit_transform(fitur_train)
svd_test = svd.transform(fitur_test)

variance = sum(svd.explained_variance_ratio_) * 100
print(f"\nVariance explained: {variance:.2f}%")

# Buat DataFrame hasil: ID | SVD1 | SVD2 | ... | SVD200 | Label
kolom_svd = [f'SVD{i+1}' for i in range(N_KOMPONEN)]

df_svd_train = pd.DataFrame(svd_train, columns=kolom_svd)
df_svd_train.insert(0, 'ID', df_train['ID'].values)
df_svd_train['Label'] = df_train['Label'].values

df_svd_test = pd.DataFrame(svd_test, columns=kolom_svd)
df_svd_test.insert(0, 'ID', df_test['ID'].values)
df_svd_test['Label'] = df_test['Label'].values

print(f"Training sesudah reduksi: {df_svd_train.shape}")
print(f"Testing sesudah reduksi : {df_svd_test.shape}")

# Simpan
df_svd_train.to_csv("tfidf_svd_training.csv", index=False)
df_svd_train.to_excel("tfidf_svd_training.xlsx", index=False)
print(f"\nTersimpan: tfidf_svd_training.csv / .xlsx")

df_svd_test.to_csv("tfidf_svd_testing.csv", index=False)
df_svd_test.to_excel("tfidf_svd_testing.xlsx", index=False)
print(f"Tersimpan: tfidf_svd_testing.csv / .xlsx")
