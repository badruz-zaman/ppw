# Reduksi Dimensi dengan PCA
# Membaca tfidf_training/testing, mereduksi kolom kata unik dari ribuan menjadi 200 dimensi

import pandas as pd
from sklearn.decomposition import PCA

N_KOMPONEN = 160

# Baca data TF-IDF
df_train = pd.read_csv("tfidf_training.csv")
df_test = pd.read_csv("tfidf_testing.csv")
print(f"Training sebelum reduksi: {df_train.shape}")
print(f"Testing sebelum reduksi : {df_test.shape}")

# Pisahkan kolom fitur (tanpa ID dan Label)
fitur_train = df_train.drop(columns=['ID', 'Label']).values
fitur_test = df_test.drop(columns=['ID', 'Label']).values

# Terapkan PCA
pca = PCA(n_components=N_KOMPONEN, random_state=42)
pca_train = pca.fit_transform(fitur_train)
pca_test = pca.transform(fitur_test)

variance = sum(pca.explained_variance_ratio_) * 100
print(f"\nVariance explained: {variance:.2f}%")

# Buat DataFrame hasil: ID | PC1 | PC2 | ... | PC200 | Label
kolom_pca = [f'PC{i+1}' for i in range(N_KOMPONEN)]

df_pca_train = pd.DataFrame(pca_train, columns=kolom_pca)
df_pca_train.insert(0, 'ID', df_train['ID'].values)
df_pca_train['Label'] = df_train['Label'].values

df_pca_test = pd.DataFrame(pca_test, columns=kolom_pca)
df_pca_test.insert(0, 'ID', df_test['ID'].values)
df_pca_test['Label'] = df_test['Label'].values

print(f"Training sesudah reduksi: {df_pca_train.shape}")
print(f"Testing sesudah reduksi : {df_pca_test.shape}")

# Simpan
df_pca_train.to_csv("tfidf_pca_training.csv", index=False)
df_pca_train.to_excel("tfidf_pca_training.xlsx", index=False)
print(f"\nTersimpan: tfidf_pca_training.csv / .xlsx")

df_pca_test.to_csv("tfidf_pca_testing.csv", index=False)
df_pca_test.to_excel("tfidf_pca_testing.xlsx", index=False)
print(f"Tersimpan: tfidf_pca_testing.csv / .xlsx")
