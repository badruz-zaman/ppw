# Laporan Analisis Komparatif: Word2Vec Skip-Gram vs TF-IDF

**Penyusun:** Badruz Zaman Ash Sholih (240411100140)  
**Mata Kuliah:** Penambangan Web (PPW)  
**Dataset:** 200 Artikel Berita detik.com (100 Kategori *Finance*, 100 Kategori *Sport*)  
**Metode Evaluasi:** *Test on test data* (160 Data Latih / 80%, 40 Data Uji / 20%, Stratified)  
**Perangkat Lunak Evaluasi:** Orange Data Mining  

---

## 1. Ikhtisar Perbandingan Karakteristik Metode

| Parameter Evaluasi | TF-IDF (Term Frequency–Inverse Document Frequency) | Word2Vec Skip-Gram (Neural Word Embedding) |
| :--- | :--- | :--- |
| **Paradigma Dasar** | Statistik frekuensi kemunculan kata (*Lexical Frequency-based*) | Jaringan saraf tiruan / pembelajaran representasi (*Neural Representation Learning*) |
| **Bentuk Representasi** | **Matriks Jarang (Sparse Matrix)** — mayoritas elemen bernilai 0 | **Matriks Padat (Dense Matrix)** — seluruh nilai berupa vektor riil kontinu |
| **Dimensi Fitur** | **7.424 dimensi** (setiap kata unik dalam korpus menjadi 1 kolom) | **100 dimensi** (tetap, **74× lebih ringkas** dibanding TF-IDF) |
| **Kapasitas Semantik** | **Tidak ada (Ortogonal)**: kata sinonim/terkait dianggap independen (*saham* &ne; *dividen*) | **Tinggi (Distributional Semantics)**: kata dalam konteks serupa berada berdekatan di ruang laten |
| **Ketergantungan Reduksi** | Sangat membutuhkan reduksi fitur (PCA / Chi-Square) agar komputasi tidak membengkak | **Bawaan (Native)**: langsung berdimensi kompak tanpa memerlukan reduksi tambahan |
| **Skalabilitas Data** | Rentan terhadap *Curse of Dimensionality* jika jumlah dokumen bertambah banyak | Sangat terukur (*scalable*), dimensi tetap konstan berapapun ukuran korpus teks |

---

## 2. Perbandingan Hasil Klasifikasi di Orange Data Mining

Evaluasi performa klasifikasi dilakukan menggunakan 40 dokumen data uji independen (*Test on test data*):

### Tabel Hasil Pengujian Lengkap

| Skenario Representasi Fitur | Jumlah Dimensi | Algoritma Klasifikasi | AUC | CA (Akurasi) | F1-Score | Precision | Recall | MCC | Salah Prediksi |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Raw TF-IDF (Tanpa Reduksi)** | 7.424 | **Naive Bayes** | 1,000 | **1,000 (100,0%)** | 1,000 | 1,000 | 1,000 | 1,000 | 0 / 40 |
| | | **kNN** | 1,000 | **1,000 (100,0%)** | 1,000 | 1,000 | 1,000 | 1,000 | 0 / 40 |
| **TF-IDF + PCA 50 Komponen** | 50 | **kNN 🌟** | 1,000 | **1,000 (100,0%)** | 1,000 | 1,000 | 1,000 | 1,000 | 0 / 40 |
| | | **Naive Bayes** | 1,000 | **0,975 (97,5%)** | 0,975 | 0,976 | 0,975 | 0,951 | 1 / 40 |
| **TF-IDF + PCA 10 Komponen** | 10 | **kNN** | 1,000 | **0,975 (97,5%)** | 0,975 | 0,976 | 0,975 | 0,951 | 1 / 40 |
| | | **Naive Bayes** | 0,990 | **0,925 (92,5%)** | 0,925 | 0,935 | 0,925 | 0,860 | 3 / 40 |
| **Skip-Gram Versi 1 (Dengan Angka)** | 100 | **Naive Bayes** | 1,000 | **0,975 (97,5%)** | 0,975 | 0,976 | 0,975 | 0,951 | 1 / 40 |
| | | **kNN** | 0,989 | **0,925 (92,5%)** | 0,925 | 0,935 | 0,925 | 0,860 | 3 / 40 |
| **Skip-Gram Versi 2 (Tanpa Angka)** | 100 | **kNN 🌟** | **1,000** | **1,000 (100,0%)** | **1,000** | **1,000** | **1,000** | **1,000** | **0 / 40 (Sempurna)** |
| | | **Naive Bayes** | 1,000 | **0,975 (97,5%)** | 0,975 | 0,976 | 0,975 | 0,951 | 1 / 40 |

---

## 3. Analisis Mendalam: Mengapa Hasilnya Bisa Berbeda?

### Faktor 1: Struktur Ruang Fitur (Sparse 7.424D vs Dense 100D)
- **Pada Raw TF-IDF**, setiap kata unik membentuk satu sumbu dimensi tersendiri. Berita *Sport* dan *Finance* memiliki leksikon kata kunci yang sangat berbeda secara mencolok (misal: *pelatih, gawang, skor, liga* vs *inflasi, obligasi, emiten, laba*). Karena ruang fiturnya berukuran 7.424 dimensi, dokumen kedua kategori terpisah sangat jauh di sudut-sudut ruang ortogonal, sehingga Naive Bayes dan kNN dengan mudah mencapai akurasi 100%. Namun, kelemahannya adalah **sangat boros memori dan komputasi**.
- **Pada Word2Vec Skip-Gram**, 7.424 kata dipadatkan ke dalam ruang kontinu berdimensi 100 melalui mekanisme proyeksi bobot jaringan saraf. Pada Skip-Gram Versi 2, model membuktikan bahwa **100 dimensi padat sudah lebih dari cukup untuk mempertahankan daya pemisah (separabilitas) sempurna (kNN 100%)**, dengan efisiensi memori 74 kali lebih hemat.

### Faktor 2: Mekanisme Sensitivitas Noise Angka (Mean Pooling vs Bobot IDF)
Inilah alasan krusial mengapa performa kNN berbeda antara Skip-Gram Versi 1 dan Versi 2:
- **Pada TF-IDF:** Angka-angka acak (seperti tanggal "2024", waktu "90", angka nominal transaksi) tersebar di kolom-kolom tersendiri. Karena angka-angka umum muncul di kedua kategori berita, rumus **IDF (Inverse Document Frequency)** secara otomatis menekan bobot nilai angka tersebut mendekati nol, sehingga angka tidak merusak performa klasifikasi TF-IDF.
- **Pada Skip-Gram:** Representasi tingkat dokumen dibentuk menggunakan **Mean Pooling (Perataan Vektor Kata)**:
  $$d_k = \frac{1}{N_k} \sum_{i=1}^{N_k} v(w_i)$$
  Pada **Versi 1 (Dengan Angka)**, vektor representasi kata angka ikut dijumlahkan dan dirata-ratakan ke dalam vektor dokumen. Karena angka muncul acak di teks sport dan finance, vektor angka bertindak sebagai **derau (noise) spasial** yang sedikit menggeser koordinat dokumen, menyebabkan 3 dokumen testing salah dikelompokkan oleh kNN (akurasi 92,5%).
- Saat angka dihapus secara total pada **Versi 2**, rata-rata vektor murni merefleksikan kata-kata semantik topik, sehingga kluster spasial antarkategori terpisah bersih dan kNN melompat menjadi **100,0% sempurna**.

### Faktor 3: Perilaku Algoritma (kNN vs Naive Bayes)
- **k-Nearest Neighbors (kNN):** Bekerja berdasarkan jarak geometris *Euclidean*. kNN sangat diuntungkan oleh ruang representasi Skip-Gram Versi 2 karena dokumen dengan topik yang sama membentuk kluster yang padat dan terisolasi dari kelas lain, memastikan seluruh 5 tetangga terdekat selalu tepat sasaran (100%).
- **Naive Bayes:** Bekerja berdasarkan estimasi probabilitas bersyarat dengan asumsi distribusi Gaussian pada fitur kontinu. Naive Bayes menunjukkan stabilitas luar biasa dengan akurasi tetap **97,5% (hanya 1 salah prediksi)** baik pada Versi 1 maupun Versi 2, membuktikan ketahanannya terhadap fluktuasi lokal nilai fitur.

### Faktor 4: Perbandingan dengan Reduksi PCA pada TF-IDF
- Untuk mengecilkan TF-IDF agar setara dengan dimensi Skip-Gram (sekitar 50–100 dimensi), diperlukan transformasi PCA.
- Pada **PCA 150**, akurasi Naive Bayes sempat anjlok ke **65,0%** akibat komponen minor berbobot variansi kecil yang mengacaukan estimasi probabilitas.
- Sebaliknya, **Skip-Gram 100 dimensi secara langsung (*native*) stabil di rentang 97,5% – 100%** tanpa pernah mengalami degradasi performa drastis seperti PCA 150.

---

## 4. Kesimpulan Rekomendasi Praktis

1. **Gunakan TF-IDF jika:**
   - Dataset berukuran kecil hingga menengah dengan topik yang kosakatanya sangat berbeda secara leksikal.
   - Komputasi memori bukan merupakan kendala.
   - Diperlukan interpretasi langsung fitur kata mana yang paling berkontribusi terhadap klasifikasi.

2. **Gunakan Word2Vec Skip-Gram jika:**
   - Mengutamakan efisiensi ruang fitur (hanya 100 dimensi padat vs ribuan dimensi jarang).
   - Ingin menangkap hubungan semantik dan asosiasi antarkata (menghubungkan sinonim atau kata dalam konteks serupa).
   - Korpus teks akan terus bertambah besar secara dinamis di masa depan tanpa ingin ukuran dimensi matriks meledak (*curse of dimensionality*).
   - **Catatan implementasi:** Karakter angka sebaiknya dibersihkan pada tahap *preprocessing* jika menggunakan agregasi Mean Pooling agar tidak menimbulkan noise spasial pada model berbasis jarak seperti kNN.
