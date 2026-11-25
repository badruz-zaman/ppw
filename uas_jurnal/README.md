# UAS Jurnal — Analisis Sitasi Jurnal & PageRank (ScienceDirect / Elsevier)

Aplikasi Web interaktif berbasis Streamlit untuk menganalisis **jaringan sitasi artikel ilmiah (Citation Network Analysis)** dan menentukan **artikel ilmiah paling berpengaruh (*Most Influential Paper*)** menggunakan algoritma **PageRank**.

---

## 1. Fitur Utama

* **Objek Simpul (Node):** Judul Artikel / Paper Ilmiah secara individual.
* **Garis Relasi (Edge):** Hubungan Sitasi / Daftar Pustaka ($A \to B$ artinya Paper A menyitasi Paper B).
* **Sumber Data:** Indeks resmi penerbit ilmiah ScienceDirect / Elsevier melalui metadata sinkronisasi terbuka (Crossref & OpenAlex API) — **100% bebas blokir HTTP 403 Forbidden dan tidak memerlukan API Key**.
* **Dukungan Input:**
  * DOI resmi (misal `10.1016/j.compag.2018.02.016`).
  * Link URL artikel ScienceDirect.
  * Kata kunci judul atau topik penelitian.
* **Algoritma Otoritas:** PageRank dengan *Power Iteration* konvergensi tinggi ($\text{tolerance} < 10^{-9}, d = 0.85$).
* **Visualisasi Animasi Fisika:** Kanvas graf interaktif Vis.js model *ForceAtlas2* dengan fitur *CitationGraph* (klik simpul untuk menyorot jalur sitasi terhubung).
* **Ekspor Data:** Download peringkat paper dan relasi sitasi ke format `.csv`.

---

## 2. Cara Menjalankan Aplikasi

Pastikan virtual environment aktif, lalu jalankan perintah berikut dari direktori utama (`e:\ppw`):

```bash
python -m streamlit run uas_jurnal/app.py
```
atau:
```bash
streamlit run uas_jurnal/app.py
```

Aplikasi akan otomatis terbuka di browser pada alamat:
👉 `http://localhost:8501`

---

## 3. Struktur File di Folder `uas_jurnal/`

* [`uas_jurnal/app.py`](file:///e:/ppw/uas_jurnal/app.py): Antarmuka web pengguna (Dashboard Streamlit, slider kontrol, tabel metrik, visualisasi graf, dan tab metodologi).
* [`uas_jurnal/crawler_citation.py`](file:///e:/ppw/uas_jurnal/crawler_citation.py): Mesin backend (Pengunduh sitasi ScienceDirect, pembangun graf berarah NetworkX, dan penghitung PageRank).
