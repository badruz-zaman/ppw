"""
Script Pembuat Website Multi-Page Statis (PPW)
Membagi website menjadi beberapa file HTML agar rapi, ringan, dan tidak ribuan baris:
1. index.html    -> Halaman Pengantar, Konsep Dasar, Taksonomi, dan Navigasi Tugas
2. crawling.html -> Halaman Khusus Tabel Data Crawling Berita Detik.com (200 Data)
3. tfidf.html    -> Halaman Khusus Pembobotan Kata TF-IDF Testing (40 Data x 7.426 Kolom)
4. pca.html      -> Halaman Khusus Reduksi Dimensi PCA Testing (40 Data x 160 PC)
Style dipisah ke style.css
"""

import pandas as pd
import html
import os
import time

t0 = time.time()
print("Memulai build website multi-page...")

# Helper untuk membuat header navbar
def buat_navbar(active_page):
    links = [
        ("index.html", "Beranda", active_page == "index"),
        ("crawling.html", "Data Crawling", active_page == "crawling"),
        ("tfidf.html", "TF-IDF Testing", active_page == "tfidf"),
        ("pca.html", "PCA Testing", active_page == "pca"),
    ]
    html_links = []
    for url, text, is_active in links:
        cls = "nav-link active" if is_active else "nav-link"
        html_links.append(f'<a href="{url}" class="{cls}">{text}</a>')
    
    return f"""  <div class="topbar">Pencarian dan Penambangan Web · 2026</div>
  <nav class="navbar">
    {" ".join(html_links)}
  </nav>"""

# ==============================================================================
# 1. GENERATE index.html (Halaman Pengantar & Konsep - Sangat Rapi & Ringkas)
# ==============================================================================
index_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pengantar Web Mining — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

{buat_navbar("index")}

  <div class="hero">
    <h1>Pengantar Web Mining</h1>
    <p class="subtitle">Konsep, taksonomi, dan proses penambangan informasi dari ekosistem World Wide Web.</p>
  </div>

  <div class="divider"></div>

  <div class="author">
    <div class="name">Badruz Zaman Ash Sholih</div>
    <div class="meta">240411100140 · <a href="mailto:badruzzamannnnn@gmail.com">badruzzamannnnn@gmail.com</a></div>
  </div>

  <main class="content content-narrow">

    <h2>Konsep Dasar</h2>
    <p>
      <strong>Web Mining</strong> adalah pemanfaatan teknik <em>data mining</em> secara otomatis untuk menemukan, mengekstraksi, dan memahami pola serta informasi tersembunyi dari dokumen, layanan, dan repositori World Wide Web (Liu, 2011).
    </p>
    <p>
      Web merupakan ekosistem data yang sangat besar, heterogen, dinamis, dan sebagian besar bersifat tidak terstruktur. Karakteristik inilah yang membedakannya dari data mining konvensional yang umumnya beroperasi pada basis data relasional.
    </p>

    <h2>Taksonomi</h2>
    <p>
      Berdasarkan literatur ilmiah (Kosala &amp; Blockeel, 2000), Web Mining diklasifikasikan ke dalam tiga domain utama:
    </p>

    <div class="taxonomy-item">
      <h4>Web Content Mining</h4>
      <p>
        Penambangan dan ekstraksi informasi dari konten halaman web — teks, gambar, audio, maupun video. Mengintegrasikan teknik <em>Natural Language Processing</em>, <em>Information Retrieval</em>, serta klasifikasi dan klasterisasi teks.
      </p>
    </div>

    <div class="taxonomy-item">
      <h4>Web Structure Mining</h4>
      <p>
        Menganalisis topologi hubungan antar dokumen web melalui representasi <em>hyperlink</em>. Algoritma fundamental di bidang ini adalah <strong>PageRank</strong> dan <strong>HITS</strong>, yang mengukur otoritas suatu halaman berdasarkan graf tautannya.
      </p>
    </div>

    <div class="taxonomy-item">
      <h4>Web Usage Mining</h4>
      <p>
        Meneliti jejak interaksi pengguna dengan server web. Sumber datanya meliputi log server, sesi interaksi, dan <em>clickstream</em>. Tujuannya mencakup analisis perilaku pengguna, personalisasi konten, dan sistem rekomendasi (Cooley et al., 1999).
      </p>
    </div>

    <h2>Siklus Proses</h2>
    <p>Secara umum, penambangan web mengikuti alur sistematis berikut:</p>
    <ol>
      <li><strong>Pengumpulan Data</strong> — crawling dan scraping dokumen dari internet.</li>
      <li><strong>Pra-pemrosesan</strong> — pembersihan HTML, tokenisasi, <em>stopword removal</em>, <em>stemming</em>, dan parsing log.</li>
      <li><strong>Penerapan Algoritma</strong> — asosiasi, klasifikasi, regresi, atau klasterisasi untuk menemukan pola.</li>
      <li><strong>Evaluasi &amp; Interpretasi</strong> — validasi pola yang ditemukan untuk pengambilan keputusan.</li>
    </ol>

    <h2>Tahapan Praktikum &amp; Dataset</h2>
    <p>Pilih halaman di bawah ini untuk melihat data dan hasil komputasi pada masing-masing tahapan:</p>

    <div class="grid-cards">
      <a href="crawling.html" class="card-nav">
        <span class="badge-stage">Tahap 1 · Crawling</span>
        <h3>Data Berita Detik.com</h3>
        <p>200 data artikel berita (100 kategori sport &amp; 100 kategori finance).</p>
      </a>

      <a href="tfidf.html" class="card-nav">
        <span class="badge-stage">Tahap 2 · Vektorisasi</span>
        <h3>TF-IDF Testing</h3>
        <p>Matriks pembobotan kata 40 data testing × 7.424 kata unik (vocabulary).</p>
      </a>

      <a href="pca.html" class="card-nav">
        <span class="badge-stage">Tahap 3 · Reduksi</span>
        <h3>PCA Testing</h3>
        <p>Hasil reduksi dimensi 40 data testing menjadi 160 Principal Components.</p>
      </a>
    </div>

    <div class="references">
      <h3>Daftar Pustaka</h3>
      <ol>
        <li>Liu, B. (2011). <em>Web Data Mining: Exploring Hyperlinks, Contents, and Usage Data</em> (2nd ed.). Springer-Verlag.</li>
        <li>Kosala, R., &amp; Blockeel, H. (2000). Web mining research: A survey. <em>ACM SIGKDD Explorations Newsletter</em>, 2(1), 1–15.</li>
        <li>Cooley, R., Mobasher, B., &amp; Srivastava, J. (1999). Data preparation for mining World Wide Web browsing patterns. <em>Knowledge and Information Systems</em>, 1(1), 5–32.</li>
        <li>Han, J., Kamber, M., &amp; Pei, J. (2011). <em>Data Mining: Concepts and Techniques</em> (3rd ed.). Morgan Kaufmann.</li>
      </ol>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman · 240411100140</footer>

</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(index_content)
print(f"-> index.html berhasil dibuat ({len(index_content.splitlines())} baris)")

# ==============================================================================
# 2. GENERATE crawling.html (Halaman Data Crawling)
# ==============================================================================
df_crawl = pd.read_csv("01_crawling/data_berita_detik.csv")
rows_crawl = []
for idx, row in df_crawl.iterrows():
    row_id = row['id']
    isi = html.escape(str(row['isi_berita']))
    label = html.escape(str(row['label']))
    badge_cls = f"badge badge-{label.lower()}"
    rows_crawl.append(f"""            <tr>
              <td class="col-id">{row_id}</td>
              <td class="col-berita">{isi}</td>
              <td class="col-label"><span class="{badge_cls}">{label}</span></td>
            </tr>""")
table_crawl_str = "\n".join(rows_crawl)

crawling_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Data Crawling Berita Detik.com — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

{buat_navbar("crawling")}

  <main class="content">

    <h2>Data Hasil Crawling Berita Detik.com</h2>
    <p>
      Berikut merupakan dataset berita hasil penambangan data (<em>web scraping/crawling</em>) dari portal Detik.com yang terdiri atas <strong>200 data artikel berita</strong> (100 berita kategori <strong>sport</strong> dan 100 berita kategori <strong>finance</strong>):
    </p>

    <div class="download-card">
      <div class="download-info">
        <strong>File Mentah Crawling:</strong> 200 baris × 3 kolom (id, isi_berita, label)
      </div>
      <div class="download-actions">
        <a href="01_crawling/data_berita_detik.xlsx" class="btn-download" download>📥 Download .xlsx</a>
        <a href="01_crawling/data_berita_detik.csv" class="btn-download btn-download-alt" download>📄 Download .csv</a>
      </div>
    </div>

    <div class="table-container">
      <table class="data-table">
        <thead>
          <tr>
            <th class="col-id">id</th>
            <th class="col-berita">isi_berita</th>
            <th class="col-label">label</th>
          </tr>
        </thead>
        <tbody>
{table_crawl_str}
        </tbody>
      </table>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman · 240411100140</footer>

</body>
</html>
"""

with open("crawling.html", "w", encoding="utf-8") as f:
    f.write(crawling_content)
print("-> crawling.html berhasil dibuat")

# ==============================================================================
# 3. GENERATE tfidf.html (Halaman TF-IDF Testing)
# ==============================================================================
df_tfidf = pd.read_csv("03_tfidf/tfidf_testing.csv")
sample_words = df_tfidf.columns[1:11].tolist()

tfidf_th_list = ['<th class="col-matrix-id">ID</th>']
for col in sample_words:
    tfidf_th_list.append(f'<th>{html.escape(col)}</th>')
tfidf_th_list.append('<th class="col-ellipsis">... [7.414 kata lainnya] ...</th>')
tfidf_th_list.append('<th class="col-matrix-label">Label</th>')
tfidf_thead = "<tr>" + "".join(tfidf_th_list) + "</tr>"

tfidf_rows = []
for row in df_tfidf.itertuples(index=False):
    cells = []
    cells.append(f'<td class="col-matrix-id">{row[0]}</td>')
    for val in row[1:11]:
        if val == 0.0 or val == 0:
            cells.append('<td>0</td>')
        else:
            cells.append(f'<td>{val:.4f}</td>')
    cells.append('<td class="col-ellipsis">...</td>')
    label_val = row[-1]
    badge_cls = "badge badge-sport" if label_val == 1 else "badge badge-finance"
    cells.append(f'<td class="col-matrix-label"><span class="{badge_cls}">{label_val}</span></td>')
    tfidf_rows.append("<tr>" + "".join(cells) + "</tr>")
tfidf_tbody = "\n".join(tfidf_rows)

tfidf_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pembobotan TF-IDF Testing — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

{buat_navbar("tfidf")}

  <main class="content">

    <h2>TF-IDF Tanpa Reduksi Dimensi (Data Testing)</h2>
    <p>
      Berikut merupakan representasi matriks pembobotan kata menggunakan metode <strong>TF-IDF</strong> tanpa reduksi dimensi untuk <strong>data testing</strong> (40 data dokumen berita dan 7.424 kata kosakata unik / <em>vocabulary</em>):
    </p>

    <div class="download-card">
      <div class="download-info">
        <strong>Dataset Lengkap:</strong> 40 baris × 7.426 kolom (7.424 fitur kata unik)
      </div>
      <div class="download-actions">
        <a href="03_tfidf/tfidf_testing.xlsx" class="btn-download" download>📥 Download .xlsx</a>
        <a href="03_tfidf/tfidf_testing.csv" class="btn-download btn-download-alt" download>📄 Download .csv</a>
      </div>
    </div>

    <div class="table-container">
      <table class="data-table matrix-table">
        <thead>
          {tfidf_thead}
        </thead>
        <tbody>
{tfidf_tbody}
        </tbody>
      </table>
    </div>
    <p class="preview-note">
      * Menampilkan sampel 10 kata representatif pertama dan kolom label. Seluruh 7.424 fitur kata lengkap dapat diunduh melalui tombol Excel/CSV di atas.
    </p>

  </main>

  <footer>&copy; 2026 Badruz Zaman · 240411100140</footer>

</body>
</html>
"""

with open("tfidf.html", "w", encoding="utf-8") as f:
    f.write(tfidf_content)
print("-> tfidf.html berhasil dibuat")

# ==============================================================================
# 4. GENERATE pca.html (Halaman PCA Testing)
# ==============================================================================
df_pca = pd.read_csv("04_reduksi_dimensi/tfidf_pca_testing.csv")
pca_cols = df_pca.columns.tolist()

pca_th_list = []
for i, col in enumerate(pca_cols):
    if col == 'ID':
        pca_th_list.append('<th class="col-matrix-id">ID</th>')
    elif col == 'Label':
        pca_th_list.append('<th class="col-matrix-label">Label</th>')
    else:
        pca_th_list.append(f'<th>{html.escape(col)}</th>')
pca_thead = "<tr>" + "".join(pca_th_list) + "</tr>"

pca_rows = []
for row in df_pca.itertuples(index=False):
    cells = []
    cells.append(f'<td class="col-matrix-id">{row[0]}</td>')
    for val in row[1:-1]:
        cells.append(f'<td>{val:.4f}</td>')
    label_val = row[-1]
    badge_cls = "badge badge-sport" if label_val == 1 else "badge badge-finance"
    cells.append(f'<td class="col-matrix-label"><span class="{badge_cls}">{label_val}</span></td>')
    pca_rows.append("<tr>" + "".join(cells) + "</tr>")
pca_tbody = "\n".join(pca_rows)

pca_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reduksi Dimensi PCA Testing — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

{buat_navbar("pca")}

  <main class="content">

    <h2>TF-IDF dengan Reduksi Dimensi PCA (Data Testing)</h2>
    <p>
      Berikut merupakan hasil reduksi dimensi menggunakan metode <strong>Principal Component Analysis (PCA)</strong> untuk <strong>data testing</strong> dari 7.424 fitur kata menjadi 160 komponen utama (<em>Principal Components</em>, PC1 s/d PC160):
    </p>

    <div class="download-card">
      <div class="download-info">
        <strong>Dataset PCA:</strong> 40 baris × 162 kolom (160 Principal Components)
      </div>
      <div class="download-actions">
        <a href="04_reduksi_dimensi/tfidf_pca_testing.xlsx" class="btn-download" download>📥 Download .xlsx</a>
        <a href="04_reduksi_dimensi/tfidf_pca_testing.csv" class="btn-download btn-download-alt" download>📄 Download .csv</a>
      </div>
    </div>

    <div class="table-container">
      <table class="data-table matrix-table">
        <thead>
          {pca_thead}
        </thead>
        <tbody>
{pca_tbody}
        </tbody>
      </table>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman · 240411100140</footer>

</body>
</html>
"""

with open("pca.html", "w", encoding="utf-8") as f:
    f.write(pca_content)
print("-> pca.html berhasil dibuat")

print(f"\nSelesai! Seluruh file website berhasil dibuat dalam {time.time()-t0:.2f} detik.")
