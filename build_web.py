"""
Script Pembuat Website Multi-Page Statis (PPW)
Redesain dengan Floating Action Button (FAB), narasi tiap halaman,
data training + testing, dan halaman eksperimen.

Output:
1. index.html      -> Pengantar Web Mining (konsep, taksonomi, siklus)
2. crawling.html   -> Data Crawling + narasi proses
3. tfidf.html      -> TF-IDF Training & Testing + narasi proses
4. pca.html        -> PCA Training & Testing + narasi proses
5. eksperimen.html -> Hasil perbandingan akurasi kNN vs Naive Bayes
Style: style.css (dibuat terpisah)
"""

import pandas as pd
import html
import time

t0 = time.time()
print("Memulai build website multi-page (redesain)...")

# ==============================================================================
# HELPER: Floating Action Button (FAB) Menu Component
# ==============================================================================
def buat_fab_menu(active_page):
    items = [
        ("index.html",      "Pengantar",  "\U0001F4D6", "index"),
        ("crawling.html",   "Crawling",   "\U0001F577\uFE0F", "crawling"),
        ("tfidf.html",      "TF-IDF",     "\U0001F4CA", "tfidf"),
        ("pca.html",        "PCA",        "\U0001F52C", "pca"),
        ("eksperimen.html", "Eksperimen", "\U0001F9EA", "eksperimen"),
    ]
    fab_items = []
    for i, (url, label, emoji, page_id) in enumerate(items):
        active_cls = " active" if page_id == active_page else ""
        fab_items.append(
            f'    <a href="{url}" class="fab-item{active_cls}" '
            f'data-tooltip="{label}" style="--i:{i}">'
            f'<span class="fab-emoji">{emoji}</span></a>'
        )
    items_html = "\n".join(fab_items)
    return f'''
  <div class="fab-container" id="fabNav">
    <div class="fab-overlay" onclick="document.getElementById('fabNav').classList.remove('open')"></div>
    <nav class="fab-menu">
{items_html}
    </nav>
    <button class="fab-toggle" onclick="document.getElementById('fabNav').classList.toggle('open')" aria-label="Menu navigasi">
      <div class="fab-moon"></div>
      <span class="fab-close">\u2715</span>
    </button>
  </div>'''


# HELPER: Tab switching script (inline JS)
TAB_SCRIPT = '''
  <script>
    function switchTab(btn, panelId) {
      var section = btn.closest('.data-section');
      section.querySelectorAll('.data-tab').forEach(function(t) { t.classList.remove('active'); });
      section.querySelectorAll('.tab-panel').forEach(function(p) { p.style.display = 'none'; });
      btn.classList.add('active');
      document.getElementById(panelId).style.display = 'block';
    }
  </script>'''


# HELPER: Generate preview table rows for matrix data (TF-IDF / PCA)
def buat_matrix_preview(df, n_preview_cols=10):
    """Generate thead + tbody HTML showing first n_preview_cols feature columns + ellipsis + label."""
    cols = df.columns.tolist()
    id_col = cols[0]
    label_col = cols[-1]
    feature_cols = cols[1:-1]
    preview_cols = feature_cols[:n_preview_cols]
    remaining = len(feature_cols) - n_preview_cols

    # Thead
    th_parts = [f'<th class="col-matrix-id">{html.escape(str(id_col))}</th>']
    for c in preview_cols:
        th_parts.append(f'<th>{html.escape(str(c))}</th>')
    if remaining > 0:
        th_parts.append(f'<th class="col-ellipsis">... [{remaining:,} kolom lainnya] ...</th>')
    th_parts.append(f'<th class="col-matrix-label">{html.escape(str(label_col))}</th>')
    thead = "<tr>" + "".join(th_parts) + "</tr>"

    # Tbody
    rows = []
    for row_data in df.itertuples(index=False):
        cells = [f'<td class="col-matrix-id">{row_data[0]}</td>']
        for idx in range(1, 1 + len(preview_cols)):
            val = row_data[idx]
            if val == 0.0 or val == 0:
                cells.append('<td>0</td>')
            else:
                cells.append(f'<td>{val:.4f}</td>')
        if remaining > 0:
            cells.append('<td class="col-ellipsis">...</td>')
        label_val = row_data[-1]
        badge_cls = "badge badge-1" if label_val == 1 else "badge badge-2"
        cells.append(f'<td class="col-matrix-label"><span class="{badge_cls}">{label_val}</span></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "\n".join(rows)

    return thead, tbody


# ==============================================================================
# 1. GENERATE index.html — Halaman Pengantar
# ==============================================================================
print("  Generating index.html...")

index_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pengantar Web Mining — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="hero">
    <h1>Pengantar Web Mining</h1>
    <p class="subtitle">Konsep, taksonomi, dan proses penambangan informasi dari ekosistem World Wide Web.</p>
  </div>

  <div class="divider"></div>

  <div class="author">
    <div class="name">Badruz Zaman Ash Sholih</div>
    <div class="meta">240411100140 &middot; <a href="mailto:badruzzamannnnn@gmail.com">badruzzamannnnn@gmail.com</a></div>
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
        Penambangan dan ekstraksi informasi dari konten halaman web &mdash; teks, gambar, audio, maupun video. Mengintegrasikan teknik <em>Natural Language Processing</em>, <em>Information Retrieval</em>, serta klasifikasi dan klasterisasi teks.
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
      <li><strong>Pengumpulan Data</strong> &mdash; crawling dan scraping dokumen dari internet.</li>
      <li><strong>Pra-pemrosesan</strong> &mdash; pembersihan HTML, tokenisasi, <em>stopword removal</em>, <em>stemming</em>, dan parsing log.</li>
      <li><strong>Penerapan Algoritma</strong> &mdash; asosiasi, klasifikasi, regresi, atau klasterisasi untuk menemukan pola.</li>
      <li><strong>Evaluasi &amp; Interpretasi</strong> &mdash; validasi pola yang ditemukan untuk pengambilan keputusan.</li>
    </ol>

    <div class="references">
      <h3>Daftar Pustaka</h3>
      <ol>
        <li>Liu, B. (2011). <em>Web Data Mining: Exploring Hyperlinks, Contents, and Usage Data</em> (2nd ed.). Springer-Verlag.</li>
        <li>Kosala, R., &amp; Blockeel, H. (2000). Web mining research: A survey. <em>ACM SIGKDD Explorations Newsletter</em>, 2(1), 1&ndash;15.</li>
        <li>Cooley, R., Mobasher, B., &amp; Srivastava, J. (1999). Data preparation for mining World Wide Web browsing patterns. <em>Knowledge and Information Systems</em>, 1(1), 5&ndash;32.</li>
        <li>Han, J., Kamber, M., &amp; Pei, J. (2011). <em>Data Mining: Concepts and Techniques</em> (3rd ed.). Morgan Kaufmann.</li>
      </ol>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("index")}
</body>
</html>
'''

with open("index.html", "w", encoding="utf-8") as f:
    f.write(index_html)
print(f"  -> index.html ({len(index_html.splitlines())} baris)")


# ==============================================================================
# 2. GENERATE crawling.html — Halaman Data Crawling + Narasi
# ==============================================================================
print("  Generating crawling.html...")

df_crawl = pd.read_csv("01_crawling/data_berita_detik.csv")
rows_crawl = []
for _, row in df_crawl.iterrows():
    row_id = row['id']
    isi = html.escape(str(row['isi_berita']))
    label = html.escape(str(row['label']))
    badge_cls = f"badge badge-{label.lower()}"
    rows_crawl.append(f'''            <tr>
              <td class="col-id">{row_id}</td>
              <td class="col-berita">{isi}</td>
              <td class="col-label"><span class="{badge_cls}">{label}</span></td>
            </tr>''')
table_crawl = "\n".join(rows_crawl)

crawling_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Data Crawling Berita Detik.com — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="page-header">
    <h1>Data Crawling Berita</h1>
    <p class="subtitle">Pengumpulan data mentah dari portal berita Detik.com secara otomatis</p>
  </div>

  <main class="content">

    <div class="narasi">
      <span class="narasi-label">Tentang Proses Ini</span>
      <p>
        Tahap pertama dalam proses web mining adalah <strong>pengumpulan data mentah</strong> dari sumber yang relevan. Dalam praktikum ini, data dikumpulkan dari portal berita <strong>Detik.com</strong> &mdash; salah satu situs berita daring terbesar di Indonesia &mdash; menggunakan teknik <em>web crawling</em> secara otomatis.
      </p>
      <p>
        Proses crawling memanfaatkan <strong>Selenium WebDriver</strong> untuk mengendalikan browser secara programatis dan <strong>BeautifulSoup</strong> untuk mem-parsing struktur HTML halaman berita. Secara otomatis, program mengunjungi halaman-halaman berita pada dua kategori yang dipilih &mdash; <em>Sport</em> dan <em>Finance</em> &mdash; lalu mengekstraksi isi teks lengkap setiap artikel.
      </p>
      <p>
        Hasilnya berupa <strong>200 artikel berita mentah</strong> (100 dari kategori Sport dan 100 dari kategori Finance) yang tersimpan dalam format tabular dengan tiga kolom: nomor identifikasi (ID), isi lengkap berita, dan label kategori. Dataset inilah yang menjadi bahan baku untuk seluruh tahapan analisis selanjutnya.
      </p>
    </div>

    <div class="download-card">
      <div class="download-info">
        <strong>Dataset Crawling:</strong> 200 baris &times; 3 kolom (id, isi_berita, label)
      </div>
      <div class="download-actions">
        <a href="01_crawling/data_berita_detik.xlsx" class="btn-download" download>&#128229; Download .xlsx</a>
        <a href="01_crawling/data_berita_detik.csv" class="btn-download btn-download-alt" download>&#128196; Download .csv</a>
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
{table_crawl}
        </tbody>
      </table>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("crawling")}
</body>
</html>
'''

with open("crawling.html", "w", encoding="utf-8") as f:
    f.write(crawling_html)
print(f"  -> crawling.html ({len(crawling_html.splitlines())} baris)")


# ==============================================================================
# 3. GENERATE tfidf.html — Halaman TF-IDF Training & Testing + Narasi
# ==============================================================================
print("  Generating tfidf.html...")

df_tfidf_train = pd.read_csv("03_tfidf/tfidf_training.csv")
df_tfidf_test = pd.read_csv("03_tfidf/tfidf_testing.csv")

train_thead, train_tbody = buat_matrix_preview(df_tfidf_train, n_preview_cols=10)
test_thead, test_tbody = buat_matrix_preview(df_tfidf_test, n_preview_cols=10)

n_vocab = len(df_tfidf_train.columns) - 2  # minus ID and Label

tfidf_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pembobotan TF-IDF — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="page-header">
    <h1>Pembobotan TF-IDF</h1>
    <p class="subtitle">Transformasi teks berita menjadi representasi numerik berbasis frekuensi kata</p>
  </div>

  <main class="content">

    <div class="narasi">
      <span class="narasi-label">Tentang Proses Ini</span>
      <p>
        Setelah data berita mentah terkumpul, langkah selanjutnya adalah mengubah teks menjadi <strong>representasi numerik</strong> agar dapat diproses oleh algoritma <em>machine learning</em>. Mesin komputer tidak memahami kata-kata &mdash; ia hanya memahami angka. Di sinilah peran metode <strong>TF-IDF</strong> (<em>Term Frequency&ndash;Inverse Document Frequency</em>).
      </p>
      <p>
        Sebelum menghitung bobot TF-IDF, seluruh teks berita melewati tahapan <em>preprocessing</em> yang mencakup: (1) <strong>case folding</strong> &mdash; mengubah semua huruf menjadi huruf kecil, (2) <strong>tokenisasi</strong> &mdash; memecah kalimat menjadi kata-kata individual, (3) <strong>stopword removal</strong> &mdash; menghapus kata-kata umum yang tidak informatif seperti &ldquo;yang&rdquo;, &ldquo;dan&rdquo;, &ldquo;di&rdquo;, serta (4) <strong>stemming</strong> &mdash; mereduksi kata ke bentuk dasarnya.
      </p>
      <p>
        Dari 200 dokumen berita, diperoleh <strong>{n_vocab:,} kata unik</strong> (<em>vocabulary</em>) yang membentuk ruang fitur. Setiap dokumen kemudian direpresentasikan sebagai vektor dengan {n_vocab:,} dimensi, di mana setiap elemen berisi bobot TF-IDF dari kata tersebut. Bobot ini mencerminkan seberapa penting suatu kata dalam sebuah dokumen relatif terhadap keseluruhan <em>corpus</em>.
      </p>
      <p>
        Dataset dibagi menjadi dua bagian: <strong>160 dokumen untuk data training</strong> (80%) dan <strong>40 dokumen untuk data testing</strong> (20%). Pembagian ini memastikan model belajar dari sebagian besar data dan dievaluasi pada data yang belum pernah dilihat sebelumnya.
      </p>
    </div>

    <div class="data-section">
      <div class="data-tabs">
        <button class="data-tab active" onclick="switchTab(this, 'tfidf-training')">Data Training (160 dokumen)</button>
        <button class="data-tab" onclick="switchTab(this, 'tfidf-testing')">Data Testing (40 dokumen)</button>
      </div>

      <div class="tab-panel" id="tfidf-training">
        <div class="download-card">
          <div class="download-info">
            <strong>TF-IDF Training:</strong> 160 baris &times; {len(df_tfidf_train.columns):,} kolom ({n_vocab:,} fitur kata unik)
          </div>
          <div class="download-actions">
            <a href="03_tfidf/tfidf_training.xlsx" class="btn-download" download>&#128229; Download .xlsx</a>
            <a href="03_tfidf/tfidf_training.csv" class="btn-download btn-download-alt" download>&#128196; Download .csv</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{train_thead}</thead>
            <tbody>
{train_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">
          * Menampilkan 10 kata representatif pertama dari {n_vocab:,} kata unik. Dataset lengkap tersedia melalui tombol download di atas.
        </p>
      </div>

      <div class="tab-panel" id="tfidf-testing" style="display:none;">
        <div class="download-card">
          <div class="download-info">
            <strong>TF-IDF Testing:</strong> 40 baris &times; {len(df_tfidf_test.columns):,} kolom ({n_vocab:,} fitur kata unik)
          </div>
          <div class="download-actions">
            <a href="03_tfidf/tfidf_testing.xlsx" class="btn-download" download>&#128229; Download .xlsx</a>
            <a href="03_tfidf/tfidf_testing.csv" class="btn-download btn-download-alt" download>&#128196; Download .csv</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{test_thead}</thead>
            <tbody>
{test_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">
          * Menampilkan 10 kata representatif pertama dari {n_vocab:,} kata unik. Dataset lengkap tersedia melalui tombol download di atas.
        </p>
      </div>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("tfidf")}
{TAB_SCRIPT}
</body>
</html>
'''

with open("tfidf.html", "w", encoding="utf-8") as f:
    f.write(tfidf_html)
print(f"  -> tfidf.html ({len(tfidf_html.splitlines())} baris)")


# ==============================================================================
# 4. GENERATE pca.html — Halaman PCA Training & Testing + Narasi
# ==============================================================================
print("  Generating pca.html...")

df_pca_train = pd.read_csv("04_reduksi_dimensi/tfidf_pca_training.csv")
df_pca_test = pd.read_csv("04_reduksi_dimensi/tfidf_pca_testing.csv")

n_pc = len(df_pca_train.columns) - 2  # minus ID and Label

pca_train_thead, pca_train_tbody = buat_matrix_preview(df_pca_train, n_preview_cols=10)
pca_test_thead, pca_test_tbody = buat_matrix_preview(df_pca_test, n_preview_cols=10)

pca_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Reduksi Dimensi PCA — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="page-header">
    <h1>Reduksi Dimensi PCA</h1>
    <p class="subtitle">Menyederhanakan ribuan fitur kata menjadi komponen utama tanpa kehilangan informasi kunci</p>
  </div>

  <main class="content">

    <div class="narasi">
      <span class="narasi-label">Tentang Proses Ini</span>
      <p>
        Dengan {n_vocab:,} dimensi fitur, data TF-IDF memiliki ukuran yang sangat besar &mdash; sebuah kondisi yang dikenal sebagai <em>curse of dimensionality</em>. Jumlah fitur yang jauh melebihi jumlah data training dapat menyebabkan <em>overfitting</em> dan memperlambat proses komputasi secara signifikan. Untuk mengatasi hal ini, digunakan metode <strong>Principal Component Analysis (PCA)</strong> untuk mereduksi dimensi data.
      </p>
      <p>
        PCA bekerja dengan mentransformasi fitur-fitur asli menjadi sekumpulan fitur baru yang disebut <strong>Principal Components</strong> (PC). Setiap PC merupakan kombinasi linear dari seluruh fitur asli, disusun sedemikian rupa sehingga PC pertama menangkap variansi data terbesar, PC kedua menangkap variansi terbesar berikutnya yang tegak lurus terhadap PC pertama, dan seterusnya.
      </p>
      <p>
        Terdapat <strong>batasan matematis</strong> yang penting: jumlah maksimum komponen PCA yang dapat dihasilkan adalah min(N, D), di mana N adalah jumlah data training dan D adalah jumlah fitur. Karena data training berjumlah 160 dokumen, maka PCA menghasilkan maksimal <strong>{n_pc} Principal Components</strong> &mdash; mereduksi dimensi secara drastis dari {n_vocab:,} menjadi {n_pc}.
      </p>
      <p>
        PCA di-<em>fitting</em> pada data training terlebih dahulu, kemudian transformasi yang sama diterapkan pada data testing untuk menjaga konsistensi representasi antar kedua set data.
      </p>
    </div>

    <div class="data-section">
      <div class="data-tabs">
        <button class="data-tab active" onclick="switchTab(this, 'pca-training')">Data Training ({len(df_pca_train)} dokumen)</button>
        <button class="data-tab" onclick="switchTab(this, 'pca-testing')">Data Testing ({len(df_pca_test)} dokumen)</button>
      </div>

      <div class="tab-panel" id="pca-training">
        <div class="download-card">
          <div class="download-info">
            <strong>PCA Training:</strong> {len(df_pca_train)} baris &times; {len(df_pca_train.columns)} kolom ({n_pc} Principal Components)
          </div>
          <div class="download-actions">
            <a href="04_reduksi_dimensi/tfidf_pca_training.xlsx" class="btn-download" download>&#128229; Download .xlsx</a>
            <a href="04_reduksi_dimensi/tfidf_pca_training.csv" class="btn-download btn-download-alt" download>&#128196; Download .csv</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{pca_train_thead}</thead>
            <tbody>
{pca_train_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">
          * Menampilkan 10 dari {n_pc} Principal Components. Dataset lengkap tersedia melalui tombol download di atas.
        </p>
      </div>

      <div class="tab-panel" id="pca-testing" style="display:none;">
        <div class="download-card">
          <div class="download-info">
            <strong>PCA Testing:</strong> {len(df_pca_test)} baris &times; {len(df_pca_test.columns)} kolom ({n_pc} Principal Components)
          </div>
          <div class="download-actions">
            <a href="04_reduksi_dimensi/tfidf_pca_testing.xlsx" class="btn-download" download>&#128229; Download .xlsx</a>
            <a href="04_reduksi_dimensi/tfidf_pca_testing.csv" class="btn-download btn-download-alt" download>&#128196; Download .csv</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{pca_test_thead}</thead>
            <tbody>
{pca_test_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">
          * Menampilkan 10 dari {n_pc} Principal Components. Dataset lengkap tersedia melalui tombol download di atas.
        </p>
      </div>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("pca")}
{TAB_SCRIPT}
</body>
</html>
'''

with open("pca.html", "w", encoding="utf-8") as f:
    f.write(pca_html)
print(f"  -> pca.html ({len(pca_html.splitlines())} baris)")


# ==============================================================================
# 5. GENERATE eksperimen.html — Halaman Hasil Eksperimen Klasifikasi
# ==============================================================================
print("  Generating eksperimen.html...")

eksperimen_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Hasil Eksperimen Klasifikasi — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="page-header">
    <h1>Eksperimen Klasifikasi</h1>
    <p class="subtitle">Perbandingan akurasi kNN dan Naive Bayes pada berbagai tingkat reduksi fitur</p>
  </div>

  <main class="content">

    <div class="narasi">
      <span class="narasi-label">Tentang Eksperimen Ini</span>
      <p>
        Eksperimen ini bertujuan untuk menguji apakah <strong>reduksi dimensi memengaruhi akurasi klasifikasi</strong>, dan bagaimana dua algoritma klasifikasi yang berbeda merespons perubahan jumlah fitur. Dua algoritma yang digunakan adalah <strong>k-Nearest Neighbors (kNN)</strong> dan <strong>Naive Bayes</strong> &mdash; masing-masing memiliki cara kerja yang fundamental berbeda.
      </p>
      <p>
        <strong>kNN</strong> mengklasifikasikan data berdasarkan kedekatan jarak (<em>distance-based</em>), di mana sebuah dokumen dikelompokkan sesuai dengan label mayoritas dari <em>k</em> tetangga terdekatnya di ruang fitur. <strong>Naive Bayes</strong>, sebaliknya, menggunakan pendekatan probabilistik berdasarkan Teorema Bayes, dengan asumsi bahwa setiap fitur bersifat independen satu sama lain (<em>feature independence</em>).
      </p>
      <p>
        Eksperimen dilakukan dalam dua skenario. <strong>Pertama</strong>, reduksi dimensi menggunakan PCA secara bertahap (150, 100, 50, 20, dan 10 komponen). <strong>Kedua</strong>, seleksi fitur menggunakan metode <strong>Chi-Square (&chi;&sup2;)</strong> yang memilih kata-kata paling diskriminatif dari 6.000 hingga 500 kata teratas. Metrik evaluasi yang digunakan adalah <em>Classification Accuracy</em> (CA).
      </p>
    </div>

    <h2>Skenario 1: Reduksi Dimensi dengan PCA</h2>
    <p>
      Tabel berikut menampilkan perbandingan performa akurasi klasifikasi kNN dan Naive Bayes yang diuji langsung pada perangkat lunak <strong>Orange Data Mining</strong> (skenario: <em>Test on test data</em> dengan 40 dokumen data uji), pada berbagai tingkatan komponen PCA:
    </p>

    <div class="experiment-table-wrap">
      <table class="experiment-table">
        <thead>
          <tr>
            <th>Metode</th>
            <th>Jumlah Komponen</th>
            <th>Informasi Variansi (%)</th>
            <th>Akurasi kNN (%)</th>
            <th>Akurasi Naive Bayes (%)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Raw TF-IDF (Tanpa Reduksi)</td>
            <td>7.424</td>
            <td>100,00</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>PCA 150 Komponen</td>
            <td>150</td>
            <td>98,73</td>
            <td>95,0</td>
            <td>65,0</td>
          </tr>
          <tr>
            <td>PCA 100 Komponen</td>
            <td>100</td>
            <td>80,48</td>
            <td>90,0</td>
            <td>95,0</td>
          </tr>
          <tr class="highlight-row">
            <td>PCA 50 Komponen 🌟</td>
            <td>50</td>
            <td>52,34</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">97,5</td>
          </tr>
          <tr>
            <td>PCA 20 Komponen</td>
            <td>20</td>
            <td>29,42</td>
            <td>95,0</td>
            <td>90,0</td>
          </tr>
          <tr>
            <td>PCA 10 Komponen</td>
            <td>10</td>
            <td>18,66</td>
            <td>97,5</td>
            <td>92,5</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="experiment-caption">Tabel 1. Perbandingan Classification Accuracy (CA) kNN vs Naive Bayes pada berbagai jumlah komponen PCA</p>

    <h3>Rincian Lengkap Metrik Evaluasi Orange Data Mining</h3>
    <p>
      Berikut adalah rincian metrik evaluasi lengkap (AUC, CA, F1-Score, Precision, Recall, dan MCC) yang diekstrak langsung dari widget <em>Test and Score</em> Orange Data Mining untuk setiap tingkat reduksi PCA:
    </p>

    <div class="experiment-table-wrap">
      <table class="experiment-table">
        <thead>
          <tr>
            <th>Eksperimen</th>
            <th>Model</th>
            <th>AUC</th>
            <th>CA (Akurasi)</th>
            <th>F1-Score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>MCC</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">PCA 10 Komponen</td>
            <td>kNN</td>
            <td>1,000</td>
            <td class="highlight-best">0,975 (97,5%)</td>
            <td>0,975</td>
            <td>0,976</td>
            <td>0,975</td>
            <td>0,951</td>
          </tr>
          <tr>
            <td>Naive Bayes</td>
            <td>0,990</td>
            <td>0,925 (92,5%)</td>
            <td>0,925</td>
            <td>0,935</td>
            <td>0,925</td>
            <td>0,860</td>
          </tr>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">PCA 20 Komponen</td>
            <td>kNN</td>
            <td>0,996</td>
            <td>0,950 (95,0%)</td>
            <td>0,950</td>
            <td>0,955</td>
            <td>0,950</td>
            <td>0,905</td>
          </tr>
          <tr>
            <td>Naive Bayes</td>
            <td>0,990</td>
            <td>0,900 (90,0%)</td>
            <td>0,899</td>
            <td>0,917</td>
            <td>0,900</td>
            <td>0,816</td>
          </tr>
          <tr class="highlight-row">
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">PCA 50 Komponen 🌟</td>
            <td><strong>kNN</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">0,975 (97,5%)</td>
            <td class="highlight-best">0,975</td>
            <td class="highlight-best">0,976</td>
            <td class="highlight-best">0,975</td>
            <td class="highlight-best">0,951</td>
          </tr>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">PCA 100 Komponen</td>
            <td>kNN</td>
            <td>0,996</td>
            <td>0,900 (90,0%)</td>
            <td>0,899</td>
            <td>0,917</td>
            <td>0,900</td>
            <td>0,816</td>
          </tr>
          <tr>
            <td>Naive Bayes</td>
            <td>0,995</td>
            <td>0,950 (95,0%)</td>
            <td>0,950</td>
            <td>0,955</td>
            <td>0,950</td>
            <td>0,905</td>
          </tr>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">PCA 150 Komponen</td>
            <td>kNN</td>
            <td>0,990</td>
            <td>0,950 (95,0%)</td>
            <td>0,950</td>
            <td>0,955</td>
            <td>0,950</td>
            <td>0,905</td>
          </tr>
          <tr>
            <td>Naive Bayes</td>
            <td>0,955</td>
            <td>0,650 (65,0%)</td>
            <td>0,601</td>
            <td>0,794</td>
            <td>0,650</td>
            <td>0,420</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="experiment-caption">Tabel 2. Evaluasi metrik lengkap Orange Data Mining pada data testing (Target: average over classes)</p>

    <h2>Skenario 2: Seleksi Fitur dengan Chi-Square</h2>
    <p>
      Berbeda dengan PCA yang mentransformasi fitur, metode Chi-Square (&chi;&sup2;) bekerja dengan <strong>memilih kata-kata yang paling berkorelasi dengan label kelas</strong>. Kata-kata yang memiliki skor Chi-Square tertinggi dianggap paling diskriminatif untuk membedakan kategori Sport dan Finance.
    </p>

    <div class="experiment-table-wrap">
      <table class="experiment-table">
        <thead>
          <tr>
            <th>Tahap</th>
            <th>Jumlah Kata</th>
            <th>Akurasi kNN (%)</th>
            <th>Akurasi Naive Bayes (%)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Data Asli (7.424 kata)</td>
            <td>7.424</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>Chi-Square Top-6000</td>
            <td>6.000</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>Chi-Square Top-4000</td>
            <td>4.000</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>Chi-Square Top-2000</td>
            <td>2.000</td>
            <td class="highlight-best">100,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>Chi-Square Top-1000</td>
            <td>1.000</td>
            <td>77,5</td>
            <td class="highlight-best">100,0</td>
          </tr>
          <tr>
            <td>Chi-Square Top-500</td>
            <td>500</td>
            <td>90,0</td>
            <td class="highlight-best">100,0</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="experiment-caption">Tabel 3. Akurasi klasifikasi pada berbagai tingkat seleksi fitur Chi-Square</p>

    <h2>Analisis Hasil</h2>

    <div class="narasi">
      <span class="narasi-label">Temuan Utama</span>
      <p>
        Pada <strong>data asli tanpa reduksi</strong> (7.424 fitur), baik kNN maupun Naive Bayes sama-sama mencapai akurasi sempurna 100%. Ini menunjukkan bahwa kedua kategori berita (Sport dan Finance) memiliki perbedaan kosakata yang sangat jelas dan mudah dipisahkan.
      </p>
      <p>
        Pada <strong>eksperimen PCA</strong>, kedua algoritma mencapai titik performa optimal (<em>sweet spot</em>) pada <strong>50 komponen PCA</strong>. Pada titik ini, <strong>kNN mencapai akurasi sempurna 100,0%</strong> (AUC: 1.000, F1: 1.000, MCC: 1.000) dan <strong>Naive Bayes mencapai akurasi tertingginya yaitu 97,5%</strong> (AUC: 1.000, F1: 0.975). Fakta bahwa 50 komponen (mewakili 52,34% variansi kumulatif) menghasilkan performa lebih baik daripada 150 komponen membuktikan bahwa reduksi dimensi efektif memangkas komponen berbobot variansi kecil yang cenderung menjadi <em>noise</em>.
      </p>
      <p>
        <strong>k-Nearest Neighbors (kNN)</strong> terbukti sangat stabil di seluruh level reduksi PCA, dengan rentang akurasi tinggi antara <strong>90,0% hingga 100,0%</strong>. Bahkan pada reduksi paling agresif (10 komponen yang hanya menyisakan 18,66% variansi data), kNN tetap mencatat akurasi sangat tinggi sebesar <strong>97,5%</strong> dengan AUC sempurna 1.000. Ini membuktikan bahwa jarak antar titik dokumen di ruang berdimensi rendah tetap mempertahankan separabilitas kelas yang sangat baik bagi kNN.
      </p>
      <p>
        <strong>Naive Bayes</strong> menunjukkan karakteristik yang sangat menarik: pada <strong>150 komponen</strong>, akurasinya turun signifikan ke <strong>65,0%</strong> (F1: 0.601, MCC: 0.420). Hal ini mengindikasikan bahwa terlalu banyak komponen minor dengan variansi sangat kecil mengganggu estimasi distribusi probabilitas fitur pada Naive Bayes. Namun, saat jumlah komponen disederhanakan ke rentang 10–100 komponen, akurasi Naive Bayes melonjak stabil di kisaran <strong>90,0% – 97,5%</strong>, seiring dengan asumsi independensi fitur (<em>feature independence</em>) yang lebih terpenuhi pada komponen-komponen utama yang orthogonal.
      </p>
      <p>
        Pada <strong>eksperimen Chi-Square</strong>, Naive Bayes mempertahankan akurasi sempurna 100% di semua level seleksi fitur &mdash; menunjukkan robustness yang luar biasa. Sementara itu, kNN mengalami penurunan pada 1.000 kata (77,5%) namun kembali membaik pada 500 kata (90,0%). Pola ini mengindikasikan bahwa pada <em>threshold</em> 1.000 kata, terdapat fitur-fitur yang justru mengaburkan jarak antar kelas bagi kNN.
      </p>
    </div>

    <div class="download-card">
      <div class="download-info">
        <strong>Notebook Eksperimen:</strong> Seluruh kode eksperimen lengkap (Jupyter Notebook)
      </div>
      <div class="download-actions">
        <a href="05_eksperimen/EksperimenPCA-Klasifikasi.ipynb" class="btn-download" download>&#128229; Download Notebook (.ipynb)</a>
      </div>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("eksperimen")}
</body>
</html>
'''

with open("eksperimen.html", "w", encoding="utf-8") as f:
    f.write(eksperimen_html)
print(f"  -> eksperimen.html ({len(eksperimen_html.splitlines())} baris)")


# ==============================================================================
# DONE
# ==============================================================================
elapsed = time.time() - t0
print(f"\nSelesai! 5 halaman website berhasil dibuat dalam {elapsed:.2f} detik.")
