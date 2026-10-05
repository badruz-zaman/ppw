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
        ("skipgram.html",   "Skip-Gram",  "\U0001F9E0", "skipgram"),
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


# HELPER: Generate preview table rows for Skip-Gram continuous vectors
def buat_skipgram_matrix_preview(df, n_preview_cols=10):
    """Generate thead + tbody HTML for Skip-Gram continuous feature columns + ellipsis + label."""
    cols = df.columns.tolist()
    feature_cols = [c for c in cols if c != 'label']
    preview_cols = feature_cols[:n_preview_cols]
    remaining = len(feature_cols) - n_preview_cols

    th_parts = ['<th class="col-matrix-id">ID</th>']
    for c in preview_cols:
        th_parts.append(f'<th>{html.escape(str(c))}</th>')
    if remaining > 0:
        th_parts.append(f'<th class="col-ellipsis">... [{remaining:,} dimensi lainnya] ...</th>')
    th_parts.append('<th class="col-matrix-label">Label</th>')
    thead = "<tr>" + "".join(th_parts) + "</tr>"

    rows = []
    for idx, row_data in enumerate(df.itertuples(index=False), start=1):
        cells = [f'<td class="col-matrix-id">{idx}</td>']
        for val in row_data[:n_preview_cols]:
            cells.append(f'<td>{val:.4f}</td>')
        if remaining > 0:
            cells.append('<td class="col-ellipsis">...</td>')
        label_val = str(row_data[-1])
        badge_cls = f"badge badge-{label_val.lower()}"
        cells.append(f'<td class="col-matrix-label"><span class="{badge_cls}">{html.escape(label_val)}</span></td>')
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
            <td>80,0</td>
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
    <p class="experiment-caption">Tabel 3. Perbandingan Classification Accuracy (CA) kNN vs Naive Bayes pada berbagai tingkat seleksi fitur Chi-Square</p>

    <h3>Rincian Lengkap Metrik Evaluasi Seleksi Fitur Chi-Square (Orange Data Mining)</h3>
    <p>
      Berikut adalah rincian metrik evaluasi lengkap (AUC, CA, F1-Score, Precision, Recall, dan MCC) dari widget <em>Test and Score</em> Orange Data Mining untuk setiap tingkat seleksi fitur Chi-Square:
    </p>

    <div class="experiment-table-wrap">
      <table class="experiment-table">
        <thead>
          <tr>
            <th>Tahap Seleksi Fitur</th>
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
          <tr class="highlight-row">
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Chi-Square Top-6000</td>
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>kNN</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Chi-Square Top-4000</td>
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>kNN</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Chi-Square Top-2000</td>
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>kNN</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Chi-Square Top-1000</td>
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr>
            <td>kNN</td>
            <td>0,955</td>
            <td>0,800 (80,0%)</td>
            <td>0,795</td>
            <td>0,830</td>
            <td>0,800</td>
            <td>0,629</td>
          </tr>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Chi-Square Top-500</td>
            <td><strong>Naive Bayes</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
          </tr>
          <tr>
            <td>kNN</td>
            <td>0,945</td>
            <td>0,900 (90,0%)</td>
            <td>0,899</td>
            <td>0,917</td>
            <td>0,900</td>
            <td>0,816</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="experiment-caption">Tabel 4. Rincian metrik evaluasi Orange Data Mining pada seleksi fitur Chi-Square (Target: average over classes)</p>

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
        Pada <strong>eksperimen Chi-Square</strong>, <strong>Naive Bayes mempertahankan akurasi sempurna 100,0%</strong> di semua level seleksi fitur (6.000 hingga 500 kata) &mdash; menunjukkan <em>robustness</em> luar biasa karena estimasi probabilitas kata kunci spesifik sudah cukup kuat mengunci keputusan kelas. Sementara itu, kNN mengalami penurunan pada 1.000 kata (80,0%) namun kembali membaik pada 500 kata (90,0%).
      </p>
      <p>
        <strong>Mengapa Hasil pada 2.000, 4.000, dan 6.000 Kata Identik (100% Sempurna)?</strong><br>
        Fenomena ini terjadi karena <strong>separabilitas kelas berita Sport dan Finance sangat tinggi</strong>. Kata-kata pembeda paling kuat (seperti istilah olahraga spesifik dan istilah perbankan/pasar modal) sudah terkumpul lengkap di dalam Top-2.000 kata. Ketika jumlah fitur ditingkatkan ke 4.000 dan 6.000 kata, fitur-fitur baru yang ditambahkan memiliki frekuensi dokumen yang sangat rendah atau berbobot TF-IDF kecil, sehingga tidak mampu mengubah probabilitas (Naive Bayes) maupun membalikkan tetangga terdekat (kNN) pada 40 dokumen data uji.
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
# 6. GENERATE skipgram.html — Halaman Word2Vec Skip-Gram + Evaluasi Orange
# ==============================================================================
print("  Generating skipgram.html...")

df_sg_v1_train = pd.read_csv("06_skipgram/skipgram_v1_training.csv")
df_sg_v1_test  = pd.read_csv("06_skipgram/skipgram_v1_testing.csv")
df_sg_v2_train = pd.read_csv("06_skipgram/skipgram_v2_training.csv")
df_sg_v2_test  = pd.read_csv("06_skipgram/skipgram_v2_testing.csv")

sg_v1_train_thead, sg_v1_train_tbody = buat_skipgram_matrix_preview(df_sg_v1_train, n_preview_cols=10)
sg_v1_test_thead,  sg_v1_test_tbody  = buat_skipgram_matrix_preview(df_sg_v1_test,  n_preview_cols=10)
sg_v2_train_thead, sg_v2_train_tbody = buat_skipgram_matrix_preview(df_sg_v2_train, n_preview_cols=10)
sg_v2_test_thead,  sg_v2_test_tbody  = buat_skipgram_matrix_preview(df_sg_v2_test,  n_preview_cols=10)

skipgram_html = f'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Representasi Word2Vec Skip-Gram — Badruz Zaman</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>

  <div class="page-header">
    <h1>Representasi Dokumen dengan Word2Vec Skip-Gram</h1>
    <p class="subtitle">Transformasi fitur teks berita ke ruang vektor kontinu 100 dimensi menggunakan arsitektur neural Skip-Gram dan evaluasi performa klasifikasi di Orange Data Mining</p>
  </div>

  <main class="content">

    <div class="narasi">
      <span class="narasi-label">Konsep &amp; Teori Dasar Word Embedding</span>
      <p>
        Dalam pemrosesan bahasa alami (<em>Natural Language Processing</em>), metode berbasis frekuensi kata tradisional seperti <strong>TF-IDF</strong> (<em>Term Frequency&ndash;Inverse Document Frequency</em>) merepresentasikan dokumen dalam bentuk matriks yang sangat besar dan bersifat <strong>sparse</strong> (kebanyakan elemen bernilai 0). Pada dataset proyek ini, representasi TF-IDF menghasilkan ruang fitur sebesar <strong>7.424 dimensi</strong>. Kelemahan mendasar dari model representasi frekuensi adalah sifatnya yang <em>orthogonal</em> &mdash; model tidak memiliki pemahaman semantik bahwa kata &ldquo;saham&rdquo; memiliki keterkaitan erat dengan &ldquo;dividen&rdquo;, atau kata &ldquo;atlet&rdquo; berkaitan erat dengan &ldquo;pertandingan&rdquo;.
      </p>
      <p>
        Untuk mengatasi keterbatasan tersebut, diperkenalkan paradigma <strong>Word Embedding</strong> melalui algoritma <strong>Word2Vec</strong> (Mikolov et al., 2013). Word2Vec didasarkan pada <em>Distributional Hypothesis</em> (J.R. Firth, 1957) yang menyatakan bahwa <em>&ldquo;a word is characterized by the company it keeps&rdquo;</em> &mdash; kata-kata yang muncul dalam konteks kalimat yang serupa cenderung memiliki makna semantik yang mirip pula.
      </p>
      <p>
        Word2Vec memetakan setiap kata ke dalam ruang vektor laten kontinu berdimensi rendah (<em>d = 100 &ll; 7.424</em>). Berbeda dengan TF-IDF yang menghasilkan matriks jarang, vektor Word2Vec bersifat <strong>padat (dense)</strong> dan bernilai riil kontinu, di mana kedekatan makna semantik antar kata direfleksikan secara matematis melalui nilai <em>cosine similarity</em> atau jarak Euclidean.
      </p>
    </div>

    <h2>Arsitektur Model: Skip-Gram</h2>
    <p>
      Word2Vec menyediakan dua varian arsitektur neural network utama: <strong>Continuous Bag-of-Words (CBOW)</strong> dan <strong>Skip-Gram</strong>. Pada penelitian/praktikum ini, arsitektur yang digunakan adalah <strong>Skip-Gram</strong>:
    </p>

    <div class="taxonomy-item">
      <h4>Mekanisme Prediksi Terbalik (Target &rarr; Konteks)</h4>
      <p>
        Jika CBOW memprediksi satu kata target dari sekumpulan kata-kata konteks di sekelilingnya, maka arsitektur <strong>Skip-Gram membalik mekanisme tersebut</strong>: model menerima satu kata target <em>w<sub>t</sub></em> pada lapisan input (<em>input layer</em>), kemudian memprediksi probabilitas kemunculan kata-kata konteks <em>w<sub>t+j</sub></em> di sekitarnya dalam radius jendela tertentu (<em>context window</em>).
      </p>
    </div>

    <div class="taxonomy-item">
      <h4>Ketangguhan terhadap Kosakata Jarang (Rare Words)</h4>
      <p>
        Skip-Gram sangat unggul untuk dataset berukuran kecil hingga menengah karena setiap pasangan kata target dan kata konteks diperlakukan sebagai satu sampel pelatihan tersendiri. Hal ini memberikan kesempatan yang seimbang bagi kata-kata yang frekuensinya relatif jarang (<em>rare words</em>) untuk memperbarui bobot vektor representasinya secara optimal tanpa tertutupi oleh kata-kata yang sangat sering muncul.
      </p>
    </div>

    <div class="taxonomy-item">
      <h4>Fungsi Objektif &amp; Optimasi Matematis</h4>
      <p>
        Tujuan pelatihan Skip-Gram adalah memaksimalkan rata-rata log-probabilitas berikut di seluruh korpus teks dengan panjang <em>T</em> kata dan ukuran jendela konteks <em>c</em>:
        <br /><br />
        <span style="display:block; text-align:center; font-family:'Inter',sans-serif; font-size:0.95rem; font-weight:600; padding:10px 0; background:#fdfcf9; border:1px solid #e0ddd8; border-radius:4px;">
          L(&theta;) = (1 / T) &times; &sum;<sub>t=1</sub><sup>T</sup> &sum;<sub>-c &le; j &le; c, j &ne; 0</sub> log P(w<sub>t+j</sub> | w<sub>t</sub>)
        </span>
        <br />
        Probabilitas kondisional dasar dihitung menggunakan fungsi <strong>Softmax</strong>:
        <br /><br />
        <span style="display:block; text-align:center; font-family:'Inter',sans-serif; font-size:0.92rem; font-weight:600; padding:10px 0; background:#fdfcf9; border:1px solid #e0ddd8; border-radius:4px;">
          P(w<sub>O</sub> | w<sub>I</sub>) = exp(v'&lsquo;<sub>wO</sub><sup>T</sup> &middot; v<sub>wI</sub>) / &sum;<sub>w=1</sub><sup>W</sup> exp(v'&lsquo;<sub>w</sub><sup>T</sup> &middot; v<sub>wI</sub>)
        </span>
        <br />
        Untuk mempercepat komputasi dari penyebut Softmax yang melibatkan seluruh kosakata <em>W</em>, pustaka <strong>Gensim</strong> mengimplementasikan optimasi <strong>Negative Sampling (NEG)</strong>, di mana model hanya memperbarui bobot untuk pasangan konteks aktual ditambah sejumlah kecil kata acak sebagai sampel negatif.
      </p>
    </div>

    <h2>Konfigurasi Hyperparameter Pelatihan</h2>
    <p>Model Skip-Gram dilatih menggunakan modul <code>gensim.models.Word2Vec</code> dengan parameter arsitektur sebagai berikut:</p>
    <ol>
      <li><strong>vector_size = 100:</strong> Menetapkan dimensi ruang laten kontinu sebanyak 100 dimensi. Ukuran ini ideal untuk menangkap semantik topik tanpa membebani kompleksitas komputasi.</li>
      <li><strong>window = 5:</strong> Jendela konteks sebesar 5 kata ke kiri dan 5 kata ke kanan dari kata target (total jangkauan rentang konteks maksimum adalah 10 kata).</li>
      <li><strong>min_count = 1:</strong> Ambang batas frekuensi kata minimal bernilai 1, memastikan seluruh kosakata yang ada pada 200 dokumen berita tetap dipertahankan tanpa ada kata yang dieliminasi.</li>
      <li><strong>sg = 1:</strong> Menandakan pemilihan arsitektur Skip-Gram (jika <code>sg = 0</code> adalah arsitektur CBOW).</li>
      <li><strong>epochs = 30:</strong> Jumlah siklus iterasi penuh (epoch) pelatihan model neural network pada seluruh korpus teks.</li>
      <li><strong>seed = 42 &amp; workers = 1:</strong> Mengunci pengacakan inisialisasi bobot agar proses pelatihan sepenuhnya deterministik dan dapat direproduksi secara konsisten (<em>fully reproducible</em>).</li>
    </ol>

    <h2>Siklus Proses: Dari Teks Mentah ke Evaluasi Klasifikasi</h2>
    <p>Transformasi artikel berita mentah menjadi representasi vektor hingga evaluasi performa klasifikasi dilakukan melalui 6 tahapan komputasi yang terstruktur:</p>
    <ol>
      <li>
        <strong>Tahap 1 &mdash; Preprocessing Teks (Dua Skenario Komparatif):</strong>
        <br />Teks mentah dibersihkan dari seluruh tanda baca (<em>punctuation</em>) dan dilakukan normalisasi terhadap sekitar 80 kata tidak baku / kata gaul bahasa Indonesia (misal: <em>yg &rarr; yang, gak &rarr; tidak, bgt &rarr; sangat</em>) serta pembersihan spasi berlebih. Pada tahap ini, eksperimen dibagi menjadi dua skenario:
        <ul style="margin: 8px 0 8px 24px; font-size: 0.98rem; line-height: 1.7; color: #444;">
          <li><strong>Versi 1 (Dengan Angka):</strong> Karakter angka (0&ndash;9) dipertahankan di dalam teks kalimat. Format huruf besar/kecil (kapital) tidak diubah (tanpa <em>case folding</em>).</li>
          <li><strong>Versi 2 (Tanpa Angka):</strong> Seluruh karakter angka (0&ndash;9) dihapus secara total dari korpus menggunakan ekspresi reguler. Huruf kapital tetap dipertahankan.</li>
        </ul>
      </li>
      <li><strong>Tahap 2 &mdash; Tokenisasi Dokumen:</strong> Setiap artikel berita dipecah menjadi deretan token kata (<em>list of words</em>) yang menyusun korpus kalimat.</li>
      <li>
        <strong>Tahap 3 &mdash; Pembagian Dataset (Split Data 80:20 Terstratifikasi):</strong>
        <br />Sebelum melatih model Skip-Gram, dataset 200 dokumen dibagi menjadi <strong>80% data latih (160 dokumen)</strong> dan <strong>20% data uji (40 dokumen)</strong> menggunakan fungsi <code>train_test_split</code> dengan parameter <code>train_size=0.8</code>, <code>test_size=0.2</code>, <code>random_state=42</code>, dan <code>stratify=label</code>. Pembagian terstratifikasi ini memastikan proporsi kelas seimbang sempurna (80 Sport &amp; 80 Finance pada data latih; 20 Sport &amp; 20 Finance pada data uji).
      </li>
      <li>
        <strong>Tahap 4 &mdash; Pelatihan Word2Vec Skip-Gram (Hanya pada Data Latih):</strong>
        <br />Untuk mencegah kebocoran data (<em>data leakage</em>), model neural Skip-Gram dilatih secara ketat <strong>hanya pada 160 dokumen data latih</strong>. 40 dokumen data uji tidak pernah dilihat atau dipelajari oleh model selama proses pembentukan ruang vektor kata.
      </li>
      <li>
        <strong>Tahap 5 &mdash; Document Embedding via Mean Pooling:</strong>
        <br />Karena Word2Vec menghasilkan representasi untuk masing-masing kata, representasi tingkat dokumen dibentuk menggunakan teknik <strong>Mean Pooling (Perataan Vektor)</strong>:
        <br /><br />
        <span style="display:block; text-align:center; font-family:'Inter',sans-serif; font-size:0.92rem; font-weight:600; padding:10px 0; background:#fdfcf9; border:1px solid #e0ddd8; border-radius:4px;">
          d&#8407;<sub>k</sub> = (1 / N<sub>k</sub>) &times; &sum;<sub>i=1</sub><sup>N<sub>k</sub></sup> v&#8407;(w<sub>i</sub>)
        </span>
        <br />
        Di mana <em>d&#8407;<sub>k</sub> &isin; &#8477;<sup>100</sup></em> adalah vektor akhir dokumen ke-<em>k</em>, <em>N<sub>k</sub></em> adalah total kata dalam dokumen tersebut, dan <em>v&#8407;(w<sub>i</sub>)</em> adalah vektor kata dari model Skip-Gram. Setiap dimensi dari dokumen merupakan nilai rata-rata dari dimensi yang bersesuaian pada seluruh kata penyusunnya. Fungsi ini diterapkan secara terpisah untuk menghasilkan matriks data latih (160 baris &times; 100 dimensi) dan matriks data uji (40 baris &times; 100 dimensi).
      </li>
      <li>
        <strong>Tahap 6 &mdash; Ekspor CSV &amp; Klasifikasi di Orange Data Mining:</strong>
        <br />Matriks fitur digabungkan dengan kolom target <code>label</code> dan diekspor ke file CSV terpisah (<code>skipgram_v1_training.csv</code>, <code>skipgram_v1_testing.csv</code>, <code>skipgram_v2_training.csv</code>, dan <code>skipgram_v2_testing.csv</code>). Selanjutnya dilakukan evaluasi klasifikasi menggunakan <strong>Orange Data Mining</strong> dengan skenario <strong>Test on test data</strong> untuk algoritma <strong>Naive Bayes</strong> dan <strong>k-Nearest Neighbors (kNN)</strong>.
      </li>
    </ol>

    <h2>Dataset Matriks Vektor Dokumen (Skip-Gram 100 Dimensi)</h2>
    <p>
      Berikut adalah pratinjau data representasi vektor dokumen hasil pemodelan Skip-Gram untuk kedua versi preprocessing, terbagi menjadi Data Training (160 dokumen) dan Data Testing (40 dokumen). Setiap dokumen diwakili oleh 100 dimensi numerik kontinu beserta target label kategorinya:
    </p>

    <div class="data-section">
      <div class="data-tabs">
        <button class="data-tab active" onclick="switchTab(this, 'sg-v1-train')">Versi 1: Training (160 dok)</button>
        <button class="data-tab" onclick="switchTab(this, 'sg-v1-test')">Versi 1: Testing (40 dok)</button>
        <button class="data-tab" onclick="switchTab(this, 'sg-v2-train')">Versi 2: Training (160 dok)</button>
        <button class="data-tab" onclick="switchTab(this, 'sg-v2-test')">Versi 2: Testing (40 dok)</button>
      </div>

      <div class="tab-panel" id="sg-v1-train">
        <div class="download-card">
          <div class="download-info">
            <strong>Skip-Gram Versi 1 (Training):</strong> 160 baris &times; 101 kolom (100 dimensi fitur + target label, mempertahankan angka)
          </div>
          <div class="download-actions">
            <a href="06_skipgram/skipgram_v1_training.csv" class="btn-download" download>&#128196; Download Training .csv</a>
            <a href="06_skipgram/skipgram_versi_1.py" class="btn-download btn-download-alt" download>&#128013; Script Python (V1)</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{sg_v1_train_thead}</thead>
            <tbody>
{sg_v1_train_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">* Menampilkan 160 baris data latih dengan 10 dimensi awal (dim_1 &ndash; dim_10), 90 dimensi lainnya diwakili ellipsis (...), dan target label.</p>
      </div>

      <div class="tab-panel" id="sg-v1-test" style="display: none;">
        <div class="download-card">
          <div class="download-info">
            <strong>Skip-Gram Versi 1 (Testing):</strong> 40 baris &times; 101 kolom (100 dimensi fitur + target label, mempertahankan angka)
          </div>
          <div class="download-actions">
            <a href="06_skipgram/skipgram_v1_testing.csv" class="btn-download" download>&#128196; Download Testing .csv</a>
            <a href="06_skipgram/skipgram_versi_1.py" class="btn-download btn-download-alt" download>&#128013; Script Python (V1)</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{sg_v1_test_thead}</thead>
            <tbody>
{sg_v1_test_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">* Menampilkan 40 baris data uji dengan 10 dimensi awal (dim_1 &ndash; dim_10), 90 dimensi lainnya diwakili ellipsis (...), dan target label.</p>
      </div>

      <div class="tab-panel" id="sg-v2-train" style="display: none;">
        <div class="download-card">
          <div class="download-info">
            <strong>Skip-Gram Versi 2 (Training):</strong> 160 baris &times; 101 kolom (100 dimensi fitur + target label, angka dihapus)
          </div>
          <div class="download-actions">
            <a href="06_skipgram/skipgram_v2_training.csv" class="btn-download" download>&#128196; Download Training .csv</a>
            <a href="06_skipgram/skipgram_versi_2.py" class="btn-download btn-download-alt" download>&#128013; Script Python (V2)</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{sg_v2_train_thead}</thead>
            <tbody>
{sg_v2_train_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">* Menampilkan 160 baris data latih dengan 10 dimensi awal (dim_1 &ndash; dim_10), 90 dimensi lainnya diwakili ellipsis (...), dan target label.</p>
      </div>

      <div class="tab-panel" id="sg-v2-test" style="display: none;">
        <div class="download-card">
          <div class="download-info">
            <strong>Skip-Gram Versi 2 (Testing):</strong> 40 baris &times; 101 kolom (100 dimensi fitur + target label, angka dihapus)
          </div>
          <div class="download-actions">
            <a href="06_skipgram/skipgram_v2_testing.csv" class="btn-download" download>&#128196; Download Testing .csv</a>
            <a href="06_skipgram/skipgram_versi_2.py" class="btn-download btn-download-alt" download>&#128013; Script Python (V2)</a>
          </div>
        </div>
        <div class="table-container">
          <table class="data-table matrix-table">
            <thead>{sg_v2_test_thead}</thead>
            <tbody>
{sg_v2_test_tbody}
            </tbody>
          </table>
        </div>
        <p class="preview-note">* Menampilkan 40 baris data uji dengan 10 dimensi awal (dim_1 &ndash; dim_10), 90 dimensi lainnya diwakili ellipsis (...), dan target label.</p>
      </div>
    </div>

    <h2>Hasil Eksperimen Klasifikasi pada Orange Data Mining</h2>
    <p>
      Evaluasi performa klasifikasi dilakukan menggunakan perangkat lunak <strong>Orange Data Mining</strong> dengan widget <em>Test and Score</em>. Pengujian menerapkan skenario <strong>Test on test data</strong> secara ketat &mdash; di mana model dilatih menggunakan <strong>160 dokumen data latih (80%)</strong> dan diuji performanya secara independen pada <strong>40 dokumen data uji (20%)</strong> tanpa adanya kebocoran data (<em>no data leakage</em>).
    </p>
    <p>
      Tabel berikut menampilkan perbandingan lengkap metrik evaluasi antara <strong>Versi 1 (Dengan Angka)</strong> dan <strong>Versi 2 (Tanpa Angka)</strong> untuk algoritma <strong>Naive Bayes</strong> dan <strong>k-Nearest Neighbors (kNN)</strong>:
    </p>

    <div class="experiment-table-wrap">
      <table class="experiment-table">
        <thead>
          <tr>
            <th>Skenario Preprocessing</th>
            <th>Model Klasifikasi</th>
            <th>AUC</th>
            <th>CA (Akurasi)</th>
            <th>F1-Score</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>MCC</th>
            <th>Hasil Prediksi Benar</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Versi 1<br>(Dengan Angka)</td>
            <td><strong>Naive Bayes</strong></td>
            <td>1,000</td>
            <td>0,975 (97,5%)</td>
            <td>0,975</td>
            <td>0,976</td>
            <td>0,975</td>
            <td>0,951</td>
            <td>39 / 40 data uji benar</td>
          </tr>
          <tr>
            <td>k-Nearest Neighbors (kNN)</td>
            <td>0,989</td>
            <td>0,925 (92,5%)</td>
            <td>0,925</td>
            <td>0,935</td>
            <td>0,925</td>
            <td>0,860</td>
            <td>37 / 40 data uji benar</td>
          </tr>
          <tr class="highlight-row">
            <td rowspan="2" style="vertical-align: middle; font-weight: 600;">Versi 2<br>(Tanpa Angka)</td>
            <td><strong>k-Nearest Neighbors (kNN) 🌟</strong></td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000 (100,0%)</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">1,000</td>
            <td class="highlight-best">40 / 40 data uji benar (100% Sempurna)</td>
          </tr>
          <tr class="highlight-row">
            <td><strong>Naive Bayes</strong></td>
            <td>1,000</td>
            <td>0,975 (97,5%)</td>
            <td>0,975</td>
            <td>0,976</td>
            <td>0,975</td>
            <td>0,951</td>
            <td>39 / 40 data uji benar</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="experiment-caption">Tabel 1. Perbandingan Metrik Evaluasi Klasifikasi Word2Vec Skip-Gram pada Orange Data Mining (Skenario: Test on test data, 40 Data Uji Independen)</p>

    <h2>Analisis &amp; Temuan Ilmiah</h2>

    <div class="narasi">
      <span class="narasi-label">Analisis 1 &mdash; Lonjakan Performa kNN Menjadi 100% Sempurna pada Versi 2</span>
      <p>
        Hasil pengujian empiris memperlihatkan peningkatan performa yang sangat luar biasa pada algoritma <strong>k-Nearest Neighbors (kNN)</strong>, di mana akurasinya melonjak dari <strong>92,5% pada Versi 1 menjadi 100,0% (1,000) pada Versi 2</strong>. Pada Versi 2, seluruh <strong>40 dari 40 dokumen data uji berhasil diprediksi dengan benar tanpa ada satupun kesalahan</strong> (skor AUC, CA, F1, Precision, Recall, dan MCC seluruhnya bernilai sempurna 1,000).
      </p>
      <p>
        Secara teoretis, kNN adalah model klasifikasi berbasis jarak spasial (<em>instance-based learning</em> yang menggunakan metrik <em>Euclidean distance</em>). Pada <strong>Versi 1 (Dengan Angka)</strong>, karakter angka seperti tanggal (&ldquo;2024&rdquo;), skor (&ldquo;3-0&rdquo;), waktu menit (&ldquo;90&rdquo;), maupun nominal nilai uang muncul secara acak di kedua kategori berita (Sport maupun Finance). Kehadiran vektor angka tersebut bertindak sebagai <strong>derau (noise) spasial</strong> yang menggeser posisi koordinat beberapa dokumen di ruang laten 100 dimensi, sehingga menyebabkan 3 dokumen data uji salah dikelompokkan ke tetangga kelas yang keliru.
      </p>
      <p>
        Ketika seluruh angka dibuang pada <strong>Versi 2</strong>, ruang representasi 100 dimensi murni dibentuk oleh kata-kata semantik topik inti (seperti <em>dividen, laba, bursa, saham</em> untuk Finance vs <em>gol, pemain, pelatih, juara</em> untuk Sport). Akibatnya, pemisahan antar-kluster dokumen menjadi sangat tajam dan kompak (<em>well-separated clusters</em>), memastikan kelima tetangga terdekat (k=5) selalu berasal dari kategori yang tepat.
      </p>
    </div>

    <div class="narasi">
      <span class="narasi-label">Analisis 2 &mdash; Konsistensi Tinggi Algoritma Naive Bayes (Akurasi 97,5%)</span>
      <p>
        Berbeda dengan kNN yang sangat dipengaruhi oleh posisi spasial lokal, model <strong>Naive Bayes mencatatkan akurasi yang sangat konsisten yaitu 97,5% pada kedua versi</strong> (dengan skor AUC sempurna <strong>1,000</strong>). Dari 40 dokumen data uji, Naive Bayes berhasil mengklasifikasikan 39 dokumen secara tepat dan hanya melakukan 1 kesalahan prediksi.
      </p>
      <p>
        Naive Bayes bekerja dengan pendekatan probabilistik berdasarkan Teorema Bayes yang mengasumsikan distribusi normal (Gaussian) pada setiap fitur kontinu. Sifat agregatif dari probabilitas bersyarat membuat Naive Bayes memiliki ketahanan (<em>robustness</em>) yang sangat tinggi terhadap keberadaan kata-kata angka, sehingga performanya tetap solid dan stabil di angka 97,5% baik pada Versi 1 maupun Versi 2.
      </p>
    </div>

    <div class="narasi">
      <span class="narasi-label">Analisis 3 &mdash; Validitas Metodologis: Pengujian Bebas Kebocoran Data (Zero Data Leakage)</span>
      <p>
        Salah satu keunggulan metodologis utama pada eksperimen ini adalah penerapan pemisahan data 80:20 <strong>sebelum tahap pelatihan Word2Vec Skip-Gram dilakukan</strong>. Model Skip-Gram dilatih hanya pada 160 dokumen data latih.
      </p>
      <p>
        Dengan pendekatan ini, dokumen data uji benar-benar berperan sebagai data unseen (belum pernah dilihat sama sekali oleh model neural Skip-Gram). Keberhasilan kNN mencapai akurasi 100% dan Naive Bayes mencapai 97,5% pada mode <em>Test on test data</em> membuktikan bahwa representasi semantik yang dipelajari Skip-Gram memiliki daya generalisasi yang luar biasa tinggi terhadap artikel berita baru.
      </p>
    </div>

    <div class="narasi">
      <span class="narasi-label">Analisis 4 &mdash; Efisiensi Representasi Skip-Gram (Dense 100D) vs TF-IDF (Sparse 7.424D) &amp; PCA</span>
      <p>
        Jika dikomparasikan dengan metode pembobotan frekuensi tradisional:
      </p>
      <p>
        Representasi <strong>TF-IDF</strong> menghasilkan matriks jarang (<em>sparse</em>) berdimensi sangat raksasa yaitu <strong>7.424 dimensi</strong>, yang memerlukan memori besar dan rentan terhadap fenomena <em>curse of dimensionality</em>. Sementara itu, teknik reduksi dimensi <strong>PCA</strong> memerlukan proses dekomposisi matematis (SVD) tambahan setelah TF-IDF dibentuk.
      </p>
      <p>
        Sebaliknya, <strong>Word2Vec Skip-Gram dengan agregasi Mean Pooling</strong> secara langsung menghasilkan ruang representasi padat (<em>dense</em>) hanya dengan <strong>100 dimensi</strong> (74 kali lebih ringkas dibanding TF-IDF). Meskipun dimensinya jauh lebih ringkas, representasi Skip-Gram mampu menghasilkan akurasi sempurna hingga <strong>100,0%</strong> pada kNN dan <strong>97,5%</strong> pada Naive Bayes, membuktikan keunggulan superior representasi semantik berbasis <em>neural embedding</em>.
      </p>
    </div>

    <div class="download-card">
      <div class="download-info">
        <strong>Berkas Eksperimen Lengkap:</strong> Dataset vektor CSV (Training &amp; Testing) dan Script Python untuk kedua versi preprocessing
      </div>
      <div class="download-actions">
        <a href="06_skipgram/skipgram_v1_training.csv" class="btn-download" download>&#128196; V1 Training CSV (160)</a>
        <a href="06_skipgram/skipgram_v1_testing.csv" class="btn-download" download>&#128196; V1 Testing CSV (40)</a>
        <a href="06_skipgram/skipgram_v2_training.csv" class="btn-download" download>&#128196; V2 Training CSV (160)</a>
        <a href="06_skipgram/skipgram_v2_testing.csv" class="btn-download" download>&#128196; V2 Testing CSV (40)</a>
        <a href="06_skipgram/skipgram_versi_1.py" class="btn-download btn-download-alt" download>&#128013; Script Python V1</a>
        <a href="06_skipgram/skipgram_versi_2.py" class="btn-download btn-download-alt" download>&#128013; Script Python V2</a>
      </div>
    </div>

    <div class="references">
      <h3>Daftar Pustaka</h3>
      <ol>
        <li>Mikolov, T., Chen, K., Corrado, G., &amp; Dean, J. (2013). <em>Efficient Estimation of Word Representations in Vector Space</em>. arXiv preprint arXiv:1301.3781.</li>
        <li>Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., &amp; Dean, J. (2013). Distributed representations of words and phrases and their compositionality. <em>Advances in Neural Information Processing Systems (NeurIPS)</em>, 26, 3111&ndash;3119.</li>
        <li>Firth, J. R. (1957). A synopsis of linguistic theory, 1930-1955. <em>Studies in Linguistic Analysis</em>, 1&ndash;32.</li>
        <li>Rehurek, R., &amp; Sojka, P. (2010). Software framework for topic modelling with large corpora. In <em>Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks</em> (pp. 45&ndash;50).</li>
        <li>Dem&scaron;ar, J., Curk, T., Erjavec, A., Gorup, &Ccaron;., Ho&ccaron;evar, T., Milutinovi&ccaron;, M., ... &amp; Zupan, B. (2013). Orange: data mining toolbox in Python. <em>Journal of Machine Learning Research</em>, 14(1), 2349&ndash;2353.</li>
      </ol>
    </div>

  </main>

  <footer>&copy; 2026 Badruz Zaman &middot; 240411100140</footer>

{buat_fab_menu("skipgram")}
{TAB_SCRIPT}
</body>
</html>
'''

with open("skipgram.html", "w", encoding="utf-8") as f:
    f.write(skipgram_html)
print(f"  -> skipgram.html ({len(skipgram_html.splitlines())} baris)")


# ==============================================================================
# DONE
# ==============================================================================
elapsed = time.time() - t0
print(f"\nSelesai! 6 halaman website berhasil dibuat dalam {elapsed:.2f} detik.")
