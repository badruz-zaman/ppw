# Aplikasi Klasifikasi Berita (Sport atau Finance)

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import trafilatura
from gensim.models import Word2Vec

# ──────────────────────────────────────────────
# 1. Konfigurasi Halaman
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Berita — Skip-Gram & ML",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# 2. Custom CSS (Rapi, Bersih, & Tanpa Merusak Ikon)
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Typography & Container ── */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 950px !important;
    }

    /* ── Header Bersih & Statis ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        line-height: 1.5;
    }

    /* ── Kotak Informasi Artikel ── */
    .article-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .article-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.4rem;
    }
    .article-meta {
        font-size: 0.85rem;
        color: #64748b;
    }
    .article-meta a {
        color: #818cf8;
        text-decoration: none;
    }
    .article-meta a:hover {
        text-decoration: underline;
    }

    /* ── Card Hasil Prediksi — Sport ── */
    .card-sport {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 2px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .badge-sport {
        background: #10b981;
        color: #ffffff;
        padding: 0.5rem 1.8rem;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.6rem;
        display: inline-block;
        letter-spacing: 1px;
    }

    /* ── Card Hasil Prediksi — Finance ── */
    .card-finance {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(29, 78, 216, 0.08) 100%);
        border: 2px solid #3b82f6;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .badge-finance {
        background: #3b82f6;
        color: #ffffff;
        padding: 0.5rem 1.8rem;
        border-radius: 30px;
        font-weight: 800;
        font-size: 1.6rem;
        display: inline-block;
        letter-spacing: 1px;
    }

    .result-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-bottom: 0.8rem;
    }
    .result-desc {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-top: 1rem;
    }

    /* ── Tombol Klasifikasi ── */
    .stButton > button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 0.5rem 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 3. Stopwords / Kata Tidak Baku
# ──────────────────────────────────────────────
KATA_TIDAK_BAKU = {
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

# ──────────────────────────────────────────────
# 4. Preprocessing & Vektor Dokumen
# ──────────────────────────────────────────────
def preprocess_teks(teks, varian="v1"):
    """Tokenisasi teks: v1 menyertakan angka, v2 hanya huruf."""
    pattern = r'\b\w+\b' if varian == "v1" else r'\b[a-zA-Z]+\b'
    tokens = re.findall(pattern, str(teks))
    return [k for k in tokens if k.lower() not in KATA_TIDAK_BAKU]

def hitung_vektor_dokumen(list_token, model_sg, dim=100):
    """Mean-pooling vektor Word2Vec dari dokumen."""
    vektor_valid = [model_sg.wv[kata] for kata in list_token if kata in model_sg.wv]
    if vektor_valid:
        return np.mean(vektor_valid, axis=0)
    return np.zeros(dim)

# ──────────────────────────────────────────────
# 5. Load Model (Cache)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_all_models():
    models = {}
    for ver in ("v1", "v2"):
        sg_path  = os.path.join(MODELS_DIR, f"skipgram_{ver}.model")
        knn_path = os.path.join(MODELS_DIR, f"knn_{ver}.joblib")
        nb_path  = os.path.join(MODELS_DIR, f"nb_{ver}.joblib")
        
        if os.path.exists(sg_path) and os.path.exists(knn_path) and os.path.exists(nb_path):
            models[f"sg_{ver}"]  = Word2Vec.load(sg_path)
            models[f"knn_{ver}"] = joblib.load(knn_path)
            models[f"nb_{ver}"]  = joblib.load(nb_path)
    return models

# ──────────────────────────────────────────────
# 6. Scraping dengan Trafilatura
# ──────────────────────────────────────────────
def ambil_berita(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return None, "Gagal mengunduh halaman. Pastikan link aktif dan valid."
        
        teks = trafilatura.extract(downloaded)
        if not teks or len(teks.strip()) < 50:
            return None, "Teks artikel tidak ditemukan atau terlalu pendek."
        
        meta = trafilatura.extract_metadata(downloaded)
        judul = meta.title if (meta and meta.title) else "Judul Artikel Berita"
        tanggal = meta.date if (meta and meta.date) else None
        
        return {"judul": judul, "tanggal": tanggal, "isi": teks, "url": url}, None
    except Exception as e:
        return None, f"Terjadi kesalahan saat scraping: {e}"

# ──────────────────────────────────────────────
# 7. Sidebar (Pengaturan Model)
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    
    # Versi 1 di atas, Versi 2 di bawah
    pilihan_varian = st.radio(
        "Varian Skip-Gram:",
        [
            "Versi 1 — Dengan Angka",
            "Versi 2 — Tanpa Angka"
        ],
        index=0
    )
    kode_varian = "v1" if "Versi 1" in pilihan_varian else "v2"
    
    # Algoritma klasifikasi pakai radio button
    pilihan_algoritma = st.radio(
        "Algoritma Klasifikasi:",
        [
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes"
        ],
        index=0
    )
    kode_algo = "knn" if "knn" in pilihan_algoritma.lower() else "nb"

# ──────────────────────────────────────────────
# 8. Halaman Utama
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">Klasifikasi Berita</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">'
    'Klasifikasi kategori berita (<strong>Sport</strong> atau <strong>Finance</strong>) '
    'langsung dari link artikel berita menggunakan Skip-Gram dan Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

models = load_all_models()

# ──────────────────────────────────────────────
# 9. Input Link Berita
# ──────────────────────────────────────────────
url_input = st.text_input(
    "Masukkan Link URL Berita:",
    placeholder="https://sport.detik.com/... atau https://finance.detik.com/..."
)

submit_url = st.button("🚀 Klasifikasikan Berita", type="primary")

data_input_siap = None

if submit_url:
    if not url_input.strip():
        st.warning("⚠️ Silakan masukkan URL berita terlebih dahulu.")
    else:
        with st.spinner("Mengambil konten berita dari link..."):
            artikel, err = ambil_berita(url_input.strip())
            if err:
                st.error(f"❌ {err}")
            else:
                data_input_siap = artikel

# ──────────────────────────────────────────────
# 10. Pemrosesan & Tampilan Hasil
# ──────────────────────────────────────────────
if data_input_siap is not None:
    st.markdown("---")
    
    # ── Info Artikel yang Dibaca ──
    st.subheader("📰 Artikel yang Diproses")
    
    meta_bar = []
    if data_input_siap.get("tanggal"):
        meta_bar.append(f"📅 {data_input_siap['tanggal']}")
    meta_bar.append(f"🔗 [Buka Sumber Asli]({data_input_siap['url']})")
    
    st.markdown(f"""
    <div class="article-box">
        <div class="article-title">{data_input_siap['judul']}</div>
        <div class="article-meta">{' &nbsp;•&nbsp; '.join(meta_bar)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Lihat Isi Berita Lengkap"):
        st.write(data_input_siap['isi'])
    
    # ── Proses Prediksi ──
    with st.spinner("Memproses klasifikasi..."):
        tokens = preprocess_teks(data_input_siap['isi'], varian=kode_varian)
        
        key_sg = f"sg_{kode_varian}"
        key_clf = f"{kode_algo}_{kode_varian}"
        
        if key_sg not in models or key_clf not in models:
            st.error(f"❌ Model `{pilihan_varian}` belum ditemukan di folder `models/`.")
            st.stop()
        
        model_sg = models[key_sg]
        model_clf = models[key_clf]
        
        vektor_dok = hitung_vektor_dokumen(tokens, model_sg)
        prediksi = model_clf.predict([vektor_dok])[0]
        
    
    # ── Hasil Klasifikasi Utama ──
    st.markdown("---")
    st.subheader("🎯 Hasil Prediksi")
    
    label = str(prediksi).strip().lower()
    
    if label == "sport":
        st.markdown("""
        <div class="card-sport">
            <div class="result-label">Kategori Berita</div>
            <div style="margin: 0.8rem 0;"><span class="badge-sport">⚽ SPORT</span></div>
            <div class="result-desc">Berita ini tergolong ke dalam kategori <strong>Sport</strong>.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card-finance">
            <div class="result-label">Kategori Berita</div>
            <div style="margin: 0.8rem 0;"><span class="badge-finance">💰 FINANCE</span></div>
            <div class="result-desc">Berita ini tergolong ke dalam kategori <strong>Finance</strong>.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── HASIL EVALUASI MODEL YANG DIPILIH (ORANGE DATA MINING) ──
    st.markdown("---")
    st.subheader("📊 Hasil Evaluasi Model")

    # Metrik sesuai kombinasi varian + algoritma yang dipilih user
    metrik_orange = {
        ("v1", "knn"): {"AUC": "0.989", "CA": "0.925", "F1": "0.925", "Precision": "0.935", "Recall": "0.925", "MCC": "0.860"},
        ("v1", "nb"):  {"AUC": "1.000", "CA": "0.975", "F1": "0.975", "Precision": "0.976", "Recall": "0.975", "MCC": "0.951"},
        ("v2", "knn"): {"AUC": "1.000", "CA": "1.000", "F1": "1.000", "Precision": "1.000", "Recall": "1.000", "MCC": "1.000"},
        ("v2", "nb"):  {"AUC": "1.000", "CA": "0.975", "F1": "0.975", "Precision": "0.976", "Recall": "0.975", "MCC": "0.951"},
    }

    m = metrik_orange[(kode_varian, kode_algo)]

    st.caption(f"Pengujian *Test on test data* — 160 Data Latih (80%) / 40 Data Uji (20%) — **{pilihan_algoritma}** • **{pilihan_varian}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("AUC", m["AUC"])
    with col2:
        st.metric("CA (Akurasi)", m["CA"])
    with col3:
        st.metric("F1-Score", m["F1"])
    with col4:
        st.metric("Precision", m["Precision"])
    with col5:
        st.metric("Recall", m["Recall"])
    with col6:
        st.metric("MCC", m["MCC"])