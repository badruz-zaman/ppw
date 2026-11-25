"""Antarmuka Dashboard Streamlit untuk Analisis Sitasi Jurnal & PageRank (ScienceDirect / Elsevier)."""

from __future__ import annotations

import os
import sys

# Tambahkan direktori file saat ini ke sys.path agar impor modul selalu berhasil
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

from crawler_citation import (
    crawl_and_build_citation_graph,
    generate_citation_vis_html,
)


def _apply_custom_css() -> None:
    """Terapkan tema warm paper, font Consolas monospace, dan kontras tajam."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');

            html, body, [data-testid="stAppViewContainer"], .main {
                background-color: #faf9f7 !important;
                color: #111111 !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
            }

            /* ── Sembunyikan Header Bawaan Streamlit (Kotak Hitam Horizontal) ── */
            header[data-testid="stHeader"],
            [data-testid="stHeader"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }

            /* ── Sembunyikan Tombol Collapse Sidebar agar tidak muncul teks 'key' / 'keyb' ── */
            [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"],
            button[kind="header"] {
                display: none !important;
                visibility: hidden !important;
            }

            /* Lindungi font icon bawaan Streamlit agar ligatur icon tidak berubah menjadi teks */
            [class*="material-symbols"],
            [data-testid="stIconMaterial"] {
                font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            }

            [data-testid="stSidebar"] {
                background-color: #f3f0ea !important;
                border-right: 1px solid #e5e0d8 !important;
            }
            [data-testid="stSidebar"] * {
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #222222 !important;
            }

            .block-container {
                padding-top: 2.0rem !important;
                padding-bottom: 4.5rem !important;
                max-width: 980px !important;
                margin: 0 auto !important;
            }

            /* ── Hero / Page Header ── */
            .hero {
                text-align: center;
                padding: 10px 0 0;
            }
            .hero-kicker {
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.82rem !important;
                font-weight: 700 !important;
                text-transform: uppercase !important;
                letter-spacing: 2px !important;
                color: #777777 !important;
                margin-bottom: 0.5rem !important;
            }
            .hero h1 {
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 2.1rem !important;
                font-weight: 700 !important;
                line-height: 1.3 !important;
                color: #111111 !important;
                margin-bottom: 12px !important;
            }
            .hero-subtitle {
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.92rem !important;
                color: #555555 !important;
                line-height: 1.6 !important;
                max-width: 840px;
                margin: 0 auto 0.5rem auto !important;
            }

            .divider {
                width: 40px;
                height: 2px;
                background: #bbb;
                margin: 20px auto 24px;
            }

            /* ── Styling Input Teks ── */
            .stTextInput label,
            .stTextInput label p {
                color: #111111 !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
            }
            .stTextInput div[data-baseweb="input"],
            div[data-baseweb="input"] {
                background-color: #ffffff !important;
                background: #ffffff !important;
                border: 1.5px solid #c8c3ba !important;
                border-radius: 6px !important;
                height: 44px !important;
            }
            .stTextInput input,
            div[data-baseweb="input"] input {
                background-color: #ffffff !important;
                background: #ffffff !important;
                color: #111111 !important;
                caret-color: #111111 !important;
                cursor: text !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.95rem !important;
                font-weight: 600 !important;
                padding: 0 0.85rem !important;
            }

            /* ── Tombol Primer (#141425) ── */
            .stButton > button[kind="primary"] {
                font-family: 'Consolas', 'Courier New', monospace !important;
                background-color: #141425 !important;
                color: #ffffff !important;
                border: none !important;
                border-radius: 6px !important;
                font-weight: 700 !important;
                font-size: 0.92rem !important;
                height: 44px !important;
                padding: 0 1.4rem !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button[kind="primary"] * {
                color: #ffffff !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
            }
            .stButton > button[kind="primary"]:hover {
                background-color: #2a2a44 !important;
            }

            /* ── Tombol Download CSV ── */
            .stDownloadButton > button {
                font-family: 'Consolas', 'Courier New', monospace !important;
                background: #ffffff !important;
                color: #141425 !important;
                border: 1.5px solid #d5d0c8 !important;
                border-radius: 6px !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                padding: 0.55rem 1.3rem !important;
                margin: 0.6rem 0 !important;
            }
            .stDownloadButton > button:hover {
                border-color: #141425 !important;
                background-color: #f4f1eb !important;
            }

            /* ── Kotak Metrik Ringkas ── */
            .metric-pill {
                font-family: 'Consolas', 'Courier New', monospace !important;
                background: #ffffff;
                border: 1px solid #dcd8d0;
                border-radius: 6px;
                padding: 0.8rem 1.1rem;
                font-size: 0.85rem;
                color: #555555;
                text-align: center;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            }
            .metric-pill-val {
                font-size: 1.4rem;
                font-weight: 700;
                color: #111111;
                margin-top: 0.2rem;
            }

            /* ── Kotak Penjelasan Metodologi & Kontainer Border ── */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: #ffffff !important;
                border: 1px solid #e0ddd7 !important;
                border-radius: 6px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_methodology_tab(result: dict, alpha: float) -> None:
    """Tampilkan penjelasan bagaimana jaringan sitasi jurnal dibangun dan jejak konvergensi Power Iteration."""
    st.markdown("### Bagaimana Jaringan Sitasi Jurnal Dibangun?")
    with st.container(border=True):
        st.markdown(
            r"""
1. **Input Paper Utama (Seed Paper):** Pengguna memasukkan DOI (misal `10.1016/...`), tautan artikel ScienceDirect, atau kata kunci judul artikel ilmiah.
2. **Pengambilan Metadata Resmi:** Sistem menghubungi basis data ilmiah terbuka (OpenAlex & Crossref) untuk mendapatkan judul artikel, nama penulis, tahun rilis, wadah jurnal, dan seluruh daftar pustaka (*references*).
3. **Ekstraksi Hubungan Sitasi (Depth 1):** Seluruh artikel ilmiah yang disitasi langsung oleh paper utama diekstrak sebagai simpul (*nodes*) anak.
4. **Deteksi Sitasi Silang & Kedalaman Lanjutan (Depth 2):** Sistem memeriksa apakah sesama artikel referensi saling mengutip satu sama lain (*cross-citations*), serta menelusuri artikel cucu yang dikutip oleh artikel referensi tersebut hingga batas kuota.
5. **Pembentukan Graf Berarah Sitasi (DiGraph):** Jika Paper **A** mengutip Paper **B** (**A ≠ B**), maka ditambahkan sebuah relasi berarah **A → B** (artinya Paper A memberikan suara/bobot sitasi kepada Paper B).
6. **Perhitungan PageRank Jurnal:** Algoritma PageRank (*Power Iteration*) dijalankan pada graf sitasi hingga konvergen untuk mengukur prestise ilmiah dan menemukan artikel mana yang paling penting / paling banyak menjadi fondasi rujukan.
"""
        )

    st.markdown("### Rumus PageRank Jurnal yang Diterapkan")
    with st.container(border=True):
        st.latex(r"PR(i) = \frac{1 - d}{N} + d \sum_{(j,i) \in E} \frac{PR(j)}{D_{\text{out}}(j)}")

        g = result.get("graph")
        total_nodes = g.number_of_nodes() if g is not None else 0
        total_edges = g.number_of_edges() if g is not None else 0

        st.markdown(
            "**Keterangan Variabel & Komponen Rumus:**\n\n"
            f"- **$PR(i)$** : Skor PageRank (tingkat pengaruh/otoritas ilmiah) dari artikel ilmiah $i$.\n"
            f"- **$d$** : *Damping Factor* ($d = {alpha:.2f}$), probabilitas seorang akademisi terus menelusuri rantai sitasi dari artikel saat ini.\n"
            f"- **$1 - d$** : Probabilitas teleportasi acak ($1 - d = {1 - alpha:.2f}$), peluang akademisi mencari literatur baru di luar rantai sitasi saat ini.\n"
            f"- **$N$** : Total seluruh artikel ilmiah unik yang ditemukan dalam graf jaringan sitasi ($N = {total_nodes}$).\n"
            r"- **$\frac{1 - d}{N}$** : Bobot dasar minimum (*uniform teleportation*) yang dibagikan merata ke seluruh artikel agar tidak terjadi kebuntuan (*citation dead-end*)." + "\n"
            f"- **$E$** : Himpunan seluruh relasi sitasi berarah antar-artikel ilmiah ($|E| = {total_edges}$).\n"
            r"- **$D_{\text{out}}(j)$** : *Out-degree* (jumlah seluruh daftar pustaka / referensi keluar) yang dikutip oleh artikel $j$." + "\n"
            r"- **$\sum_{(j,i) \in E} \frac{PR(j)}{D_{\text{out}}(j)}$** : Akumulasi transfer bobot dari seluruh artikel $j$ yang menyitasi atau memberikan rujukan (*in-link citation*) menuju artikel $i$."
        )

    trace = result.get("power_trace", [])
    if trace:
        st.markdown("#### Jejak Konvergensi Power Iteration pada Graf Sitasi")
        st.dataframe(trace, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Analisis Sitasi Jurnal & PageRank (ScienceDirect)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_custom_css()

    with st.sidebar:
        st.markdown("### Setting")
        max_papers = st.slider(
            "Max Papers",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Jumlah maksimal artikel ilmiah yang dianalisis dalam jaringan sitasi.",
        )
        max_depth = st.slider(
            "Depth",
            min_value=1,
            max_value=2,
            value=2,
            help="Kedalaman penelusuran sitasi (Depth 1 = referensi langsung, Depth 2 = sitasi silang & cucu).",
        )
        alpha = 0.85

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">PROJECT UAS PPW — JURNAL</div>
            <h1>Analisis Sitasi Jurnal &amp; PageRank</h1>
            <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, btn_col = st.columns([4.2, 1.8], vertical_alignment="bottom")
    with input_col:
        target_paper = st.text_input(
            "Masukkan Link ScienceDirect / DOI / Judul Paper:",
            value="10.1016/j.compag.2018.02.016",
            placeholder="Misal: 10.1016/j.compag.2018.02.016 atau judul paper...",
        )
    with btn_col:
        run_clicked = st.button(
            "Ekstrak & Analisis Sitasi",
            type="primary",
            use_container_width=True,
        )

    if run_clicked:
        if not target_paper.strip():
            st.warning("Silakan masukkan DOI, link ScienceDirect, atau judul paper terlebih dahulu.")
        else:
            progress_bar = st.progress(0.0, text="Menghubungi indeks ilmiah ScienceDirect...")

            def _update_progress(done_cnt: int, total_cnt: int, status_text: str, n_nodes: int, n_edges: int) -> None:
                ratio = min(max(done_cnt / max(total_cnt, 1), 0.0), 1.0)
                progress_bar.progress(
                    ratio,
                    text=f"Progress [{done_cnt}/{total_cnt}]: {status_text} | Simpul: {n_nodes}, Sitasi: {n_edges}",
                )

            with st.spinner("Sedang mengekstrak daftar sitasi, membangun graf, dan menghitung PageRank..."):
                res = crawl_and_build_citation_graph(
                    seed_input=target_paper.strip(),
                    max_papers=max_papers,
                    max_depth=max_depth,
                    alpha=alpha,
                    on_progress=_update_progress,
                )

            progress_bar.empty()
            if res.get("error"):
                st.error(f"{res['error']}")
            else:
                st.session_state["last_citation_result"] = res

    result = st.session_state.get("last_citation_result")
    if not result:
        return

    DG = result["graph"]
    top10_df = result["top10_df"]
    ranking_df = result["ranking_df"]
    edges_df = result["edges_df"]
    seed_paper = result["seed_paper"]

    # 4 Kotak Metrik Ringkas
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="metric-pill">Total Paper (Simpul)<div class="metric-pill-val">{DG.number_of_nodes()}</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-pill">Sitasi (Sisi)<div class="metric-pill-val">{DG.number_of_edges()}</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-pill">Tahun Paper Awal<div class="metric-pill-val">{seed_paper.get('year', '-')}</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-pill">Waktu Eksekusi<div class="metric-pill-val">{result['elapsed_sec']}s</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("10 Paper Paling Berpengaruh (Peringkat PageRank Tertinggi)")
    st.dataframe(
        top10_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "PageRank": st.column_config.NumberColumn(format="%.6f"),
            "Persentase (%)": st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=float(max(top10_df["Persentase (%)"].max(), 1.0)),
                format="%.2f%%",
            ),
        },
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.subheader("Visualisasi Graf Berarah Jaringan Sitasi Antar-Paper")

    total_nodes = DG.number_of_nodes()
    if total_nodes <= 5:
        top_n_vis = total_nodes
    else:
        top_n_vis = st.slider(
            "Jumlah Simpul pada Visualisasi Graf Sitasi:",
            min_value=5,
            max_value=total_nodes,
            value=total_nodes,
            help=f"Geser untuk menyesuaikan jumlah paper yang ditampilkan (maksimal {total_nodes} simpul).",
        )

    vis_html = generate_citation_vis_html(
        DG,
        result["pagerank"],
        seed_id=result["seed_id"],
        top_n=top_n_vis,
        height=620,
    )
    components.html(vis_html, height=640)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    tab_all, tab_edges, tab_method = st.tabs([
        f"Semua Paper / Artikel ({len(ranking_df)})",
        f"Daftar Hubungan Sitasi ({len(edges_df)})",
        "Cara Kerja Sitasi & Rumus PageRank Jurnal",
    ])

    with tab_all:
        search_node = st.text_input(
            "Cari paper pada seluruh hasil:",
            placeholder="Ketik judul paper atau nama penulis...",
        )
        filtered_nodes = ranking_df
        if search_node.strip():
            q = search_node.strip()
            mask = (
                ranking_df["Judul Paper / Jurnal"].str.contains(q, case=False, na=False)
                | ranking_df["Penulis"].str.contains(q, case=False, na=False)
            )
            filtered_nodes = ranking_df[mask]
        st.dataframe(filtered_nodes, hide_index=True, use_container_width=True)
        st.download_button(
            label="Download Semua Peringkat Paper (CSV)",
            data=ranking_df.to_csv(index=False).encode("utf-8"),
            file_name="semua_peringkat_paper.csv",
            mime="text/csv",
        )

    with tab_edges:
        search_edge = st.text_input(
            "Filter relasi sitasi (Menyitasi atau Disitasi):",
            placeholder="Misal: deep learning atau nama penulis...",
        )
        filtered_edges = edges_df
        if search_edge.strip():
            q = search_edge.strip()
            mask = (
                edges_df["Paper yang Menyitasi (Source)"].str.contains(q, case=False, na=False)
                | edges_df["Paper yang Disitasi (Target)"].str.contains(q, case=False, na=False)
            )
            filtered_edges = edges_df[mask]
        st.dataframe(filtered_edges, hide_index=True, use_container_width=True)
        st.download_button(
            label="Download Daftar Relasi Sitasi (CSV)",
            data=edges_df.to_csv(index=False).encode("utf-8"),
            file_name="relasi_sitasi_jurnal.csv",
            mime="text/csv",
        )

    with tab_method:
        _render_methodology_tab(result, alpha=alpha)


if __name__ == "__main__":
    main()
