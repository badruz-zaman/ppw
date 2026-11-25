from __future__ import annotations

import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

import streamlit as st
import streamlit.components.v1 as components

from crawler_pagerank import (
    crawl_and_build_graph,
    generate_citationgraph_vis_html,
)


def _apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            /* ── Reset Seluruh Font ke Consolas & Monospace Murni ── */
            * {
                font-family: 'Consolas', 'Courier New', monospace !important;
            }

            html, body, [data-testid="stAppViewContainer"], .main {
                background-color: #faf9f7 !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #222222 !important;
                -webkit-font-smoothing: antialiased;
            }

            header[data-testid="stHeader"],
            [data-testid="stHeader"] {
                display: none !important;
                visibility: hidden !important;
                height: 0 !important;
            }

            /* Sembunyikan tombol collapse sidebar agar tidak muncul teks 'key' */
            [data-testid="stSidebarCollapseButton"],
            [data-testid="collapsedControl"] {
                display: none !important;
            }

            /* Lindungi font icon bawaan Streamlit agar ligatur icon tidak berubah menjadi teks */
            [class*="material-symbols"],
            [data-testid="stIconMaterial"] {
                font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
            }

            /* ── Sidebar Style (Warm Paper Tone & Consolas) ── */
            [data-testid="stSidebar"] {
                background-color: #f4f1eb !important;
                border-right: 1px solid #e5e0d8 !important;
            }
            [data-testid="stSidebar"] * {
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #222222 !important;
            }

            .block-container {
                padding-top: 3.2rem !important;
                padding-bottom: 4.5rem !important;
                max-width: 960px !important;
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
                font-size: 2.2rem !important;
                font-weight: 700 !important;
                line-height: 1.3 !important;
                color: #111111 !important;
                margin-bottom: 12px !important;
                letter-spacing: -0.01em !important;
            }
            .hero-subtitle {
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.94rem !important;
                color: #555555 !important;
                line-height: 1.6 !important;
                max-width: 820px;
                margin: 0 auto 0.5rem auto !important;
            }

            /* ── Divider Garis Tipis ── */
            .divider {
                width: 40px;
                height: 2px;
                background: #bbb;
                margin: 22px auto 26px;
            }

            /* ── Styling Input Teks URL (Kotak Putih, Teks Hitam Kontras Tinggi) ── */
            .stTextInput,
            .stTextInput > div,
            .stTextInput > div > div {
                background-color: transparent !important;
            }
            .stTextInput label,
            .stTextInput label p,
            div[data-testid="stTextInput"] label,
            div[data-testid="stTextInput"] label p {
                color: #111111 !important;
                font-weight: 700 !important;
                font-size: 0.95rem !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                margin-bottom: 0.4rem !important;
            }
            .stTextInput div[data-baseweb="input"],
            .stTextInput div[data-baseweb="base-input"],
            div[data-baseweb="input"],
            div[data-baseweb="base-input"] {
                background-color: #ffffff !important;
                background: #ffffff !important;
                border: 1.5px solid #c8c3ba !important;
                border-radius: 6px !important;
                box-shadow: none !important;
                height: 44px !important;
                min-height: 44px !important;
                max-height: 44px !important;
                box-sizing: border-box !important;
            }
            .stTextInput div[data-baseweb="input"]:focus-within,
            div[data-baseweb="input"]:focus-within {
                border-color: #141425 !important;
                box-shadow: 0 0 0 1px #141425 !important;
            }
            .stTextInput input,
            div[data-baseweb="input"] input,
            div[data-baseweb="base-input"] input {
                background-color: #ffffff !important;
                background: #ffffff !important;
                color: #111111 !important;
                -webkit-text-fill-color: #111111 !important;
                caret-color: #111111 !important;
                cursor: text !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.95rem !important;
                font-weight: 600 !important;
                padding: 0 0.85rem !important;
                height: 40px !important;
                line-height: 40px !important;
            }
            .stTextInput input::placeholder,
            div[data-baseweb="input"] input::placeholder {
                color: #777777 !important;
                -webkit-text-fill-color: #777777 !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
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
                letter-spacing: 0.5px !important;
                height: 44px !important;
                min-height: 44px !important;
                max-height: 44px !important;
                padding: 0 1.4rem !important;
                white-space: nowrap !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                transition: all 0.2s ease !important;
            }
            .stButton > button[kind="primary"] * {
                white-space: nowrap !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #ffffff !important;
            }
            .stButton > button[kind="primary"]:hover {
                background-color: #2a2a44 !important;
                color: #ffffff !important;
            }

            /* ── Tombol Download CSV (st.download_button) ── */
            .stDownloadButton,
            .stDownloadButton > button,
            [data-testid="stDownloadButton"] > button {
                font-family: 'Consolas', 'Courier New', monospace !important;
                background-color: #ffffff !important;
                background: #ffffff !important;
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
                border: 1.5px solid #d5d0c8 !important;
                border-radius: 6px !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
                padding: 0.55rem 1.3rem !important;
                margin: 0.6rem 0 !important;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
                transition: all 0.15s ease !important;
            }
            .stDownloadButton > button *,
            [data-testid="stDownloadButton"] > button * {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
                font-family: 'Consolas', 'Courier New', monospace !important;
                font-size: 0.88rem !important;
                font-weight: 600 !important;
            }
            .stDownloadButton > button:hover,
            [data-testid="stDownloadButton"] > button:hover {
                border-color: #141425 !important;
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
                background-color: #f4f1eb !important;
                background: #f4f1eb !important;
            }
            .stDownloadButton > button:hover *,
            [data-testid="stDownloadButton"] > button:hover * {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
            }

            /* ── Styling Tab Streamlit (st.tabs) ── */
            div[data-baseweb="tab-list"] {
                gap: 8px;
                border-bottom: 2px solid #e0ddd7 !important;
            }
            button[data-baseweb="tab"] {
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #555555 !important;
                -webkit-text-fill-color: #555555 !important;
                font-weight: 600 !important;
                font-size: 0.9rem !important;
                background: transparent !important;
                padding: 0.5rem 1rem !important;
            }
            button[data-baseweb="tab"] * {
                font-family: 'Consolas', 'Courier New', monospace !important;
                color: #555555 !important;
                -webkit-text-fill-color: #555555 !important;
                font-weight: 600 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
                border-bottom: 2px solid #141425 !important;
            }
            button[data-baseweb="tab"][aria-selected="true"] * {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
                font-weight: 700 !important;
            }
            button[data-baseweb="tab"]:hover {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
            }
            button[data-baseweb="tab"]:hover * {
                color: #141425 !important;
                -webkit-text-fill-color: #141425 !important;
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
                font-size: 1.45rem;
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
    """Tampilkan penjelasan bagaimana hubungan website dibangun dan jejak konvergensi Power Iteration."""
    st.markdown("### Bagaimana Membangun Hubungan Antar-Website?")
    with st.container(border=True):
        st.markdown(
            r"""
1. **Input Seed URL:** Pengguna memasukkan sembarang URL website publik awal (portal berita, website kampus, Wikipedia, blog, atau portal kementerian/organisasi).
2. **HTTP Request & Penanganan Redirect:** Sistem menghubungi website target via HTTP GET menggunakan User-Agent browser modern dan mencatat domain kanonikal bila terjadi pengalihan (*redirect*).
3. **Ekstraksi Hyperlink (`<a href>`):** Seluruh tag tautan di dalam dokumen HTML diekstrak dengan `BeautifulSoup`, dikonversi menjadi URL absolut (`urljoin`), dan dibersihkan dari fragment (`#`), tautan non-HTTP (`mailto:`, `tel:`, `javascript:`), serta file statis (gambar, PDF, video).
4. **Normalisasi Node Website:** Target tautan dipetakan menjadi nama simpul graf (berdasarkan *Host / Subdomain* atau *Root Domain*), sekaligus menyaring tautan noise (tombol share media sosial, tracker analitik, CDN) jika opsi filter aktif.
5. **Crawling Berulang (BFS Multi-Depth):** Sistem mengunjungi website-website tujuan yang ditemukan secara paralel hingga batas kedalaman (**Depth = 2**). Dari kunjungan berulang ini, tautan balik (*backlink*) dan tautan silang antar-website berhasil terekstrak sehingga graf keterhubungan benar-benar terjalin.
6. **Pembentukan Graf Berarah (DiGraph):** Jika website **A** memiliki tautan menuju website **B** (**A ≠ B**), maka ditambahkan sebuah sisi berarah **A → B**.
7. **Perhitungan PageRank:** Algoritma PageRank (*Power Iteration*) dijalankan pada graf berarah hingga konvergen untuk menghitung distribusi bobot otoritas seluruh simpul.
"""
        )

    st.markdown("### Rumus PageRank yang Diterapkan")
    with st.container(border=True):
        st.latex(r"PR(i) = \frac{1 - d}{N} + d \sum_{(j,i) \in E} \frac{PR(j)}{D_{\text{out}}(j)}")

        g = result.get("graph")
        total_nodes = g.number_of_nodes() if g is not None else 0
        total_edges = g.number_of_edges() if g is not None else 0

        st.markdown(
            "**Keterangan Variabel & Komponen Rumus:**\n\n"
            f"- **$PR(i)$** : Skor PageRank (bobot otoritas/kepentingan) dari website $i$.\n"
            f"- **$d$** : *Damping Factor* ($d = {alpha:.2f}$), probabilitas penelusur terus mengikuti tautan di halaman web.\n"
            f"- **$1 - d$** : Probabilitas teleportasi acak ($1 - d = {1 - alpha:.2f}$), peluang melompat acak ke sembarang website lain.\n"
            f"- **$N$** : Jumlah total website unik yang ditemukan dalam graf ($N = {total_nodes}$).\n"
            r"- **$\frac{1 - d}{N}$** : Bobot dasar minimum (*uniform teleportation*) yang dibagikan merata ke seluruh simpul agar surfer tidak terjebak di halaman buntu (*dead end*)." + "\n"
            f"- **$E$** : Himpunan seluruh relasi tautan berarah antar-website ($|E| = {total_edges}$).\n"
            r"- **$D_{\text{out}}(j)$** : *Out-degree* (jumlah seluruh tautan keluar) dari website $j$." + "\n"
            r"- **$\sum_{(j,i) \in E} \frac{PR(j)}{D_{\text{out}}(j)}$** : Akumulasi transfer bobot dari setiap website $j$ yang memberikan tautan masuk (*in-link*) menuju website $i$."
        )

    trace = result.get("power_trace", [])
    if trace:
        st.markdown("#### Jejak Konvergensi Power Iteration pada Graf Hasil Crawling")
        st.dataframe(trace, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="Web Crawling & Analisis PageRank Website",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _apply_custom_css()

    with st.sidebar:
        st.markdown("### Setting")
        max_pages = st.slider(
            "Max Pages",
            min_value=8,
            max_value=50,
            value=25,
            step=5,
            help="Jumlah maksimal website yang dikunjungi crawler untuk mengekstrak tautan keluar.",
        )
        max_depth = st.slider(
            "Depth",
            min_value=1,
            max_value=3,
            value=2,
            help="Kedalaman penelusuran tautan (1 atau 2). Depth = 2 sangat disarankan agar terbentuk tautan balik dan hubungan silang antar-website.",
        )
        alpha = 0.85

    # Hero / Header Bersih (Senada dengan PSD App)
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">PROJECT UAS PPW</div>
            <h1>Crawling &amp; PageRank Web</h1>
            <p class="hero-subtitle">
                Sistem penelusuran tautan otomatis yang mengekstrak hyperlink antar website secara iteratif.
            </p>
            <div class="divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    input_col, btn_col = st.columns([4.0, 1.8], vertical_alignment="bottom")
    with input_col:
        target_url = st.text_input(
            "Masukkan Link Web:",
            value="https://www.trunojoyo.ac.id",
            placeholder="https://www.trunojoyo.ac.id",
        )
    with btn_col:
        run_clicked = st.button(
            "Ekstrak & Analisis",
            type="primary",
            use_container_width=True,
        )
    if run_clicked:
        if not target_url.strip():
            st.warning("Silakan masukkan URL website terlebih dahulu.")
        else:
            progress_bar = st.progress(0.0, text="Menghubungi website awal...")

            def _update_progress(done_cnt: int, total_cnt: int, current_host: str, n_nodes: int, n_edges: int) -> None:
                ratio = min(max(done_cnt / max(total_cnt, 1), 0.0), 1.0)
                progress_bar.progress(
                    ratio,
                    text=f"Crawling [{done_cnt}/{total_cnt}]: {current_host} | Simpul: {n_nodes}, Hubungan: {n_edges}",
                )

            with st.spinner("Sedang mengekstrak tautan, menelusuri hubungan, dan menghitung PageRank..."):
                res = crawl_and_build_graph(
                    seed_url=target_url.strip(),
                    mode="root",
                    max_pages=max_pages,
                    max_depth=max_depth,
                    filter_noise=True,
                    alpha=alpha,
                    use_weight=False,
                    on_progress=_update_progress,
                )

            progress_bar.empty()
            if res.get("error"):
                st.error(f"{res['error']}")
            else:
                st.session_state["last_pagerank_result"] = res

    result = st.session_state.get("last_pagerank_result")
    if not result:
        return

    DG = result["graph"]
    top10_df = result["top10_df"]
    ranking_df = result["ranking_df"]
    edges_df = result["edges_df"]

    # 4 Kotak Metrik Ringkas
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""<div class="metric-pill">Simpul (Node)<div class="metric-pill-val">{DG.number_of_nodes()}</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-pill">Sisi (Edge)<div class="metric-pill-val">{DG.number_of_edges()}</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-pill">Website Terkunjungi<div class="metric-pill-val">{result['visited_count']}</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""<div class="metric-pill">Waktu Eksekusi<div class="metric-pill-val">{result['elapsed_sec']}s</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.subheader("10 Website Terpenting (Peringkat PageRank Tertinggi)")
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
    st.subheader("Visualisasi Graf Berarah Hubungan Antar-Website")

    total_nodes = DG.number_of_nodes()
    if total_nodes <= 5:
        top_n_vis = total_nodes
    else:
        top_n_vis = st.slider(
            "Jumlah Simpul pada Visualisasi Graf:",
            min_value=5,
            max_value=total_nodes,
            value=total_nodes,
            help=f"Geser untuk menyesuaikan jumlah simpul yang ditampilkan (maksimal {total_nodes} simpul).",
        )

    vis_html = generate_citationgraph_vis_html(
        DG,
        result["pagerank"],
        seed_node=result["seed_node"],
        top_n=top_n_vis,
        height=620,
    )
    components.html(vis_html, height=640)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    tab_all, tab_edges, tab_method = st.tabs([
        f"Semua Website ({len(ranking_df)})",
        f"Daftar Relasi Tautan ({len(edges_df)})",
        "Cara Kerja Crawling & Rumus PageRank",
    ])

    with tab_all:
        search_node = st.text_input(
            "Cari website pada seluruh hasil:",
            placeholder="Ketik nama domain atau subdomain...",
        )
        filtered_nodes = ranking_df
        if search_node.strip():
            filtered_nodes = ranking_df[
                ranking_df["Website"].str.contains(search_node.strip(), case=False, na=False)
            ]
        st.dataframe(filtered_nodes, hide_index=True, use_container_width=True)
        st.download_button(
            label="Download Semua Peringkat Website (CSV)",
            data=ranking_df.to_csv(index=False).encode("utf-8"),
            file_name="semua_peringkat_pagerank.csv",
            mime="text/csv",
        )

    with tab_edges:
        search_edge = st.text_input(
            "Filter relasi tautan (Asal atau Tujuan):",
            placeholder="Misal: trunojoyo atau kompas",
        )
        filtered_edges = edges_df
        if search_edge.strip():
            q = search_edge.strip()
            mask = (
                edges_df["Website Asal"].str.contains(q, case=False, na=False)
                | edges_df["Website Tujuan"].str.contains(q, case=False, na=False)
            )
            filtered_edges = edges_df[mask]
        st.dataframe(filtered_edges, hide_index=True, use_container_width=True)
        st.download_button(
            label="Download Daftar Relasi Tautan (CSV)",
            data=edges_df.to_csv(index=False).encode("utf-8"),
            file_name="relasi_tautan_antar_website.csv",
            mime="text/csv",
        )

    with tab_method:
        _render_methodology_tab(result, alpha=alpha)


if __name__ == "__main__":
    main()
