"""Engine crawling hyperlink, pembangunan graf berarah, dan perhitungan PageRank."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import urllib3
from bs4 import BeautifulSoup

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

NOISE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
    "tiktok.com",
    "linkedin.com",
    "wa.me",
    "api.whatsapp.com",
    "whatsapp.com",
    "t.me",
    "telegram.me",
    "pinterest.com",
    "threads.net",
    "line.me",
    "play.google.com",
    "apps.apple.com",
    "itunes.apple.com",
    "fonts.googleapis.com",
    "g.co",
    "goo.gl",
    "bit.ly",
    "tinyurl.com",
    "templatemo.com",
    "themewagon.com",
    "wordpress.org",
    "forums.colorlib.com",
    "colorlib.com",
    "w3.org",
    "creativecommons.org",
    "schema.org",
    "doubleclick.net",
    "googlesyndication.com",
    "googletagmanager.com",
    "support.google.com",
    "accounts.google.com",
    "google.com",
    "google.co.id",
    "apple.com",
    "cloudflare.com",
    "enable-javascript.com",
    "bunkerweb.io",
}

IGNORED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".pdf", ".css", ".js", ".zip", ".rar", ".7z",
    ".mp4", ".mp3", ".avi", ".mov", ".xml", ".json",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)

ID_SECOND_LEVEL_DOMAINS = {"ac", "co", "go", "or", "sch", "web", "net", "my", "mil", "biz"}


def normalize_url_input(raw_url: str) -> str:
    """Pastikan URL input memiliki skema http/https yang valid."""
    cleaned = raw_url.strip()
    if not cleaned:
        return ""
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    return cleaned


def get_root_domain(host: str) -> str:
    """Ambil root domain (menjaga domain tingkat dua Indonesia seperti .ac.id/.co.id)."""
    parts = [p for p in host.split(".") if p]
    if len(parts) >= 3 and parts[-1] == "id" and parts[-2] in ID_SECOND_LEVEL_DOMAINS:
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def normalize_host(url: str, seed_aliases: Optional[dict[str, str]] = None) -> str:
    """Ekstrak dan normalkan hostname dari URL (tanpa port serta prefix www/m umum)."""
    host = urlparse(url).netloc.lower().split(":")[0]
    for prefix in ("www.", "www2.", "www3.", "m."):
        if host.startswith(prefix):
            host = host[len(prefix):]
            break
    if seed_aliases and host in seed_aliases:
        return seed_aliases[host]
    return host


def resolve_node_name(
    url: str,
    mode: str = "host",
    seed_aliases: Optional[dict[str, str]] = None,
) -> str:
    """Ubah URL menjadi nama node graf sesuai mode ('host' atau 'root')."""
    host = normalize_host(url, seed_aliases=seed_aliases)
    if not host:
        return ""
    return get_root_domain(host) if mode == "root" else host


def is_ignored_domain(host: str) -> bool:
    """Periksa apakah host termasuk media sosial, CDN, atau noise umum."""
    if not host or "." not in host:
        return True
    return any(host == noise or host.endswith("." + noise) for noise in NOISE_DOMAINS)


def classify_relation(node: str, seed_node: str, seed_root: str) -> str:
    """Kelompokkan relasi node terhadap website awal (Seed, Internal Ekosistem, Eksternal)."""
    if node == seed_node:
        return "Web Input (Seed)"
    if node == seed_root or node.endswith("." + seed_root):
        return "Internal Ekosistem"
    return "Website Eksternal"


def extract_outbound_links(
    html: str,
    final_url: str,
    mode: str,
    seed_aliases: Optional[dict[str, str]],
    filter_noise: bool,
) -> tuple[str, dict[str, dict]]:
    """Ekstrak semua tautan keluar antar-website dari dokumen HTML."""
    src_node = resolve_node_name(final_url, mode=mode, seed_aliases=seed_aliases)
    soup = BeautifulSoup(html, "lxml")
    out_links: dict[str, dict] = {}

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
            continue

        target_url = urljoin(final_url, href).split("#")[0]
        parsed = urlparse(target_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            continue
        if parsed.path.lower().endswith(IGNORED_EXTENSIONS):
            continue

        dst_host = normalize_host(target_url, seed_aliases=seed_aliases)
        if filter_noise and is_ignored_domain(dst_host):
            continue

        dst_node = get_root_domain(dst_host) if mode == "root" else dst_host
        if not dst_node or dst_node == src_node:
            continue

        anchor_text = " ".join(tag.get_text(" ", strip=True).split())[:80]
        if dst_node not in out_links:
            out_links[dst_node] = {
                "count": 1,
                "sample_url": target_url,
                "anchor_text": anchor_text or "(tanpa teks)",
            }
        else:
            out_links[dst_node]["count"] += 1

    return src_node, out_links


def fetch_page_links(
    url: str,
    mode: str,
    seed_aliases: Optional[dict[str, str]],
    filter_noise: bool,
    timeout: float = 5.0,
) -> tuple[Optional[str], Optional[str], dict[str, dict]]:
    """Ambil halaman web dan ekstrak link keluarnya dengan timeout aman."""
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, verify=False)
        content_type = resp.headers.get("Content-Type", "").lower()
        if resp.status_code != 200 or ("text/html" not in content_type and "xhtml" not in content_type):
            return None, None, {}
    except Exception:
        return None, None, {}

    src_node, out_links = extract_outbound_links(
        resp.text,
        resp.url,
        mode=mode,
        seed_aliases=seed_aliases,
        filter_noise=filter_noise,
    )
    return src_node, resp.url, out_links


def compute_power_iteration_trace(
    DG: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 15,
) -> list[dict]:
    """Hitung jejak iterasi manual Power Iteration PageRank (termasuk dangling mass)."""
    nodes = list(DG.nodes())
    n_nodes = len(nodes)
    if n_nodes == 0:
        return []

    out_deg = dict(DG.out_degree())
    dangling = [node for node in nodes if out_deg[node] == 0]
    pr = {node: 1.0 / n_nodes for node in nodes}
    trace = []

    for it in range(1, max_iter + 1):
        dangling_sum = alpha * sum(pr[d] for d in dangling) / n_nodes
        teleport = (1.0 - alpha) / n_nodes
        pr_next = {}
        for i in nodes:
            rank_sum = sum(pr[j] / out_deg[j] for j in DG.predecessors(i))
            pr_next[i] = teleport + dangling_sum + alpha * rank_sum

        delta = sum(abs(pr_next[node] - pr[node]) for node in nodes)
        pr = pr_next
        top_node, top_val = max(pr.items(), key=lambda item: item[1])
        trace.append({
            "Iterasi": it,
            "Delta Konvergensi": round(delta, 8),
            "Web Tertinggi Sementara": top_node,
            "Skor Tertinggi": round(top_val, 6),
        })
        if delta < 1e-8:
            break

    return trace


def _build_ranking_dataframe(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
    seed_node: str,
    seed_root: str,
    visited_nodes: set[str],
) -> pd.DataFrame:
    """Susun DataFrame peringkat seluruh website berdasarkan PageRank tertinggi."""
    rows = []
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
    for idx, (node, score) in enumerate(sorted_pr, start=1):
        in_neighbors = sorted(DG.predecessors(node))
        preview_sources = ", ".join(in_neighbors[:4])
        if len(in_neighbors) > 4:
            preview_sources += f", +{len(in_neighbors) - 4} lainnya"

        rows.append({
            "Rank": idx,
            "Website": node,
            "PageRank": round(score, 6),
            "Persentase (%)": round(score * 100, 2),
            "Link Masuk (In-Degree)": DG.in_degree(node),
            "Link Keluar (Out-Degree)": DG.out_degree(node),
            "Kategori": classify_relation(node, seed_node, seed_root),
            "Status Crawl": "Terkunjungi" if node in visited_nodes else "Referensi Ditemukan",
            "Contoh Pemberi Link": preview_sources or "-",
        })
    return pd.DataFrame(rows)


def _build_edges_dataframe(DG: nx.DiGraph, pagerank: dict[str, float]) -> pd.DataFrame:
    """Susun DataFrame seluruh relasi berarah antar-website hasil ekstraksi."""
    rows = []
    for u, v, data in DG.edges(data=True):
        rows.append({
            "Website Asal": u,
            "Website Tujuan": v,
            "Jumlah Tautan": data.get("weight", 1),
            "PageRank Tujuan": round(pagerank.get(v, 0.0), 6),
            "Contoh Teks Link": data.get("anchor_text", "-"),
            "Contoh URL Tautan": data.get("sample_url", "-"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["PageRank Tujuan", "Jumlah Tautan"],
            ascending=[False, False],
        ).reset_index(drop=True)
    return df


def crawl_and_build_graph(
    seed_url: str,
    mode: str = "host",
    max_pages: int = 25,
    max_depth: int = 2,
    filter_noise: bool = True,
    alpha: float = 0.85,
    use_weight: bool = False,
    on_progress: Optional[Callable[[int, int, str, int, int], None]] = None,
) -> dict:
    """Crawl website secara paralel (prioritas in-degree BFS), bangun graf, dan hitung PageRank."""
    t_start = time.time()
    normalized_seed = normalize_url_input(seed_url)
    if not normalized_seed:
        return {"error": "URL tidak boleh kosong."}

    try:
        seed_resp = requests.get(normalized_seed, headers=DEFAULT_HEADERS, timeout=8, verify=False)
        if seed_resp.status_code != 200:
            return {"error": f"Website mengembalikan status HTTP {seed_resp.status_code}."}
    except Exception as exc:
        return {"error": f"Gagal mengakses URL awal: {exc}"}

    raw_seed_host = normalize_host(normalized_seed)
    final_seed_host = normalize_host(seed_resp.url)
    seed_aliases = {final_seed_host: raw_seed_host, raw_seed_host: raw_seed_host}

    seed_node = resolve_node_name(normalized_seed, mode=mode, seed_aliases=seed_aliases)
    seed_root = get_root_domain(seed_node)

    DG = nx.DiGraph()
    attempted_nodes: set[str] = set()
    visited_nodes: set[str] = set()
    visited_urls: list[str] = []
    candidates: dict[str, tuple[str, int]] = {}

    # Proses halaman seed langsung agar wave pertama hemat 1 request HTTP
    src0, out0 = extract_outbound_links(
        seed_resp.text,
        seed_resp.url,
        mode=mode,
        seed_aliases=seed_aliases,
        filter_noise=filter_noise,
    )
    attempted_nodes.add(seed_node)
    visited_nodes.add(seed_node)
    visited_urls.append(seed_resp.url)
    DG.add_node(seed_node)

    for dst_node, info in out0.items():
        DG.add_edge(
            seed_node,
            dst_node,
            weight=info["count"],
            sample_url=info["sample_url"],
            anchor_text=info["anchor_text"],
        )
        if max_depth >= 1 and dst_node not in attempted_nodes and dst_node not in candidates:
            candidates[dst_node] = (info["sample_url"], 1)

    if on_progress:
        on_progress(len(visited_nodes), max_pages, seed_node, DG.number_of_nodes(), DG.number_of_edges())

    # Gelombang BFS paralel memprioritaskan node ber-in-degree tertinggi & ekosistem internal dulu
    while candidates and len(visited_nodes) < max_pages:
        ranked_cands = sorted(
            candidates.items(),
            key=lambda kv: (
                -(DG.in_degree(kv[0]) if kv[0] in DG else 0),
                0 if (kv[0] == seed_root or kv[0].endswith("." + seed_root)) else 1,
                kv[1][1],
            ),
        )
        remaining = max_pages - len(visited_nodes)
        batch_size = min(len(ranked_cands), min(remaining, 8))
        batch = []
        for node_key, (target_url, depth_val) in ranked_cands[:batch_size]:
            candidates.pop(node_key)
            if node_key not in attempted_nodes:
                attempted_nodes.add(node_key)
                batch.append((node_key, target_url, depth_val))

        if not batch:
            break

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(fetch_page_links, target_url, mode, seed_aliases, filter_noise, 5.0): (
                    node_key,
                    depth_val,
                )
                for node_key, target_url, depth_val in batch
            }
            for fut in as_completed(futures):
                node_key, depth_val = futures[fut]
                src_node, final_url, out_links = fut.result()
                if not src_node:
                    continue

                visited_nodes.add(node_key)
                visited_nodes.add(src_node)
                if final_url:
                    visited_urls.append(final_url)
                DG.add_node(src_node)

                for dst_node, info in out_links.items():
                    if DG.has_edge(src_node, dst_node):
                        DG[src_node][dst_node]["weight"] += info["count"]
                    else:
                        DG.add_edge(
                            src_node,
                            dst_node,
                            weight=info["count"],
                            sample_url=info["sample_url"],
                            anchor_text=info["anchor_text"],
                        )
                    if (
                        depth_val + 1 <= max_depth
                        and dst_node not in attempted_nodes
                        and dst_node not in candidates
                    ):
                        candidates[dst_node] = (info["sample_url"], depth_val + 1)

                if on_progress:
                    on_progress(
                        min(len(visited_nodes), max_pages),
                        max_pages,
                        src_node,
                        DG.number_of_nodes(),
                        DG.number_of_edges(),
                    )

    if DG.number_of_nodes() == 0 or DG.number_of_edges() == 0:
        return {
            "error": "Tidak ditemukan cukup tautan keluar pada URL tersebut untuk membentuk graf.",
        }

    pagerank = nx.pagerank(
        DG,
        alpha=alpha,
        weight="weight" if use_weight else None,
        max_iter=500,
        tol=1e-9,
    )
    ranking_df = _build_ranking_dataframe(DG, pagerank, seed_node, seed_root, visited_nodes)
    top10_df = ranking_df.head(10).copy()
    edges_df = _build_edges_dataframe(DG, pagerank)
    power_trace = compute_power_iteration_trace(DG, alpha=alpha, max_iter=15)

    return {
        "error": None,
        "seed_url": seed_resp.url,
        "seed_node": seed_node,
        "seed_root": seed_root,
        "graph": DG,
        "pagerank": pagerank,
        "ranking_df": ranking_df,
        "top10_df": top10_df,
        "edges_df": edges_df,
        "power_trace": power_trace,
        "visited_count": len(visited_nodes),
        "attempted_count": len(attempted_nodes),
        "visited_urls": visited_urls,
        "elapsed_sec": round(time.time() - t_start, 2),
    }


def render_directed_graph_figure(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
    seed_node: str,
    top_n: int = 15,
    title: str = "Graf Berarah Hubungan Antar-Website",
) -> plt.Figure:
    """Visualisasikan subgraf Top-N website terpenting dengan tata letak rapi tanpa tumpang tindih."""
    ranked_nodes = [node for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)]
    selected_set = set(ranked_nodes[:top_n])
    if seed_node in DG:
        selected_set.add(seed_node)

    # Sertakan node perantara pada lintasan terpendek dari seed agar tidak muncul pulau terisolasi
    for target in list(selected_set):
        if target != seed_node and seed_node in DG and not nx.has_path(DG.subgraph(selected_set), seed_node, target):
            try:
                path = nx.shortest_path(DG, source=seed_node, target=target)
                if len(path) <= 4:
                    selected_set.update(path)
            except nx.NetworkXNoPath:
                pass

    ordered_nodes = [n for n in ranked_nodes if n in selected_set]
    subG = DG.subgraph(ordered_nodes).copy()

    fig, ax = plt.subplots(figsize=(11.5, 8.0), dpi=130)
    fig.patch.set_facecolor("#faf9f7")
    ax.set_facecolor("#faf9f7")
    top1_node = ranked_nodes[0] if ranked_nodes else None

    if len(ordered_nodes) >= 4 and top1_node in subG:
        outer_nodes = [n for n in ordered_nodes if n != top1_node]
        pos = nx.shell_layout(subG, nlist=[[top1_node], outer_nodes])
    else:
        pos = nx.circular_layout(subG)

    max_pr = max((pagerank.get(n, 0.0) for n in ordered_nodes), default=1.0) or 1.0
    node_sizes = [850 + (pagerank[n] / max_pr) * 2400 for n in ordered_nodes]

    top10_set = set(ranked_nodes[:10])
    node_colors = []
    edge_colors_node = []
    for n in ordered_nodes:
        if n == top1_node:
            node_colors.append("#f59e0b")
            edge_colors_node.append("#b45309")
        elif n == seed_node:
            node_colors.append("#10b981")
            edge_colors_node.append("#047857")
        elif n in top10_set:
            node_colors.append("#38bdf8")
            edge_colors_node.append("#0369a1")
        else:
            node_colors.append("#cbd5e1")
            edge_colors_node.append("#64748b")

    nx.draw_networkx_nodes(
        subG,
        pos,
        ax=ax,
        nodelist=ordered_nodes,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors=edge_colors_node,
        linewidths=2.0,
    )
    nx.draw_networkx_edges(
        subG,
        pos,
        ax=ax,
        width=1.35,
        alpha=0.55,
        edge_color="#334155",
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.14",
        node_size=node_sizes,
    )

    rank_lookup = {n: i for i, n in enumerate(ranked_nodes, start=1)}
    labels = {
        n: f"#{rank_lookup[n]} {n}\n({pagerank[n]:.4f})"
        for n in ordered_nodes
    }
    nx.draw_networkx_labels(
        subG,
        pos,
        ax=ax,
        labels=labels,
        font_size=8.0,
        font_weight="bold",
        font_color="#111111",
        font_family="Consolas",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#ffffff", "ec": "#d5d0c8", "alpha": 0.95},
    )

    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=16, color="#111111", fontfamily="Consolas")
    ax.margins(0.22)
    ax.axis("off")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Rank #1 PageRank", markerfacecolor="#f59e0b", markeredgecolor="#b45309", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Web Input (Jika bukan #1)", markerfacecolor="#10b981", markeredgecolor="#047857", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Masuk Top 10", markerfacecolor="#38bdf8", markeredgecolor="#0369a1", markersize=11),
        Line2D([0], [0], marker="o", color="w", label="Peringkat #11+ / Perantara", markerfacecolor="#cbd5e1", markeredgecolor="#64748b", markersize=11),
    ]
    leg = ax.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=8.5)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#d5d0c8")
    for text in leg.get_texts():
        text.set_fontfamily("Consolas")
    fig.tight_layout()
    return fig


def render_pagerank_barchart(top10_df: pd.DataFrame) -> plt.Figure:
    """Diagram batang horizontal skor PageRank Top 10 berlatar #faf9f7 tanpa teks terpotong."""
    df_sorted = top10_df.sort_values(by="PageRank", ascending=True)

    fig, ax = plt.subplots(figsize=(11.0, 4.8), dpi=130)
    fig.patch.set_facecolor("#faf9f7")
    ax.set_facecolor("#faf9f7")

    y_pos = list(range(len(df_sorted)))
    bars = ax.barh(y_pos, df_sorted["PageRank"], color="#141425", edgecolor="#334155", height=0.62)

    # Beri warna emas (#f59e0b) pada peringkat #1 (bar paling atas)
    if len(bars) > 0:
        bars[-1].set_color("#f59e0b")
        bars[-1].set_edgecolor("#b45309")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        df_sorted["Website"],
        fontfamily="Consolas",
        fontsize=9.5,
        fontweight="bold",
        color="#111111",
    )
    ax.set_xlabel("Skor PageRank", fontfamily="Consolas", fontsize=10, color="#555555", labelpad=8)

    max_val = max(df_sorted["PageRank"]) if not df_sorted.empty else 1.0
    for bar, val, pct in zip(bars, df_sorted["PageRank"], df_sorted["Persentase (%)"]):
        ax.text(
            bar.get_width() + (max_val * 0.015),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.6f} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontfamily="Consolas",
            fontsize=8.5,
            color="#111111",
            fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d5d0c8")
    ax.spines["bottom"].set_color("#d5d0c8")
    ax.tick_params(colors="#555555", which="both")
    ax.margins(x=0.22)
    fig.tight_layout()
    return fig


def generate_citationgraph_vis_html(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
    seed_node: str,
    top_n: int = 25,
    height: int = 620,
) -> str:
    """Hasilkan HTML graf interaktif force-directed (Vis.js) mirip CitationGraph.org."""
    import json

    ranked_nodes = [node for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)]
    selected_set = set(ranked_nodes[:top_n])
    if seed_node in DG:
        selected_set.add(seed_node)

    for target in list(selected_set):
        if target != seed_node and seed_node in DG and not nx.has_path(DG.subgraph(selected_set), seed_node, target):
            try:
                path = nx.shortest_path(DG, source=seed_node, target=target)
                if len(path) <= 4:
                    selected_set.update(path)
            except nx.NetworkXNoPath:
                pass

    ordered_nodes = [n for n in ranked_nodes if n in selected_set]
    subG = DG.subgraph(ordered_nodes).copy()
    rank_lookup = {n: i for i, n in enumerate(ranked_nodes, start=1)}
    top1_node = ranked_nodes[0] if ranked_nodes else None
    max_pr = max((pagerank.get(n, 0.0) for n in ordered_nodes), default=1.0) or 1.0

    nodes_data = []
    for node in ordered_nodes:
        val = pagerank.get(node, 0.0)
        rank_idx = rank_lookup.get(node, 999)
        size = int(14 + (val / max_pr) * 32)

        if node == top1_node:
            bg_color = "#f59e0b"
            border_color = "#b45309"
            category_tag = "Rank #1 PageRank"
        elif node == seed_node:
            bg_color = "#10b981"
            border_color = "#047857"
            category_tag = "Web Input (Seed)"
        elif rank_idx <= 10:
            bg_color = "#38bdf8"
            border_color = "#0369a1"
            category_tag = f"Masuk Top 10 (#{rank_idx})"
        else:
            bg_color = "#cbd5e1"
            border_color = "#64748b"
            category_tag = f"Peringkat #{rank_idx}"

        tooltip_html = (
            f"<div style='font-family: Consolas, monospace; line-height: 1.5; color: #111111;'>"
            f"<div style='font-size: 13px; font-weight: bold; border-bottom: 1.5px solid #e2e8f0; padding-bottom: 4px; margin-bottom: 6px; color: #141425;'>{node}</div>"
            f"<div style='display: flex; justify-content: space-between; gap: 12px; margin-bottom: 2px;'><span style='color: #64748b;'>Peringkat:</span><b style='color: #0369a1;'>#{rank_idx}</b></div>"
            f"<div style='display: flex; justify-content: space-between; gap: 12px; margin-bottom: 2px;'><span style='color: #64748b;'>PageRank:</span><b>{val:.6f} ({val * 100:.2f}%)</b></div>"
            f"<div style='display: flex; justify-content: space-between; gap: 12px; margin-bottom: 2px;'><span style='color: #64748b;'>Link Masuk:</span><b>{DG.in_degree(node)}</b></div>"
            f"<div style='display: flex; justify-content: space-between; gap: 12px; margin-bottom: 2px;'><span style='color: #64748b;'>Link Keluar:</span><b>{DG.out_degree(node)}</b></div>"
            f"<div style='margin-top: 6px; padding-top: 4px; border-top: 1px dashed #e2e8f0; font-size: 11px; color: #64748b;'>Kategori: <b style='color: #334155;'>{category_tag}</b></div>"
            f"</div>"
        )

        nodes_data.append({
            "id": node,
            "label": f"#{rank_idx} {node}" if len(node) <= 22 else f"#{rank_idx} {node[:19]}..",
            "size": size,
            "color": {
                "background": bg_color,
                "border": border_color,
                "highlight": {"background": "#f59e0b", "border": "#b45309"},
                "hover": {"background": "#f59e0b", "border": "#b45309"},
            },
            "titleHtml": tooltip_html,
        })

    edges_data = []
    for edge_id, (u, v) in enumerate(subG.edges(), start=1):
        edges_data.append({
            "id": f"e{edge_id}",
            "from": u,
            "to": v,
        })

    nodes_json_str = json.dumps(nodes_data)
    edges_json_str = json.dumps(edges_data)

    html_template = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ box-sizing: border-box; font-family: 'Consolas', 'Courier New', monospace; }}
    html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background-color: #faf9f7; overflow: hidden; }}
    #graph-wrapper {{ width: 100%; height: {height}px; background-color: #faf9f7; border: 1.5px solid #d5d0c8; border-radius: 8px; position: relative; }}
    #mynetwork {{ width: 100%; height: 100%; }}

    .controls {{
      position: absolute;
      top: 12px;
      right: 12px;
      z-index: 100;
      display: flex;
      gap: 6px;
      background: rgba(255, 255, 255, 0.95);
      padding: 6px 10px;
      border: 1px solid #d5d0c8;
      border-radius: 6px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }}
    .ctrl-btn {{
      background: #ffffff;
      border: 1px solid #c8c3ba;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 11px;
      font-weight: 700;
      color: #141425;
      cursor: pointer;
      font-family: 'Consolas', 'Courier New', monospace !important;
      transition: all 0.15s ease;
    }}
    .ctrl-btn:hover {{
      background: #141425;
      color: #ffffff;
      border-color: #141425;
    }}

    .graph-legend {{
      position: absolute;
      bottom: 12px;
      left: 12px;
      z-index: 100;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid #d5d0c8;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11px;
      color: #222222;
      box-shadow: 0 2px 6px rgba(0,0,0,0.06);
      line-height: 1.6;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 2px; }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; }}

    div.vis-tooltip {{
      position: absolute;
      visibility: hidden;
      padding: 10px 14px;
      font-family: 'Consolas', 'Courier New', monospace !important;
      font-size: 12px !important;
      color: #111111 !important;
      background-color: #ffffff !important;
      border: 1.5px solid #d5d0c8 !important;
      border-radius: 6px !important;
      box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
      z-index: 1000;
      pointer-events: none;
      line-height: 1.5;
      max-width: 320px;
    }}
    div.vis-tooltip * {{
      font-family: 'Consolas', 'Courier New', monospace !important;
    }}
  </style>
</head>
<body>
  <div id="graph-wrapper">
    <div class="controls">
      <button class="ctrl-btn" onclick="network.fit({{animation: {{duration: 500, easingFunction: 'easeInOutQuad'}} }})">Fit View</button>
      <button class="ctrl-btn" id="physics-toggle" onclick="togglePhysics()">Pause</button>
      <button class="ctrl-btn" onclick="network.moveTo({{scale: network.getScale() * 1.3}})">Zoom In</button>
      <button class="ctrl-btn" onclick="network.moveTo({{scale: network.getScale() * 0.7}})">Zoom Out</button>
    </div>

    <div class="graph-legend">
      <div class="legend-item"><span class="dot" style="background:#f59e0b; border: 1.5px solid #b45309;"></span> <b>Rank #1 PageRank</b></div>
      <div class="legend-item"><span class="dot" style="background:#10b981; border: 1.5px solid #047857;"></span> <b>Web Input (Seed)</b></div>
      <div class="legend-item"><span class="dot" style="background:#38bdf8; border: 1.5px solid #0369a1;"></span> <b>Masuk Top 10</b></div>
      <div class="legend-item"><span class="dot" style="background:#cbd5e1; border: 1.5px solid #64748b;"></span> <b>Peringkat #11+</b></div>
      <div style="margin-top: 4px; font-size: 10px; color: #666666;">Klik simpul untuk highlight koneksinya &bull; Scroll untuk zoom</div>
    </div>

    <div id="mynetwork"></div>
  </div>

  <script type="text/javascript">
    var rawNodes = {nodes_json_str};
    var rawEdges = {edges_json_str};

    // Konversi string HTML ke HTMLElement agar Vis.js merendernya sebagai rich HTML tooltip (bukan teks mentah)
    rawNodes.forEach(function(node) {{
      if (node.titleHtml) {{
        var el = document.createElement("div");
        el.innerHTML = node.titleHtml;
        node.title = el;
      }}
    }});

    var nodes = new vis.DataSet(rawNodes);
    var edges = new vis.DataSet(rawEdges);
    var container = document.getElementById('mynetwork');
    var data = {{ nodes: nodes, edges: edges }};

    var options = {{
      nodes: {{
        shape: 'dot',
        font: {{
          face: 'Consolas, Courier New, monospace',
          size: 11.5,
          color: '#111111',
          strokeWidth: 2.5,
          strokeColor: '#ffffff'
        }},
        borderWidth: 2,
        shadow: {{ enabled: true, color: 'rgba(0,0,0,0.06)', size: 5, x: 2, y: 2 }}
      }},
      edges: {{
        arrows: {{
          to: {{ enabled: true, scaleFactor: 0.75, type: 'arrow' }}
        }},
        color: {{
          color: '#94a3b8',
          highlight: '#141425',
          hover: '#0f172a',
          opacity: 0.65
        }},
        width: 1.3,
        hoverWidth: 2.5,
        smooth: {{
          type: 'curvedCW',
          roundness: 0.15
        }}
      }},
      physics: {{
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{
          gravitationalConstant: -70,
          centralGravity: 0.008,
          springLength: 120,
          springConstant: 0.06,
          damping: 0.55
        }},
        maxVelocity: 35,
        minVelocity: 0.2,
        stabilization: {{
          enabled: true,
          iterations: 180,
          updateInterval: 25
        }}
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 80,
        zoomView: true,
        dragView: true,
        dragNodes: true,
        multiselect: false,
        navigationButtons: false
      }}
    }};

    var network = new vis.Network(container, data, options);

    // Fitur CitationGraph: Klik simpul untuk sorot koneksinya
    var allNodes = rawNodes;
    var highlightActive = false;

    network.on("click", function(params) {{
      if (params.nodes.length > 0) {{
        highlightActive = true;
        var selectedNode = params.nodes[0];
        var connectedNodes = network.getConnectedNodes(selectedNode);
        var connectedEdges = network.getConnectedEdges(selectedNode);

        var updateNodes = [];
        allNodes.forEach(function(node) {{
          if (node.id === selectedNode) {{
            updateNodes.push({{ id: node.id, opacity: 1.0, font: {{ color: '#000000', size: 13 }} }});
          }} else if (connectedNodes.indexOf(node.id) !== -1) {{
            updateNodes.push({{ id: node.id, opacity: 0.95, font: {{ color: '#111111', size: 11.5 }} }});
          }} else {{
            updateNodes.push({{ id: node.id, opacity: 0.15, font: {{ color: 'rgba(0,0,0,0.15)', size: 9 }} }});
          }}
        }});
        nodes.update(updateNodes);

        var updateEdges = [];
        rawEdges.forEach(function(edge) {{
          if (connectedEdges.indexOf(edge.id) !== -1) {{
            updateEdges.push({{ id: edge.id, color: {{ color: '#141425', opacity: 0.95 }}, width: 2.5 }});
          }} else {{
            updateEdges.push({{ id: edge.id, color: {{ color: '#e2e8f0', opacity: 0.10 }}, width: 0.5 }});
          }}
        }});
        edges.update(updateEdges);
      }} else if (highlightActive) {{
        highlightActive = false;
        var resetNodes = allNodes.map(function(n) {{
          return {{ id: n.id, opacity: 1.0, font: {{ color: '#111111', size: 11.5 }} }};
        }});
        nodes.update(resetNodes);

        var resetEdges = rawEdges.map(function(e) {{
          return {{ id: e.id, color: {{ color: '#94a3b8', opacity: 0.65 }}, width: 1.3 }};
        }});
        edges.update(resetEdges);
      }}
    }});

    var isPhysicsPaused = false;
    function togglePhysics() {{
      isPhysicsPaused = !isPhysicsPaused;
      network.setOptions({{ physics: {{ enabled: !isPhysicsPaused }} }});
      var btn = document.getElementById('physics-toggle');
      btn.innerText = isPhysicsPaused ? 'Resume' : 'Pause';
    }}
  </script>
</body>
</html>"""
    return html_template


