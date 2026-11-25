"""Engine analisis sitasi jurnal, pembangunan graf sitasi berarah, dan perhitungan PageRank."""

from __future__ import annotations

import json
import re
import time
from typing import Callable, Optional
import networkx as nx
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Lineage ID Elsevier di OpenAlex untuk filter jurnal ScienceDirect/Elsevier
ELSEVIER_LINEAGE_ID = "https://openalex.org/P4310320990"


def extract_doi_or_clean(input_str: str) -> tuple[Optional[str], str]:
    """Ekstrak DOI jika ada, atau kembalikan teks pencarian bersih."""
    cleaned = input_str.strip()
    doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", cleaned)
    if doi_match:
        doi = doi_match.group(1).rstrip("/.")
        return doi, doi
    return None, cleaned


def resolve_seed_paper(input_str: str) -> Optional[dict]:
    """Cari metadata paper awal (Seed) dari DOI, link ScienceDirect, atau kata kunci judul."""
    doi, clean_query = extract_doi_or_clean(input_str)

    if doi:
        url = f"https://api.openalex.org/works/doi:{doi}"
        try:
            r = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            if r.status_code == 200:
                return _parse_openalex_work(r.json())
        except Exception:
            pass

    # Jika bukan DOI langsung atau DOI tidak ditemukan, lakukan pencarian berdasarkan judul/topik
    # Prioritaskan paper penerbit Elsevier / ScienceDirect
    search_url = (
        f"https://api.openalex.org/works?"
        f"filter=primary_location.source.host_organization_lineage:{ELSEVIER_LINEAGE_ID}"
        f"&search={clean_query}&per-page=1"
    )
    try:
        r = requests.get(search_url, headers=DEFAULT_HEADERS, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results:
                return _parse_openalex_work(results[0])
    except Exception:
        pass

    # Pencarian umum jika filter Elsevier kosong
    general_url = f"https://api.openalex.org/works?search={clean_query}&per-page=1"
    try:
        r2 = requests.get(general_url, headers=DEFAULT_HEADERS, timeout=10)
        if r2.status_code == 200:
            results2 = r2.json().get("results", [])
            if results2:
                return _parse_openalex_work(results2[0])
    except Exception:
        pass

    return None


def _parse_openalex_work(data: dict) -> dict:
    """Konversi respons data OpenAlex menjadi format kamus paper standar."""
    raw_id = data.get("id", "")
    clean_id = raw_id.replace("https://openalex.org/", "")

    authorships = data.get("authorships", [])
    author_names = [a.get("author", {}).get("display_name", "") for a in authorships if a.get("author")]
    if len(author_names) > 3:
        authors_str = ", ".join(author_names[:3]) + " et al."
    elif author_names:
        authors_str = ", ".join(author_names)
    else:
        authors_str = "Peneliti"

    raw_doi = data.get("doi", "") or ""
    clean_doi = raw_doi.replace("https://doi.org/", "")

    journal_name = (
        data.get("primary_location", {})
        .get("source", {})
        .get("display_name", "ScienceDirect / Elsevier")
        or "ScienceDirect / Elsevier"
    )

    referenced = [
        ref.replace("https://openalex.org/", "")
        for ref in data.get("referenced_works", [])
        if ref
    ]

    title = data.get("title") or "Tanpa Judul"

    return {
        "id": clean_id,
        "title": title,
        "authors": authors_str,
        "year": data.get("publication_year", 0),
        "journal": journal_name,
        "doi": clean_doi,
        "cited_by_count": data.get("cited_by_count", 0),
        "referenced_works": referenced,
    }


def fetch_batch_papers(openalex_ids: list[str]) -> dict[str, dict]:
    """Ambil metadata banyak paper sekaligus dalam 1 request batch OpenAlex."""
    if not openalex_ids:
        return {}

    clean_ids = [i.replace("https://openalex.org/", "").strip() for i in openalex_ids if i]
    # Ambil maksimal 50 per batch agar tidak melebihi URL limit
    batch = clean_ids[:50]
    id_filter = "|".join(batch)
    url = f"https://api.openalex.org/works?filter=openalex:{id_filter}&per-page=50"

    results_map: dict[str, dict] = {}
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=12)
        if r.status_code == 200:
            for item in r.json().get("results", []):
                parsed = _parse_openalex_work(item)
                results_map[parsed["id"]] = parsed
    except Exception:
        pass

    return results_map


def crawl_and_build_citation_graph(
    seed_input: str,
    max_papers: int = 25,
    max_depth: int = 2,
    alpha: float = 0.85,
    on_progress: Optional[Callable[[int, int, str, int, int], None]] = None,
) -> dict:
    """Ekstrak jaringan sitasi jurnal (Breadth-First Search sitasi) dan hitung PageRank."""
    t_start = time.time()

    if not seed_input.strip():
        return {"error": "Input judul atau DOI paper tidak boleh kosong."}

    seed_paper = resolve_seed_paper(seed_input.strip())
    if not seed_paper:
        return {
            "error": "Paper tidak ditemukan di ScienceDirect / indeks ilmiah. Pastikan DOI atau kata kunci judul valid.",
        }

    DG = nx.DiGraph()
    seed_id = seed_paper["id"]

    DG.add_node(
        seed_id,
        title=seed_paper["title"],
        authors=seed_paper["authors"],
        year=seed_paper["year"],
        journal=seed_paper["journal"],
        doi=seed_paper["doi"],
        is_seed=True,
    )

    visited_papers: set[str] = {seed_id}
    all_paper_metadata: dict[str, dict] = {seed_id: seed_paper}

    seed_refs = seed_paper.get("referenced_works", [])
    if on_progress:
        on_progress(1, max_papers, seed_paper["title"][:40], 1, 0)

    # ── Tahap 1: Ekstraksi Paper yang Disitasi Langsung oleh Seed (Depth 1) ──
    limit_depth1 = min(len(seed_refs), max_papers - 1)
    target_depth1_ids = seed_refs[:limit_depth1]

    depth1_papers = fetch_batch_papers(target_depth1_ids)

    depth1_ids = []
    for pid, pdata in depth1_papers.items():
        if pid not in DG:
            DG.add_node(
                pid,
                title=pdata["title"],
                authors=pdata["authors"],
                year=pdata["year"],
                journal=pdata["journal"],
                doi=pdata["doi"],
                is_seed=False,
            )
        DG.add_edge(seed_id, pid, relation="Menyitasi Langsung")
        visited_papers.add(pid)
        all_paper_metadata[pid] = pdata
        depth1_ids.append(pid)

    if on_progress:
        on_progress(
            len(visited_papers),
            max_papers,
            f"Selesai Depth 1 ({len(depth1_ids)} referensi)",
            DG.number_of_nodes(),
            DG.number_of_edges(),
        )

    # ── Tahap 2: Cek Tautan Silang Antar-Referensi & Penelusuran Cucu (Depth 2) ──
    # Pertama, periksa apakah sesama paper di Depth 1 saling menyitasi (Cross-Citations)
    for pid in depth1_ids:
        pdata = all_paper_metadata.get(pid, {})
        p_refs = set(pdata.get("referenced_works", []))
        for other_id in depth1_ids:
            if other_id != pid and other_id in p_refs:
                DG.add_edge(pid, other_id, relation="Sitasi Silang Antar-Referensi")

    # Jika Depth >= 2 dan kuota max_papers masih tersisa, telusuri referensi generasi berikutnya
    if max_depth >= 2 and len(visited_papers) < max_papers:
        remaining_quota = max_papers - len(visited_papers)
        # Kumpulkan kandidat referensi generasi ke-2 (cucu) dari paper depth 1
        depth2_candidate_ids: list[tuple[str, str]] = []  # (source_pid, target_ref_id)
        for pid in depth1_ids:
            pdata = all_paper_metadata.get(pid, {})
            for r_id in pdata.get("referenced_works", []):
                if r_id not in visited_papers:
                    depth2_candidate_ids.append((pid, r_id))

        # Ambil batch unik untuk cucu
        unique_cucu_ids = list(dict.fromkeys([t[1] for t in depth2_candidate_ids]))[:remaining_quota]
        depth2_papers = fetch_batch_papers(unique_cucu_ids)

        for src_pid, dst_pid in depth2_candidate_ids:
            if dst_pid in depth2_papers:
                cucu_data = depth2_papers[dst_pid]
                if dst_pid not in DG:
                    DG.add_node(
                        dst_pid,
                        title=cucu_data["title"],
                        authors=cucu_data["authors"],
                        year=cucu_data["year"],
                        journal=cucu_data["journal"],
                        doi=cucu_data["doi"],
                        is_seed=False,
                    )
                    visited_papers.add(dst_pid)
                    all_paper_metadata[dst_pid] = cucu_data
                DG.add_edge(src_pid, dst_pid, relation="Sitasi Lanjutan (Depth 2)")

    if on_progress:
        on_progress(
            len(visited_papers),
            max_papers,
            "Menghitung PageRank...",
            DG.number_of_nodes(),
            DG.number_of_edges(),
        )

    if DG.number_of_nodes() == 0 or DG.number_of_edges() == 0:
        return {
            "error": "Tidak ditemukan cukup relasi sitasi pada paper tersebut untuk membangun graf.",
        }

    # ── Tahap 3: Perhitungan Algoritma PageRank ──
    pagerank = nx.pagerank(
        DG,
        alpha=alpha,
        max_iter=500,
        tol=1e-9,
    )

    ranking_df = _build_ranking_dataframe(DG, pagerank, seed_id)
    top10_df = ranking_df.head(10).copy()
    edges_df = _build_edges_dataframe(DG, pagerank)
    power_trace = compute_power_iteration_trace(DG, alpha=alpha, max_iter=15)

    return {
        "error": None,
        "seed_paper": seed_paper,
        "seed_id": seed_id,
        "graph": DG,
        "pagerank": pagerank,
        "ranking_df": ranking_df,
        "top10_df": top10_df,
        "edges_df": edges_df,
        "power_trace": power_trace,
        "total_nodes": DG.number_of_nodes(),
        "total_edges": DG.number_of_edges(),
        "elapsed_sec": round(time.time() - t_start, 2),
    }


def compute_power_iteration_trace(
    DG: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 15,
) -> list[dict]:
    """Hitung jejak konvergensi manual Power Iteration pada graf sitasi."""
    nodes = list(DG.nodes())
    N = len(nodes)
    if N == 0:
        return []

    pr = {n: 1.0 / N for n in nodes}
    trace = [{
        "Iterasi": 0,
        "Delta Konvergensi": 1.0,
        "Paper Otoritas Tertinggi Sementara": DG.nodes[nodes[0]].get("title", nodes[0])[:45],
        "Skor Tertinggi": round(1.0 / N, 6),
    }]

    for it in range(1, max_iter + 1):
        pr_next = {}
        dangling_sum = sum(pr[n] for n in nodes if DG.out_degree(n) == 0)
        base = (1.0 - alpha) / N + (alpha * dangling_sum / N)

        for i in nodes:
            in_sum = sum(pr[j] / DG.out_degree(j) for j in DG.predecessors(i))
            pr_next[i] = base + (alpha * in_sum)

        delta = sum(abs(pr_next[n] - pr[n]) for n in nodes)
        pr = pr_next
        top_node, top_val = max(pr.items(), key=lambda item: item[1])
        top_title = DG.nodes[top_node].get("title", top_node)

        trace.append({
            "Iterasi": it,
            "Delta Konvergensi": round(delta, 8),
            "Paper Otoritas Tertinggi Sementara": top_title[:45] + ("..." if len(top_title) > 45 else ""),
            "Skor Tertinggi": round(top_val, 6),
        })

        if delta < 1e-8:
            break

    return trace


def _build_ranking_dataframe(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
    seed_id: str,
) -> pd.DataFrame:
    """Susun DataFrame peringkat seluruh paper berdasarkan skor PageRank tertinggi."""
    rows = []
    sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

    for idx, (node_id, score) in enumerate(sorted_pr, start=1):
        node_info = DG.nodes[node_id]
        is_seed = (node_id == seed_id)

        rows.append({
            "Rank": idx,
            "Judul Paper / Jurnal": node_info.get("title", "Tanpa Judul"),
            "Penulis": node_info.get("authors", "-"),
            "Tahun": node_info.get("year", "-"),
            "Wadah Jurnal / Penerbit": node_info.get("journal", "-"),
            "PageRank": round(score, 6),
            "Persentase (%)": round(score * 100, 2),
            "Sitasi Masuk (In-Degree)": DG.in_degree(node_id),
            "Referensi Keluar (Out-Degree)": DG.out_degree(node_id),
            "Peran Paper": "Paper Utama (Seed)" if is_seed else "Paper Rujukan / Sitasi",
            "DOI": node_info.get("doi", "-"),
        })

    return pd.DataFrame(rows)


def _build_edges_dataframe(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
) -> pd.DataFrame:
    """Susun DataFrame seluruh hubungan sitasi berarah antar-paper."""
    rows = []
    for u, v, data in DG.edges(data=True):
        src_info = DG.nodes[u]
        dst_info = DG.nodes[v]

        rows.append({
            "Paper yang Menyitasi (Source)": src_info.get("title", u),
            "Paper yang Disitasi (Target)": dst_info.get("title", v),
            "Tipe Hubungan": data.get("relation", "Menyitasi"),
            "PageRank Target": round(pagerank.get(v, 0.0), 6),
            "Penulis Target": dst_info.get("authors", "-"),
            "Tahun Target": dst_info.get("year", "-"),
            "Jurnal Target": dst_info.get("journal", "-"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["PageRank Target"],
            ascending=[False],
        ).reset_index(drop=True)
    return df


def generate_citation_vis_html(
    DG: nx.DiGraph,
    pagerank: dict[str, float],
    seed_id: str,
    top_n: Optional[int] = None,
    height: int = 620,
) -> str:
    """Bangun template visualisasi graf animasi Vis.js ForceAtlas2 khusus sitasi ilmiah."""
    ranked_nodes = [node for node, _ in sorted(pagerank.items(), key=lambda x: x[1], reverse=True)]

    if top_n is not None and top_n < len(ranked_nodes):
        selected_nodes = set(ranked_nodes[:top_n])
        selected_nodes.add(seed_id)
        subG = DG.subgraph(selected_nodes).copy()
    else:
        subG = DG.copy()

    top1_node = ranked_nodes[0] if ranked_nodes else None
    top10_set = set(ranked_nodes[:10])

    max_pr = max((pagerank.get(n, 0.0) for n in subG.nodes()), default=1.0) or 1.0

    nodes_data = []
    for n in subG.nodes():
        info = subG.nodes[n]
        score = pagerank.get(n, 0.0)
        norm_score = score / max_pr
        size = 14 + (norm_score * 32)

        if n == top1_node:
            color_bg = "#f59e0b"
            color_border = "#b45309"
        elif n == seed_id:
            color_bg = "#10b981"
            color_border = "#047857"
        elif n in top10_set:
            color_bg = "#38bdf8"
            color_border = "#0369a1"
        else:
            color_bg = "#cbd5e1"
            color_border = "#64748b"

        full_title = info.get("title", n)
        short_label = full_title[:28] + ("..." if len(full_title) > 28 else "")

        title_html = (
            f"<div style='font-family:Consolas,monospace; font-size:12px; line-height:1.5; color:#111; max-width:320px;'>"
            f"<b>{full_title}</b><br>"
            f"<span style='color:#555;'>Penulis: {info.get('authors', '-')}</span><br>"
            f"<span style='color:#555;'>Tahun: {info.get('year', '-')} | Jurnal: {info.get('journal', '-')}</span><br>"
            f"<hr style='margin:4px 0; border:0; border-top:1px solid #ccc;'>"
            f"<b>Skor PageRank:</b> {score:.6f} ({score * 100:.2f}%)<br>"
            f"<b>Disitasi (In-Degree):</b> {subG.in_degree(n)} kali<br>"
            f"<b>Referensi (Out-Degree):</b> {subG.out_degree(n)} paper"
            f"</div>"
        )

        nodes_data.append({
            "id": n,
            "label": short_label,
            "size": round(size, 1),
            "color": {
                "background": color_bg,
                "border": color_border,
                "highlight": {"background": "#141425", "border": "#ffffff"},
            },
            "titleHtml": title_html,
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
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 7px; }}
  </style>
</head>
<body>
  <div id="graph-wrapper">
    <div class="graph-legend">
      <div class="legend-item"><span class="dot" style="background:#f59e0b; border: 1.5px solid #b45309;"></span> <b>Rank #1 Most Influential Paper</b></div>
      <div class="legend-item"><span class="dot" style="background:#10b981; border: 1.5px solid #047857;"></span> <b>Paper Awal (Seed Input)</b></div>
      <div class="legend-item"><span class="dot" style="background:#38bdf8; border: 1.5px solid #0369a1;"></span> <b>Top 10 Paper Berpengaruh</b></div>
      <div class="legend-item"><span class="dot" style="background:#cbd5e1; border: 1.5px solid #64748b;"></span> <b>Peringkat #11+</b></div>
    </div>
    <div id="mynetwork"></div>
  </div>

  <script type="text/javascript">
    var rawNodes = {nodes_json_str};
    var rawEdges = {edges_json_str};

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
          springLength: 130,
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

    // Sorot koneksi sitasi saat simpul paper diklik
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
  </script>
</body>
</html>"""
    return html_template
