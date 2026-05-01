
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Headless matplotlib (works on servers, in CI, etc.)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import networkx as nx
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
from matplotlib.patches import Patch



INPUT_CSV       = "apt_papers_clean.csv"
DENDROGRAM_PNG  = "apt_dendrogram.png"
TREE_PNG        = "apt_taxonomy_tree.png"
SILHOUETTE_PNG  = "silhouette_scores.png"
CORPUS_DIST_PNG = "corpus_distribution.png"
MAPPING_CSV     = "final_taxonomy_mapping.csv"
LOG_FILE        = "taxonomy_builder.log"


EMBED_MODEL_NAME = "all-mpnet-base-v2"


N_MAIN_CLUSTERS  = 7       
N_SUB_PER_MAIN   = 2        
MIN_FOR_SUBSPLIT = 8       
LINKAGE_METHOD   = "ward"
LINKAGE_METRIC   = "euclidean"


GOLD_LABELS = [
    "Cyber Espionage & Nation-State Attribution",
    "Financial Theft & Ransomware Operations",
    "Initial Access: Spear-Phishing & Social Engineering",
    "Zero-Day Exploits & Vulnerability Analysis",
    "Supply Chain & Third-Party Compromise",
    "Lateral Movement & Command-and-Control Infrastructure",
    "Malware Analysis & Reverse Engineering",
    "Provenance Graphs & Attack Forensics",
    "Threat Intelligence & Kill Chain Modeling",
    "ML-Based Intrusion Detection Systems",
    "Credential Theft & Privilege Escalation",
    "Data Exfiltration & Persistent Access",
]

# Distinct color for each main cluster (used in dendrogram + tree)
CLUSTER_COLORS = [
    "#e6194b",   # red
    "#3cb44b",   # green
    "#4363d8",   # blue
    "#f58231",   # orange
    "#911eb4",   # purple
    "#42d4f4",   # cyan
    "#f032e6",   # magenta
    "#bfef45",   # lime
    "#fabed4",   # pink
    "#469990",   # teal
]

# Diagnostic
SILHOUETTE_K_RANGE = range(4, 11)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)



def build_embeddings(texts: List[str], model_name: str = EMBED_MODEL_NAME) -> Tuple[np.ndarray, object]:
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        log.error(
            "sentence-transformers is not installed.\n"
            "  pip install sentence-transformers==2.7.0 torch==2.2.0"
        )
        raise SystemExit(1) from e

    log.info(f"Loading sentence-transformer: {model_name}")
    model = SentenceTransformer(model_name)
    log.info(f"Encoding {len(texts)} abstracts (batch=16) ...")
    emb = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,   
        convert_to_numpy=True,
    )
    log.info(f"Embedding matrix: {emb.shape}  dtype={emb.dtype}")
    return emb, model



def hac_linkage(emb: np.ndarray) -> np.ndarray:
    """Compute the Ward-linkage matrix Z from an embedding matrix."""
    log.info(f"Computing condensed pairwise distances ({LINKAGE_METRIC}) ...")
    dist = pdist(emb, metric=LINKAGE_METRIC)
    log.info(f"Running HAC: linkage={LINKAGE_METHOD}, metric={LINKAGE_METRIC} ...")
    Z = linkage(dist, method=LINKAGE_METHOD)
    log.info(f"Linkage matrix shape: {Z.shape}  (n-1 merges expected)")
    return Z


def cut_clusters(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cut the dendrogram into exactly `n_clusters` flat labels (1..k)."""
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def silhouette_sweep(emb: np.ndarray, Z: np.ndarray,
                     k_range=SILHOUETTE_K_RANGE) -> Dict[int, float]:
    
    scores: Dict[int, float] = {}
    for k in k_range:
        labels = cut_clusters(Z, k)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(emb, labels, metric="euclidean")
        scores[k] = float(s)
        log.info(f"  k={k:>2}: silhouette={s:+.4f}")
    if scores:
        best_k = max(scores, key=scores.get)
        log.info(f"  → silhouette-optimal k = {best_k} (score={scores[best_k]:+.4f})")
    return scores



def assign_labels_by_centroid(
    emb: np.ndarray,
    cluster_ids: np.ndarray,
    gold_labels: List[str],
    model,
) -> Tuple[Dict[int, str], Dict[int, float]]:
    
    unique_clusters = sorted(set(cluster_ids))
    n_clusters = len(unique_clusters)

    # Step 1 — Cluster centroids (mean of member embeddings, re-normalized)
    centroids = np.zeros((n_clusters, emb.shape[1]), dtype=np.float32)
    for i, c in enumerate(unique_clusters):
        mask = cluster_ids == c
        centroids[i] = emb[mask].mean(axis=0)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.clip(norms, 1e-9, None)

    # Step 2 — Embed gold-standard labels with the same model
    log.info(f"  Encoding {len(gold_labels)} gold-standard labels ...")
    label_emb = model.encode(
        gold_labels,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Step 3 — Cosine similarity matrix (clusters × labels)
    sim_matrix = centroids @ label_emb.T   # (n_clusters, n_labels)

    # Step 4 — Hungarian assignment (maximize similarity = minimize -sim)
    row_ind, col_ind = linear_sum_assignment(-sim_matrix)

    label_map: Dict[int, str] = {}
    sim_map: Dict[int, float] = {}
    for r, c in zip(row_ind, col_ind):
        cid = unique_clusters[r]
        label_map[cid] = gold_labels[c]
        sim_map[cid] = float(sim_matrix[r, c])
        log.info(f"    C{cid} → {gold_labels[c]!r:50s}  "
                 f"(cosine={sim_matrix[r, c]:.4f})")

    return label_map, sim_map



def _leaves_under(node_id: int, Z: np.ndarray, n_samples: int) -> set:
    """Recursively collect all leaf indices beneath a linkage node."""
    if node_id < n_samples:
        return {node_id}
    row = Z[node_id - n_samples]
    return (_leaves_under(int(row[0]), Z, n_samples) |
            _leaves_under(int(row[1]), Z, n_samples))


def _make_link_color_func(
    Z: np.ndarray,
    cluster_labels: np.ndarray,
    color_map: Dict[int, str],
    n_samples: int,
):
   
    def link_color_func(node_id: int) -> str:
        leaves = _leaves_under(node_id, Z, n_samples)
        clusters = {cluster_labels[l] for l in leaves}
        if len(clusters) == 1:
            return color_map[clusters.pop()]
        return "#888888"
    return link_color_func


def plot_dendrogram(Z: np.ndarray, leaf_labels: List[str],
                    out_path: str, n_clusters: int,
                    cluster_labels: np.ndarray,
                    label_map: Dict[int, str]) -> None:
    
    n_samples = len(leaf_labels)

    # Build color map: cluster_id → hex color
    unique_clusters = sorted(set(cluster_labels))
    color_map: Dict[int, str] = {}
    for i, cid in enumerate(unique_clusters):
        color_map[cid] = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]

    fig, ax = plt.subplots(figsize=(22, 11), dpi=140)
    color_thresh = Z[-(n_clusters - 1), 2] if n_clusters > 1 else 0.0

    dendrogram(
        Z,
        labels=leaf_labels,
        leaf_rotation=90,
        leaf_font_size=6,
        color_threshold=0,         # disable default coloring
        above_threshold_color="#888888",
        link_color_func=_make_link_color_func(Z, cluster_labels,
                                              color_map, n_samples),
        ax=ax,
    )
    ax.axhline(y=color_thresh, color="#dc2626", linewidth=1.2, linestyle="--",
               label=f"Cut at k={n_clusters}  (height={color_thresh:.3f})")

    # Legend: one colored patch per cluster → semantic label
    legend_patches = [
        Patch(facecolor=color_map[cid],
              label=f"C{cid}: {label_map.get(cid, 'Unknown')}")
        for cid in unique_clusters
    ]
    legend_patches.append(
        Patch(facecolor="#888888", label="Inter-cluster merges")
    )
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7.5,
              frameon=True, framealpha=0.9, ncol=1,
              title="Cluster → Taxonomy Label", title_fontsize=8)

    ax.set_title("APT Research Corpus — HAC Ward Linkage with Semantic Labels",
                 fontsize=13, pad=14)
    ax.set_xlabel("Paper")
    ax.set_ylabel("Ward linkage distance")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  ✓ Dendrogram → {out_path}")


def _hierarchical_layout(G: nx.DiGraph, root: str) -> Dict[str, Tuple[float, float]]:
    
    levels: Dict[int, List[str]] = defaultdict(list)

    def walk(n: str, depth: int) -> None:
        levels[depth].append(n)
        for child in G.successors(n):
            walk(child, depth + 1)
    walk(root, 0)

    width = max(len(v) for v in levels.values())
    pos: Dict[str, Tuple[float, float]] = {}
    for depth, nodes in levels.items():
        y = -depth * 1.5
        if len(nodes) == 1:
            xs = [0.0]
        else:
            xs = np.linspace(-width / 2.0, width / 2.0, len(nodes))
        for x, n in zip(xs, nodes):
            pos[n] = (float(x), y)
    return pos


def plot_taxonomy_tree(
    root_label: str,
    main_labels_map: Dict[int, str],
    sub_labels_map: Dict[Tuple[int, int], str],
    main_sizes: Dict[int, int],
    sub_sizes: Dict[Tuple[int, int], int],
    out_path: str,
) -> None:
    """Render Root → Cluster → Sub-cluster as a clean networkx tree."""
    G = nx.DiGraph()
    G.add_node("ROOT", label=root_label, level=0)

    for cid, clabel in main_labels_map.items():
        node = f"C{cid}"
        G.add_node(node,
                   label=f"{clabel}\n(n={main_sizes.get(cid, 0)})",
                   level=1)
        G.add_edge("ROOT", node)

        for (parent_cid, sid), slabel in sub_labels_map.items():
            if parent_cid != cid:
                continue
            sn = f"C{cid}.S{sid}"
            G.add_node(
                sn,
                label=f"{slabel}\n(n={sub_sizes.get((parent_cid, sid), 0)})",
                level=2,
            )
            G.add_edge(node, sn)

    pos = _hierarchical_layout(G, "ROOT")

    fig, ax = plt.subplots(figsize=(22, 12), dpi=140)

    
    unique_main = sorted(main_labels_map.keys())
    cid_color = {cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                 for i, cid in enumerate(unique_main)}

    node_colors = []
    node_sizes = []
    for n in G.nodes:
        lvl = G.nodes[n]["level"]
        if lvl == 0:
            node_colors.append("#0f172a")
            node_sizes.append(4200)
        elif lvl == 1:
            cid = int(n.replace("C", ""))
            node_colors.append(cid_color.get(cid, "#0ea5e9"))
            node_sizes.append(2600)
        else:
            
            parent_cid = int(n.split(".")[0].replace("C", ""))
            node_colors.append(cid_color.get(parent_cid, "#84cc16"))
            node_sizes.append(1700)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#94a3b8",
                           arrows=False, width=1.4)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors="white", linewidths=1.8)
    nx.draw_networkx_labels(
        G, pos,
        labels={n: G.nodes[n]["label"] for n in G.nodes},
        ax=ax, font_size=8.5, font_color="white",
        bbox=dict(boxstyle="round,pad=0.35", ec="none", fc="#0f172abb"),
    )
    ax.set_title("APT Research Taxonomy — HAC + Semantic Centroid Matching",
                 fontsize=13, pad=14)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  ✓ Taxonomy tree → {out_path}")


def plot_silhouette(scores: Dict[int, float], chosen_k: int,
                    out_path: str) -> None:
    """Plot silhouette score vs number of clusters (k) with chosen k marked."""
    if not scores:
        log.warning("No silhouette scores to plot.")
        return

    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.plot(ks, vals, "o-", color="#2ca02c", linewidth=2, markersize=8)
    ax.axvline(x=chosen_k, color="#dc2626", linewidth=2, linestyle="--",
               label=f"chosen k = {chosen_k}")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Silhouette score (cosine)", fontsize=12)
    ax.set_title("Silhouette Score vs Number of Clusters (k)", fontsize=13)
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  ✓ Silhouette plot → {out_path}")


def plot_corpus_distribution(df: pd.DataFrame, out_path: str) -> None:
    """Bar chart of paper count per publication year."""
    year_counts = df["year"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    bars = ax.bar(year_counts.index.astype(str), year_counts.values,
                  color="#4682b4", edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, year_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Number of Papers", fontsize=12)
    ax.set_title("Corpus Distribution by Publication Year", fontsize=13)
    ax.set_ylim(0, max(year_counts.values) + 3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info(f"  ✓ Corpus distribution → {out_path}")



def main() -> None:
    log.info("=" * 64)
    log.info("  APT Taxonomy — Stage 3: Hierarchical Clustering & Labeling")
    log.info("  Labeling method: Semantic Centroid Matching (Hungarian)")
    log.info("=" * 64)

    if not Path(INPUT_CSV).exists():
        log.error(f"'{INPUT_CSV}' not found. Run preprocess.py first.")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8").reset_index(drop=True)
    log.info(f"Loaded {len(df)} papers from '{INPUT_CSV}'")
    if "abstract" not in df.columns:
        log.error("Expected column 'abstract' missing.")
        return

    if len(df) < 20:
        log.error(f"Corpus too small (n={len(df)}); HAC needs ≥20 for meaningful tree.")
        return

    # ── Step 1 — Embeddings (RAW abstracts; transformers expect grammar) ──
    log.info("\nStep 1 — Building semantic embeddings (using RAW abstracts) ...")
    raw_texts = df["abstract"].fillna("").astype(str).tolist()
    emb, model = build_embeddings(raw_texts)

    # ── Step 2 — HAC ──
    log.info("\nStep 2 — Hierarchical Agglomerative Clustering ...")
    Z = hac_linkage(emb)

    # ── Step 2b — Silhouette diagnostic ──
    log.info("\nStep 2b — Silhouette diagnostic (defensible k selection) ...")
    sil_scores = silhouette_sweep(emb, Z)
    plot_silhouette(sil_scores, N_MAIN_CLUSTERS, SILHOUETTE_PNG)

    # ── Step 3 — Cut to N main clusters ──
    log.info(f"\nStep 3 — Cutting dendrogram at k = {N_MAIN_CLUSTERS} ...")
    main_labels = cut_clusters(Z, N_MAIN_CLUSTERS)
    df["cluster_id"] = main_labels
    main_counts = Counter(main_labels)
    log.info(f"Main cluster sizes: {dict(sorted(main_counts.items()))}")

    # ── Step 4 — Semantic Centroid Matching for main clusters ──
    log.info("\nStep 4 — Semantic Centroid Matching (main clusters → gold labels) ...")
    main_labels_map, main_sim_map = assign_labels_by_centroid(
        emb, main_labels, GOLD_LABELS, model
    )
    df["cluster_label"] = df["cluster_id"].map(main_labels_map)

    # ── Step 5 — Sub-clustering inside each main (depth-2 hierarchy) ──
    log.info("\nStep 5 — Sub-clustering inside each main cluster ...")
    df["sub_id"] = 0
    sub_labels_map: Dict[Tuple[int, int], str] = {}
    sub_sizes: Dict[Tuple[int, int], int] = {}

    
    SUB_GOLD_LABELS = [
        "Detection & Classification",
        "Behavioral Analysis",
        "Attribution & Tracking",
        "Defense & Mitigation",
        "Forensic Investigation",
        "Technique Analysis",
    ]

    for c in sorted(df["cluster_id"].unique()):
        subset_idx = df.index[df["cluster_id"] == c].to_numpy()
        n = len(subset_idx)

        if n < MIN_FOR_SUBSPLIT:
            df.loc[subset_idx, "sub_id"] = 1
            sub_labels_map[(c, 1)] = main_labels_map[c]
            sub_sizes[(c, 1)] = n
            log.info(f"  C{c}: n={n} < {MIN_FOR_SUBSPLIT}, no sub-split")
            continue

        sub_emb = emb[subset_idx]
        sub_Z = linkage(pdist(sub_emb, metric=LINKAGE_METRIC),
                        method=LINKAGE_METHOD)
        n_sub = min(N_SUB_PER_MAIN, max(2, n // 6))
        sub_lbl = fcluster(sub_Z, t=n_sub, criterion="maxclust")
        df.loc[subset_idx, "sub_id"] = sub_lbl

        for sid in sorted(set(sub_lbl)):
            sub_sizes[(c, int(sid))] = int((sub_lbl == sid).sum())

        # Centroid-match sub-clusters against sub-gold-labels
        if len(set(sub_lbl)) >= 2:
            sub_label_map_local, _ = assign_labels_by_centroid(
                sub_emb, sub_lbl, SUB_GOLD_LABELS, model
            )
            for sid, slbl in sub_label_map_local.items():
                sub_labels_map[(c, sid)] = slbl
        else:
            for sid in set(sub_lbl):
                sub_labels_map[(c, int(sid))] = main_labels_map[c]

        log.info(f"  C{c} → {n_sub} sub-clusters: " +
                 ", ".join(f"S{sid}={sub_sizes[(c, sid)]}"
                           f"({sub_labels_map[(c, sid)]})"
                           for sid in sorted(set(int(s) for s in sub_lbl))))

    df["sub_label"] = df.apply(
        lambda r: sub_labels_map.get((int(r["cluster_id"]), int(r["sub_id"])),
                                     r["cluster_label"]),
        axis=1,
    )

    # ── Step 6 — Visual outputs ──
    log.info("\nStep 6 — Generating visual outputs ...")

    # Dendrogram
    leaf_labels: List[str] = []
    for i, row in df.iterrows():
        pid = str(row.get("paper_id", i))[:8]
        leaf_labels.append(f"{pid}|C{row['cluster_id']}")
    plot_dendrogram(Z, leaf_labels, DENDROGRAM_PNG, N_MAIN_CLUSTERS,
                    cluster_labels=main_labels,
                    label_map=main_labels_map)

    # Tree
    plot_taxonomy_tree(
        root_label="APT Research\nTaxonomy",
        main_labels_map=main_labels_map,
        sub_labels_map=sub_labels_map,
        main_sizes=dict(main_counts),
        sub_sizes=sub_sizes,
        out_path=TREE_PNG,
    )

    # Corpus year distribution
    plot_corpus_distribution(df, CORPUS_DIST_PNG)

    # ── Step 7 — Mapping CSV ──
    log.info("\nStep 7 — Writing final taxonomy mapping ...")
    out_cols = [
        "paper_id", "title", "year", "authors", "venue",
        "cluster_id", "cluster_label", "sub_id", "sub_label",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(MAPPING_CSV, index=False, encoding="utf-8")
    log.info(f"  ✓ Mapping → {MAPPING_CSV}")

    # ── Final summary ──
    log.info("\n" + "=" * 64)
    log.info("  FINAL TAXONOMY SUMMARY")
    log.info("=" * 64)
    for c in sorted(main_labels_map.keys()):
        sim = main_sim_map.get(c, 0.0)
        log.info(f"  C{c} — {main_labels_map[c]}  "
                 f"({main_counts[c]} papers, cosine={sim:.4f})")
        for (pc, sid), slbl in sorted(sub_labels_map.items()):
            if pc != c:
                continue
            log.info(f"      └── S{sid}: {slbl}  ({sub_sizes[(pc, sid)]})")
    log.info("\nDone. Review cosine scores — values below 0.25 indicate")
    log.info("poor label fit; consider adding more specific gold labels.")


if __name__ == "__main__":
    main()
