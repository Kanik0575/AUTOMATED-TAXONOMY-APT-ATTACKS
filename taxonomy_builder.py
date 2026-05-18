"""
taxonomy_builder.py — Stage 3: Hierarchical Taxonomy Construction
==================================================================
Builds an automated, two-level hierarchical taxonomy over the cleaned
APT research corpus (`apt_papers_clean.csv`).

Pipeline
--------
  1. Load corpus (raw abstract for embedding)
  2. Encode abstracts with a sentence-transformer (semantic embeddings)
  3. Hierarchical Agglomerative Clustering (Ward linkage, Euclidean)
  4. Diagnostic silhouette sweep across candidate k
  5. Cut to N_MAIN_CLUSTERS top-level groups
  6. Sub-cluster each main group (depth-2 hierarchy)
  7. Label clusters via Semantic Centroid Matching against gold-standard
     APT taxonomy labels (cosine similarity + Hungarian assignment)
  8. Render dendrogram (scipy) + taxonomy tree (networkx)
  9. Persist final mapping to CSV

Outputs
-------
  • apt_dendrogram.png       — scientific dendrogram with cut line
  • apt_taxonomy_tree.png    — Root → Cluster → Sub-cluster network
  • final_taxonomy_mapping.csv  — per-paper assignments

Design notes (read the README before defending this to your professor)
----------------------------------------------------------------------
  • Embeddings use `all-mpnet-base-v2` — a *general-purpose* English
    sentence transformer. It is NOT cybersecurity-specific. True security
    models (SecBERT, SecureBERT) are masked-LM BERTs requiring custom
    mean-pooling and have weaker public validation. mpnet is the
    pragmatically defensible choice for a small academic corpus.
  • Embeddings are L2-normalized so Ward's required Euclidean distance
    is monotonic with cosine distance on the unit hypersphere.
  • Labeling uses Semantic Centroid Matching: each cluster centroid is
    compared via cosine similarity against embedded gold-standard APT
    taxonomy labels. The Hungarian algorithm (scipy.optimize.linear_sum_assignment)
    guarantees globally optimal, non-duplicate label assignment.
  • This is semi-supervised taxonomy alignment — domain expertise defines
    candidate categories; the algorithm decides which ones the data
    supports and how papers map to them.

Required extra packages (not in requirements.txt yet)
-----------------------------------------------------
  pip install sentence-transformers==2.7.0 networkx==3.2.1 torch==2.2.0

Author: <you>
"""

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


# ╭──────────────────────────────────────────────────────────────────╮
# │  CONFIGURATION                                                   │
# ╰──────────────────────────────────────────────────────────────────╯
INPUT_CSV       = "apt_papers_clean.csv"
DENDROGRAM_PNG  = "apt_dendrogram.png"
TREE_PNG        = "apt_taxonomy_tree.png"
SILHOUETTE_PNG  = "silhouette_scores.png"
CORPUS_DIST_PNG = "corpus_distribution.png"
MAPPING_CSV     = "final_taxonomy_mapping.csv"
LOG_FILE        = "taxonomy_builder.log"

# General-purpose semantic embedder. Honest disclosure in module docstring.
EMBED_MODEL_NAME = "all-mpnet-base-v2"

# HAC parameters
N_MAIN_CLUSTERS  = 7        # top-level taxonomy buckets (sweet spot for ~120 papers)
N_SUB_PER_MAIN   = 2        # sub-clusters under each main (only if main is large enough)
MIN_FOR_SUBSPLIT = 8        # do not split a main cluster smaller than this
LINKAGE_METHOD   = "ward"
LINKAGE_METRIC   = "euclidean"

# ── GOLD-STANDARD APT TAXONOMY LABELS ──
# These are expert-defined candidate categories for Semantic Centroid Matching.
# The Hungarian algorithm will assign the best-fit label to each cluster;
# surplus labels (len > N_MAIN_CLUSTERS) are simply unassigned.
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


# ╭──────────────────────────────────────────────────────────────────╮
# │  LOGGING                                                         │
# ╰──────────────────────────────────────────────────────────────────╯
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


# ╭──────────────────────────────────────────────────────────────────╮
# │  EMBEDDINGS                                                      │
# ╰──────────────────────────────────────────────────────────────────╯
def build_embeddings(texts: List[str], model_name: str = EMBED_MODEL_NAME) -> Tuple[np.ndarray, object]:
    """Encode abstracts into dense semantic vectors.

    Uses sentence-transformers, which handles tokenization, attention,
    and mean-pooling internally. Outputs are L2-normalized so that
    Euclidean distance becomes a monotonic function of cosine distance —
    essential because Ward linkage requires Euclidean.

    Returns the embedding matrix AND the loaded model (reused later
    for encoding gold-standard labels in Semantic Centroid Matching).
    """
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
        normalize_embeddings=True,   # unit vectors → cos ≡ Euclidean (monotone)
        convert_to_numpy=True,
    )
    log.info(f"Embedding matrix: {emb.shape}  dtype={emb.dtype}")
    return emb, model


# ╭──────────────────────────────────────────────────────────────────╮
# │  CLUSTERING                                                      │
# ╰──────────────────────────────────────────────────────────────────╯
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
    """Diagnostic: silhouette score across candidate k values.

    Higher = tighter, better-separated clusters. Use this to defend
    your choice of k to a sceptical examiner.
    """
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


# ╭──────────────────────────────────────────────────────────────────╮
# │  SEMANTIC CENTROID MATCHING (replaces c-TF-IDF)                  │
# ╰──────────────────────────────────────────────────────────────────╯
def assign_labels_by_centroid(
    emb: np.ndarray,
    cluster_ids: np.ndarray,
    gold_labels: List[str],
    model,
) -> Tuple[Dict[int, str], Dict[int, float]]:
    """Assign each cluster the gold label whose embedding is closest
    to that cluster's centroid, using the Hungarian algorithm for
    globally optimal one-to-one assignment.

    How it works
    ------------
    1. Compute the centroid (mean vector) of all paper embeddings in
       each cluster. L2-normalize so cosine similarity = dot product.
    2. Embed the gold-standard label strings with the SAME model.
    3. Build a (n_clusters × n_labels) cosine similarity matrix.
    4. Run the Hungarian algorithm (linear_sum_assignment) on the
       negated matrix to find the assignment that maximizes total
       similarity while guaranteeing no two clusters share a label.

    Returns
    -------
    label_map   : {cluster_id: assigned_label}
    sim_map     : {cluster_id: cosine_similarity_score}
    """
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


# ╭──────────────────────────────────────────────────────────────────╮
# │  VISUALIZATION                                                   │
# ╰──────────────────────────────────────────────────────────────────╯
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
    """Return a callable for scipy dendrogram's `link_color_func`.

    Colors a branch by its cluster if ALL leaves below it belong to
    the same cluster; otherwise gray (inter-cluster merge).
    """
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
    """Scientific dendrogram with cluster-colored branches and a legend
    mapping each color to its semantic taxonomy label."""
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


def _wrap_label(text: str, max_chars: int = 28) -> str:
    """Word-wrap a label so no line exceeds max_chars.
    Preserves existing newlines (e.g. the '(n=X)' part)."""
    import textwrap
    parts = text.split("\n")
    wrapped = []
    for part in parts:
        wrapped.append("\n".join(textwrap.wrap(part, width=max_chars)))
    return "\n".join(wrapped)


def plot_taxonomy_tree(
    root_label: str,
    main_labels_map: Dict[int, str],
    sub_labels_map: Dict[Tuple[int, int], str],
    main_sizes: Dict[int, int],
    sub_sizes: Dict[Tuple[int, int], int],
    out_path: str,
) -> None:
    """Publication-quality taxonomy tree using raw matplotlib patches + text.

    Why not networkx drawing?
    -------------------------
    nx.draw_networkx_labels auto-sizes boxes in font-space, giving zero
    control over rendered box dimensions. When a 50-inch figure is scaled
    to fit a paper column, text becomes unreadable. By drawing FancyBboxPatch
    and ax.text manually in DATA coordinates, we guarantee every box is
    large enough to read at the target print size.
    """
    from matplotlib.patches import FancyBboxPatch

    # ── Collect nodes ──
    nodes = {}   # id → {label, level, cid}
    edges = []   # (parent_id, child_id)

    nodes["ROOT"] = {"label": root_label, "level": 0, "cid": None}

    for cid, clabel in main_labels_map.items():
        nid = f"C{cid}"
        nodes[nid] = {
            "label": f"{clabel}\n(n={main_sizes.get(cid, 0)})",
            "level": 1, "cid": cid,
        }
        edges.append(("ROOT", nid))

        for (parent_cid, sid), slabel in sub_labels_map.items():
            if parent_cid != cid:
                continue
            snid = f"C{cid}.S{sid}"
            nodes[snid] = {
                "label": f"{slabel}\n(n={sub_sizes.get((parent_cid, sid), 0)})",
                "level": 2, "cid": cid,
            }
            edges.append((nid, snid))

    # ── Pastel color palette ──
    PASTEL_COLORS = [
        "#fecaca", "#bbf7d0", "#bfdbfe", "#fed7aa", "#e9d5ff",
        "#a5f3fc", "#f9a8d4", "#d9f99d", "#fde68a", "#c7d2fe",
    ]
    unique_main = sorted(main_labels_map.keys())
    cid_pastel = {cid: PASTEL_COLORS[i % len(PASTEL_COLORS)]
                  for i, cid in enumerate(unique_main)}

    # ── Layout parameters (all in data-coordinate units) ──
    # Designed for figsize≈(18,8) at 300 DPI → full-page landscape figure.
    # Fonts are set to their FINAL print size (7-9pt), so what you see
    # in the PNG is what prints in the paper.
    BOX_W_L1      = 1.55    # width of level-1 boxes
    BOX_H_L1      = 0.75    # height of level-1 boxes
    BOX_W_L2      = 1.3     # width of level-2 boxes
    BOX_H_L2      = 0.65    # height of level-2 boxes
    BOX_W_ROOT    = 1.2
    BOX_H_ROOT    = 0.55

    SLOT_W_L2     = BOX_W_L2 + 0.25  # horizontal pitch per sub-cluster slot
    MIN_SLOT_L1   = BOX_W_L1 + 0.35  # minimum pitch per main cluster
    PARENT_GAP    = 0.35              # extra gap between main-cluster regions
    Y_ROOT        = 6.5
    Y_L1          = 4.0
    Y_L2          = 1.0

    # ── Bottom-up layout ──
    l1_nodes = [nid for nid, d in nodes.items() if d["level"] == 1]
    children_of = defaultdict(list)
    for p, c in edges:
        if nodes[c]["level"] == 2:
            children_of[p].append(c)

    # Width each L1 node needs = max(own box, children slots)
    l1_widths = {}
    for nid in l1_nodes:
        nc = len(children_of.get(nid, []))
        l1_widths[nid] = max(MIN_SLOT_L1, nc * SLOT_W_L2)

    # Place L1 left-to-right
    total_w = sum(l1_widths[n] for n in l1_nodes) + PARENT_GAP * max(0, len(l1_nodes) - 1)
    pos = {}
    cursor = -total_w / 2
    for nid in l1_nodes:
        w = l1_widths[nid]
        pos[nid] = (cursor + w / 2, Y_L1)
        cursor += w + PARENT_GAP

    # Root centred
    pos["ROOT"] = (0.0, Y_ROOT)

    # L2 centred under parent
    for parent, children in children_of.items():
        px, _ = pos[parent]
        n = len(children)
        if n == 1:
            pos[children[0]] = (px, Y_L2)
        else:
            span = (n - 1) * SLOT_W_L2
            for i, c in enumerate(children):
                pos[c] = (px - span / 2 + i * SLOT_W_L2, Y_L2)

    # ── Figure — sized for direct inclusion in a landscape paper figure ──
    # At ~18 inches wide, a full-page landscape figure keeps fonts readable.
    fig_w = max(18, total_w + 3)
    fig, ax = plt.subplots(figsize=(fig_w, 9), dpi=300)

    # ── Draw edges (thin lines from bottom-centre of parent box to
    #    top-centre of child box) ──
    for p, c in edges:
        px, py = pos[p]
        cx, cy = pos[c]
        if nodes[p]["level"] == 0:
            py_bot = py - BOX_H_ROOT / 2
        else:
            py_bot = py - BOX_H_L1 / 2
        if nodes[c]["level"] == 1:
            cy_top = cy + BOX_H_L1 / 2
        else:
            cy_top = cy + BOX_H_L2 / 2
        ax.plot([px, cx], [py_bot, cy_top],
                color="#64748b", linewidth=1.0, zorder=1)

    # ── Draw boxes + text ──
    font_family = "DejaVu Sans"

    def draw_box(nid):
        x, y = pos[nid]
        info = nodes[nid]
        lvl = info["level"]

        if lvl == 0:
            bw, bh = BOX_W_ROOT, BOX_H_ROOT
            fc, ec, tc = "#1e293b", "#000000", "white"
            fs, fw = 10, "bold"
            wrap_at = 18
        elif lvl == 1:
            bw, bh = BOX_W_L1, BOX_H_L1
            fc = cid_pastel.get(info["cid"], "#bfdbfe")
            ec, tc = "#334155", "#0f172a"
            fs, fw = 8, "semibold"
            wrap_at = 18
        else:
            bw, bh = BOX_W_L2, BOX_H_L2
            fc = cid_pastel.get(info["cid"], "#e2e8f0")
            ec, tc = "#334155", "#0f172a"
            fs, fw = 7.5, "normal"
            wrap_at = 16

        label = _wrap_label(info["label"], max_chars=wrap_at)

        rect = FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y, label,
                ha="center", va="center",
                fontsize=fs, fontweight=fw, fontfamily=font_family,
                color=tc, zorder=3,
                linespacing=1.15)

    for nid in nodes:
        draw_box(nid)

    ax.set_title("APT Research Taxonomy — HAC + Semantic Centroid Matching",
                 fontsize=13, fontweight="bold", pad=12, fontfamily=font_family)
    ax.set_axis_off()
    ax.autoscale_view()
    # Add a little padding around the content
    xvals = [p[0] for p in pos.values()]
    yvals = [p[1] for p in pos.values()]
    ax.set_xlim(min(xvals) - 2.5, max(xvals) + 2.5)
    ax.set_ylim(min(yvals) - 1.5, max(yvals) + 1.5)
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


# ╭──────────────────────────────────────────────────────────────────╮
# │  MAIN                                                            │
# ╰──────────────────────────────────────────────────────────────────╯
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

    # Build sub-gold-labels per main cluster: narrower, more specific
    # variants that match the parent theme
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
