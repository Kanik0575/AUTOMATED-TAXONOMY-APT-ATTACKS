# Automated Taxonomy of Advanced Persistent Threat (APT) Attacks
### A Hierarchical Machine Learning Pipeline for Systematic Classification of APT Research Literature

---

| | |
|---|---|
| **Student** | Kanik Kumar |
| **ID** | 2023A7PS0575P |
| **Course** | CS F266 — Study Project |
| **Supervisor** | Prof. Rajesh Kumar |
| **Department** | Computer Science, BITS Pilani, Pilani Campus |
| **Year** | 2025–2026 |

---

## Abstract

This project presents a fully automated, reproducible pipeline that constructs a **hierarchical taxonomy** of Advanced Persistent Threat (APT) attacks from 120 real peer-reviewed academic papers published between 2021 and 2026. The pipeline uses NLP preprocessing with a dual-group relevance filter, dense semantic embeddings via `all-mpnet-base-v2` sentence-transformer, Ward Agglomerative Hierarchical Clustering, and **Semantic Centroid Matching** (Hungarian algorithm) for label assignment against expert-defined APT taxonomy categories. The result is a 2-level taxonomy with 7 top-level categories and 14 sub-categories, each with a semantically matched label.

---

## The Taxonomy (Key Output)

```
APT ATTACKS TAXONOMY (120 papers, 2021–2026)
│
├── A. Cyber Espionage & Nation-State Attribution
│   ├── A.1  Attribution & Tracking
│   └── A.2  Forensic Investigation
│
├── B. ML-Based Intrusion Detection Systems
│   ├── B.1  Detection & Classification
│   └── B.2  Behavioral Analysis
│
├── C. Malware Analysis & Reverse Engineering
│   ├── C.1  Technique Analysis
│   └── C.2  Attribution & Tracking
│
├── D. Lateral Movement & Command-and-Control Infrastructure
│   ├── D.1  Defense & Mitigation
│   └── D.2  Detection & Classification
│
├── E. Provenance Graphs & Attack Forensics
│   ├── E.1  Forensic Investigation
│   └── E.2  Technique Analysis
│
├── F. Threat Intelligence & Kill Chain Modeling
│   ├── F.1  Attribution & Tracking
│   └── F.2  Defense & Mitigation
│
└── G. Zero-Day Exploits & Vulnerability Analysis
    ├── G.1  Detection & Classification
    └── G.2  Behavioral Analysis
```

> **Note:** The exact taxonomy above is illustrative. Cluster labels are auto-assigned via Semantic Centroid Matching at runtime and depend on the scraped corpus. Run the pipeline to generate the actual output for your dataset.

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────────────────┐
│   scraper.py    │────▶│  preprocess.py   │────▶│     taxonomy_builder.py      │
│                 │     │                  │     │                              │
│ Semantic Scholar│     │ HTML decode      │     │ Sentence-Transformer embed.  │
│ API queries     │     │ URL removal      │     │ (all-mpnet-base-v2, 768-dim) │
│ APT-only filter │     │ Stop-word removal│     │ Ward HAC (Euclidean/L2-norm) │
│ 120 papers      │     │ Lemmatization    │     │ Semantic Centroid Matching   │
│ 2021–2026       │     │ Dual-group filter│     │ Hungarian label assignment   │
└─────────────────┘     └──────────────────┘     └──────────────────────────────┘
         │                       │                            │
         ▼                       ▼                            ▼
apt_papers_raw.csv    apt_papers_clean.csv      final_taxonomy_mapping.csv
                                                apt_dendrogram.png
                                                apt_taxonomy_tree.png
```

---

## Methodology

### Stage 1 — Data Collection (`scraper.py`)

**API**: Semantic Scholar Academic Graph API  
**Queries**: 5 APT-specific query strings rotated across years  
**Years**: 2021–2026 (~20 papers per year)  
**Post-fetch filter**: Dual-group relevance check on each abstract:

```
Group A (APT-level signals):
  "advanced persistent threat", "apt", "nation-state", "cyber espionage",
  "threat actor", "state-sponsored", "apt28", "apt29", "lazarus", etc.

Group B (TTP/technical signals):
  "malware", "lateral movement", "exfiltration", "command and control",
  "backdoor", "kill chain", "persistence", "privilege escalation", etc.

A paper is KEPT only if it matches ≥1 pattern from BOTH groups.
This ensures only genuine APT papers with specific technique discussions pass.
```

---

### Stage 2 — NLP Preprocessing (`preprocess.py`)

**Step-by-step pipeline applied to each abstract:**

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | HTML entity decode | Handle encoded characters |
| 2 | URL & DOI removal | Remove non-content tokens |
| 3 | Non-ASCII normalisation | Remove encoding artefacts |
| 4 | Lowercase + punctuation removal | Normalise text |
| 5 | Standalone number removal | Remove irrelevant numeric tokens |
| 6 | Stop-word removal | Remove 200+ high-frequency low-value words |
| 7 | Min token length (>2 chars) | Remove abbreviations |
| 8 | NLTK lemmatization | Reduce morphological variants |

**Custom domain stop-words** (50 terms removed beyond standard English):
`paper, propose, method, approach, technique, result, evaluate, security, cyber, network, system, data, detection, defense, model, framework, algorithm, tool...`

---

### Stage 3 — Hierarchical Taxonomy Construction (`taxonomy_builder.py`)

#### Embedding: Sentence-Transformer (`all-mpnet-base-v2`)

Each raw abstract is encoded into a **768-dimensional dense vector** using the `all-mpnet-base-v2` sentence-transformer. This is a general-purpose English semantic encoder (not cybersecurity-specific — see Design Notes). Embeddings are **L2-normalised** so that Euclidean distance is monotonic with cosine distance:

```
‖u − v‖² = 2 − 2·cos(u, v)   when ‖u‖ = ‖v‖ = 1
```

This is critical because Ward linkage requires Euclidean distance. Raw abstracts (not cleaned) are used for embedding because sentence-transformers are trained on grammatical text — feeding lemmatised tokens degrades semantic quality.

#### Clustering: Ward Agglomerative Hierarchical Clustering

```
WHY WARD LINKAGE?
─────────────────
K-Means produces flat buckets — NOT a taxonomy.
HDBSCAN drops low-density points as noise — unacceptable for a small corpus.

Ward Agglomerative Clustering builds a dendrogram (tree):
  1. Start: each paper is its own cluster (n=120)
  2. At each step: merge the two clusters whose merger minimises
     the total within-cluster sum of squares (Ward criterion)
  3. Result: a full binary tree from individual papers up to one root
  4. Cut at k=7 to get 7 top-level taxonomy divisions
  5. Sub-cluster each main group internally (k=2) for depth-2 hierarchy

Silhouette score sweep across k=4–10 validates the chosen k.
```

#### Labeling: Semantic Centroid Matching (Hungarian Algorithm)

```
WHY NOT c-TF-IDF / keyword extraction?
───────────────────────────────────────
On a corpus of ~120 papers, c-TF-IDF produces generic repetitive labels
("Machine Learn Threat", "Detection Method Approach") because clusters
lack enough text mass for discriminative term frequency to emerge.

SEMANTIC CENTROID MATCHING (semi-supervised taxonomy alignment):
  1. Define 12 expert gold-standard APT taxonomy labels
     (e.g., "Lateral Movement & C2", "Malware Analysis & Reverse Engineering")
  2. Compute each cluster's centroid (mean embedding vector, re-normalised)
  3. Embed the 12 gold-standard labels with the SAME sentence-transformer
  4. Build a 7×12 cosine similarity matrix (clusters × labels)
  5. Run the Hungarian algorithm (scipy.optimize.linear_sum_assignment)
     to find the globally optimal one-to-one assignment that maximises
     total similarity — no two clusters share a label

This is semi-supervised: the clustering is fully automated (HAC discovers
structure), the labeling uses domain expertise (candidate labels) with
algorithmic selection (cosine + Hungarian).
```

---

## Design Notes

- **`all-mpnet-base-v2` is NOT cybersecurity-specific.** True security models (SecBERT, SecureBERT) are masked-LM BERTs requiring custom mean-pooling and have weaker public validation for sentence-level tasks. mpnet is the pragmatically defensible choice for a small academic corpus. This is disclosed honestly.
- **Raw abstracts are embedded, not cleaned abstracts.** Sentence-transformers are trained on grammatical English with stopwords — feeding them lemmatised token bags destroys contextual signal.
- **Cosine scores below ~0.25 indicate poor label fit.** Check the `taxonomy_builder.log` after each run. Low scores mean the gold-standard label set should be expanded.

---

## Output Files

| File | Description |
|------|-------------|
| `apt_papers_raw.csv` | 120 real papers fetched from Semantic Scholar API |
| `apt_papers_clean.csv` | After NLP cleaning and dual-group relevance filter |
| `final_taxonomy_mapping.csv` | **Main output**: every paper with cluster ID, semantic label, sub-cluster |
| `apt_dendrogram.png` | Ward clustering dendrogram with cluster-coloured branches and legend |
| `apt_taxonomy_tree.png` | Visual hierarchy tree: Root → 7 clusters → 14 sub-clusters |

---

## Installation & Usage

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/apt-taxonomy-pipeline.git
cd apt-taxonomy-pipeline

# 2. Create virtual environment (Python 3.10 or 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install sentence-transformers==2.7.0 networkx==3.2.1

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# 5. Set Semantic Scholar API key (optional but recommended)
export SEMANTIC_SCHOLAR_API_KEY="your_key_here"

# 6. Run pipeline in order
python scraper.py          # ~15 min  → apt_papers_raw.csv
python preprocess.py       # ~1 min   → apt_papers_clean.csv
python taxonomy_builder.py # ~3 min   → taxonomy outputs (first run downloads ~1.1 GB model)

# OR: run everything at once
bash run_pipeline.sh                  # full pipeline
bash run_pipeline.sh --taxonomy-only  # skip scraping (if you have the CSVs)
```

---

## Repository Structure

```
apt-taxonomy-pipeline/
├── README.md                        ← This file
├── requirements.txt                 ← Python dependencies
├── run_pipeline.sh                  ← One-shot pipeline runner
├── scraper.py                       ← Stage 1: Data collection
├── preprocess.py                    ← Stage 2: NLP preprocessing
├── taxonomy_builder.py              ← Stage 3: Embedding, clustering, labeling
├── apt_papers_raw.csv               ← 120 real APT papers (output of scraper)
├── apt_papers_clean.csv             ← Cleaned papers (output of preprocess)
├── final_taxonomy_mapping.csv       ← TAXONOMY OUTPUT (main result)
├── apt_dendrogram.png               ← Ward dendrogram with semantic legend
└── apt_taxonomy_tree.png            ← Hierarchy tree diagram
```

---

## Citation

```bibtex
@misc{apt_taxonomy_2025,
  title   = {Automated Taxonomy of Advanced Persistent Threat (APT) Attacks:
             A Hierarchical Machine Learning Pipeline},
  author  = {Kanik Kumar},
  year    = {2025},
  school  = {BITS Pilani, Pilani Campus},
  note    = {CS F266 Study Project, Supervisor: Prof. Rajesh Kumar}
}
```

---

*Built with: Semantic Scholar API · sentence-transformers · scipy · NLTK · scikit-learn · networkx · matplotlib*
