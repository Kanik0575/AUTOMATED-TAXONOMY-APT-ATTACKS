

import re, html, logging
import pandas as pd
from pathlib import Path


import nltk
for pkg, path in [("stopwords","corpora/stopwords"),
                   ("wordnet",  "corpora/wordnet"),
                   ("omw-1.4",  "corpora/omw-1.4")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

INPUT  = "apt_papers_raw.csv"
OUTPUT = "apt_papers_clean.csv"
MIN_WORDS = 15   # minimum content words after NLP cleaning


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("preprocess.log", mode="w")],
)
log = logging.getLogger(__name__)


DOMAIN_STOP = {
    "paper","propose","proposed","present","study","approach","method",
    "technique","result","show","demonstrate","evaluate","analysis","analyze",
    "based","use","used","using","work","novel","new","existing","previous",
    "recent","research","provide","achieve","improve","effective","efficient",
    "different","various","multiple","include","enable","make","significant",
    "important","security","cyber","network","system","data","information",
    "user","attack","detection","detect","defense","model","framework",
    "algorithm","tool","provide","high","low","real","time","case","type",
    "level","number","rate","well","due","via","two","three","one","first",
    "second","large","small","find","found","address","develop",
}
ALL_STOP  = set(stopwords.words("english")) | DOMAIN_STOP
lemmatizer = WordNetLemmatizer()


GROUP_A = [                           
    r"\bapt\b",
    r"advanced persistent threat",
    r"nation.?state",
    r"cyber.?espionage",
    r"threat\s+actor",
    r"threat\s+group",
    r"targeted\s+attack",
    r"state.?sponsored",
    r"threat\s+intelligence",
    r"apt2[0-9]|apt41|lazarus|turla|cozy\s+bear|fancy\s+bear",
    r"intrusion\s+set",
    r"\bcampaign\b",
]
GROUP_B = [                           
    r"\bmalware\b",
    r"\bexploit\b",
    r"lateral\s+movement",
    r"\bexfiltrat",
    r"command.and.control",
    r"\bc2\b",
    r"spear.?phishing",
    r"\bphishing\b",
    r"\bbackdoor\b",
    r"\bransomware\b",
    r"zero.?day",
    r"\bvulnerabilit",
    r"\bintrusion\b",
    r"privilege\s+escalation",
    r"reconnaissance",
    r"supply\s+chain",
    r"\bpersistence\b",
    r"\btrojan\b",
    r"\bimplant\b",
    r"kill\s+chain",
    r"\bcredential",
    r"provenance\s+graph",
    r"threat\s+hunting",
]


def relevance_filter(abstract: str) -> bool:
    """Return True if abstract has ≥1 Group A match AND ≥1 Group B match."""
    t = (abstract or "").lower()
    has_a = any(re.search(p, t) for p in GROUP_A)
    has_b = any(re.search(p, t) for p in GROUP_B)
    return has_a and has_b


def clean_text(text: str) -> str:
    """Full NLP cleaning pipeline: decode → normalise → tokenise → filter → lemmatize."""
    if not isinstance(text, str): return ""
    text = html.unescape(text)                          
    text = re.sub(r"https?://\S+|doi:\S+", " ", text)  
    text = re.sub(r"[^\x00-\x7F]+", " ", text)         
    text = re.sub(r"[^\w\s\-]", " ", text)             
    text = re.sub(r"\b\d+\b", " ", text)                
    text = re.sub(r"\s+", " ", text).strip().lower()    
    tokens = []
    for tok in text.split():
        tok = tok.strip("-_")
        if len(tok) <= 2:         continue              
        if tok in ALL_STOP:       continue              
        if not re.match(r"^[a-z][a-z0-9\-]*$", tok): continue
        lemma = lemmatizer.lemmatize(tok, pos="v")      
        if lemma == tok:
            lemma = lemmatizer.lemmatize(tok, pos="n")  
        tokens.append(lemma)
    return " ".join(tokens)


def main():
    log.info("=" * 60)
    log.info("  APT Taxonomy — Stage 2: NLP Preprocessing")
    log.info("=" * 60)

    if not Path(INPUT).exists():
        log.error(f"'{INPUT}' not found. Run scraper.py first.")
        return

    df = pd.read_csv(INPUT, encoding="utf-8")
    log.info(f"Loaded {len(df)} papers from '{INPUT}'")
    log.info(f"Year distribution: {df['year'].value_counts().sort_index().to_dict()}")

    # Step 1 — Drop missing/short abstracts
    df = df.dropna(subset=["abstract"])
    df = df[df["abstract"].str.strip().str.len() > 80].copy()
    log.info(f"\nStep 1 (abstract quality filter): {len(df)} papers remain")

    # Step 2 — Dual-group relevance filter
    log.info("\nStep 2 — Applying dual-group relevance filter...")
    df["_pass"] = df["abstract"].apply(relevance_filter)
    log.info(f"  Papers failing filter: {(~df['_pass']).sum()}")
    df = df[df["_pass"]].drop(columns=["_pass"]).copy()
    log.info(f"  Papers passing filter: {len(df)}")

    # Step 3 — NLP cleaning
    log.info("\nStep 3 — Applying NLP cleaning pipeline...")
    df["abstract_clean"] = df["abstract"].apply(clean_text)

    # Step 4 — Minimum word count
    df["_wc"] = df["abstract_clean"].str.split().str.len()
    df = df[df["_wc"] >= MIN_WORDS].drop(columns=["_wc"]).copy()
    log.info(f"Step 4 (min {MIN_WORDS} content words): {len(df)} papers remain")

    # Step 5 — Title deduplication
    before = len(df)
    df["_tl"] = df["title"].str.lower().str.strip()
    df = df.drop_duplicates("_tl").drop(columns=["_tl"]).reset_index(drop=True)
    if len(df) < before:
        log.info(f"Step 5 (title dedup): removed {before-len(df)}")

    # Output
    out_cols = [c for c in ["paper_id","title","year","authors",
                             "venue","url","abstract","abstract_clean"]
                if c in df.columns]
    df[out_cols].to_csv(OUTPUT, index=False, encoding="utf-8")

    log.info("\n" + "="*60)
    log.info("  PREPROCESSING SUMMARY")
    log.info(f"  Final papers: {len(df)}")
    for yr, cnt in df["year"].value_counts().sort_index().items():
        log.info(f"    {yr}: {'█'*cnt} ({cnt})")
    log.info(f"\n  ✓ Saved → {OUTPUT}")
    log.info("  Next: python taxonomy_builder.py")
    log.info("="*60)


if __name__ == "__main__":
    main()
