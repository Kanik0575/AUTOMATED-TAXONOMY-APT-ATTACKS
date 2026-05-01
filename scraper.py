

import os, time, logging, requests, pandas as pd


API_KEY   = os.environ.get("SEMANTIC_SCHOLAR_API_KEY",
                            "s2k-8u5d36z91DTEJL7N1aEzukKwp6S8SXv0eUjaGBMh")
BASE_URL  = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS    = "paperId,title,year,authors,abstract,venue,openAccessPdf"
OUTPUT    = "apt_papers_raw.csv"

TARGET_YEARS    = [2021, 2022, 2023, 2024, 2025, 2026]
TARGET_PER_YEAR = 20          
BATCH_SIZE      = 25
DELAY           = 1.2        

APT_QUERIES = [
    "Advanced Persistent Threat detection machine learning",
    "APT attack campaign malware intrusion",
    "Advanced Persistent Threat provenance graph forensics",
    "nation state cyber espionage threat actor attribution",
    "APT command control lateral movement exfiltration",
]


GROUP_A = [                     
    "advanced persistent threat", "apt ", " apt", "nation-state",
    "nation state", "cyber espionage", "threat actor", "state-sponsored",
    "targeted attack", "apt28", "apt29", "apt41", "lazarus",
]
GROUP_B = [                     
    "malware", "backdoor", "lateral movement", "exfiltration",
    "command and control", " c2 ", "c&c", "spear phishing", "phishing",
    "rootkit", "ransomware", "zero-day", "exploit", "intrusion",
    "persistence", "privilege escalation", "reconnaissance",
    "supply chain", "trojan", "implant", "credential", "kill chain",
    "provenance", "threat intelligence", "attribution",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(),
              logging.FileHandler("scraper.log", mode="w")],
)
log = logging.getLogger(__name__)

HEADERS = {"Accept": "application/json", "x-api-key": API_KEY}


def passes_filter(abstract: str) -> bool:
    """Return True only if abstract has both APT-level AND TTP-level signal."""
    t = (abstract or "").lower()
    return any(k in t for k in GROUP_A) and any(k in t for k in GROUP_B)


def fetch_batch(query: str, year: int, offset: int) -> list:
    """Fetch one batch from Semantic Scholar API with retry logic."""
    params = {"query": query, "fields": FIELDS,
               "year": str(year), "offset": offset, "limit": BATCH_SIZE}
    for attempt in range(1, 6):
        try:
            r = requests.get(BASE_URL, params=params,
                              headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 60 * attempt
                log.warning(f"Rate limited. Waiting {wait}s (attempt {attempt}/5)...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                log.error(f"HTTP {r.status_code} — attempt {attempt}/5")
                time.sleep(DELAY * attempt)
                continue
            return r.json().get("data", [])
        except Exception as e:
            log.warning(f"Error: {e} — retrying ({attempt}/5)")
            time.sleep(DELAY * attempt)
    return []


def collect_year(year: int) -> list:
    """Collect target number of filtered APT papers for one year."""
    log.info(f"\n── Year {year} (target {TARGET_PER_YEAR}) ──")
    seen, results = set(), []

    for qi, query in enumerate(APT_QUERIES):
        if len(results) >= TARGET_PER_YEAR:
            break
        offset = 0
        while len(results) < TARGET_PER_YEAR:
            log.info(f"  Query {qi+1}: '{query[:50]}' | "
                     f"offset={offset} | collected={len(results)}/{TARGET_PER_YEAR}")

            batch = fetch_batch(query, year, offset)
            time.sleep(DELAY)

            if not batch:
                break

            for p in batch:
                pid       = p.get("paperId", "")
                title     = (p.get("title") or "").strip()
                abstract  = (p.get("abstract") or "").strip()
                year_val  = p.get("year")
                venue     = (p.get("venue") or "Unknown").strip()
                authors   = "; ".join(
                    a.get("name","") for a in (p.get("authors") or []))
                oa  = p.get("openAccessPdf") or {}
                url = oa.get("url") or \
                      (f"https://www.semanticscholar.org/paper/{pid}" if pid else "")

                if not pid or pid in seen:       continue
                if not title:                    continue
                if len(abstract) < 100:          continue
                if not passes_filter(abstract):  continue   # ← KEY FILTER

                seen.add(pid)
                results.append({
                    "paper_id": pid, "title": title,
                    "year": int(year_val) if year_val else year,
                    "authors": authors or "Unknown",
                    "abstract": abstract, "venue": venue, "url": url,
                })
                if len(results) >= TARGET_PER_YEAR:
                    break

            offset += BATCH_SIZE
            if offset >= 950:
                break

    log.info(f"  Year {year}: {len(results)} papers collected.")
    return results


def main():
    log.info("=" * 60)
    log.info("  APT Taxonomy — Stage 1: Data Collection")
    log.info(f"  API Key: {API_KEY[:16]}...")
    log.info("=" * 60)

    all_papers = []
    for year in TARGET_YEARS:
        all_papers.extend(collect_year(year))

    df = pd.DataFrame(all_papers)
    if df.empty:
        log.error("No papers collected. Check API key and connection.")
        return

    before = len(df)
    df = df.drop_duplicates(subset="paper_id").reset_index(drop=True)
    log.info(f"\nDeduplication removed {before - len(df)} papers.")
    df = df.sort_values(["year", "title"]).reset_index(drop=True)

    log.info("\n" + "="*60)
    log.info(f"  TOTAL PAPERS COLLECTED: {len(df)}")
    for yr, cnt in df["year"].value_counts().sort_index().items():
        log.info(f"    {yr}: {'█'*cnt} ({cnt})")
    log.info("="*60)

    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    log.info(f"\n  ✓ Saved → {OUTPUT}")
    log.info("  Next: python preprocess.py")


if __name__ == "__main__":
    main()
