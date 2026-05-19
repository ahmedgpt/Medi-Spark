"""
WHO Fact Sheet Scraper
======================
Scrapes all WHO disease fact sheets and saves them to:
    data/medical_knowledge/who_factsheets.json

The RAG engine (app/services/rag_engine.py) automatically ingests this file
alongside MedQuAD when building the ChromaDB vector store.

Usage
-----
    # From the project root:
    python scrappers/who_scraper.py

    # Force re-scrape even if output already exists:
    python scrappers/who_scraper.py --force

Requirements
------------
    pip install requests beautifulsoup4 trafilatura
    (already in requirements if you added them, otherwise install separately)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    print("❌  trafilatura not found. Install it:\n    pip install trafilatura")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.who.int"
FACTSHEET_URL   = "https://www.who.int/news-room/fact-sheets"

# Output goes into the project's medical knowledge folder (used by RAG engine)
PROJECT_ROOT    = Path(__file__).resolve().parent.parent
OUTPUT_DIR      = PROJECT_ROOT / "data" / "medical_knowledge"
OUTPUT_FILE     = OUTPUT_DIR / "who_factsheets.json"

# Scraper behaviour
REQUEST_DELAY   = 1.2   # seconds between requests (be polite to WHO servers)
MAX_RETRIES     = 3
RETRY_BACKOFF   = 2.0   # seconds — doubles on each retry

HEADERS = {
    "User-Agent": (
        "MediSpark-Academic-Scraper/1.0 "
        "(https://github.com/your-repo; educational use only)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DISCOVER FACT-SHEET LINKS
# ══════════════════════════════════════════════════════════════════════════════
def discover_links() -> set[str]:
    """Return all unique WHO fact-sheet URLs found on the index page."""
    print(f"[WHO] Fetching index page: {FACTSHEET_URL}")
    resp = requests.get(FACTSHEET_URL, headers=HEADERS, timeout=20)
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "html.parser")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/news-room/fact-sheets/detail/" in href:
            full_url = urljoin(BASE_URL, href)
            links.add(full_url)

    print(f"[WHO] Found {len(links)} fact-sheet links.")
    return links


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCRAPE ONE PAGE
# ══════════════════════════════════════════════════════════════════════════════
def scrape_page(url: str) -> dict | None:
    """
    Fetch a single WHO fact sheet and extract its clean text.
    Returns None if the page yields no usable content.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
            )
            if not text or len(text.strip()) < 100:
                # Skip pages with almost no content
                return None

            # Try to pull a title from the URL slug
            slug    = url.rstrip("/").split("/")[-1]
            title   = slug.replace("-", " ").title()

            return {
                "title":   title,
                "url":     url,
                "source":  "WHO Fact Sheets",
                "content": text.strip(),
            }

        except Exception as exc:
            wait = RETRY_BACKOFF * attempt
            print(f"  [WARN] Attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    return None  # all retries exhausted


# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN SCRAPE LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run(force: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if data already exists and --force not set
    if OUTPUT_FILE.exists() and not force:
        existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        print(
            f"[WHO] Output already exists with {len(existing)} documents.\n"
            f"      Use --force to re-scrape.\n"
            f"      Path: {OUTPUT_FILE}"
        )
        return

    links    = discover_links()
    all_docs = []
    skipped  = 0

    for i, url in enumerate(sorted(links), 1):
        print(f"[WHO] ({i}/{len(links)}) Scraping: {url}")
        doc = scrape_page(url)

        if doc:
            all_docs.append(doc)
        else:
            skipped += 1
            print(f"  [SKIP] No usable content extracted.")

        time.sleep(REQUEST_DELAY)

    # Save result
    OUTPUT_FILE.write_text(
        json.dumps(all_docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"✅  WHO scraping complete!")
    print(f"    Documents saved : {len(all_docs)}")
    print(f"    Skipped (empty) : {skipped}")
    print(f"    Output file     : {OUTPUT_FILE}")
    print("=" * 60)
    print("\nNext step: rebuild the RAG vector store to include this data:")
    print("    python -m app.services.rag_engine --rebuild")


# ══════════════════════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape WHO disease fact sheets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape even if the output file already exists.",
    )
    args = parser.parse_args()
    run(force=args.force)
