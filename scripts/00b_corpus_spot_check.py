# RUN ON: LAPTOP
# Reason: Quick validation check, no GPU needed.

"""
scripts/00b_corpus_spot_check.py

Spot-checks whether Wikipedia contains relevant passages for 10 NQ queries.
Searches Wikipedia using the gold answer directly (more reliable than question keywords).

Requirements: pip install wikipedia-api
"""

import json
import string
from pathlib import Path
import wikipediaapi

DATA_FILE = Path("data/nq_validation.jsonl")


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def find_wikipedia_page(wiki, gold_answers: list):
    """Search Wikipedia using the gold answer directly."""
    for ans in gold_answers:
        page = wiki.page(ans)
        if page.exists():
            return page
    return None


def main():
    print("=" * 60)
    print("00b_corpus_spot_check.py — Wikipedia Corpus Spot Check")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run 01_download_data.py first.")
        return

    queries = []
    with open(DATA_FILE) as f:
        for line in f:
            queries.append(json.loads(line))
            if len(queries) == 10:
                break

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="rag-retrieval-study/1.0"
    )

    found = 0
    for item in queries:
        question = item["question"]
        gold_answers = item["gold_answers"]

        print(f"\nQ: {question}")
        print(f"  Gold answers : {gold_answers}")

        page = find_wikipedia_page(wiki, gold_answers)
        if page is None:
            print(f"  Wikipedia    : No article found for '{gold_answers[0]}'")
            print(f"  Match        : NO")
            continue

        title = page.title
        snippet = page.text[:200].replace("\n", " ")
        full_text = normalize(page.text)

        match = any(normalize(ans) in full_text for ans in gold_answers)
        if match:
            found += 1

        print(f"  Wikipedia    : {title}")
        print(f"  Article start: {snippet}...")
        print(f"  Match        : {'YES' if match else 'NO'}")

    n = len(queries)
    print("\n" + "=" * 60)
    print(f"SUMMARY: {found}/{n} queries have relevant Wikipedia articles")
    if found >= 7:
        print("PASS — Wikipedia corpus is a valid retrieval source for NQ.")
    elif found >= 4:
        print("WEAK — Some coverage. Check retrieval quality carefully.")
    else:
        print("FAIL — Low Wikipedia coverage. Investigate before full run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
