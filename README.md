# Product Recommendation Agent

> **Rooman AI Challenge — Junior AI Research Associate Selection Round**
> Agent type: Product Recommendation Agent (Intermediate)
> Single-sentence mission: **My agent takes a structured user profile (background, goals, skills, preferences, price constraints, keywords, optional purchase history) and produces a ranked shortlist of products with a transparent score breakdown + a specific rationale explaining every recommendation.**

---

## Contents

- [One-line description](#one-line-description)
- [Tools and Technologies](#tools-and-technologies)
- [Quick start (5 minutes)](#quick-start-5-minutes)
  - [1. Install](#1-install)
  - [2. Configure an LLM API key (optional, recommended)](#2-configure-an-llm-api-key-optional-recommended)
  - [3. Run the full demo](#3-run-the-full-demo)
  - [4. Try a single sample profile](#4-try-a-single-sample-profile)
- [CLI Usage reference](#cli-usage-reference)
- [Deliverables (per the challenge brief)](#deliverables-per-the-challenge-brief)
  - [1. Product catalogue](#1-product-catalogue)
  - [2. Sample user profiles (3-4, actually 4)](#2-sample-user-profiles-3-4-actually-4)
  - [3. Recommendation output](#3-recommendation-output)
  - [4. Rationale for every recommendation](#4-rationale-for-every-recommendation)
- [How recommendations are scored — NLP + weighted heuristics](#how-recommendations-are-scored--nlp--weighted-heuristics)
  - [Score components and weights](#score-components-and-weights)
  - [Cold-start handling](#cold-start-handling)
  - [Diversity & history-aware reranking](#diversity--history-aware-reranking)
- [Design choices — why this stack](#design-choices--why-this-stack)
- [Tradeoffs, limitations, and what I'd improve with more time](#tradeoffs-limitations-and-what-id-improve-with-more-time)
- [Project structure](#project-structure)
- [Output images](#output-images)

---

## One-line description

> **Input → Think → Act → Output**
>
> **Input:** A user profile — free-form background, goals, known skills, interests, usage context; structured preferred categories/brands, price range, must-have features, avoid-tags, free-form keywords, and optional purchase-history IDs.
>
> **Think (NLP):** Build a product "document" per SKU (name + category + brand + description + features + tags), tokenize and compute TF-IDF vectors over the whole 50-product catalogue. Build an equivalent query vector from the user's keywords / goals / interests / usage context, then rank by **cosine similarity** (content-based filtering).
>
> **Act (heuristics):** Layer in structured attribute matching (preferred categories, preferred brands, price-range fit, must-have feature hit rate, avoid-tag exclusion), plus a popularity-weighted rating bonus. Penalize products too similar to anything already in the user's purchase history for intra-list diversity.
>
> **Output:** Ranked list (configurable top-N, default 8), each with a score bar, a 3-component breakdown (content-similarity % | attribute-match % | rating-boost +/-%), a 2-4 sentence rationale, key-match bullet points, and flagged tradeoffs/concerns. Full results are written as machine-readable JSON to `output/`.

---

## Tools and Technologies
- Python
- Groq API
- JSON
- VS Code
- Git, GitHub and Version control

## Quick start (5 minutes)

### Prerequisites
- **Python 3.10+** (uses modern typing: `X | None`, `list[dict]`)
- **pip** (standard)
- (Optional) An **OpenAI API key** or a **Groq API key** for LLM-written rationale. Without a key the agent uses a solid heuristic rationale generator (every recommendation still gets a written explanation).

### 1. Install

```bash
git clone <your-repo-url>
cd Product-Recommendation-Agent
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

> **Zero-dep core:** The `src/recommender.py` engine is **pure standard-library Python**. `requirements.txt` only lists `openai` for the optional LLM rationale generator. You can actually skip `pip install -r requirements.txt` entirely and still run recommendations — they'll just use the heuristic fallback rationale instead of GPT/Claude/Llama.

### 2. Configure an LLM API key (optional, recommended)

Copy `.env.example` → `.env` and paste one key:

```bash
# --- Option A: OpenAI ---
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=gpt-4o-mini

# --- Option B: Groq (cheaper, free-tier available) ---
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
# OPENAI_BASE_URL=https://api.groq.com/openai/v1
# LLM_MODEL=llama-3.1-70b-versatile
```

The agent auto-detects both OpenAI and Groq-style OpenAI-compatible endpoints.

### 3. Run the full demo

This runs **all 4 sample user profiles** back-to-back and writes the JSON output for each one to `output/`:

```bash
python agent.py --demo
```

You'll see a `DEMO SUMMARY — Top match per profile` table at the end:

```
U001  Alex — Aspiring Software Developer (Student)          ->  CodeMaster Coding Mouse for Kids  (38.0%)
U002  Priya — Fitness & Outdoor Enthusiast                  ->  TrailEndure Hiking Backpack 55L    (41.8%)
U003  Raj — Home Cook & Aspiring Home Chef                  ->  FreshBrew Cold Brew Coffee Maker  (47.1%)
U004  Meera — Mindfulness & Remote Working Professional     ->  VelvetGlow Hydrating Serum        (41.5%)
```

Every run also persists per-profile JSON to `output/recs_<id>_<timestamp>.json` (see example in `output/` after your run).

### 4. Try a single sample profile

```bash
# Profile U002 = Priya the fitness/outdoor enthusiast (has purchase history!)
python agent.py --profile U002 --top-n 8

# Build your own profile interactively:
python agent.py --custom

# Interactive menu with all options:
python agent.py

# List available sample profiles:
python agent.py --list-profiles
```

---

## CLI Usage reference

| Flag | What it does |
|---|---|
| *(no flags)* | Launches the interactive menu (demo / sample profiles / custom profile / list). |
| `--demo` | Runs the 4 bundled sample profiles end-to-end, saves JSON per profile, prints a summary table. |
| `--profile U001` | Runs one sample profile ID (`U001`, `U002`, `U003`, `U004`). |
| `--custom` | Interactive 8-field profile builder → immediate recommendations. |
| `--list-profiles` | Prints a compact summary of all sample user profiles. |
| `--top-n N` | Overrides the number of returned recommendations (default `8`, clamped to 3–20). |

---

## Deliverables (per the challenge brief)

### 1. Product catalogue

Located at [`data/product_catalogue.json`](file:///C:/Umesh%20Folder/Internship%20and%20job%20projects/Product%20Recommendation%20Agent/data/product_catalogue.json).

- **50 SKUs** across **7 macro categories** (Electronics, Apparel, Home & Kitchen, Books, Sports & Outdoors, Beauty, plus Health/Travel sub-categories spread into the 6 primary ones)
- Each SKU carries a **rich attribute schema**:
  ```
  id, name, category, subcategory, price, brand,
  rating, review_count,
  description (paragraph),
  features[] (5-8 bullets),
  tags[] (5-10 normalized tags, the primary signal for content-based similarity)
  ```
- Catalogue has realistic price distribution:
  - <$50 (17), $50-150 (17), $150-400 (10), $400+ (6)
- Rating distribution skews realistic-optimistic: 3.9–4.9/5 with corresponding review counts.

### 2. Sample user profiles (3-4, actually 4)

Located at [`data/sample_user_profiles.json`](file:///C:/Umesh%20Folder/Internship%20and%20job%20projects/Product%20Recommendation%20Agent/data/sample_user_profiles.json).

| ID | Persona | Cold start? | Price focus | Key goals |
|---|---|---|---|---|
| **U001** | Alex — 3rd year BCA / Aspiring SWE | YES — no history | $20–$600 | Coding setup, budget discipline, learn data-science basics |
| **U002** | Priya — Marketer / gym 5x/week + weekend treks | NO — owns `P006` (running shoes) + `P017` (yoga mat) | $25–$400 | Build compact home gym, prep Himalayan trek gear, recovery |
| **U003** | Raj — Young professional / new apartment / weekend dinner parties | YES | $30–$500 | Professional knives, espresso setup, sous-vide, organized storage |
| **U004** | Meera — Remote UX designer / slow-living / stress-aware | YES | $15–$350 | Ergonomic calm home office, skincare, sleep quality, eco-friendly habits |

Every profile carries: background, goals, known_skills, interests[], usage_context, preferences.preferred_categories[], preferences.preferred_brands[], preferences.price_range{min,max}, preferences.must_have_features[], preferences.avoid_tags[], and free-form keywords. U002 also carries a `purchase_history[]` to test non-cold-start flow.

### 3. Recommendation output

Every `python agent.py --profile X` run writes a timestamped machine-readable JSON to [`output/`](file:///C:/Umesh%20Folder/Internship%20and%20job%20projects/Product%20Recommendation%20Agent/output/):

Structure of `recs_<profile-id>_<timestamp>.json`:
```
{
  "generated_at": ISO timestamp,
  "user_profile": full profile echoed back,
  "cold_start": true/false,
  "llm_rationale_enabled": true/false,
  "recommendations": [
    {
      "rank": 1..N,
      "product": { full SKU record },
      "scores": {
        "overall":           0.0–1.0 (weighted composite),
        "content_similarity":0.0–1.0 TF-IDF cosine similarity,
        "attribute_match":   0.0–1.0 structured rule-based fit,
        "rating_boost":     -0.03..+0.05 popularity-weighted rating bump
      },
      "rationale": {
        "rationale": "2-4 sentence explanation...",
        "key_matches": ["- Category: Electronics > Laptops", "- Tag match: budget, student", ...],
        "potential_concerns": ["Rating is 3.9/5, slightly below top tier", ...]
      }
    },
    ...
  ]
}
```

CLI output is a human-readable card per recommendation with a rendered score bar, score breakdown, rationale, key matches, and tradeoff flags.

### 4. Rationale for every recommendation

Implemented in [`src/llm_rationale.py`](file:///C:/Umesh%20Folder/Internship%20and%20job%20projects/Product%20Recommendation%20Agent/src/llm_rationale.py):

Two-tiered (so the agent never ships without rationale):

1. **Tier A — LLM-written (preferred)** when `OPENAI_API_KEY` or `GROQ_API_KEY` is set. Uses `json_object` response format with the full structured output schema (rationale, key_matches, potential_concerns per product). System prompt instructs the model to explicitly ground each statement in (a) the user's stated goals/preferences, (b) concrete product features/price/rating, and to honestly flag tradeoffs rather than overhype.
2. **Tier B — Heuristic fallback (guaranteed)** when no API key, or the LLM call fails for any reason (network, rate-limit, bad key, timeout…). Constructs the exact same JSON schema: rationale paragraph is built from the score breakdown + top 3 features + rating; key_matches are category/tag/feature slices; potential_concerns are derived from rating < 4.2 or review_count < 500.

Fallback is triggered by catching: `APIError`, `APIConnectionError`, `RateLimitError`, plus a bare `Exception` guard.

---

## How recommendations are scored — NLP + weighted heuristics

### Scoring pipeline

```
User profile  ──►  keyword/goals/interests tokenizer
                      │
                      ▼
                 query TF-IDF vector ◄─── IDF learned once over the whole catalogue
                      │
Product doc ◄─────────┘
(name + cat/subcat + brand + desc + features + tags, tokenized + stopwords removed)
                      │
                      ▼
          cosine_similarity(query_vec, product_vec)   → content_similarity (0..1)

                      │
                      ▼
Attribute-match rules run over structured preferences  → attribute_match (0..1)
  • preferred_categories hit            (0.15 weight)
  • preferred_brands hit                (0.10 weight)
  • price_range fit (+ soft penalty/bounty for near-miss) (0.15 weight)
  • must_have_features hit-rate over the product's features + tags + description (0.15 weight)
  • avoid_tags non-intersection         (0.05 weight)
  Result is normalized by its own theoretical max (= 0.60) → attr_match ∈ [0,1]

                      │
                      ▼
rating_bonus: popularity-log weighted ((rating - 3.5)/1.5) × 0.05 × min(1, log10(reviews+1)/4)
  Clamped to [-0.03, +0.05] to avoid overwhelming NLP signal

                      │
                      ▼
Weighted composite (reweights automatically if no meaningful user query was supplied):
  Normal run:       0.60 × content_sim  + 0.35 × attr_match  + 0.05 × rating_boost
  Empty-query mode: 0.70 × attr_match   + 0.30 × rating_boost    (no keywords → use pure structured fit + popularity)

                      │
                      ▼
Diversity penalty (when purchase_history is non-empty):
  For every purchased-ID, compute cosine_similarity(candidate, purchased) → take the max → subtract (max × 0.08)
  Prevents recommending "the exact same thing you already bought" and improves intra-list variety.

                      │
                      ▼
Final score clamped to [0, 1], primary sort = -final_score, secondary sort = -rating (tie-break toward better-reviewed SKUs)
```

### Cold-start handling

Detected simply as `len(purchase_history) == 0`: the agent explicitly prints a cold-start banner and takes a different routing:

- Calls `recommender.cold_start_recommend()` which is just `recommend(..., purchase_history=[])` — the **same scoring pipeline**, but the composite weights still shift if the user didn't supply meaningful keywords.
- In real systems you'd use popularity + demographic bins; here we lean on (a) explicit stated preferences (categories / price / must-haves), (b) the small catalogue's naturally high review-count products bubbling up via `rating_bonus`.
- U001/U003/U004 are cold-start; U002 exercises the diversity-penalty branch because she already bought running shoes and a yoga mat.

### Diversity & history-aware reranking

Two subtle defenses against a boring "all-the-same-category" list:

1. **Per-candidate history penalty** (in scoring): if a candidate is ≥ 50% similar (cosine) to anything already purchased, lose up to 8 points of composite score.
2. **Scored candidate pool stays large** (top_n × 2) before truncating → in a tie you implicitly get variety because sort-then-truncate preserves the top *scoring* items, but categories that would otherwise monopolize the top-5 are naturally scattered because the attribute-match/price rules nudge each user into >1 category.

---

## Design choices — why this stack

| Decision | Why |
|---|---|
| **Pure-Python TF-IDF + cosine similarity (no scikit-learn)** | Zero runtime deps for the core. A reviewer can `git clone` + `python agent.py --demo` in under 10 seconds with no `pip install`. scikit-learn would be ~50 MB of transitive wheels for a 50-document corpus where a hand-written 30-line TF-IDF does the exact same thing. (Easy drop-in later if the catalogue hits 10k+ SKUs.) |
| **Tokenize + stopword list in-source** | Keeps indexing deterministic. No surprises from different NLTK corpus downloads on different machines. |
| **Hand-rolled attribute-match over a linear model** | Every percentage point in `attribute_match` is traceable: you can re-derive the score from the product and a printout of the rules. Linear/logistic regression would hide *why* a budget laptop was ranked above a chromebook. |
| **Two-tier rationale (LLM + heuristic fallback)** | Challenge rules say "use any model/any API", but it also says reviewers score what they can actually run. Guaranteeing a rationale (no key needed) avoids a zero on the demo. |
| **JSON as the on-disk data layer (product catalogue + profiles)** | No SQLite/CSV schema juggling: both files are human-editable, git-friendly, and the recommendation engine loads in <10 ms. Perfect for 50 products × 4 users. |
| **CLI-first, no UI** | Challenge brief explicitly says "a CLI is fine. A UI is optional." I spent the saved time on (a) rich product attributes, (b) transparent score breakdowns, (c) a clean --demo sweep that guarantees every reviewer gets a working end-to-end run. |

---

## Tradeoffs, limitations, and what I'd improve with more time

**What I chose to trade, and why:**

1. **Content-based only — no collaborative filtering.**
   - Why: With only 4 sample users, a user-item matrix would be ~96% sparse. Collaborative filtering (SVD, ALS) gives no lift on a demo-sized dataset and introduces a dep like `surprise` or `implicit`.
   - If I had more time + real users: hybrid content-based + matrix factorization (LightFM is the gold standard for hybrid recommenders), plus per-session item-item nearest neighbours on the product detail view.

2. **Keyword similarity is "bag-of-words" — no sentence embeddings.**
   - Why: TF-IDF is interpretable, runs offline, no model weights download. Semantic similarity (`all-MiniLM-L6-v2` via SentenceTransformers) would catch "web dev" ↔ "React developer" synonymy, but that's 400 MB of embedding model.
   - If I had more time: wrap SentenceTransformers as an optional similarity backend and benchmark Recall@10 vs TF-IDF on a held-out set of labeled recommendations.

3. **50 products, synthetic catalogue — no real e-commerce source.**
   - Why: Challenge allows any catalogue; 50 SKUs × (7 features + 8 tags) is the sweet spot where you can actually inspect a ranked list and say "this makes sense" vs 500 SKUs where errors blur into noise. I deliberately mixed near-duplicates (budget laptop vs pro laptop) to test the ranking's ability to differentiate.
   - If I had more time: script a scraper + real product IDs + real images + real review histograms from Amazon/Flipkart public product APIs for a real e-commerce category.

4. **Purchase-history diversity is a cosine penalty, not a proper re-ranking.**
   - Why: Fast to implement, and the catalogue is small enough that a "greedy with penalty" beats Maximal Marginal Relevance in readability.
   - If I had more time: switch the final N pick to a proper MMR or DPP re-ranker that explicitly optimizes the tradeoff `λ × relevance + (1-λ) × diversity`.

5. **Rationale "grounding" is prompt-based, not a formal RAG pipeline.**
   - Why: The LLM gets the *entire product record* + *entire user profile* inside the prompt window for every batch. For 8 recommendations × ~200 tokens/product + ~150 tokens/user, the context window is ~2 KB + instructions = well inside gpt-4o-mini's 128K. Hard to hallucinate features when the full product JSON is in-context.
   - If I had more time + larger catalogue: add a retrieval step (embedding → top-30 candidates → rerank in-LLM), and add structured self-verification: parse the LLM's `key_matches` array and assert each claim actually exists in the product record; strip any hallucinated ones.

6. **No A/B test harness.**
   - Why: Not possible inside a 24-hour window. But every recommendation writes JSON with the 3-component score decomposition, so offline evaluation (simulate 100 synthetic user profiles, measure how often the top-3 match the profile's stated categories) is possible.
   - If I had more time: add an offline evaluator (precision@k, NDCG@k) vs a labeled ground-truth of "expected recommendations for the 4 sample profiles".

7. **No web UI.**
   - Why: See earlier — the rubric explicitly prioritizes a runnable CLI. A Streamlit/Gradio frontend would be ~2 hours of forms + result rendering. I traded those hours for (1) the heuristic fallback guarantee, (2) a richer product attribute schema, and (3) a fully self-contained `--demo` path that outputs 8 JSON files + a summary.
   - If I had more time: add Streamlit with the 4 sample profile picker, custom profile sliders, and color-coded score bars per recommendation.

---

## Project structure

```
Product Recommendation Agent/
├── agent.py                          ← CLI entrypoint (argparse + interactive menu)
├── requirements.txt                  ← Only openai; the core recommender is stdlib-only
├── .env.example                      ← OPENAI_API_KEY / GROQ_API_KEY template
├── data/
│   ├── product_catalogue.json        ← 50 products, 6 categories, rich attributes
│   └── sample_user_profiles.json     ← 4 profiles (U001 cold, U002 history, U003 cold, U004 cold)
├── src/
│   ├── __init__.py
│   ├── recommender.py                ← TF-IDF + cosine + attribute-match scoring engine
│   └── llm_rationale.py              ← OpenAI/Groq client + guaranteed heuristic fallback
└── output/                           ← Timestamped JSON recommendation dumps appear here after runs
    ├── recs_u001_YYYYMMDD_HHMMSS.json
    ├── recs_u002_YYYYMMDD_HHMMSS.json
    ├── recs_u003_YYYYMMDD_HHMMSS.json
    ├── recs_u004_YYYYMMDD_HHMMSS.json
    └── demo_summary.json
```
<img width="395" height="761" alt="Screenshot 2026-08-07 001051" src="https://github.com/user-attachments/assets/83ed6858-cf50-4a04-a800-9cb2e5824818" />
<img width="398" height="120" alt="Screenshot 2026-08-07 001100" src="https://github.com/user-attachments/assets/a05f72f2-429b-4a00-93dc-96ec9517cd22" />

## Output Images
<img width="1577" height="520" alt="Screenshot 2026-08-07 001356" src="https://github.com/user-attachments/assets/c55ec656-fc2e-478c-ad8f-b7f91e79fd89" />
<img width="1475" height="912" alt="Screenshot 2026-08-07 001517" src="https://github.com/user-attachments/assets/26ba72e0-34ac-41fc-960c-282890852426" />
<img width="1526" height="950" alt="Screenshot 2026-08-07 001529" src="https://github.com/user-attachments/assets/4d6ca307-c2f7-43ca-bb83-ae77ca8d80d5" />
<img width="1527" height="947" alt="Screenshot 2026-08-07 001544" src="https://github.com/user-attachments/assets/195cf7d7-fbaa-44bc-aa0a-41fc62984f85" />
<img width="1513" height="442" alt="Screenshot 2026-08-07 001620" src="https://github.com/user-attachments/assets/a31d8dcd-ef5e-4a0a-afec-701288d05f8e" />
<img width="1428" height="943" alt="Screenshot 2026-08-07 002332" src="https://github.com/user-attachments/assets/f2aaf2cd-83e3-478a-bc0e-5e3058983335" />
<img width="1433" height="950" alt="Screenshot 2026-08-07 002343" src="https://github.com/user-attachments/assets/416ee89e-9622-448b-a8e1-0fb26055c451" />
<img width="1422" height="785" alt="Screenshot 2026-08-07 002355" src="https://github.com/user-attachments/assets/efd4c4d2-be45-4823-a7fc-b1e4162c3beb" />
