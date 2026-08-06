#!/usr/bin/env python3
"""Product Recommendation Agent — CLI entrypoint.

Usage:
    python agent.py                      # Interactive mode
    python agent.py --demo               # Run all 4 sample profiles & save JSON
    python agent.py --profile U001       # Run a specific sample profile
    python agent.py --custom             # Build a profile interactively
    python agent.py --list-profiles      # List sample user profiles
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.recommender import ProductRecommender
from src.llm_rationale import RationaleGenerator

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROFILES_PATH = DATA_DIR / "sample_user_profiles.json"

SEPARATOR = "=" * 78
THIN_SEP = "-" * 78
ROBOT = "[ROBOT]"
BULLET_OK = "[OK]"
BULLET_WARN = "[WARN]"
BULLET_INFO = "[i]"
BULLET_SAVE = "[SAVE]"
CHECK = "[+]"
CAUTION = "[!]"
BYE = "[BYE]"


def load_sample_profiles() -> List[Dict[str, Any]]:
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["profiles"]


def print_header(text: str) -> None:
    print()
    print(SEPARATOR)
    print(f"  {text}")
    print(SEPARATOR)


def score_bar(score: float, width: int = 28) -> str:
    filled = int(round(score * width))
    return "█" * filled + "░" * (width - filled)


def print_product_card(
    rank: int,
    rec: Dict[str, Any],
    rationale_data: Dict[str, Any] | None,
) -> None:
    p = rec["product"]
    overall = rec["score"]
    pid = p["id"]
    r = rationale_data.get(pid, {}) if rationale_data else {}

    print(f"\n  #{rank:>2}  {p['name']}  [${p['price']:.2f}]  |  {p['rating']:.1f}/5 ({p['review_count']} rev)")
    print(f"       Brand: {p['brand']}   |   {p['category']} > {p['subcategory']}")
    print(f"       Score: {score_bar(overall)} {overall:6.1%}")
    print(f"           -> content-similarity {rec['similarity_score']:5.1%}  |  attribute-match {rec['attribute_score']:5.1%}  |  rating-boost {rec['rating_boost']:+5.1%}")
    if r:
        rationale_text = r.get("rationale", "")
        if rationale_text:
            print(f"\n       Why it's a match:\n         {rationale_text}")
        key_matches = r.get("key_matches", [])
        if key_matches:
            print(f"       {CHECK} Key matches: {', '.join('- ' + km for km in key_matches)}")
        concerns = r.get("potential_concerns", [])
        if concerns:
            print(f"       {CAUTION} Trade-offs: {', '.join('- ' + c for c in concerns)}")
    print(f"\n       Tags: {', '.join('#' + t for t in p['tags'][:8])}")
    print(f"       Top features: {', '.join(p['features'][:4])}")
    print(THIN_SEP)


def print_user_profile(profile: Dict[str, Any]) -> None:
    print(f"\n  User  : {profile.get('name', 'Unnamed')}")
    if "background" in profile:
        print(f"  About : {profile['background']}")
    if "goals" in profile:
        print(f"  Goals : {profile['goals']}")
    if "interests" in profile:
        interests = profile["interests"]
        if isinstance(interests, list):
            interests = ", ".join(interests)
        print(f"  Interests : {interests}")
    prefs = profile.get("preferences", {})
    pref_bits = []
    if prefs.get("preferred_categories"):
        pref_bits.append(f"categories={','.join(prefs['preferred_categories'])}")
    if prefs.get("price_range"):
        pr = prefs["price_range"]
        pref_bits.append(f"price=${pr.get('min', 0)}-${pr.get('max', 'N/A')}")
    if prefs.get("must_have_features"):
        pref_bits.append(f"must_have=({','.join(prefs['must_have_features'])})")
    if pref_bits:
        print(f"  Prefs : {' | '.join(pref_bits)}")
    if profile.get("purchase_history"):
        print(f"  History (already purchased, excluded from ranking): {', '.join(profile['purchase_history'])}")


def build_custom_profile() -> Dict[str, Any]:
    print_header("Build Your Recommendation Profile")
    print("  (Press Enter to skip any field.)\n")
    name = input("  Your name / title (e.g., 'B.M. - Grad Student'): ").strip()
    background = input("  Your background (studies, job, experience): ").strip()
    goals = input("  Your goals with these purchases: ").strip()
    interests = input("  Your interests (comma-separated): ").strip()
    categories_raw = input("  Preferred categories (comma-separated, e.g. Electronics,Books): ").strip()
    brands_raw = input("  Preferred brands (comma-separated, optional): ").strip()
    min_p_str = input("  Min price (USD, default 0): ").strip() or "0"
    max_p_str = input("  Max price (USD, default 2000): ").strip() or "2000"
    must_have_raw = input("  Must-have features (comma-separated, optional): ").strip()
    keywords = input("  Free-form keywords (e.g., 'budget laptop coding'): ").strip()
    context = input("  Usage context (when/how you'll use these): ").strip()

    interests_list = [s.strip() for s in interests.split(",") if s.strip()]
    cats = [s.strip() for s in categories_raw.split(",") if s.strip()]
    brands = [s.strip() for s in brands_raw.split(",") if s.strip()]
    must_have = [s.strip() for s in must_have_raw.split(",") if s.strip()]
    try:
        min_p = float(min_p_str)
    except ValueError:
        min_p = 0.0
    try:
        max_p = float(max_p_str)
    except ValueError:
        max_p = 2000.0

    return {
        "id": "CUSTOM",
        "name": name or "Custom Profile",
        "background": background,
        "goals": goals,
        "interests": interests_list,
        "usage_context": context,
        "preferences": {
            "preferred_categories": cats,
            "preferred_brands": brands,
            "price_range": {"min": min_p, "max": max_p},
            "must_have_features": must_have,
        },
        "keywords": keywords,
    }


def run_profile(
    profile: Dict[str, Any],
    recommender: ProductRecommender,
    generator: RationaleGenerator,
    top_n: int = 8,
    save_output: bool = True,
) -> Dict[str, Any]:
    print_header(f"Recommendations for -> {profile.get('name', 'Unnamed')}")
    print_user_profile(profile)

    purchase_history = profile.get("purchase_history", [])
    cold_start = len(purchase_history) == 0
    if cold_start:
        print(f"\n  {BULLET_INFO} Cold-start detected: no purchase history. Will use explicit profile preferences + catalogue popularity.")
        recs = recommender.cold_start_recommend(profile, top_n=top_n)
    else:
        recs = recommender.recommend(profile, top_n=top_n, purchase_history=purchase_history)

    print(f"\n  Generating recommendation rationale ...")
    rationale_data = generator.generate_rationales(profile, recs)
    print("  Done.\n")

    for i, rec in enumerate(recs, 1):
        print_product_card(i, rec, rationale_data.get("rationales", {}))

    print(f"\n  Summary:")
    print(f"    Products ranked : {len(recs)}")
    print(f"    Cold-start mode : {'Yes' if cold_start else 'No (used history for diversity)'}")
    print(f"    LLM rationale   : {'Yes (GPT-powered)' if generator.is_available() else 'No (heuristic fallback -- set OPENAI_API_KEY/GROQ_API_KEY to enable)'}")
    print(f"    Top match score : {recs[0]['score']:.1%} -- {recs[0]['product']['name']}")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "user_profile": profile,
        "cold_start": cold_start,
        "llm_rationale_enabled": generator.is_available(),
        "recommendations": [
            {
                "rank": i + 1,
                "product": rec["product"],
                "scores": {
                    "overall": rec["score"],
                    "content_similarity": rec["similarity_score"],
                    "attribute_match": rec["attribute_score"],
                    "rating_boost": rec["rating_boost"],
                },
                "rationale": rationale_data.get("rationales", {}).get(rec["product"]["id"], {}),
            }
            for i, rec in enumerate(recs)
        ],
    }

    if save_output:
        OUTPUT_DIR.mkdir(exist_ok=True)
        pid = profile.get("id", "profile").lower()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"recs_{pid}_{stamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n  {BULLET_SAVE} Full output saved -> {out_path}")

    return result


def run_demo(recommender: ProductRecommender, generator: RationaleGenerator) -> None:
    profiles = load_sample_profiles()
    print_header("DEMO MODE -- Running all 4 sample user profiles")
    print(f"  Profiles loaded: {len(profiles)}")
    if not generator.is_available():
        print(f"  {BULLET_WARN} LLM API key not configured -- rationale will use heuristic fallback.")
        print("     Set OPENAI_API_KEY or GROQ_API_KEY in a .env file to enable.")
    all_results = {}
    for profile in profiles:
        res = run_profile(profile, recommender, generator, top_n=8, save_output=True)
        all_results[profile["id"]] = {
            "user_name": profile.get("name"),
            "top_match": {
                "name": res["recommendations"][0]["product"]["name"],
                "score": res["recommendations"][0]["scores"]["overall"],
            },
        }
    print_header("DEMO SUMMARY -- Top match per profile")
    for pid, info in all_results.items():
        print(f"  {pid}  {info['user_name']:<45} ->  {info['top_match']['name']}  ({info['top_match']['score']:.1%})")
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path = OUTPUT_DIR / "demo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  {BULLET_SAVE} Summary saved -> {summary_path}")


def list_profiles_action() -> None:
    profiles = load_sample_profiles()
    print_header("Available Sample User Profiles")
    for p in profiles:
        print(f"\n  [{p['id']}]  {p['name']}")
        if p.get("goals"):
            print(f"         Goals: {p['goals']}")
        prefs = p.get("preferences", {})
        pr = prefs.get("price_range")
        if pr:
            print(f"         Budget: ${pr.get('min', 0)}-${pr.get('max', 'N/A')}  |  Categories: {', '.join(prefs.get('preferred_categories', []))}")


def interactive_mode(recommender: ProductRecommender, generator: RationaleGenerator) -> None:
    print_header(f"{ROBOT} Product Recommendation Agent -- Interactive Mode")
    print("  Type a number or a letter + Enter to choose. Press Q to quit.\n")
    profiles = load_sample_profiles()
    while True:
        print("\n  1) Use a sample profile")
        print("  2) Build my own profile")
        print("  3) Run full demo (all 4 profiles)")
        print("  4) List profiles")
        print("  Q) Quit")
        choice = input("\n  Choose: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            print(f"\n  {BYE} Goodbye!")
            sys.exit(0)
        elif choice == "1":
            for p in profiles:
                print(f"    {p['id']} -- {p['name']}")
            sel = input("  Enter profile ID: ").strip().upper()
            match = next((p for p in profiles if p["id"] == sel), None)
            if match:
                try:
                    tn = int(input("  How many recommendations (default 8): ").strip() or "8")
                except ValueError:
                    tn = 8
                run_profile(match, recommender, generator, top_n=max(3, min(tn, 20)))
            else:
                print(f"  Unknown profile ID: {sel}")
        elif choice == "2":
            custom = build_custom_profile()
            try:
                tn = int(input("  How many recommendations (default 8): ").strip() or "8")
            except ValueError:
                tn = 8
            run_profile(custom, recommender, generator, top_n=max(3, min(tn, 20)))
        elif choice == "3":
            run_demo(recommender, generator)
        elif choice == "4":
            list_profiles_action()
        else:
            print("  Invalid choice. Try 1-4 or Q.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Product Recommendation Agent")
    parser.add_argument("--demo", action="store_true", help="Run demo across all 4 sample profiles")
    parser.add_argument("--profile", type=str, default=None, help="Run a specific sample profile ID (U001..U004)")
    parser.add_argument("--custom", action="store_true", help="Build a profile interactively and get recommendations")
    parser.add_argument("--list-profiles", action="store_true", help="List sample user profiles")
    parser.add_argument("--top-n", type=int, default=8, help="Number of recommendations to return (default 8)")
    args = parser.parse_args()

    print(f"{ROBOT} ROOMAN AI -- Product Recommendation Agent (Intermediate)")
    print(f"   Initializing recommender and rationale generator ...")

    recommender = ProductRecommender()
    generator = RationaleGenerator()
    print(f"   Catalogue loaded: {len(recommender.products)} products across {len(recommender.catalogue_data['categories'])} categories")
    print(f"   LLM API: {'Available (' + generator.model + ')' if generator.is_available() else 'Not configured (using heuristic fallback)'}")

    if args.list_profiles:
        list_profiles_action()
    elif args.demo:
        run_demo(recommender, generator)
    elif args.profile:
        profiles = load_sample_profiles()
        sel = args.profile.strip().upper()
        match = next((p for p in profiles if p["id"] == sel), None)
        if match:
            run_profile(match, recommender, generator, top_n=args.top_n)
        else:
            print(f"ERROR: No profile with ID {sel}")
            list_profiles_action()
            sys.exit(1)
    elif args.custom:
        custom = build_custom_profile()
        run_profile(custom, recommender, generator, top_n=args.top_n)
    else:
        interactive_mode(recommender, generator)


if __name__ == "__main__":
    main()
