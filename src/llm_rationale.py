import os
import json
from typing import Dict, List, Any, Optional

try:
    from openai import OpenAI, APIError, APIConnectionError, RateLimitError
except ImportError:
    OpenAI = None

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> Dict[str, str]:
    env_path = PROJECT_ROOT / ".env"
    env = {}
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key not in env:
            env[key] = value
    return env


class RationaleGenerator:
    SYSTEM_PROMPT = """You are a senior product recommendation specialist. Your job is to explain WHY a product is a good match for a specific user based on:
- The user's profile (background, goals, known skills, preferences, price range)
- The product's attributes (features, category, price, brand, tags, description)
- The matching scores (similarity score, attribute match score, overall score)

Write clear, concise, honest rationale (2-4 sentences per product). Focus on specific, concrete reasons, not vague praise. Explicitly reference:
1. Which of the user's needs/goals this product addresses
2. Key product features that match the user's preferences
3. Value proposition given the price range
4. Any tradeoffs the user should be aware of

Do NOT invent features the product doesn't have. If a product is less aligned in some areas, mention that honestly.

Format: Return JSON with a "rationales" array where each entry has keys:
- product_id: string
- rationale: string (the explanation)
- key_matches: array of 2-4 specific strings like "matches goal: build home gym" or "fits budget: under $500"
- potential_concerns: array of 0-2 strings (minor tradeoffs, empty if none)
"""

    def __init__(self):
        self.env = load_env()
        self.api_key = self.env.get("OPENAI_API_KEY") or self.env.get("GROQ_API_KEY")
        self.base_url = self.env.get("OPENAI_BASE_URL")
        self.model = self.env.get("LLM_MODEL", "gpt-4o-mini")
        self.client = None
        if OpenAI and self.api_key:
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            try:
                self.client = OpenAI(**kwargs)
            except Exception:
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def _build_user_context(self, user_profile: Dict[str, Any]) -> str:
        parts = []
        if "name" in user_profile:
            parts.append(f"User name: {user_profile['name']}")
        if "background" in user_profile:
            parts.append(f"Background: {user_profile['background']}")
        if "goals" in user_profile:
            parts.append(f"Goals: {user_profile['goals']}")
        if "known_skills" in user_profile:
            skills = user_profile['known_skills']
            if isinstance(skills, list):
                parts.append(f"Known skills: {', '.join(skills)}")
            else:
                parts.append(f"Known skills: {skills}")
        if "preferences" in user_profile:
            prefs = user_profile["preferences"]
            if "preferred_categories" in prefs and prefs["preferred_categories"]:
                parts.append(f"Preferred categories: {', '.join(prefs['preferred_categories'])}")
            if "preferred_brands" in prefs and prefs["preferred_brands"]:
                parts.append(f"Preferred brands: {', '.join(prefs['preferred_brands'])}")
            if "price_range" in prefs and prefs["price_range"]:
                pr = prefs["price_range"]
                parts.append(f"Price range: ${pr.get('min', 0)} - ${pr.get('max', 'N/A')}")
            if "must_have_features" in prefs and prefs["must_have_features"]:
                parts.append(f"Must have: {', '.join(prefs['must_have_features'])}")
        if "usage_context" in user_profile:
            parts.append(f"Usage context: {user_profile['usage_context']}")
        if "interests" in user_profile:
            interests = user_profile['interests']
            if isinstance(interests, list):
                parts.append(f"Interests: {', '.join(interests)}")
            else:
                parts.append(f"Interests: {interests}")
        return "\n".join(parts)

    def _build_product_context(self, recs: List[Dict[str, Any]]) -> str:
        entries = []
        for rec in recs:
            p = rec["product"]
            entries.append({
                "rank": len(entries) + 1,
                "product_id": p["id"],
                "product_name": p["name"],
                "category": f"{p['category']} > {p['subcategory']}",
                "price": f"${p['price']}",
                "brand": p["brand"],
                "rating": f"{p['rating']}/5 ({p['review_count']} reviews)",
                "description": p["description"],
                "features": p["features"],
                "tags": p["tags"],
                "scores": {
                    "overall_score": rec["score"],
                    "content_similarity": rec["similarity_score"],
                    "attribute_match": rec["attribute_score"],
                    "rating_boost": rec["rating_boost"],
                }
            })
        return json.dumps(entries, indent=2)

    def generate_rationales(
        self,
        user_profile: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.is_available():
            return self._fallback_rationales(user_profile, recommendations)

        user_ctx = self._build_user_context(user_profile)
        product_ctx = self._build_product_context(recommendations)
        user_prompt = f"""
USER PROFILE:
{user_ctx}

RECOMMENDED PRODUCTS (ranked):
{product_ctx}

Generate the rationale for each product. Return ONLY valid JSON. No markdown, no commentary.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            parsed = json.loads(content)
            return self._normalize(parsed, recommendations)
        except (APIError, APIConnectionError, RateLimitError, Exception) as e:
            print(f"  [LLM warning: {type(e).__name__}] Falling back to heuristic rationale.")
            return self._fallback_rationales(user_profile, recommendations)

    def _normalize(self, parsed: Dict, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        result = {"rationales": {}}
        rationales_list = parsed.get("rationales", [])
        for r in rationales_list:
            pid = r.get("product_id")
            if pid:
                result["rationales"][pid] = {
                    "rationale": r.get("rationale", ""),
                    "key_matches": r.get("key_matches", []),
                    "potential_concerns": r.get("potential_concerns", []),
                }
        for rec in recommendations:
            pid = rec["product"]["id"]
            if pid not in result["rationales"]:
                result["rationales"][pid] = self._single_fallback(rec)
        return result

    def _fallback_rationales(
        self, user_profile: Dict[str, Any], recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        result = {"rationales": {}}
        for rec in recommendations:
            pid = rec["product"]["id"]
            result["rationales"][pid] = self._single_fallback(rec)
        return result

    def _single_fallback(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        p = rec["product"]
        overall = rec["score"]
        content_sim = rec["similarity_score"]
        attr_match = rec["attribute_score"]
        top_features = p["features"][:3] if p["features"] else []
        top_tags = p["tags"][:2] if p["tags"] else []
        rationale = (
            f"{p['name']} (${p['price']}) matches with an overall score of {overall:.0%}. "
            f"Content similarity is {content_sim:.0%} and attribute alignment is {attr_match:.0%}. "
            f"Key features: {'; '.join(top_features)}. "
            f"Rated {p['rating']}/5 from {p['review_count']} reviews, indicating strong market confidence."
        )
        key_matches = [f"Category: {p['category']} > {p['subcategory']}"]
        if top_tags:
            key_matches.append(f"Tag match: {', '.join(top_tags)}")
        if top_features:
            key_matches.append(f"Feature highlight: {top_features[0]}")
        concerns = []
        if p.get("rating", 4.0) < 4.2:
            concerns.append(f"Rating is {p['rating']}/5, slightly below the top tier")
        if p.get("review_count", 0) < 500:
            concerns.append(f"Fewer reviews ({p['review_count']}) — consider evaluating with more user data")
        return {
            "rationale": rationale,
            "key_matches": key_matches,
            "potential_concerns": concerns,
        }
