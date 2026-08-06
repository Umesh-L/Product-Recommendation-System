import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CATALOGUE_PATH = DATA_DIR / "product_catalogue.json"


def load_catalogue() -> Dict[str, Any]:
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s&/-]", " ", text)
    tokens = text.split()
    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall", "can",
        "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into",
        "through", "during", "before", "after", "above", "below", "between",
        "about", "against", "this", "that", "these", "those", "it", "its",
        "which", "what", "who", "whom", "whose", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their",
        "not", "no", "nor", "too", "very", "s", "t", "just", "don", "now",
        "up", "down", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "so", "than", "also", "includes", "included", "including",
        "fits", "fit", "set", "pack", "perfect", "great", "good", "nice"
    }
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def build_document(product: Dict[str, Any]) -> str:
    parts = [
        product.get("name", ""),
        product.get("category", ""),
        product.get("subcategory", ""),
        product.get("brand", ""),
        product.get("description", ""),
        " ".join(product.get("features", [])),
        " ".join(product.get("tags", [])),
    ]
    return " ".join(parts)


def compute_tf(tokens: List[str]) -> Counter:
    return Counter(tokens)


def compute_idf(documents: List[str]) -> Dict[str, float]:
    num_docs = len(documents)
    doc_freq: Dict[str, int] = {}
    for doc in documents:
        unique_tokens = set(tokenize(doc))
        for token in unique_tokens:
            doc_freq[token] = doc_freq.get(token, 0) + 1
    idf = {}
    for token, freq in doc_freq.items():
        idf[token] = math.log((1 + num_docs) / (1 + freq)) + 1
    return idf


def vectorize(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = compute_tf(tokens)
    total_tokens = len(tokens) if tokens else 1
    vector = {}
    for token, count in tf.items():
        if token in idf:
            tf_normalized = count / total_tokens
            vector[token] = tf_normalized * idf[token]
    return vector


def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    shared = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in shared)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def build_user_query_vector(preferences: Dict[str, Any], idf: Dict[str, float]) -> Dict[str, float]:
    query_parts = []
    if "keywords" in preferences and preferences["keywords"]:
        if isinstance(preferences["keywords"], list):
            query_parts.extend(preferences["keywords"])
        else:
            query_parts.append(str(preferences["keywords"]))
    if "goals" in preferences and preferences["goals"]:
        query_parts.append(str(preferences["goals"]))
    if "interests" in preferences and preferences["interests"]:
        if isinstance(preferences["interests"], list):
            query_parts.extend(preferences["interests"])
        else:
            query_parts.append(str(preferences["interests"]))
    if "usage_context" in preferences and preferences["usage_context"]:
        query_parts.append(str(preferences["usage_context"]))
    query_text = " ".join(query_parts)
    tokens = tokenize(query_text)
    return vectorize(tokens, idf)


def attribute_match_score(product: Dict[str, Any], preferences: Dict[str, Any]) -> float:
    score = 0.0
    max_score = 0.0

    if "preferred_categories" in preferences and preferences["preferred_categories"]:
        max_score += 0.15
        if product.get("category") in preferences["preferred_categories"]:
            score += 0.15
        elif product.get("subcategory") in preferences["preferred_categories"]:
            score += 0.10

    if "preferred_brands" in preferences and preferences["preferred_brands"]:
        max_score += 0.10
        if product.get("brand") in preferences["preferred_brands"]:
            score += 0.10

    if "price_range" in preferences and preferences["price_range"]:
        max_score += 0.15
        pr = preferences["price_range"]
        product_price = product.get("price", 0)
        min_price = pr.get("min", 0)
        max_price = pr.get("max", float("inf"))
        if min_price <= product_price <= max_price:
            score += 0.15
        elif product_price > max_price:
            penalty = min(0.15, (product_price - max_price) / max_price * 0.5)
            score += max(0.0, 0.15 - penalty)
        else:
            bonus = min(0.05, (min_price - product_price) / min_price * 0.3)
            score += 0.10 + bonus

    if "must_have_features" in preferences and preferences["must_have_features"]:
        max_score += 0.15
        product_features = set(f.lower() for f in product.get("features", []))
        product_tags = set(t.lower() for t in product.get("tags", []))
        required = preferences["must_have_features"]
        matches = sum(
            1 for req in required
            if any(req.lower() in pf for pf in product_features) or
               any(req.lower() in pt for pt in product_tags) or
               req.lower() in product.get("description", "").lower()
        )
        if len(required) > 0:
            score += 0.15 * (matches / len(required))

    if "avoid_tags" in preferences and preferences["avoid_tags"]:
        max_score += 0.05
        product_tags = set(t.lower() for t in product.get("tags", []))
        avoid_lower = set(a.lower() for a in preferences["avoid_tags"])
        if not product_tags & avoid_lower:
            score += 0.05
        else:
            score -= 0.05

    score = max(0.0, min(max_score, score))
    normalized = score / max_score if max_score > 0 else 0.0
    return normalized


def rating_bonus(product: Dict[str, Any]) -> float:
    rating = product.get("rating", 0)
    reviews = product.get("review_count", 0)
    if rating == 0 or reviews == 0:
        return 0.0
    popularity_factor = min(1.0, math.log10(reviews + 1) / 4.0)
    bonus = ((rating - 3.5) / 1.5) * 0.05 * popularity_factor
    return max(-0.03, min(0.05, bonus))


class ProductRecommender:
    def __init__(self):
        self.catalogue_data = load_catalogue()
        self.products: List[Dict[str, Any]] = self.catalogue_data["products"]
        self.documents: List[str] = [build_document(p) for p in self.products]
        self.idf: Dict[str, float] = compute_idf(self.documents)
        self.product_vectors: List[Dict[str, float]] = [
            vectorize(tokenize(doc), self.idf) for doc in self.documents
        ]

    def recommend(
        self,
        preferences: Dict[str, Any],
        top_n: int = 10,
        purchase_history: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        user_vector = build_user_query_vector(preferences, self.idf)
        purchase_history = purchase_history or []
        purchased_ids: Set[str] = set(purchase_history)

        attr_prefs = preferences.get("preferences", preferences)

        scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, product in enumerate(self.products):
            if product["id"] in purchased_ids:
                continue

            sim = cosine_similarity(user_vector, self.product_vectors[idx])
            attr_score = attribute_match_score(product, attr_prefs)
            rating_boost = rating_bonus(product)

            content_weight = 0.60
            attr_weight = 0.35
            rating_weight = 0.05

            has_query = bool(user_vector)
            if not has_query:
                content_weight = 0.0
                attr_weight = 0.70
                rating_weight = 0.30
                total_w = attr_weight + rating_weight
                attr_weight /= total_w
                rating_weight /= total_w

            total_w = content_weight + attr_weight + rating_weight
            final_score = (
                (sim * content_weight) +
                (attr_score * attr_weight) +
                (rating_boost * rating_weight)
            ) / total_w

            diversity_penalty = 0.0
            if purchase_history:
                for hist_id in purchase_history:
                    hist_idx = next(
                        (i for i, p in enumerate(self.products) if p["id"] == hist_id), None
                    )
                    if hist_idx is not None:
                        hist_sim = cosine_similarity(
                            self.product_vectors[idx], self.product_vectors[hist_idx]
                        )
                        diversity_penalty = max(diversity_penalty, hist_sim * 0.08)
            final_score -= diversity_penalty

            final_score = max(0.0, min(1.0, final_score))
            scored.append((final_score, idx, product))

        scored.sort(key=lambda x: (-x[0], -x[2].get("rating", 0)))

        seen_categories: Set[str] = set()
        diversity_count = 0
        results: List[Dict[str, Any]] = []
        for score, idx, product in scored:
            if len(results) < top_n * 2:
                results.append({
                    "product": product,
                    "score": round(score, 4),
                    "similarity_score": round(cosine_similarity(user_vector, self.product_vectors[idx]), 4),
                    "attribute_score": round(attribute_match_score(product, attr_prefs), 4),
                    "rating_boost": round(rating_bonus(product), 4),
                })
            cat = product.get("category", "")
            if cat not in seen_categories:
                seen_categories.add(cat)
                diversity_count += 1

        return results[:top_n]

    def cold_start_recommend(
        self,
        preferences: Dict[str, Any],
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        return self.recommend(preferences, top_n=top_n, purchase_history=[])

    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        for p in self.products:
            if p["id"] == product_id:
                return p
        return None
