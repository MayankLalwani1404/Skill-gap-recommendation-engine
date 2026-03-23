"""
Recommendation Engine
Provides:
  - TF-IDF vectorization of role skill profiles
  - Cosine similarity-based role matching
  - Weighted hybrid scoring (skills + job_zone + domain)
  - Skill gap computation
  - Learning roadmap generation
  - Related roles suggestion
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Education level mapping (O*NET categories → readable labels)
EDU_LEVEL_MAP = {
    1: "Less than High School",
    2: "High School Diploma",
    3: "Some College",
    4: "Associate's Degree",
    5: "Bachelor's Degree",
    6: "Post-Bachelor's Certificate",
    7: "Master's Degree",
    8: "Post-Master's Certificate",
    9: "First Professional Degree",
    10: "Doctoral Degree",
    11: "Post-Doctoral Training",
    12: "Other",
}

JOB_ZONE_MAP = {
    1: "🟢 Entry Level – Little or no preparation",
    2: "🟡 Some Preparation Needed",
    3: "🟠 Medium Preparation Needed",
    4: "🔵 Considerable Preparation Needed",
    5: "🔴 Extensive Preparation Needed",
}

USER_EDU_MAP = {
    "High School / GED": 2,
    "Some College": 3,
    "Associate's Degree": 4,
    "Bachelor's Degree": 5,
    "Master's Degree": 7,
    "PhD / Doctoral": 10,
}

USER_EXP_MAP = {
    "No Experience (Student)": 1,
    "< 1 Year": 1,
    "1–2 Years": 2,
    "3–5 Years": 3,
    "5–10 Years": 4,
    "10+ Years": 5,
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


class SkillRecommender:
    def __init__(self, role_df: pd.DataFrame, long_df: pd.DataFrame, role_skills: dict):
        self.role_df = role_df.copy()
        self.long_df = long_df.copy()
        self.role_skills = role_skills  # {onet_code: [skill_clean, ...]}

        # Build TF-IDF matrix once
        self._build_tfidf()
        # Build skill importance lookup
        self._build_importance_lookup()
        # Cluster roles
        self._cluster_roles()

    # ─────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────

    def _build_tfidf(self):
        """Create TF-IDF matrix with one document per role (skills as words)."""
        # Sort roles consistently
        codes = self.role_df["onet_code"].tolist()
        self._codes = codes

        docs = [" ".join(self.role_skills.get(c, [])) for c in codes]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def _build_importance_lookup(self):
        """Build {onet_code: {skill_clean: importance}} lookup."""
        self._imp = (
            self.long_df
            .groupby(["onet_code", "skill_clean"])["importance"]
            .max()
            .reset_index()
        )
        self._imp_dict = {}
        for row in self._imp.itertuples():
            if row.onet_code not in self._imp_dict:
                self._imp_dict[row.onet_code] = {}
            self._imp_dict[row.onet_code][row.skill_clean] = row.importance

    def _cluster_roles(self, n_clusters: int = 20):
        """KMeans clustering over TF-IDF for career family grouping."""
        X = normalize(self.tfidf_matrix, norm="l2")
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        km.fit(X)
        self.role_df["cluster"] = km.labels_
        # Cluster labels: pick most common SOC major group
        cluster_names = {}
        for cl in range(n_clusters):
            mask = self.role_df["cluster"] == cl
            top_title = self.role_df.loc[mask, "title_clean"].iloc[0] if mask.sum() > 0 else f"Cluster {cl}"
            cluster_names[cl] = top_title
        self._cluster_names = cluster_names

    # ─────────────────────────────────────────────────────────────────────
    # CORE RECOMMENDATION
    # ─────────────────────────────────────────────────────────────────────

    def recommend(
        self,
        user_skills: list[str],
        education: str = "Bachelor's Degree",
        experience: str = "1–2 Years",
        domain_keywords: list[str] = None,
        top_n: int = 10,
        filter_by_domain: str = None,
    ) -> pd.DataFrame:
        """
        Recommend top-N careers for a user profile.

        Returns a DataFrame with columns:
          onet_code, title, match_score, skill_match_pct,
          zone_score, edu_score, matched_skills, missing_skills,
          job_zone, edu_level, cluster, description
        """
        # Normalize user skills
        user_skills_clean = [s.strip().lower() for s in user_skills if s.strip()]
        if not user_skills_clean:
            return pd.DataFrame()

        # --- TF-IDF Cosine Similarity ---
        user_doc = " ".join(user_skills_clean)
        user_vec = self.vectorizer.transform([user_doc])
        cosine_scores = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # --- Skill Overlap Score ---
        user_set = set(user_skills_clean)
        overlap_scores = np.array([
            len(user_set & set(self.role_skills.get(c, []))) /
            max(len(self.role_skills.get(c, [])), 1)
            for c in self._codes
        ])

        # --- Job Zone Score ---
        user_zone = USER_EXP_MAP.get(experience, 2)
        zone_scores = np.array([
            1.0 - abs(row.job_zone - user_zone) / 4.0
            for _, row in self.role_df.iterrows()
        ])

        # --- Education Score ---
        user_edu = USER_EDU_MAP.get(education, 5)
        edu_scores = np.array([
            1.0 - min(abs(row.edu_level - user_edu) / 10.0, 1.0)
            for _, row in self.role_df.iterrows()
        ])

        # --- Domain Keyword Boost ---
        domain_scores = np.zeros(len(self._codes))
        if domain_keywords:
            kw_clean = [k.strip().lower() for k in domain_keywords]
            for i, code in enumerate(self._codes):
                title_lower = self.role_df.iloc[i]["title_clean"].lower()
                desc_lower = str(self.role_df.iloc[i].get("description", "")).lower()
                hit = sum(1 for kw in kw_clean if kw in title_lower or kw in desc_lower)
                domain_scores[i] = min(hit / max(len(kw_clean), 1), 1.0)

        # --- Hybrid Score ---
        hybrid = (
            0.35 * cosine_scores +
            0.35 * overlap_scores +
            0.12 * zone_scores +
            0.08 * edu_scores +
            0.10 * domain_scores
        )

        # Build results
        top_idx = np.argsort(hybrid)[::-1][:top_n * 3]
        rows = []
        for i in top_idx:
            code = self._codes[i]
            role = self.role_df.iloc[i]
            role_skills_set = set(self.role_skills.get(code, []))
            matched = sorted(user_set & role_skills_set)
            missing_all = sorted(role_skills_set - user_set)

            # Rank missing by importance
            imp_dict = self._imp_dict.get(code, {})
            missing_ranked = sorted(
                missing_all,
                key=lambda s: imp_dict.get(s, 0),
                reverse=True,
            )

            rows.append({
                "onet_code": code,
                "title": role["title_clean"],
                "description": str(role.get("description", ""))[:300],
                "match_score": round(hybrid[i] * 100, 1),
                "cosine_score": round(cosine_scores[i] * 100, 1),
                "skill_match_pct": round(overlap_scores[i] * 100, 1),
                "zone_score": round(zone_scores[i] * 100, 1),
                "edu_score": round(edu_scores[i] * 100, 1),
                "matched_skills": matched,
                "missing_skills": missing_ranked,
                "job_zone": int(role["job_zone"]),
                "job_zone_label": JOB_ZONE_MAP.get(int(role["job_zone"]), ""),
                "edu_level": int(role["edu_level"]),
                "edu_level_label": EDU_LEVEL_MAP.get(int(role["edu_level"]), ""),
                "cluster": int(role.get("cluster", 0)),
                "skill_count": int(role.get("skill_count", 0)),
                "soc_major": role.get("soc_major", ""),
            })

        results = pd.DataFrame(rows).sort_values("match_score", ascending=False)

        # Optional domain filter
        if filter_by_domain:
            results = results[
                results["title"].str.lower().str.contains(filter_by_domain.lower())
            ]

        return results.head(top_n).reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────
    # SKILL GAP & ROADMAP
    # ─────────────────────────────────────────────────────────────────────

    def get_skill_gap(self, user_skills: list[str], target_code: str) -> dict:
        """
        Returns a detailed skill gap analysis for a specific occupation.
        """
        user_set = set(s.strip().lower() for s in user_skills)
        role_skills_set = set(self.role_skills.get(target_code, []))
        matched = sorted(user_set & role_skills_set)
        missing_all = sorted(role_skills_set - user_set)

        imp_dict = self._imp_dict.get(target_code, {})
        missing_ranked = sorted(missing_all, key=lambda s: imp_dict.get(s, 0), reverse=True)

        match_pct = len(matched) / max(len(role_skills_set), 1) * 100

        return {
            "onet_code": target_code,
            "matched_skills": matched,
            "missing_skills": missing_ranked,
            "match_percentage": round(match_pct, 1),
            "total_required": len(role_skills_set),
            "user_has": len(matched),
            "user_missing": len(missing_ranked),
        }

    def generate_roadmap(
        self, user_skills: list[str], target_code: str, max_skills: int = 12
    ) -> list[dict]:
        """
        Generate a tiered learning roadmap for the skill gap.

        Returns a list of roadmap steps:
          { priority, skill, category, importance, rationale }
        """
        gap = self.get_skill_gap(user_skills, target_code)
        missing = gap["missing_skills"][:max_skills]
        imp_dict = self._imp_dict.get(target_code, {})

        roadmap = []
        for rank, skill in enumerate(missing, 1):
            imp = imp_dict.get(skill, 2.5)
            if rank <= 3:
                tier = "🔴 Foundation (Learn First)"
                urgency = "Critical"
            elif rank <= 7:
                tier = "🟠 Core Skills (Next)"
                urgency = "High"
            else:
                tier = "🟢 Advanced / Nice-to-have"
                urgency = "Medium"

            # Infer source category
            row = self.long_df[
                (self.long_df["onet_code"] == target_code) &
                (self.long_df["skill_clean"] == skill)
            ]
            source = row["source"].values[0] if len(row) > 0 else "skill"
            category = {"skill": "Core Skill", "knowledge": "Domain Knowledge", "tech": "Tool / Technology"}.get(source, "Skill")

            roadmap.append({
                "priority": rank,
                "skill": skill.title(),
                "category": category,
                "importance": round(imp, 2),
                "tier": tier,
                "urgency": urgency,
            })

        return roadmap

    # ─────────────────────────────────────────────────────────────────────
    # RELATED ROLES
    # ─────────────────────────────────────────────────────────────────────

    def get_related_roles(self, onet_code: str, top_n: int = 5) -> list[str]:
        """
        Get related occupations based on TF-IDF similarity.
        """
        if onet_code not in self._codes:
            return []
        idx = self._codes.index(onet_code)
        sims = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        ).flatten()
        related_idx = np.argsort(sims)[::-1][1: top_n + 1]
        return [self.role_df.iloc[i]["title_clean"] for i in related_idx]

    # ─────────────────────────────────────────────────────────────────────
    # EDA HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def top_skills_overall(self, n: int = 30) -> pd.DataFrame:
        """Most common skills across all occupations."""
        counts = (
            self.long_df.groupby("skill_clean")["onet_code"]
            .nunique()
            .reset_index()
            .rename(columns={"onet_code": "role_count"})
            .sort_values("role_count", ascending=False)
        )
        counts["skill"] = counts["skill_clean"].str.title()
        return counts.head(n)

    def skills_by_cluster(self) -> pd.DataFrame:
        """Top skills per cluster."""
        merged = self.long_df.merge(
            self.role_df[["onet_code", "cluster"]], on="onet_code", how="left"
        )
        top = (
            merged.groupby(["cluster", "skill_clean"])["importance"]
            .mean()
            .reset_index()
            .sort_values(["cluster", "importance"], ascending=[True, False])
        )
        return top

    def get_all_skills(self) -> list[str]:
        """Sorted unique skills for autocomplete widgets."""
        return sorted(self.long_df["skill_clean"].str.title().unique().tolist())

    def get_roles_in_cluster(self, cluster_id: int) -> pd.DataFrame:
        return self.role_df[self.role_df["cluster"] == cluster_id][
            ["onet_code", "title_clean", "job_zone", "skill_count"]
        ].sort_values("skill_count", ascending=False)

    # ─────────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str = None):
        if path is None:
            path = os.path.join(MODEL_DIR, "recommender.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Recommender saved to {path}")

    @classmethod
    def load(cls, path: str = None) -> "SkillRecommender":
        if path is None:
            path = os.path.join(MODEL_DIR, "recommender.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
