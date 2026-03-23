"""
Evaluation Module
Validates recommendation quality using curated test profiles.
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TEST_PROFILES = [
    {
        "name": "Data Analyst (Junior)",
        "skills": ["SQL", "Excel", "Data Analysis", "Python", "Statistics", "Tableau"],
        "education": "Bachelor's Degree",
        "experience": "1–2 Years",
        "domain": ["data", "analytics", "business"],
        "expected_roles": ["data analyst", "business analyst", "operations research"],
    },
    {
        "name": "Machine Learning Engineer",
        "skills": ["Python", "Machine Learning", "Deep Learning", "TensorFlow", "Statistics",
                   "Data Science", "NumPy", "Scikit-Learn", "SQL", "Feature Engineering"],
        "education": "Master's Degree",
        "experience": "3–5 Years",
        "domain": ["machine learning", "AI", "data science"],
        "expected_roles": ["machine learning", "data scientist", "artificial intelligence"],
    },
    {
        "name": "Software Developer",
        "skills": ["Python", "Java", "JavaScript", "Git", "Linux", "REST APIs",
                   "Databases", "Agile", "Object-Oriented Programming"],
        "education": "Bachelor's Degree",
        "experience": "3–5 Years",
        "domain": ["software", "web", "programming"],
        "expected_roles": ["software", "developer", "engineer"],
    },
    {
        "name": "Healthcare Manager",
        "skills": ["Healthcare Management", "Patient Care", "Medical Records",
                   "Administration", "Budgeting", "Communication", "Leadership"],
        "education": "Master's Degree",
        "experience": "5–10 Years",
        "domain": ["healthcare", "medical", "hospital"],
        "expected_roles": ["medical", "health", "clinical"],
    },
    {
        "name": "Finance Analyst",
        "skills": ["Financial Analysis", "Excel", "Accounting", "Forecasting",
                   "SQL", "Statistics", "Risk Management", "Financial Reporting"],
        "education": "Bachelor's Degree",
        "experience": "1–2 Years",
        "domain": ["finance", "investment", "banking"],
        "expected_roles": ["financial", "analyst", "accountant"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_recommendations(recommender, top_n: int = 10) -> pd.DataFrame:
    """Run all test profiles through the recommender and check quality."""
    results = []
    for profile in TEST_PROFILES:
        recs = recommender.recommend(
            user_skills=profile["skills"],
            education=profile["education"],
            experience=profile["experience"],
            domain_keywords=profile["domain"],
            top_n=top_n,
        )
        if recs.empty:
            results.append({
                "persona": profile["name"],
                "top_match": "No results",
                "top_score": 0,
                "hit": False,
                "avg_match_pct": 0,
                "diversity_clusters": 0,
                "n_recommendations": 0,
            })
            continue

        top_titles = recs["title"].str.lower().tolist()

        # Hit rate: at least one expected role appears in top-N titles
        hit = any(
            any(exp.lower() in t for t in top_titles)
            for exp in profile["expected_roles"]
        )

        results.append({
            "persona": profile["name"],
            "top_match": recs.iloc[0]["title"],
            "top_score": recs.iloc[0]["match_score"],
            "hit": hit,
            "avg_match_pct": round(recs["match_score"].mean(), 1),
            "diversity_clusters": recs["cluster"].nunique(),
            "n_recommendations": len(recs),
        })

    return pd.DataFrame(results)


def sanity_check(recommender) -> list[str]:
    """
    Quick sanity checks:
    - SQL skills → Data/Database roles?
    - Teaching skills → Education roles?
    """
    checks = []

    # SQL → Data
    recs = recommender.recommend(
        ["SQL", "Databases", "Data Analysis", "Reporting"],
        top_n=5,
    )
    if not recs.empty:
        top = recs.iloc[0]["title"]
        checks.append(f"SQL/Data skills → Top role: '{top}' ✓" if any(
            k in top.lower() for k in ["data", "database", "analyst"]
        ) else f"SQL/Data skills → '{top}' (unexpected)")

    # Teaching → Education
    recs = recommender.recommend(
        ["Teaching", "Curriculum Development", "Instructional Design", "Classroom Management"],
        top_n=5,
    )
    if not recs.empty:
        top = recs.iloc[0]["title"]
        checks.append(f"Teaching skills → Top role: '{top}' ✓" if any(
            k in top.lower() for k in ["teacher", "instructor", "education", "training"]
        ) else f"Teaching skills → '{top}' (unexpected)")

    # Medical → Healthcare
    recs = recommender.recommend(
        ["Patient Care", "Medical Terminology", "Clinical Procedures", "Nursing"],
        top_n=5,
    )
    if not recs.empty:
        top = recs.iloc[0]["title"]
        checks.append(f"Medical skills → Top role: '{top}' ✓" if any(
            k in top.lower() for k in ["nurse", "medical", "health", "clinical"]
        ) else f"Medical skills → '{top}' (unexpected)")

    return checks


def print_evaluation_report(recommender):
    print("\n" + "="*60)
    print("  RECOMMENDATION ENGINE EVALUATION REPORT")
    print("="*60)

    df = evaluate_recommendations(recommender, top_n=10)
    hit_rate = df["hit"].mean() * 100

    print(f"\n📊 Hit Rate (top-10): {hit_rate:.0f}% ({df['hit'].sum()}/{len(df)} personas)")
    print(f"📈 Avg Match Score: {df['top_score'].mean():.1f}%")
    print(f"🎯 Avg Diversity (clusters in top-10): {df['diversity_clusters'].mean():.1f}")

    print("\n── Per Persona Results ──")
    for _, row in df.iterrows():
        icon = "✅" if row["hit"] else "⚠️"
        print(f"  {icon} {row['persona']}")
        print(f"     Top: {row['top_match']} ({row['top_score']}%)")

    print("\n── Sanity Checks ──")
    checks = sanity_check(recommender)
    for c in checks:
        print(f"  {c}")

    print("\n" + "="*60)
    return df


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.data_preprocessing import get_processed_data
    from src.recommender import SkillRecommender

    role_df, long_df, role_skills = get_processed_data()
    rec = SkillRecommender(role_df, long_df, role_skills)
    print_evaluation_report(rec)
