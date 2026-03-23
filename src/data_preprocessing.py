"""
Data Preprocessing Module
Loads and processes O*NET 30.2 database files into structured,
recommendation-ready formats.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "db_30_2_excel")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _read(filename: str) -> pd.DataFrame:
    """Read an xlsx file from the O*NET data directory."""
    return pd.read_excel(os.path.join(DATA_DIR, filename))


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _filter_importance(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """Keep only Importance-scale rows above threshold."""
    return df[
        (df["Scale ID"] == "IM") &
        (df["Data Value"] >= threshold) &
        (df["Recommend Suppress"] == "N")
    ].copy()


# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_occupations() -> pd.DataFrame:
    """Load occupation titles and descriptions."""
    df = _read("Occupation Data.xlsx")
    df.columns = ["onet_code", "title", "description"]
    df["title_clean"] = df["title"].str.strip()
    df["onet_code"] = df["onet_code"].str.strip()
    return df.drop_duplicates("onet_code").reset_index(drop=True)


def load_job_zones() -> pd.DataFrame:
    """Load job zone (experience level) per occupation."""
    df = _read("Job Zones.xlsx")
    df.columns = [c.strip() for c in df.columns]
    df = df[["O*NET-SOC Code", "Job Zone"]].copy()
    df.columns = ["onet_code", "job_zone"]
    df["onet_code"] = df["onet_code"].str.strip()
    df["job_zone"] = pd.to_numeric(df["job_zone"], errors="coerce").fillna(3).astype(int)
    return df.drop_duplicates("onet_code")


def load_skills() -> pd.DataFrame:
    """Load filtered skill scores per occupation."""
    df = _read("Skills.xlsx")
    df = _filter_importance(df, threshold=2.5)
    out = df[["O*NET-SOC Code", "Title", "Element Name", "Data Value"]].copy()
    out.columns = ["onet_code", "title", "skill", "importance"]
    out["onet_code"] = out["onet_code"].str.strip()
    out["skill_clean"] = _normalize_text(out["skill"])
    return out


def load_knowledge() -> pd.DataFrame:
    """Load filtered knowledge scores per occupation."""
    df = _read("Knowledge.xlsx")
    df = _filter_importance(df, threshold=2.5)
    out = df[["O*NET-SOC Code", "Title", "Element Name", "Data Value"]].copy()
    out.columns = ["onet_code", "title", "skill", "importance"]
    out["onet_code"] = out["onet_code"].str.strip()
    out["skill_clean"] = _normalize_text(out["skill"])
    return out


def load_tech_skills() -> pd.DataFrame:
    """Load technology skills (tools/software) per occupation."""
    df = _read("Technology Skills.xlsx")
    df.columns = [c.strip() for c in df.columns]

    # Technology Skills has: O*NET-SOC Code, Title, Commodity Code, Commodity Title, Example, Hot Technology
    onet_col = "O*NET-SOC Code"
    example_col = "Example"
    commodity_col = "Commodity Title"

    df["onet_code"] = df[onet_col].astype(str).str.strip()
    # Use Example (specific tool) if available, else Commodity Title
    df["skill"] = df[example_col].fillna(df[commodity_col]).astype(str).str.strip()
    df["skill"] = df["skill"].replace("nan", np.nan)
    df = df.dropna(subset=["skill"])
    df["skill_clean"] = _normalize_text(df["skill"])
    df["importance"] = 3.0  # assign a neutral importance score
    df["title"] = df["Title"].astype(str).str.strip()
    return df[["onet_code", "title", "skill", "skill_clean", "importance"]].drop_duplicates()


def load_work_activities() -> pd.DataFrame:
    """Load top work activities per occupation."""
    df = _read("Work Activities.xlsx")
    df = _filter_importance(df, threshold=3.0)
    out = df[["O*NET-SOC Code", "Element Name", "Data Value"]].copy()
    out.columns = ["onet_code", "activity", "importance"]
    out["onet_code"] = out["onet_code"].str.strip()
    return out


def load_education() -> pd.DataFrame:
    """Load education requirement distribution per occupation."""
    df = _read("Education, Training, and Experience.xlsx")
    df.columns = [c.strip() for c in df.columns]

    # Filter to Required Level of Education rows only
    edu_df = df[df["Element Name"] == "Required Level of Education"].copy()
    edu_df = edu_df[edu_df["Scale ID"] == "RL"].copy()

    edu_df["onet_code"] = edu_df["O*NET-SOC Code"].astype(str).str.strip()
    edu_df["category"] = pd.to_numeric(edu_df["Category"], errors="coerce")
    edu_df["data_value"] = pd.to_numeric(edu_df["Data Value"], errors="coerce").fillna(0)

    # Pick the education level with the highest % for each occupation
    idx = edu_df.groupby("onet_code")["data_value"].idxmax()
    top_edu = edu_df.loc[idx, ["onet_code", "category"]].copy()
    top_edu.columns = ["onet_code", "edu_level"]
    top_edu["edu_level"] = top_edu["edu_level"].fillna(3).astype(int)
    return top_edu.drop_duplicates("onet_code")


def load_related_occupations() -> pd.DataFrame:
    """Load related occupation pairs."""
    df = _read("Related Occupations.xlsx")
    df.columns = [c.strip() for c in df.columns]
    df["onet_code"] = df["O*NET-SOC Code"].astype(str).str.strip()
    df["related_code"] = df["Related O*NET-SOC Code"].astype(str).str.strip()
    df["related_title"] = df["Related Title"].astype(str).str.strip()
    df["tier"] = df["Relatedness Tier"].astype(str).str.strip()
    return df[["onet_code", "related_code", "related_title", "tier"]]


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BUILD
# ─────────────────────────────────────────────────────────────────────────────

def build_role_skill_matrix() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Builds:
    - role_df: one row per occupation with metadata
    - long_df: long-format (onet_code, skill_clean, importance, source)
    - role_skills: dict { onet_code: [list of skill strings] }
    """
    print("Loading occupations …")
    occ = load_occupations()

    print("Loading job zones …")
    jz = load_job_zones()

    print("Loading education …")
    edu = load_education()

    print("Loading skills …")
    sk = load_skills()
    sk["source"] = "skill"

    print("Loading knowledge …")
    kn = load_knowledge()
    kn["source"] = "knowledge"

    print("Loading tech skills …")
    ts = load_tech_skills()
    ts["source"] = "tech"

    # Combine all skill sources
    long_df = pd.concat(
        [
            sk[["onet_code", "skill", "skill_clean", "importance", "source"]],
            kn[["onet_code", "skill", "skill_clean", "importance", "source"]],
            ts[["onet_code", "skill", "skill_clean", "importance", "source"]],
        ],
        ignore_index=True,
    )
    long_df = long_df.dropna(subset=["onet_code", "skill_clean"])
    long_df = long_df[long_df["onet_code"].isin(occ["onet_code"])]

    # Aggregate: keep max importance when skill appears multiple times per role
    long_df = (
        long_df.groupby(["onet_code", "skill_clean"], as_index=False)
        .agg(importance=("importance", "max"), skill=("skill", "first"), source=("source", "first"))
    )

    print("Building role_df …")
    role_df = occ.copy()
    role_df = role_df.merge(jz, on="onet_code", how="left")
    role_df = role_df.merge(edu, on="onet_code", how="left")
    role_df["job_zone"] = role_df["job_zone"].fillna(3).astype(int)
    role_df["edu_level"] = role_df["edu_level"].fillna(3).astype(int)

    # Add SOC family (first 2 digits of code)
    role_df["soc_major"] = role_df["onet_code"].str[:2]

    print("Building role→skills dict …")
    role_skills = (
        long_df.sort_values("importance", ascending=False)
        .groupby("onet_code")["skill_clean"]
        .apply(list)
        .to_dict()
    )

    # Roles with no skills at all get an empty list
    for code in role_df["onet_code"]:
        if code not in role_skills:
            role_skills[code] = []

    # Add skill count to role_df
    role_df["skill_count"] = role_df["onet_code"].map(lambda c: len(role_skills.get(c, [])))

    print(f"✓ {len(role_df)} occupations | {long_df['skill_clean'].nunique()} unique skills")
    return role_df, long_df, role_skills


def save_processed_data(role_df, long_df, role_skills):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    role_df.to_csv(os.path.join(PROCESSED_DIR, "role_df.csv"), index=False)
    long_df.to_csv(os.path.join(PROCESSED_DIR, "long_df.csv"), index=False)
    with open(os.path.join(PROCESSED_DIR, "role_skills.pkl"), "wb") as f:
        pickle.dump(role_skills, f)
    print(f"✓ Saved processed data to {PROCESSED_DIR}")


def load_processed_data():
    """Load pre-processed data from disk (fast path)."""
    role_df = pd.read_csv(os.path.join(PROCESSED_DIR, "role_df.csv"))
    long_df = pd.read_csv(os.path.join(PROCESSED_DIR, "long_df.csv"))
    with open(os.path.join(PROCESSED_DIR, "role_skills.pkl"), "rb") as f:
        role_skills = pickle.load(f)
    return role_df, long_df, role_skills


def get_processed_data(force_rebuild: bool = False):
    """Return processed data, rebuilding from raw if needed."""
    cache_flag = os.path.join(PROCESSED_DIR, "role_skills.pkl")
    if not force_rebuild and os.path.exists(cache_flag):
        print("Loading from cache …")
        return load_processed_data()
    role_df, long_df, role_skills = build_role_skill_matrix()
    save_processed_data(role_df, long_df, role_skills)
    return role_df, long_df, role_skills


if __name__ == "__main__":
    role_df, long_df, role_skills = get_processed_data(force_rebuild=True)
    print(role_df.head())
