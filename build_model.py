"""
build_model.py
Run this once to preprocess data and build the recommender.
Usage: python build_model.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import get_processed_data
from src.recommender import SkillRecommender
from src.evaluation import print_evaluation_report

if __name__ == "__main__":
    print("=" * 55)
    print("  Career Path & Skill Gap Recommendation Engine")
    print("  Build Pipeline")
    print("=" * 55)

    print("\n[1/3] Processing O*NET data …")
    role_df, long_df, role_skills = get_processed_data(force_rebuild=True)

    print("\n[2/3] Building recommendation engine …")
    rec = SkillRecommender(role_df, long_df, role_skills)
    rec.save()

    print("\n[3/3] Running evaluation …")
    print_evaluation_report(rec)

    print("\n✅ Build complete! Run: streamlit run app.py")
