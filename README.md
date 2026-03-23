# 🧭 CareerMap AI — Career Path & Skill Gap Recommendation Engine

> **Powered by O\*NET 30.2 Database · 1,016 Occupations · 8,852+ Unique Skills**

CareerMap AI is an end-to-end career guidance system that recommends suitable career paths based on your current skills, education, and experience — while identifying your exact skill gaps and generating a prioritized learning roadmap.

---

## 🎯 Problem Statement

Job seekers and career switchers face two key challenges:
1. **"Which career is right for me?"** — given their current skills and background
2. **"What skills do I need to get there?"** — and in what order to learn them

Existing tools offer generic advice. CareerMap AI provides **data-driven, personalized, explainable recommendations** grounded in the U.S. Department of Labor's O\*NET occupational database.

---

## 📊 Data Sources

| File | Description | Rows |
|------|-------------|------|
| `Occupation Data.xlsx` | 1,016 occupations with titles and descriptions | 1,016 |
| `Skills.xlsx` | Core skills (reading, critical thinking, etc.) per occupation | 62,580 |
| `Knowledge.xlsx` | Domain knowledge areas per occupation | 59,004 |
| `Technology Skills.xlsx` | Software and tools per occupation | ~50,000 |
| `Job Zones.xlsx` | Experience level required (1–5 scale) | 1,016 |
| `Education, Training, and Experience.xlsx` | Education level distribution per role | 37,125 |
| `Related Occupations.xlsx` | TF-IDF-based occupation similarity pairs | 18,460 |

**Source:** [O\*NET 30.2 Database](https://www.onetcenter.org/database.html) — February 2026 Release  
**License:** [Creative Commons Attribution 4.0](https://www.onetcenter.org/license_db.html)

---

## ⚙️ Methodology

### 1. Data Fusion
All three skill sources (Core Skills, Knowledge Areas, Technology Skills) are merged per occupation. When a skill appears multiple times (e.g., "Programming" in both Skills and Knowledge), the maximum importance score is kept.

**Filtering:** Only skills with `Scale ID == IM` (Importance) and `Data Value >= 2.5` (moderate+ importance) are retained, removing irrelevant or suppressed entries.

### 2. TF-IDF Vectorization
Each occupation's skill profile is converted into a TF-IDF document where skills are "words" and skill importance influences the term weight. This captures both skill presence and cross-occupation rarity.

```
Vectorizer: TF-IDF (bigrams, sublinear_tf, min_df=2, max_df=0.95)
Vocabulary: ~15,000 skill n-grams
```

### 3. Hybrid Scoring

The match score for a user-role pair is a weighted sum:

| Component | Weight | Description |
|-----------|--------|-------------|
| Cosine Similarity | 35% | TF-IDF similarity between user skills and role profile |
| Skill Overlap | 35% | `|user_skills ∩ role_skills| / |role_skills|` |
| Experience Fit | 12% | Job Zone alignment with user's years of experience |
| Education Fit | 8% | Required education level vs. user's stated education |
| Domain Boost | 10% | Keyword match against role title/description |

### 4. Skill Gap Engine
For each recommended role:
- **Matched skills**: `user_skills ∩ role_skills`
- **Missing skills**: `role_skills − user_skills`, ranked by O\*NET importance score

### 5. Learning Roadmap Generation
Missing skills are tiered by importance rank:
- 🔴 **Foundation** (Top 3) — Learn these first
- 🟠 **Core Skills** (4–7) — Learn next
- 🟢 **Advanced** (8+) — Nice-to-have

### 6. Career Clustering
KMeans (k=20) on normalized TF-IDF role vectors groups similar occupations into career families for `"You might also like..."` suggestions.

---

## 📁 Project Structure

```
skill_gap_recommendation_engine/
├── db_30_2_excel/          # Raw O*NET dataset files (41 xlsx)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading, cleaning, merging
│   ├── recommender.py           # TF-IDF + hybrid scoring engine
│   └── evaluation.py            # Test personas and quality metrics
├── data/
│   └── processed/               # Cached processed data (auto-generated)
│       ├── role_df.csv
│       ├── long_df.csv
│       └── recommender.pkl
├── .streamlit/
│   └── config.toml              # Dark theme configuration
├── app.py                       # Streamlit web application
├── build_model.py               # One-shot pipeline: process + build + evaluate
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the Model (one-time)
```bash
python build_model.py
```
This processes all O\*NET files, builds the recommender, and runs evaluation. Takes ~30–60 seconds.

### 3. Launch the App
```bash
streamlit run app.py
```

---

## 🖥️ Application Features

| Tab | Description |
|-----|-------------|
| 🏆 **Career Matches** | Top-N matched careers with overall match %, skill overlap bars, matched/missing skills chips |
| 🔍 **Skill Gap** | Deep dive into one role: radar chart comparison, skill importance bars, full missing skills list |
| 🗺️ **Learning Roadmap** | Step-by-step learning plan tiered by priority with importance scores |
| 🌍 **Explore Careers** | Skill treemap, career cluster browser, role search with filters |
| 📄 **About** | Methodology, limitations, and future roadmap |

---

## 📈 Evaluation

5 curated "persona" test profiles were used for offline quality evaluation:

| Persona | Top Matched Role | Hit? |
|---------|-----------------|------|
| Data Analyst (Junior) | Data/Analytics role | ✅ |
| Machine Learning Engineer | ML/Data Science role | ✅ |
| Software Developer | Software/CS role | ✅ |
| Healthcare Manager | Medical/Health admin role | ✅ |
| Finance Analyst | Financial/Accounting role | ✅ |

**Overall Hit Rate: 80–100% across top-10 recommendations**  
**Average Career Family Diversity: 2–3 clusters per recommendation set**

---

## ⚠️ Limitations

1. **Static data**: Based on O\*NET 30.2 (Feb 2026); does not reflect real-time job market trends
2. **Skill normalization**: User must enter skills using standard O\*NET terminology for best matching
3. **No salary data**: Compensation information not included
4. **No course links**: Learning roadmap recommends skills but not specific courses
5. **English-only**: All occupational data is in English

---

## 🔮 Future Improvements

- 🔴 **NLP Resume Parser**: Auto-extract skills from uploaded resume PDF
- 🟠 **Live Job Data**: Integrate LinkedIn/Indeed APIs for real-time demand signals
- 🟡 **Course Integration**: Link roadmap skills to Coursera/Udemy/edX courses
- 🟢 **Salary Ranges**: Add BLS wage data per occupation
- 🔵 **Collaborative Filtering**: Learn from other users with similar profiles
- 🟣 **Multi-language Support**: Translate for global users

---

## 👤 Author

Built as a portfolio project demonstrating end-to-end data science engineering:
data fusion → NLP modeling → recommendation systems → production web app.

**Tech Stack:** Python · Pandas · Scikit-learn · TF-IDF · KMeans · Streamlit · Plotly
