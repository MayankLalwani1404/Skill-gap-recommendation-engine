"""
Career Path & Skill Gap Recommendation Engine
Streamlit Application
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CareerMap AI – Skill Gap & Career Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --primary: #6C63FF;
    --accent: #00D4AA;
    --warning: #FF6B6B;
    --bg-card: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hero Banner */
.hero-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 3rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid rgba(108,99,255,0.3);
    box-shadow: 0 8px 40px rgba(108,99,255,0.2);
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #00D4AA, #6C63FF);
    background-size: 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear;
    margin: 0;
}
@keyframes shimmer {
    0% { background-position: 0% }
    100% { background-position: 200% }
}
.hero-sub {
    color: rgba(255,255,255,0.7);
    font-size: 1.1rem;
    margin-top: 0.5rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: transform 0.2s, box-shadow 0.2s;
}
.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.15);
}

/* Role Card */
.role-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(0,212,170,0.05));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    position: relative;
}
.role-card-rank {
    position: absolute;
    top: 1rem;
    right: 1.2rem;
    background: linear-gradient(135deg, #6C63FF, #00D4AA);
    color: white;
    border-radius: 30px;
    padding: 0.2rem 0.9rem;
    font-size: 0.8rem;
    font-weight: 700;
}
.role-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #e0e0ff;
    margin: 0 0 0.3rem;
}
.role-meta {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.6rem;
}

/* Match Bar */
.match-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 30px;
    height: 10px;
    margin: 0.4rem 0;
}
.match-bar-fill {
    height: 10px;
    border-radius: 30px;
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
}

/* Skill Chip */
.chip-container { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.5rem; }
.chip {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.74rem;
    font-weight: 500;
}
.chip-green  { background: rgba(0,212,170,0.18);  color: #00D4AA; border: 1px solid rgba(0,212,170,0.3); }
.chip-red    { background: rgba(255,107,107,0.18); color: #FF6B6B; border: 1px solid rgba(255,107,107,0.3); }
.chip-purple { background: rgba(108,99,255,0.18); color: #a89fff; border: 1px solid rgba(108,99,255,0.3); }
.chip-gray   { background: rgba(255,255,255,0.07); color: rgba(255,255,255,0.6); border: 1px solid rgba(255,255,255,0.12); }

/* Section Headings */
.section-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0ff;
    border-left: 4px solid #6C63FF;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
}

/* Roadmap Step */
.roadmap-step {
    border-left: 3px solid #6C63FF;
    padding: 0.7rem 1rem;
    margin-bottom: 0.6rem;
    border-radius: 0 12px 12px 0;
    background: rgba(108,99,255,0.07);
}
.roadmap-step.tier-0 { border-color: #FF6B6B; background: rgba(255,107,107,0.07); }
.roadmap-step.tier-1 { border-color: #FFB347; background: rgba(255,179,71,0.07); }
.roadmap-step.tier-2 { border-color: #00D4AA; background: rgba(0,212,170,0.07); }

/* Metric Cards */
.metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-box {
    flex: 1;
    min-width: 130px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label { font-size: 0.78rem; color: rgba(255,255,255,0.5); margin-top: 0.1rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b1e 0%, #13112a 100%);
    border-right: 1px solid rgba(108,99,255,0.2);
}

/* Expander */
details { border-radius: 12px !important; }

/* Streamlit tab overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.5rem 1.2rem;
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(108,99,255,0.3), rgba(0,212,170,0.15));
    border-color: rgba(108,99,255,0.5);
    color: #ffffff;
}

/* Streamlit buttons */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #5856D6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(108,99,255,0.4);
}

/* Multiselect tag */
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(108,99,255,0.3);
    border-radius: 20px;
}

/* Hide Streamlit watermark */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "recommender.pkl")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


@st.cache_resource(show_spinner=False)
def load_recommender():
    """Load pre-built recommender or build it if not found."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    # Build on the fly
    from src.data_preprocessing import get_processed_data
    from src.recommender import SkillRecommender
    role_df, long_df, role_skills = get_processed_data()
    rec = SkillRecommender(role_df, long_df, role_skills)
    rec.save()
    return rec


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pct_color(pct):
    if pct >= 70:   return "#00D4AA"
    elif pct >= 40: return "#FFB347"
    else:           return "#FF6B6B"


def chips_html(items, cls="chip-gray", max_show=20):
    shown = items[:max_show]
    html = '<div class="chip-container">'
    for item in shown:
        html += f'<span class="chip {cls}">{item}</span>'
    if len(items) > max_show:
        html += f'<span class="chip chip-gray">+{len(items)-max_show} more</span>'
    html += '</div>'
    return html


def match_bar_html(pct, label="Match"):
    color = pct_color(pct)
    return f"""
    <div style="margin: 0.3rem 0;">
      <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:rgba(255,255,255,0.6); margin-bottom:3px;">
        <span>{label}</span><span style="color:{color}; font-weight:700;">{pct:.1f}%</span>
      </div>
      <div class="match-bar-bg">
        <div class="match-bar-fill" style="width:{min(pct,100):.1f}%; background: linear-gradient(90deg, {color}, {color}88);"></div>
      </div>
    </div>"""


def role_card_html(rank, row, user_skills_set):
    score = row["match_score"]
    color = pct_color(score)
    matched = row["matched_skills"]
    missing_top5 = row["missing_skills"][:5]

    matched_html = chips_html([s.title() for s in matched[:10]], "chip-green")
    missing_html = chips_html([s.title() for s in missing_top5], "chip-red")

    zone_label = row.get("job_zone_label", "")
    edu_label  = row.get("edu_level_label", "")

    html = f"""
<div class="role-card">
  <span class="role-card-rank">#{rank} · {score:.1f}% Match</span>
  <p class="role-title">{row['title']}</p>
  <p class="role-meta">{zone_label} &nbsp;|&nbsp; 🎓 {edu_label}</p>
  {match_bar_html(row['skill_match_pct'], '🎯 Skill Overlap')}
  {match_bar_html(row['zone_score'], '📅 Experience Fit')}
  <div style="margin-top:0.8rem;">
    <div style="font-size:0.78rem; color:#00D4AA; font-weight:600; margin-bottom:3px;">✅ Skills You Have</div>
    {matched_html if matched else '<span style="color:rgba(255,255,255,0.35);font-size:0.8rem;">None matched</span>'}
  </div>
  <div style="margin-top:0.6rem;">
    <div style="font-size:0.78rem; color:#FF6B6B; font-weight:600; margin-bottom:3px;">📚 Top Missing Skills</div>
    {missing_html if missing_top5 else '<span style="color:rgba(255,255,255,0.35);font-size:0.8rem;">None! You qualify fully.</span>'}
  </div>
</div>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

PLOT_BG = "rgba(0,0,0,0)"
PLOT_FONT = "rgba(255,255,255,0.75)"
GRID_COLOR = "rgba(255,255,255,0.08)"


def make_bar_chart(labels, values, title, color_scale=None):
    colors = []
    for v in values:
        colors.append(pct_color(v))
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(
            color=values,
            colorscale=[[0, "#FF6B6B"], [0.5, "#FFB347"], [1, "#00D4AA"]],
            cmin=0, cmax=100,
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=PLOT_FONT, size=12),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16, family="Space Grotesk")),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="Inter"),
        height=max(300, len(labels) * 45),
        margin=dict(l=10, r=60, t=50, b=20),
        xaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR,
            ticksuffix="%", tickfont=dict(color=PLOT_FONT),
            range=[0, 110],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(color="white", size=12),
            showgrid=False,
        ),
        showlegend=False,
    )
    return fig


def make_radar_chart(categories, user_vals, role_vals, role_name):
    cats = categories + [categories[0]]
    u_vals = user_vals + [user_vals[0]]
    r_vals = role_vals + [role_vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=cats, fill="toself",
        name=role_name,
        line=dict(color="#6C63FF", width=2),
        fillcolor="rgba(108,99,255,0.15)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=u_vals, theta=cats, fill="toself",
        name="Your Profile",
        line=dict(color="#00D4AA", width=2.5),
        fillcolor="rgba(0,212,170,0.1)",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color=PLOT_FONT, size=9),
                gridcolor=GRID_COLOR,
            ),
            angularaxis=dict(
                tickfont=dict(color="white", size=11),
                gridcolor=GRID_COLOR,
            ),
        ),
        paper_bgcolor=PLOT_BG,
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=30, b=30),
        height=380,
        showlegend=True,
    )
    return fig


def make_skills_treemap(top_skills_df):
    fig = px.treemap(
        top_skills_df.head(40),
        path=["skill"],
        values="role_count",
        color="role_count",
        color_continuous_scale=["#302b63", "#6C63FF", "#00D4AA"],
        title="🌐 Most In-Demand Skills Across All Roles",
    )
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        font=dict(color="white", family="Inter"),
        title_font=dict(color="white", size=16, family="Space Grotesk"),
        margin=dict(t=50, l=0, r=0, b=0),
        height=420,
        coloraxis_showscale=False,
    )
    fig.update_traces(textfont=dict(color="white"))
    return fig


def make_roadmap_figure(roadmap):
    """Horizontal timeline for learning roadmap."""
    tiers = {"🔴 Foundation (Learn First)": 0, "🟠 Core Skills (Next)": 1, "🟢 Advanced / Nice-to-have": 2}
    tier_colors = ["#FF6B6B", "#FFB347", "#00D4AA"]
    tier_labels_short = ["Foundation", "Core", "Advanced"]

    fig = go.Figure()
    for step in roadmap:
        tier_idx = tiers.get(step["tier"], 2)
        fig.add_trace(go.Bar(
            x=[step["importance"]],
            y=[step["skill"]],
            orientation="h",
            marker=dict(color=tier_colors[tier_idx], opacity=0.85),
            name=tier_labels_short[tier_idx],
            showlegend=False,
            hovertemplate=(
                f"<b>{step['skill']}</b><br>"
                f"Category: {step['category']}<br>"
                f"Importance: {step['importance']:.2f}<br>"
                f"Tier: {step['tier']}<extra></extra>"
            ),
        ))

    # Legend entries
    for i, (label, color) in enumerate(zip(tier_labels_short, tier_colors)):
        fig.add_trace(go.Bar(
            x=[None], y=[None], name=label,
            marker_color=color, showlegend=True,
        ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_FONT, family="Inter"),
        height=max(300, len(roadmap) * 42),
        margin=dict(l=10, r=20, t=20, b=20),
        xaxis=dict(title="Importance Score", gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color=PLOT_FONT)),
        yaxis=dict(autorange="reversed", tickfont=dict(color="white", size=11), showgrid=False),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.05),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR – User Profile
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar(rec):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:2.5rem;">🧭</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.2rem;
                        font-weight:700; color:#e0e0ff;">CareerMap AI</div>
            <div style="font-size:0.75rem; color:rgba(255,255,255,0.4);">
                O*NET 30.2 Powered
            </div>
        </div>
        <hr style="border-color:rgba(108,99,255,0.2); margin:0.8rem 0;">
        """, unsafe_allow_html=True)

        st.markdown("### 👤 Your Profile")

        # Skills input
        all_skills = rec.get_all_skills()
        default_candidates = ["Python", "SQL", "Sql", "Data Analysis", "Statistics"]
        valid_defaults = [s for s in default_candidates if s in all_skills]

        user_skills = st.multiselect(
            "🎯 Your Current Skills",
            options=all_skills,
            default=valid_defaults,
            placeholder="Search and select skills…",
            help="Select all skills you currently have."
        )

        # Free-text additional skills
        extra_skills_raw = st.text_input(
            "✏️ Additional Skills (comma separated)",
            placeholder="e.g. Power BI, AutoCAD, Photoshop",
            help="Enter skills not in the dropdown."
        )
        extra_skills = [s.strip().lower() for s in extra_skills_raw.split(",") if s.strip()]
        all_user_skills = [s.lower() for s in user_skills] + extra_skills

        st.markdown("---")

        education = st.selectbox(
            "🎓 Highest Education",
            ["High School / GED", "Some College", "Associate's Degree",
             "Bachelor's Degree", "Master's Degree", "PhD / Doctoral"],
            index=3,
        )

        experience = st.selectbox(
            "💼 Work Experience",
            ["No Experience (Student)", "< 1 Year", "1–2 Years",
             "3–5 Years", "5–10 Years", "10+ Years"],
            index=2,
        )

        certifications = st.text_input(
            "📜 Certifications (optional)",
            placeholder="e.g. AWS, PMP, CPA",
        )
        cert_list = [c.strip().lower() for c in certifications.split(",") if c.strip()]
        all_user_skills += cert_list

        domain_input = st.text_input(
            "🌐 Domain Interests (optional)",
            placeholder="e.g. data, healthcare, finance",
        )
        domain_keywords = [d.strip() for d in domain_input.split(",") if d.strip()]

        st.markdown("---")
        top_n = st.slider("📊 Results to show", 5, 20, 10, step=5)

        st.markdown("---")
        find_btn = st.button("🔍 Find My Career Path", use_container_width=True)

        # Quick stats
        st.markdown(f"""
        <div class="metric-box" style="margin-top:1rem;">
            <div class="metric-value">{len(all_user_skills)}</div>
            <div class="metric-label">Skills Entered</div>
        </div>
        """, unsafe_allow_html=True)

    return all_user_skills, education, experience, domain_keywords, top_n, find_btn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🧭 CareerMap AI</h1>
        <p class="hero-sub">
            Powered by O*NET 30.2 · 1,000+ Careers · AI-Driven Skill Matching
        </p>
        <p style="color:rgba(255,255,255,0.45); font-size:0.85rem; margin-top:0.3rem;">
            Enter your skills in the sidebar → Get career matches, skill gaps & a learning roadmap
        </p>
    </div>
    """, unsafe_allow_html=True)


def tab_recommendations(recs, user_skills_set):
    if recs is None or recs.empty:
        st.info("⬅️ Fill your profile in the sidebar and click **Find My Career Path**")
        return

    n = len(recs)
    avg_score = recs["match_score"].mean()
    top_role = recs.iloc[0]["title"]
    n_matched_top = len(recs.iloc[0]["matched_skills"])

    # Summary Metrics
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-box">
        <div class="metric-value">{n}</div>
        <div class="metric-label">Careers Found</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{recs.iloc[0]['match_score']:.0f}%</div>
        <div class="metric-label">Best Match Score</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{n_matched_top}</div>
        <div class="metric-label">Skills Matched (Top Role)</div>
      </div>
      <div class="metric-box">
        <div class="metric-value">{recs['cluster'].nunique()}</div>
        <div class="metric-label">Career Families</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Bar chart overview
    labels = recs["title"].tolist()
    values = recs["match_score"].tolist()
    fig = make_bar_chart(labels[::-1], values[::-1], "🏆 Career Match Rankings")
    st.plotly_chart(fig, use_container_width=True)

    # Role cards
    st.markdown('<div class="section-heading">🎯 Your Best-Fit Careers</div>', unsafe_allow_html=True)
    for i, (_, row) in enumerate(recs.iterrows(), 1):
        st.markdown(role_card_html(i, row, user_skills_set), unsafe_allow_html=True)
        with st.expander(f"📖 Full Description – {row['title']}"):
            st.markdown(f"> {row['description']}")
            related = st.session_state.rec.get_related_roles(row["onet_code"], top_n=5)
            if related:
                st.markdown("**🔗 Similar Roles You Might Also Explore:**")
                st.markdown(chips_html(related, "chip-purple"), unsafe_allow_html=True)


def tab_skill_gap(recs, user_skills, rec):
    if recs is None or recs.empty:
        st.info("⬅️ Run a search first via the sidebar.")
        return

    st.markdown('<div class="section-heading">🔍 Deep Dive: Skill Gap Analysis</div>', unsafe_allow_html=True)

    # Role selector
    role_options = recs["title"].tolist()
    selected_title = st.selectbox("Select a career to analyze:", role_options)
    selected_row = recs[recs["title"] == selected_title].iloc[0]
    selected_code = selected_row["onet_code"]

    gap = rec.get_skill_gap(user_skills, selected_code)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box" style="text-align:center;">
            <div class="metric-value" style="color:#00D4AA;">{gap['match_percentage']:.1f}%</div>
            <div class="metric-label">Skill Coverage</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box" style="text-align:center;">
            <div class="metric-value" style="color:#6C63FF;">{gap['user_has']}</div>
            <div class="metric-label">Skills You Have</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box" style="text-align:center;">
            <div class="metric-value" style="color:#FF6B6B;">{gap['user_missing']}</div>
            <div class="metric-label">Skills You Need</div>
        </div>""", unsafe_allow_html=True)

    # Radar chart: matched dimensions
    matched = gap["matched_skills"]
    missing_top = gap["missing_skills"][:8]

    # Build radar from skill categories (first 6 matched vs missing importance)
    imp_dict = rec._imp_dict.get(selected_code, {})
    all_role_skills = rec.role_skills.get(selected_code, [])
    cats = sorted(set(s.title() for s in all_role_skills[:12]))[:6]
    user_set = set(s.lower() for s in user_skills)
    u_vals = [100 if c.lower() in user_set else 0 for c in cats]
    r_vals = [min((imp_dict.get(c.lower(), 2.5) / 5.0) * 100, 100) for c in cats]

    if len(cats) >= 3:
        fig = make_radar_chart(cats, u_vals, r_vals, selected_title[:40])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Skills You Already Have**")
        st.markdown(chips_html([s.title() for s in matched], "chip-green"), unsafe_allow_html=True)
    with col2:
        st.markdown("**📚 Skills You Need to Gain**")
        st.markdown(chips_html([s.title() for s in gap["missing_skills"]], "chip-red"), unsafe_allow_html=True)

    # Skill importance bars for missing
    if missing_top:
        st.markdown("**⚡ Missing Skills by Importance**")
        imp_vals = [imp_dict.get(s, 2.5) / 5.0 * 100 for s in missing_top]
        fig2 = make_bar_chart(
            [s.title() for s in missing_top][::-1],
            imp_vals[::-1],
            "Missing Skill Importance (for this role)",
        )
        st.plotly_chart(fig2, use_container_width=True)


def tab_roadmap(recs, user_skills, rec):
    if recs is None or recs.empty:
        st.info("⬅️ Run a search first via the sidebar.")
        return

    st.markdown('<div class="section-heading">🗺️ Your Personalized Learning Roadmap</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:rgba(255,255,255,0.6); font-size:0.9rem;'>"
        "Select a target role and get a step-by-step skill learning plan ranked by importance."
        "</p>", unsafe_allow_html=True
    )

    role_options = recs["title"].tolist()
    selected_title = st.selectbox("🎯 Target Role:", role_options, key="roadmap_role")
    selected_code = recs[recs["title"] == selected_title].iloc[0]["onet_code"]

    roadmap = rec.generate_roadmap(user_skills, selected_code, max_skills=15)

    if not roadmap:
        st.success("🎉 You already have all the required skills for this role! You're ready to apply.")
        return

    # Summary banner
    tiers_counts = {}
    for step in roadmap:
        t = step["tier"]
        tiers_counts[t] = tiers_counts.get(t, 0) + 1

    foundation_n = tiers_counts.get("🔴 Foundation (Learn First)", 0)
    core_n = tiers_counts.get("🟠 Core Skills (Next)", 0)
    adv_n = tiers_counts.get("🟢 Advanced / Nice-to-have", 0)

    st.markdown(f"""
    <div class="card" style="background:linear-gradient(135deg,rgba(108,99,255,0.1),rgba(0,212,170,0.05));">
        <b style="color:#e0e0ff;">📋 Roadmap Summary for: {selected_title}</b><br>
        <span style="color:#FF6B6B;">🔴 {foundation_n} Foundation skills</span> &nbsp;→&nbsp;
        <span style="color:#FFB347;">🟠 {core_n} Core skills</span> &nbsp;→&nbsp;
        <span style="color:#00D4AA;">🟢 {adv_n} Advanced skills</span>
    </div>
    """, unsafe_allow_html=True)

    # Roadmap chart
    fig = make_roadmap_figure(roadmap)
    st.plotly_chart(fig, use_container_width=True)

    # Step-by-step cards
    st.markdown("**📌 Step-by-Step Learning Plan**")
    for step in roadmap:
        tier_class = {
            "🔴 Foundation (Learn First)": "tier-0",
            "🟠 Core Skills (Next)": "tier-1",
            "🟢 Advanced / Nice-to-have": "tier-2",
        }.get(step["tier"], "tier-2")

        st.markdown(f"""
        <div class="roadmap-step {tier_class}">
            <div style="display:flex;align-items:center;gap:0.5rem;">
                <span style="font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.9);">
                    {step['priority']}. {step['skill']}
                </span>
                <span class="chip chip-gray" style="font-size:0.72rem;">{step['category']}</span>
            </div>
            <div style="font-size:0.78rem;color:rgba(255,255,255,0.45);margin-top:2px;">
                {step['tier']} &nbsp;·&nbsp; Importance: {step['importance']:.2f}/5
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Related roles suggestion
    related = rec.get_related_roles(selected_code, top_n=5)
    if related:
        st.markdown('<div class="section-heading">🔗 Similar Careers to Also Consider</div>', unsafe_allow_html=True)
        st.markdown(chips_html(related, "chip-purple"), unsafe_allow_html=True)


def tab_explore(rec):
    st.markdown('<div class="section-heading">🌍 Explore All Careers & Skills</div>', unsafe_allow_html=True)

    subtab1, subtab2, subtab3 = st.tabs(["🏆 Most In-Demand Skills", "📂 Career Families", "🔍 Search Roles"])

    with subtab1:
        top_skills = rec.top_skills_overall(40)
        fig = make_skills_treemap(top_skills)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top 20 Skills by Role Coverage**")
        top20 = top_skills.head(20)
        fig2 = make_bar_chart(
            top20["skill"].tolist()[::-1],
            (top20["role_count"] / rec.role_df.shape[0] * 100).tolist()[::-1],
            "Skills by % of Roles Requiring Them",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with subtab2:
        # Show clusters with top roles
        role_df = rec.role_df
        cluster_ids = sorted(role_df["cluster"].unique())
        cols = st.columns(2)
        for i, cid in enumerate(cluster_ids[:20]):
            with cols[i % 2]:
                cluster_roles = role_df[role_df["cluster"] == cid].sort_values("skill_count", ascending=False)
                sample_titles = cluster_roles["title_clean"].head(5).tolist()
                st.markdown(f"""
                <div class="card" style="min-height:120px;">
                    <div style="font-weight:700;color:#a89fff;margin-bottom:0.3rem;">
                        Career Cluster {cid + 1}
                    </div>
                    <div style="font-size:0.8rem;color:rgba(255,255,255,0.6);">
                        {len(cluster_roles)} roles
                    </div>
                    {chips_html(sample_titles[:4], 'chip-gray')}
                </div>
                """, unsafe_allow_html=True)

    with subtab3:
        search_q = st.text_input("🔍 Search career titles", placeholder="e.g. engineer, nurse, analyst")
        jz_filter = st.selectbox(
            "Filter by Experience Level",
            ["All", "1 – Entry", "2 – Some Prep", "3 – Medium", "4 – Considerable", "5 – Extensive"]
        )
        filtered = rec.role_df.copy()
        if search_q:
            filtered = filtered[filtered["title_clean"].str.lower().str.contains(search_q.lower())]
        if jz_filter != "All":
            jz_num = int(jz_filter[0])
            filtered = filtered[filtered["job_zone"] == jz_num]

        st.markdown(f"**{len(filtered)} roles found**")
        display_df = filtered[["title_clean", "job_zone", "edu_level", "skill_count"]].rename(columns={
            "title_clean": "Career Title",
            "job_zone": "Job Zone (1-5)",
            "edu_level": "Education Level",
            "skill_count": "# of Skills",
        }).head(50)
        st.dataframe(display_df, use_container_width=True, height=450)


def tab_about():
    st.markdown("""
    <div class="hero-banner" style="padding:2rem; text-align:left;">
        <h2 style="font-family:'Space Grotesk',sans-serif;color:#e0e0ff;">📄 About CareerMap AI</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 What It Does
        **CareerMap AI** is a career guidance platform powered by O*NET occupational data.
        Given your current skills, education, and background, it:
        - Recommends the **most suitable careers**
        - Shows your **exact skill match %**
        - Identifies your **skill gaps**
        - Generates a **prioritized learning roadmap**

        ### 📊 Data Source
        - **O*NET 30.2 Database** (February 2026 Release)
        - 1,016 occupations · 35 dataset files
        - Skills, Knowledge, and Technology Skills combined
        - Job Zone (experience level) and Education requirements included
        - Licensed under **Creative Commons Attribution 4.0**
        """)
    with col2:
        st.markdown("""
        ### ⚙️ Methodology
        1. **Data Fusion** – Skills + Knowledge + Tech Skills merged per role
        2. **TF-IDF Vectorization** – Role skill profiles as weighted term vectors
        3. **Cosine Similarity** – User profile vs all 1,000+ role vectors
        4. **Hybrid Score** = 40% cosine + 30% overlap + 15% experience fit + 10% education fit + 5% domain boost
        5. **Skill Gap** – Set difference between user skills and role requirements, ranked by O*NET importance
        6. **Learning Roadmap** – Missing skills sorted by importance, tiered into Foundation → Core → Advanced

        ### ⚠️ Limitations
        - Recommendation quality depends on using correct skill names
        - Domain-specific tools (e.g. "Ansys", "SAP") may not always match
        - Career paths are based on static O*NET data, not real-time job market signals
        """)

    st.markdown("""
    ### 🚀 Future Improvements
    - 🔴 Live job posting integration (LinkedIn, Indeed APIs)
    - 🟠 Online course integration (Coursera, Udemy, edX links per skill)
    - 🟢 NLP resume parser: paste your resume, auto-extract skills
    - 🔵 Salary range data by role
    - 🟣 Personalized flashcard system for skill learning
    """)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load model
    with st.spinner("⚙️ Loading CareerMap AI engine…"):
        try:
            rec = load_recommender()
        except Exception as e:
            st.error(f"❌ Failed to load the recommendation engine. Run `python build_model.py` first.\n\n{e}")
            st.stop()

    st.session_state.rec = rec

    # Hero
    render_hero()

    # Sidebar – user profile
    all_user_skills, education, experience, domain_keywords, top_n, find_btn = render_sidebar(rec)

    # Session state for results
    if "recs" not in st.session_state:
        st.session_state.recs = None
    if "user_skills" not in st.session_state:
        st.session_state.user_skills = []

    if find_btn or (st.session_state.recs is not None):
        if find_btn:
            if not all_user_skills:
                st.warning("⚠️ Please enter at least one skill.")
                return
            with st.spinner("🔍 Analysing your profile and finding best-fit careers…"):
                recs = rec.recommend(
                    user_skills=all_user_skills,
                    education=education,
                    experience=experience,
                    domain_keywords=domain_keywords if domain_keywords else None,
                    top_n=top_n,
                )
            st.session_state.recs = recs
            st.session_state.user_skills = all_user_skills

        recs = st.session_state.recs
        user_skills = st.session_state.user_skills
        user_skills_set = set(s.lower() for s in user_skills)

        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Career Matches",
            "🔍 Skill Gap",
            "🗺️ Learning Roadmap",
            "🌍 Explore Careers",
            "📄 About",
        ])

        with tab1:
            tab_recommendations(recs, user_skills_set)
        with tab2:
            tab_skill_gap(recs, user_skills, rec)
        with tab3:
            tab_roadmap(recs, user_skills, rec)
        with tab4:
            tab_explore(rec)
        with tab5:
            tab_about()
    else:
        # Landing state – show explore tab
        tab4, tab5 = st.tabs(["🌍 Explore Careers", "📄 About"])
        with tab4:
            tab_explore(rec)
        with tab5:
            tab_about()


if __name__ == "__main__":
    main()
