from pathlib import Path
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import base64

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

REQUIRED_SKILLS = [
    "Python", "Machine Learning", "Scikit-learn", "Pandas", "NumPy",
    "NLP", "Natural Language Processing", "SQL", "Git", "Statistics",
    "Data Preprocessing", "Feature Engineering", "Model Evaluation"
]

PREFERRED_SKILLS = [
    "Deep Learning", "TensorFlow", "PyTorch", "Docker", "AWS"
]

ALL_SKILLS = REQUIRED_SKILLS + PREFERRED_SKILLS


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text):
    cleaned = clean_text(text)
    found = []

    for skill in ALL_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"

        if re.search(pattern, cleaned):
            found.append(skill)

    return found


def screen_candidates(resumes, job_text):
    resumes = resumes.copy()

    resumes["combined_text"] = (
        resumes["current_role"].fillna("")
        + " "
        + resumes["skills"].fillna("")
        + " "
        + resumes["resume_text"].fillna("")
    )

    resumes["clean_text"] = resumes["combined_text"].map(clean_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000
    )

    matrix = vectorizer.fit_transform(
        [job_text] + resumes["clean_text"].tolist()
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    resumes["matched_skills_list"] = resumes[
        "combined_text"
    ].map(extract_skills)

    resumes["required_matched"] = resumes[
        "matched_skills_list"
    ].map(
        lambda x: [s for s in x if s in REQUIRED_SKILLS]
    )

    resumes["missing"] = resumes[
        "required_matched"
    ].map(
        lambda x: [
            s for s in REQUIRED_SKILLS
            if s not in x
        ]
    )

    resumes["skill_coverage"] = resumes[
        "required_matched"
    ].map(
        lambda x: len(set(x)) / len(REQUIRED_SKILLS)
    )

    resumes["text_similarity"] = similarity

    resumes["fit_score"] = (
        0.70 * resumes["text_similarity"]
        + 0.30 * resumes["skill_coverage"]
    ) * 100

    resumes = resumes.sort_values(
        ["fit_score", "skill_coverage"],
        ascending=False
    ).reset_index(drop=True)

    resumes["rank"] = resumes.index + 1

    return resumes


st.set_page_config(
    page_title="AI Resume Screening",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
BG_IMAGE = BASE_DIR / "assets" / "bg.jpg"

with open(BG_IMAGE, "rb") as f:
    bg_base64 = base64.b64encode(f.read()).decode()

st.markdown(f"""
<style>

.stApp {{
    background-image:
        linear-gradient(rgba(245, 248, 252, 0.90), rgba(245, 248, 252, 0.90)),
        url("data:image/jpeg;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
}}

.subtitle {{
    color: #000000 !important;
    font-size: 16px;
    margin-top: -14px;
    margin-bottom: 28px;
}}

</style>
""", unsafe_allow_html=True)

st.title(" AI Resume Screening")

st.markdown(
    '<div class="subtitle">Compare resumes with a job description, rank candidates, and identify missing skills.</div>',
    unsafe_allow_html=True
)
with st.sidebar:
    st.header("⚙️ Screening Settings")

    st.info(
        "Scoring: 70% TF-IDF similarity + 30% required-skill coverage"
    )

    st.caption(
        "This tool supports human review; it does not replace recruitment decisions."
    )


job_default = (
    DATA_DIR / "job_description.txt"
).read_text(encoding="utf-8")

job_text = st.text_area(
    "💼 Job Description",
    value=job_default,
    height=220
)


uploaded = st.file_uploader(
    "📄 Upload a resume CSV (optional)",
    type=["csv"]
)


if uploaded:
    resumes = pd.read_csv(uploaded)
else:
    resumes = pd.read_csv(
        DATA_DIR / "resumes.csv"
    )


required_cols = {
    "candidate_id",
    "name",
    "current_role",
    "skills",
    "resume_text"
}


if not required_cols.issubset(resumes.columns):
    st.error(
        "CSV must contain: candidate_id, name, current_role, skills, resume_text"
    )
    st.stop()


if st.button(
    "🚀 Screen Candidates",
    type="primary",
    use_container_width=True
):
    results = screen_candidates(
        resumes,
        job_text
    )

    st.session_state["results"] = results


if "results" in st.session_state:

    results = st.session_state["results"]
    top = results.iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Candidates",
        len(results)
    )

    c2.metric(
        "🏆 Top Candidate",
        top["name"]
    )

    c3.metric(
        "📈 Top Fit Score",
        f"{top['fit_score']:.1f}/100"
    )

    c4.metric(
        "🎯 Skill Coverage",
        f"{top['skill_coverage'] * 100:.1f}%"
    )


    st.subheader("🏆 Candidate Ranking")

    display = results[
        [
            "rank",
            "name",
            "current_role",
            "fit_score",
            "text_similarity",
            "skill_coverage"
        ]
    ].copy()


    display["fit_score"] = (
        display["fit_score"].round(1)
    )


    display["text_similarity"] = (
        display["text_similarity"].round(3)
    )


    display["skill_coverage"] = (
        display["skill_coverage"] * 100
    ).round(1).astype(str) + "%"


    display.columns = [
        "Rank",
        "Candidate",
        "Role",
        "Fit Score",
        "Text Similarity",
        "Skill Coverage"
    ]


    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("📊 Fit Score Overview")


    chart = (
        results[
            ["name", "fit_score"]
        ]
        .set_index("name")
        .sort_values("fit_score")
    )


    st.bar_chart(chart)


    selected = st.selectbox(
        "🔎 Inspect Candidate",
        results["name"].tolist()
    )


    candidate = results[
        results["name"] == selected
    ].iloc[0]


    left, right = st.columns(2)


    with left:

        st.markdown(
            "**Matched required skills**"
        )

        skills = candidate["required_matched"]

        st.write(
            ", ".join(skills)
            if skills
            else "No required skills matched"
        )

        st.markdown(
            f'**Fit Score:** '
            f'{candidate["fit_score"]:.1f}/100'
        )

        st.markdown(
            f'**Text Similarity:** '
            f'{candidate["text_similarity"]:.3f}'
        )


    with right:

        st.markdown(
            "**Missing required skills**"
        )

        missing = candidate["missing"]

        st.write(
            ", ".join(missing)
            if missing
            else "None"
        )

        st.markdown(
            f'**Skill Coverage:** '
            f'{candidate["skill_coverage"] * 100:.1f}%'
        )


    csv = results[
        [
            "rank",
            "candidate_id",
            "name",
            "current_role",
            "fit_score",
            "text_similarity",
            "skill_coverage",
            "required_matched",
            "missing"
        ]
    ].copy()


    csv["required_matched"] = csv[
        "required_matched"
    ].map(
        lambda x: ", ".join(x)
    )


    csv["missing"] = csv[
        "missing"
    ].map(
        lambda x: ", ".join(x)
    )


    st.download_button(
        "⬇️ Download Ranking CSV",
        csv.to_csv(index=False),
        "candidate_ranking.csv",
        "text/csv"
    )


else:

    st.info(
        "Enter or edit the job description, then click "
        "**Screen Candidates** to generate the ranking."
    )


st.markdown("---")

st.caption(
    "FUTURE_ML_03 • Future Interns Machine Learning Internship "
    "• Resume / Candidate Screening System"
)