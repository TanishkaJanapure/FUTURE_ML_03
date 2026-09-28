from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

REQUIRED_SKILLS = [
    "Python", "Machine Learning", "Scikit-learn", "Pandas", "NumPy", "NLP",
    "Natural Language Processing", "SQL", "Git", "Statistics", "Data Preprocessing",
    "Feature Engineering", "Model Evaluation"
]
PREFERRED_SKILLS = ["Deep Learning", "TensorFlow", "PyTorch", "Docker", "AWS"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text):
    cleaned = clean_text(text)
    found = []
    for skill in REQUIRED_SKILLS + PREFERRED_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, cleaned):
            found.append(skill)
    return found


def main():
    df = pd.read_csv(DATA_DIR / "resumes.csv")
    job_text = (DATA_DIR / "job_description.txt").read_text(encoding="utf-8")
    df["combined_text"] = df["current_role"].fillna("") + " " + df["skills"].fillna("") + " " + df["resume_text"].fillna("")
    df["clean_text"] = df["combined_text"].map(clean_text)

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform([clean_text(job_text)] + df["clean_text"].tolist())
    df["text_similarity"] = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    df["matched_skills"] = df["combined_text"].map(extract_skills)
    df["matched_required_skills"] = df["matched_skills"].map(lambda x: [s for s in x if s in REQUIRED_SKILLS])
    df["missing_required_skills"] = df["matched_required_skills"].map(lambda x: [s for s in REQUIRED_SKILLS if s not in x])
    df["skill_coverage"] = df["matched_required_skills"].map(lambda x: len(set(x)) / len(REQUIRED_SKILLS))
    df["fit_score"] = (0.70 * df["text_similarity"] + 0.30 * df["skill_coverage"]) * 100
    df = df.sort_values(["fit_score", "skill_coverage"], ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    ranked = df[["rank", "candidate_id", "name", "current_role", "fit_score", "text_similarity", "skill_coverage", "matched_required_skills", "missing_required_skills"]].copy()
    ranked["matched_required_skills"] = ranked["matched_required_skills"].map(lambda x: ", ".join(x))
    ranked["missing_required_skills"] = ranked["missing_required_skills"].map(lambda x: ", ".join(x))
    ranked.to_csv(OUT_DIR / "candidate_ranking.csv", index=False)

    top = ranked.head(10).sort_values("fit_score")
    plt.figure(figsize=(10, 6))
    plt.barh(top["name"], top["fit_score"])
    plt.xlabel("Role Fit Score")
    plt.ylabel("Candidate")
    plt.title("Top Candidate Ranking")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "candidate_ranking.png", dpi=160)
    plt.close()

    counts = {skill: int(ranked["matched_required_skills"].str.contains(re.escape(skill), case=False, regex=True).sum()) for skill in REQUIRED_SKILLS}
    skill_df = pd.DataFrame({"skill": list(counts), "candidates": list(counts.values())}).sort_values("candidates")
    plt.figure(figsize=(10, 6))
    plt.barh(skill_df["skill"], skill_df["candidates"])
    plt.xlabel("Candidates Matching Skill")
    plt.ylabel("Required Skill")
    plt.title("Required Skill Coverage Across Candidates")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "skill_coverage.png", dpi=160)
    plt.close()

    top_candidate = ranked.iloc[0]
    summary = f"""RESUME SCREENING SUMMARY

Job role: Machine Learning Engineer
Candidates screened: {len(ranked)}

Top ranked candidate: {top_candidate['name']}
Role fit score: {top_candidate['fit_score']:.1f}/100
Text similarity: {top_candidate['text_similarity']:.3f}
Required skill coverage: {top_candidate['skill_coverage'] * 100:.1f}%

Matched required skills:
{top_candidate['matched_required_skills']}

Missing required skills:
{top_candidate['missing_required_skills'] or 'None'}

SCORING LOGIC
70% - TF-IDF cosine similarity between resume and job description
30% - Required skill coverage

This is an explainable decision-support output and should support,
not replace, human review in a real recruitment process.
"""
    (OUT_DIR / "business_summary.txt").write_text(summary, encoding="utf-8")
    print(f"Candidates screened: {len(ranked)}")
    print(f"Top candidate: {top_candidate['name']}")
    print(f"Fit score: {top_candidate['fit_score']:.1f}/100")


if __name__ == "__main__":
    main()
