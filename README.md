🚀 FUTURE_ML_03 — RESUME / CANDIDATE SCREENING SYSTEM

Future Interns | Machine Learning Internship
📌 Task 3 — Resume / Candidate Screening System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 OBJECTIVE

Build an explainable NLP-based system that compares resumes
with a job description, extracts relevant skills, scores
candidate fit, ranks candidates, and identifies missing skills.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚙️ WORKFLOW

📄 Resume Text
      ↓
🧹 Text Cleaning
      ↓
🔤 TF-IDF Vectorization
      ↓
📊 Resume-to-Role Similarity
      ↓
🛠️ Skill Extraction
      ↓
📈 Candidate Fit Score
      ↓
🏆 Candidate Ranking
      ↓
⚠️ Missing Skill Identification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🧠 NLP & ML APPROACH

TF-IDF represents resume and job-description text numerically.
Cosine similarity measures textual relevance between each resume
and the target role.

Required skills are extracted using a predefined skill dictionary
and matched against each resume.

Candidate Fit Score:

70% → TF-IDF text similarity
30% → Required skill coverage

This makes the ranking transparent and easy to explain.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ KEY FEATURES

✔ Resume text cleaning and preprocessing
✔ Skill extraction
✔ Job description parsing
✔ Resume-to-role similarity scoring
✔ Candidate ranking
✔ Required skill coverage
✔ Missing skill identification
✔ Visual candidate comparison
✔ Business-ready summary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 OUTPUTS

📄 candidate_ranking.csv
   → Ranked candidates with scores and skill gaps

📈 candidate_ranking.png
   → Visual comparison of candidate fit scores

📊 skill_coverage.png
   → Required skill coverage across candidates

📝 business_summary.txt
   → Explainable screening summary

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 PROJECT STRUCTURE

FUTURE_ML_03/
│
├── 📂 data/
│   ├── job_description.txt
│   └── resumes.csv
│
├── 📂 outputs/
│   ├── candidate_ranking.csv
│   ├── candidate_ranking.png
│   ├── skill_coverage.png
│   └── business_summary.txt
│
├── 🐍 resume_screening.py
├── 📄 requirements.txt
└── 📖 README.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️ HOW TO RUN

Install dependencies:

pip install -r requirements.txt

Run the screening system:

python resume_screening.py

Generated rankings, visualizations and the business summary
will be saved in the outputs/ folder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 DATA

The included data is a small simulated/anonymized resume set
created for demonstration and reproducibility.

The project can be extended to PDF/text resume datasets
allowed by the Future Interns task.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💼 BUSINESS USE

📌 Compare resumes against a specific role
🔎 Find relevant skills
🏆 Prioritize candidates for review
⚠️ Identify skill gaps
⏱️ Reduce manual screening effort

The output is a decision-support tool and should not replace
human review or independent hiring decisions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 INTERNSHIP

🏢 Future Interns
💻 Machine Learning Internship
📌 Task 3 — Resume / Candidate Screening System

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

👩‍💻 AUTHOR

Tanishka Janapure
Machine Learning Intern

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🖥️ FRONTEND UI

The project includes a Streamlit interface for interactive screening.

Run:

streamlit run app.py

The UI allows you to:

📄 View or edit the job description
📤 Upload a compatible resume CSV
🚀 Run candidate screening
🏆 View candidate rankings
📊 Compare fit scores
🔎 Inspect matched and missing skills
⬇️ Download the generated ranking CSV

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"# FUTURE_ML_03" 
