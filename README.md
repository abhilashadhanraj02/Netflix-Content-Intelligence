# Netflix Content Intelligence System

An end-to-end data science project using Netflix data enriched with TMDB API.

## Live Demo
[Open the App](https://netflix-content-intelligence-pk9wjthzbxb9wgzaphdbng.streamlit.app/)
[Tableau Dashboard](https://public.tableau.com/app/profile/ra.bd/viz/NetflixContentIntelligenceDashboard/MLInsights?publish=yes)

## What I Built
- ETL pipeline: cleaned 8,800+ Netflix titles + enriched with TMDB API
- EDA: 9 charts revealing content trends across 190+ countries
- ML Model 1: Content-based recommender (TF-IDF + Cosine Similarity)
- ML Model 2: Popularity predictor (Random Forest, R²=0.XX)
- ML Model 3: Genre classifier (Logistic Regression, XX% accuracy)
- Tableau: 3 interactive dashboards

## Tech Stack
Python · Pandas · Scikit-learn · Streamlit · Tableau · TMDB API

## How to Run Locally
pip install -r requirements.txt
streamlit run app.py
