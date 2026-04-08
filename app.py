# ============================================================
#   Netflix Content Intelligence — Streamlit App
#   app.py
# ============================================================
#
# FOLDER STRUCTURE (create this on your computer):
#
#   netflix-app/
#   ├── app.py                  ← this file
#   ├── netflix_cleaned.csv     ← your cleaned data from ETL
#   └── requirements.txt        ← libraries list
#
# HOW TO RUN LOCALLY:
#   pip install streamlit pandas scikit-learn
#   streamlit run app.py
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG — must be the very first Streamlit command
# ============================================================

st.set_page_config(
    page_title = "Netflix Content Intelligence",
    page_icon  = "🎬",
    layout     = "wide",
)


# ============================================================
# CUSTOM CSS — Netflix dark theme
# ============================================================

st.markdown("""
<style>
    /* Dark background like Netflix */
    .stApp { background-color: #141414; color: #ffffff; }

    /* Red header bar */
    .netflix-header {
        background-color: #E50914;
        padding: 1.2rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .netflix-header h1 {
        color: white;
        font-size: 28px;
        margin: 0;
    }
    .netflix-header p {
        color: rgba(255,255,255,0.8);
        font-size: 14px;
        margin: 4px 0 0;
    }

    /* Result cards */
    .result-card {
        background-color: #1f1f1f;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .result-card h4 { color: #ffffff; margin: 0 0 4px; font-size: 15px; }
    .result-card p  { color: #aaaaaa; margin: 0; font-size: 13px; }

    /* Score badge */
    .score-badge {
        background-color: #E50914;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1f1f1f;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-card .value { font-size: 28px; font-weight: bold; color: #E50914; }
    .metric-card .label { font-size: 13px; color: #aaaaaa; margin-top: 4px; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #aaaaaa; }
    .stTabs [aria-selected="true"] { color: #E50914 !important; border-bottom-color: #E50914 !important; }

    /* Button */
    .stButton > button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton > button:hover { background-color: #c40812; }

    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stTextArea textarea {
        background-color: #2a2a2a;
        color: white;
        border: 1px solid #444;
    }
    .stSlider > div { color: white; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD & CACHE DATA
# @st.cache_data means this only runs ONCE — makes the app fast
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_cleaned.csv")
    return df

@st.cache_resource
def build_recommender(df):
    """Build TF-IDF recommender — runs once, cached forever."""

    def make_soup(row):
        desc     = str(row["description"])   if pd.notna(row["description"])  else ""
        genres   = str(row["listed_in"]).replace(",", " ") if pd.notna(row["listed_in"]) else ""
        cast     = str(row["cast"])          if pd.notna(row["cast"])         else ""
        cast_top = " ".join([c.strip().replace(" ", "_") for c in cast.split(",")[:3]])
        director = str(row["director"]).replace(" ", "_") if pd.notna(row["director"]) else ""
        return f"{desc} {genres} {cast_top} {director}"

    df["soup"] = df.apply(make_soup, axis=1)

    tfidf     = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_mat = tfidf.fit_transform(df["soup"])
    cos_sim   = cosine_similarity(tfidf_mat, tfidf_mat)
    title_idx = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    return cos_sim, title_idx

@st.cache_resource
def build_popularity_model(df):
    """Train Random Forest popularity predictor — runs once."""

    features = ["type", "primary_genre", "primary_country", "rating",
                "cast_size", "genre_count", "release_year",
                "month_added", "duration_value"]

    model_df = df[df["tmdb_popularity"].notna()].copy()
    model_df = model_df[features + ["tmdb_popularity"]].dropna()

    encoders = {}
    for col in ["type", "primary_genre", "primary_country", "rating"]:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        encoders[col] = le

    X = model_df[features]
    y = model_df["tmdb_popularity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    avg_popularity = round(float(y.mean()), 2)
    return model, encoders, avg_popularity

@st.cache_resource
def build_genre_model(df):
    """Train genre classifier — runs once."""

    model_df   = df[df["description"].notna() & df["primary_genre"].notna()].copy()
    top_genres = model_df["primary_genre"].value_counts().head(8).index.tolist()
    model_df   = model_df[model_df["primary_genre"].isin(top_genres)]

    X = model_df["description"]
    y = model_df["primary_genre"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf  = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tf, y_train)

    accuracy = round((model.predict(X_test_tf) == y_test).mean() * 100, 1)
    return model, tfidf, accuracy


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="netflix-header">
    <h1>🎬 Netflix Content Intelligence</h1>
    <p>3 ML models · 8,800+ titles · Powered by TMDB data</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA & MODELS
# ============================================================

with st.spinner("Loading data and training models... (first load takes ~30 seconds)"):
    df = load_data()
    cos_sim, title_idx         = build_recommender(df)
    pop_model, encoders, avg_pop = build_popularity_model(df)
    genre_model, tfidf_genre, genre_accuracy = build_genre_model(df)

# Quick stats row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="value">{len(df):,}</div><div class="label">Total Titles</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="value">{df["primary_country"].nunique()}</div><div class="label">Countries</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="value">{df["primary_genre"].nunique()}</div><div class="label">Genres</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="value">{genre_accuracy}%</div><div class="label">Genre Model Accuracy</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# 3 TABS — one per model
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "🎯  Recommender",
    "📈  Popularity Predictor",
    "🏷️  Genre Classifier",
])


# ────────────────────────────────────────────────────────────
# TAB 1 — CONTENT RECOMMENDER
# ────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Content-Based Recommender")
    st.caption("Enter any Netflix title and get 10 similar recommendations based on genre, cast, and description.")

    col_input, col_num = st.columns([3, 1])
    with col_input:
        user_title = st.text_input("Enter a Netflix title", placeholder="e.g. Stranger Things, The Crown, Bird Box")
    with col_num:
        num_recs = st.slider("Number of results", min_value=5, max_value=15, value=10)

    if st.button("Get Recommendations", key="rec_btn"):
        if not user_title.strip():
            st.warning("Please enter a title first.")
        elif user_title.strip().lower() not in title_idx:
            st.error(f"'{user_title}' not found in dataset. Check the spelling.")
            # Show some example titles to help
            st.info("Some titles you can try: " +
                    ", ".join(df["title"].sample(5).tolist()))
        else:
            idx        = title_idx[user_title.strip().lower()]
            sim_scores = sorted(enumerate(cos_sim[idx]), key=lambda x: x[1], reverse=True)[1:num_recs+1]
            top_df     = df.iloc[[i[0] for i in sim_scores]][
                ["title", "type", "listed_in", "release_year", "primary_country"]
            ].copy()
            top_df["similarity"] = [round(s[1], 3) for s in sim_scores]
            top_df = top_df.reset_index(drop=True)

            st.success(f"Showing {num_recs} titles similar to **{user_title}**")

            # Display as cards
            for _, row in top_df.iterrows():
                st.markdown(f"""
                <div class="result-card">
                    <h4>{row['title']} &nbsp;
                        <span class="score-badge">Score: {row['similarity']}</span>
                    </h4>
                    <p>{row['type']} &nbsp;·&nbsp; {row['listed_in']} &nbsp;·&nbsp;
                       {row['primary_country']} &nbsp;·&nbsp; {int(row['release_year']) if pd.notna(row['release_year']) else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# TAB 2 — POPULARITY PREDICTOR
# ────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Popularity Predictor")
    st.caption("Fill in details about a hypothetical title and predict its TMDB popularity score.")

    col_a, col_b = st.columns(2)

    with col_a:
        content_type = st.selectbox("Content Type",  ["Movie", "TV Show"])
        genre        = st.selectbox("Primary Genre",  sorted(df["primary_genre"].dropna().unique().tolist()))
        country      = st.selectbox("Country",        sorted(df["primary_country"].dropna().unique().tolist()))
        rating       = st.selectbox("Age Rating",     sorted(df["rating"].dropna().unique().tolist()))

    with col_b:
        cast_size    = st.slider("Cast Size (number of actors)", 1, 20, 5)
        genre_count  = st.slider("Number of Genres",             1, 5,  2)
        release_year = st.slider("Release Year",                 2000, 2025, 2023)
        month_added  = st.slider("Month Added to Netflix",       1, 12, 10)
        duration     = st.slider("Duration (mins for movie / seasons for show)", 20, 200, 100)

    if st.button("Predict Popularity", key="pop_btn"):

        def encode_val(col, val):
            le = encoders[col]
            if val in le.classes_:
                return le.transform([val])[0]
            return 0

        input_df = pd.DataFrame([{
            "type":            encode_val("type",            content_type),
            "primary_genre":   encode_val("primary_genre",   genre),
            "primary_country": encode_val("primary_country", country),
            "rating":          encode_val("rating",          rating),
            "cast_size":       cast_size,
            "genre_count":     genre_count,
            "release_year":    release_year,
            "month_added":     month_added,
            "duration_value":  duration,
        }])

        prediction = round(float(pop_model.predict(input_df)[0]), 2)

        st.markdown("<br>", unsafe_allow_html=True)

        # Result
        if prediction >= avg_pop:
            st.success(f"🔥 Predicted Popularity Score: **{prediction}** (Average: {avg_pop})")
            st.balloons()
            st.markdown("This content is predicted to be **above average** in popularity. Could be a hit!")
        else:
            st.warning(f"📉 Predicted Popularity Score: **{prediction}** (Average: {avg_pop})")
            st.markdown("This content is predicted to be **below average** in popularity. Consider a stronger cast or more popular genre.")

        # Feature importance chart
        st.markdown("#### What matters most for popularity?")
        feature_names = ["type", "primary_genre", "primary_country", "rating",
                         "cast_size", "genre_count", "release_year", "month_added", "duration_value"]
        importance_df = pd.DataFrame({
            "Feature":    feature_names,
            "Importance": pop_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        st.bar_chart(importance_df.set_index("Feature")["Importance"])


# ────────────────────────────────────────────────────────────
# TAB 3 — GENRE CLASSIFIER
# ────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Genre Classifier")
    st.caption("Type any show or movie description and the model will predict its genre using NLP.")

    st.markdown(f"Model accuracy: **{genre_accuracy}%** on test data")

    user_desc = st.text_area(
        "Enter a description",
        placeholder="e.g. A group of teenagers discover they have supernatural powers and must fight an ancient evil...",
        height=120
    )

    if st.button("Predict Genre", key="genre_btn"):
        if not user_desc.strip():
            st.warning("Please enter a description first.")
        else:
            text_tfidf  = tfidf_genre.transform([user_desc])
            predicted   = genre_model.predict(text_tfidf)[0]
            proba       = genre_model.predict_proba(text_tfidf)[0]

            confidence_df = pd.DataFrame({
                "Genre":      genre_model.classes_,
                "Confidence": [round(p * 100, 1) for p in proba]
            }).sort_values("Confidence", ascending=False).reset_index(drop=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"🏷️ Predicted Genre: **{predicted}**")

            st.markdown("#### Confidence scores for all genres")
            for _, row in confidence_df.iterrows():
                bar_width = int(row["Confidence"] * 3)
                color = "#E50914" if row["Genre"] == predicted else "#444"
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:13px;color:#ccc;">{row['Genre']}</span>
                        <span style="font-size:13px;color:#ccc;">{row['Confidence']}%</span>
                    </div>
                    <div style="background:#333;border-radius:4px;height:8px;">
                        <div style="background:{color};width:{min(bar_width, 300)}px;height:8px;border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Show example real titles from that genre
            st.markdown(f"#### Sample Netflix titles in '{predicted}'")
            sample = df[df["primary_genre"] == predicted][["title", "release_year", "primary_country"]].sample(
                min(5, len(df[df["primary_genre"] == predicted]))
            )
            st.dataframe(sample.reset_index(drop=True), use_container_width=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#666;font-size:12px;'>"
    "Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; "
    "Data: Netflix + TMDB API"
    "</p>",
    unsafe_allow_html=True
)
