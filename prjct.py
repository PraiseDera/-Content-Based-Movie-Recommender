import streamlit as st
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender 🎬",
    page_icon="🍿",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS for style
# --------------------------------------------------
st.markdown("""
    <style>
    /* Hide hamburger menu & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Page background */
    .stApp {
        background: linear-gradient(120deg, #1c1c1c, #2c2c2c);
        color: #ffffff;
    }

    /* Titles and text */
    h1, h2, h3 {
        color: #f5c518;
        font-family: 'Arial Black', sans-serif;
    }

    /* Selectbox styling */
    div.stSelectbox > div > div > span {
        color: #f5c518;
        font-weight: bold;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #f5c518;
        color: #000000;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    div.stButton > button:hover {
        background-color: #ffea70;
        color: #000000;
    }

    /* Recommendations box */
    .recommendation {
        background-color: #333333;
        padding: 1em;
        margin-bottom: 0.5em;
        border-radius: 10px;
        font-size: 18px;
        color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🍿 Content-Based Movie Recommender")
st.markdown("""
<p style='font-size:18px;color:#ffffff'>
Select a movie below and get <b>10 similar movies</b> instantly! 
Recommendations are powered by TF-IDF + Cosine Similarity, memory-efficient for large datasets.
</p>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load data (cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    df = df[['movieId', 'title', 'genres']]
    df['genres'] = df['genres'].str.replace('|', ' ', regex=False)
    return df


df = load_data()


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df['title_clean'] = df['title'].apply(clean_text)
df['genres_clean'] = df['genres'].apply(clean_text)
df['content'] = df['title_clean'] + ' ' + df['genres_clean']


# --------------------------------------------------
# Vectorization
# --------------------------------------------------
@st.cache_resource
def vectorize(corpus):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    matrix = tfidf.fit_transform(corpus)
    return tfidf, matrix


tfidf, tfidf_matrix = vectorize(df['content'])

# --------------------------------------------------
# Index mapping
# --------------------------------------------------
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# --------------------------------------------------
# Recommendation function
# --------------------------------------------------
def recommend(movie_title, n=10):
    idx = indices[movie_title]

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][1:n + 1]

    return df['title'].iloc[top_indices]


# --------------------------------------------------
# Sidebar for selection
# --------------------------------------------------
st.sidebar.header("🎬 Movie Selection")
movie_selected = st.sidebar.selectbox("Choose a movie", sorted(df['title'].unique()))

# --------------------------------------------------
# Recommendation button
# --------------------------------------------------
if st.button("Get Recommendations 🚀"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend(movie_selected)

    st.subheader(f"Movies similar to 🎥 {movie_selected}")
    for i, movie in enumerate(recommendations, start=1):
        st.markdown(f"<div class='recommendation'>{i}. {movie}</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

