
import streamlit as st
import pandas as pd
import os
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Page configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

# Load datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies = pd.read_pickle(os.path.join(BASE_DIR, "movies.pkl"))
ratings_clean = pd.read_pickle(os.path.join(BASE_DIR, "ratings.pkl"))

# Create movie-user matrix
movie_user_matrix = ratings_clean.pivot(
    index="MovieID",
    columns="UserID",
    values="Rating"
).fillna(0)

# KNN model
sparse_matrix = csr_matrix(movie_user_matrix.values)

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6
)

knn.fit(sparse_matrix)

# Recommendation function
def recommend(movie_title):

    movie_id = movies.loc[
        movies["Title"] == movie_title,
        "MovieID"
    ].iloc[0]

    if movie_id not in movie_user_matrix.index:
        return []

    index = movie_user_matrix.index.get_loc(movie_id)

    distances, indices = knn.kneighbors(
        sparse_matrix[index],
        n_neighbors=6
    )

    ids = [
        movie_user_matrix.index[i]
        for i in indices.flatten()[1:]
    ]

    return movies[
        movies["MovieID"].isin(ids)
    ][["Title", "Genres"]].to_dict("records")


# Movie selection
movie_list = sorted(movies["Title"].unique())

selected_movie = st.selectbox(
    "Choose a movie",
    movie_list
)

if st.button("Recommend Movies"):
    results = recommend(selected_movie)

    st.subheader("Recommended Movies")

    if results:
        for movie in results:
            st.write("🎥 **" + movie["Title"] + "**")
            st.caption(movie["Genres"])
    else:
        st.warning("No recommendations found.")
