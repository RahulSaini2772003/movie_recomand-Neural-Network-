import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import base64
import streamlit as st

model = load_model('model.h5')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Function to create mappings between IDs and their encoded values
def create_id_mappings(ids):
    id2encoded = {x: i for i, x in enumerate(ids)}
    encoded2id = {i: x for i, x in enumerate(ids)}
    return id2encoded, encoded2id

user_ids = ratings["userId"].unique().tolist()
movie_ids = ratings["movieId"].unique().tolist()

user2user_encoded, userencoded2user = create_id_mappings(user_ids)
movie2movie_encoded, movie_encoded2movie = create_id_mappings(movie_ids)
@st.cache_data()
def bin_img(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

@st.cache_data
def get_recommended_movie_titles(movie_encoded2movie, movies_not_watched, top_ratings_indices):
    return [movie_encoded2movie.get(movies_not_watched[x]) for x in top_ratings_indices]

@st.cache_data
def get_top_movies_watched_by_user(movies_watched_by_user):
    return movies_watched_by_user.sort_values(by="rating", ascending=False).head(5).movieId.values

@st.cache_data
def get_movie_details(movie_ids, movies):
    return movies[movies["movieId"].isin(movie_ids)]

# Function to get recommendations
def get_recommendations(user_id):
    user_id = int(user_id)

    movies_watched_by_user = ratings[ratings.userId == user_id]
    movies_not_watched = movies[~movies["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched = [movie2movie_encoded.get(x) for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.array([[user_encoder, movie] for movie in movies_not_watched])
    user_input_array = user_movie_array[:, 0]
    movie_input_array = user_movie_array[:, 1]
    ratingss = model.predict([user_input_array, movie_input_array]).flatten()
    top_ratings_indices = ratingss.argsort()[-5:][::-1]

    recommended_movie_ids = get_recommended_movie_titles(movie_encoded2movie, movies_not_watched, top_ratings_indices)
    top_movies_user = get_top_movies_watched_by_user(movies_watched_by_user)
    movie_df_rows = get_movie_details(top_movies_user, movies)
    recommended_movies = get_movie_details(recommended_movie_ids, movies)

    return movie_df_rows, recommended_movies

png_file = '5.jpg'
bin_str = bin_img(png_file)
page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(to bottom right, rgba(0,0,0,0.9), rgba(0,0,0,0.9)), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
        width: 100vw; 
        position: fixed;
        top: 0;
        left: 0;
    }}
    </style>'''
    
st.markdown(page_bg_img, unsafe_allow_html=True)
# Input box for user ID
user_id = st.text_input("Enter User ID")

if user_id:
    movie_df_rows, recommended_movies = get_recommendations(user_id)
    
    st.write(f"**Showing recommendations for user: {user_id}**")
    st.write("=" * 32)

    st.write("**Movies with high ratings from user**")
    st.write("-" * 32)
    for row in movie_df_rows.itertuples():
        st.write(f"{row.title} : {row.genres}")

    st.write("")
    st.write("")
    st.write("")

    st.write("-" * 32)
    st.write("**Top 5 movie recommendations**")
    st.write("-" * 32)
    for row in recommended_movies.itertuples():
        st.write(f"{row.title} : {row.genres}")
else:
    st.write("Please enter a User ID to see recommendations.")
