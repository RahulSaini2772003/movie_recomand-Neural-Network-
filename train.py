import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dot, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st

# Data set Load
ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Checking for null value
movies.isnull().sum()
ratings.isnull().sum()
# checking for duplicate values
print(movies.duplicated().sum())
print(ratings.duplicated().sum())

# Function to create mappings between IDs and their encoded values
def create_id_mappings(ids):
    # Dictionary mapping original IDs to encoded values
    id2encoded = {x: i for i, x in enumerate(ids)}
    # Dictionary mapping encoded values back to original IDs
    encoded2id = {i: x for i, x in enumerate(ids)}
    return id2encoded, encoded2id

# Get unique user IDs and movie IDs from the ratings dataset
user_ids = ratings["userId"].unique().tolist()
movie_ids = ratings["movieId"].unique().tolist()

user2user_encoded, userencoded2user = create_id_mappings(user_ids)


movie2movie_encoded, movie_encoded2movie = create_id_mappings(movie_ids)

# Map user IDs to their encoded values in the ratings DataFrame
ratings["user"] = ratings["userId"].map(user2user_encoded)

# Map movie IDs to their encoded values in the ratings DataFrame
ratings["movie"] = ratings["movieId"].map(movie2movie_encoded)

# Convert ratings to float32 for compatibility with the model
ratings["rating"] = ratings["rating"].values.astype(np.float32)

# Split the ratings data into training and testing sets
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Calculate the number of unique users and movies after encoding
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)

ratings.head(5)
embedding_size = 50
# This specifies the dimensionality of the embedding vectors for both users and movies.
# Each user and movie will be represented as a dense vector of 50 dimensions.

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_size)(user_input)
# This creates an embedding layer for users.
# It maps each user ID (from 0 to num_users-1) to a dense vector of size embedding_size (50 in this case).

user_vector = Flatten()(user_embedding)
# This flattens the user embedding vector into a 1-dimensional vector.
# It prepares the data for further processing.

movie_input = Input(shape=(1,))
movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
movie_vector = Flatten()(movie_embedding)


dot_product = Dot(axes=1)([user_vector, movie_vector])
#The Dot layer in Keras computes the dot product between two tensors.
# In this case, it computes the dot product between the user_vector and movie_vector tensors.

model = Model(inputs=[user_input, movie_input], outputs=dot_product)
#The Model class in Keras allows you to define a neural network model by specifying its inputs and outputs.

model.compile(optimizer='adam', loss='mean_squared_error')
# Compilation: Configures the model for training with an optimizer and a loss function.
# The 'adam' optimizer refers to the Adam (Adaptive Moment Estimation) optimizer.
early_stopping = EarlyStopping(
    monitor='val_loss',        # Metric to monitor
    patience=10,               # Number of epochs with no improvement to wait
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
)


history = model.fit(
    [train.user, train.movie],  # Training data: user IDs and movie IDs
    train.rating,                # Training labels: ratings
    epochs=500,                  # Number of epochs (iterations over the entire datase5t)
    verbose=1,                   # Verbosity mode (1: progress bar, 0: silent)
    validation_data=([test.user, test.movie], test.rating),  # Validation data for evaluation during training
    callbacks=[early_stopping] 
)

model.save('model.h5')
