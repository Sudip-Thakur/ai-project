import pandas as pd
import numpy as np
import os
from datetime import datetime

# Extract MovieLens 100K dataset and create CSV files
def extract_movielens_data():
    ml_data_dir = "ml-100k"
    print("Starting MovieLens 100K data extraction...")
    
    #Extract ratings data (u.data)
    ratings_file = os.path.join(ml_data_dir, "u.data")
    
    if os.path.exists(ratings_file):
        ratings_df = pd.read_csv(ratings_file, 
                               sep='\t', 
                               names=['user_id', 'movie_id', 'rating', 'timestamp'],
                               header=None)
        ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        ratings_csv_path = "ratings.csv"
        ratings_df.to_csv(ratings_csv_path, index=False)
        print(f"Ratings shape: {ratings_df.shape}")
    else:
        print(f"Warning: {ratings_file} not found!")
    
    #Extract movie information (u.item)
    movies_file = os.path.join(ml_data_dir, "u.item")
    
    if os.path.exists(movies_file):
        genre_columns = ['unknown', 'Action', 'Adventure', 'Animation', 
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                        'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                        'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        movie_columns = ['movie_id', 'movie_title', 'release_date', 
                        'video_release_date', 'imdb_url'] + genre_columns
        
        movies_df = pd.read_csv(movies_file, 
                               sep='|', 
                               names=movie_columns,
                               header=None,
                               encoding='latin-1')  # Handle special characters
        
        movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], 
                                                  format='%d-%b-%Y', 
                                                  errors='coerce')
        
        movies_csv_path = "movies.csv"
        movies_df.to_csv(movies_csv_path, index=False)
        print(f"Movies shape: {movies_df.shape}")
    else:
        print(f"Warning: {movies_file} not found!")
    
    #Extract user information (u.user)
    users_file = os.path.join(ml_data_dir, "u.user")
    
    if os.path.exists(users_file):
        users_df = pd.read_csv(users_file, 
                              sep='|', 
                              names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                              header=None)
        
        # Save to CSV
        users_csv_path = "users.csv"
        users_df.to_csv(users_csv_path, index=False)
        print(f"Users shape: {users_df.shape}")
    else:
        print(f"Warning: {users_file} not found!")
    
    #Extract genre list (u.genre)
    genre_file = os.path.join(ml_data_dir, "u.genre")
    
    if os.path.exists(genre_file):
        genres_df = pd.read_csv(genre_file, 
                               sep='|', 
                               names=['genre', 'genre_id'],
                               header=None)
        
        # Save to CSV
        genres_csv_path = "genres.csv"
        genres_df.to_csv(genres_csv_path, index=False)
        print(f"Genres shape: {genres_df.shape}")
    else:
        print(f"Warning: {genre_file} not found!")
    
    
    # Create user-movie rating matrix for RBM
    if 'ratings_df' in locals():
        rating_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating', 
            fill_value=0
        )
        
        # Save rating matrix
        matrix_csv_path = "user_movie_matrix.csv"
        rating_matrix.to_csv(matrix_csv_path)
        print(f"Matrix shape: {rating_matrix.shape}")
        
        # Create binary rating matrix (for RBM - ratings 4-5 as 1, others as 0)
        binary_matrix = (rating_matrix >= 4).astype(int)
        binary_matrix_path = "user_movie_binary_matrix.csv"
        binary_matrix.to_csv(binary_matrix_path)
        print(f"Binary matrix shape: {binary_matrix.shape}")

if __name__ == "__main__":
    extract_movielens_data()