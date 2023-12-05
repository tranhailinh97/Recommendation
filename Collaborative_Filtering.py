from surprise import Dataset, Reader, KNNBasic, SVD
from surprise import dump
from surprise.model_selection import train_test_split
import pandas as pd
from flask import Flask, request, jsonify

# load cleaned dataset
song_df = pd.read_csv("Data/cleaned_song_dataset.csv")
print(song_df.head())

# Create a Reader object to define the play_count range
reader = Reader(rating_scale=(0, song_df['play_count'].max()))

# Create a dataset from song_df dataframe
data = Dataset.load_from_df(song_df[['user', 'title', 'play_count']], reader)

# Divide the data into training set and test set
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Collaborative Filtering
collab_model = KNNBasic(sim_options={'user_based': True})
collab_model.fit(trainset)


# A recommendation function using Collaborative Filtering
def collaborative_filtering_recommendation(user_id, model, df):
    # Get a list of all songs listened to by the user
    listened_songs = df[df['user'] == user_id]['song'].tolist()

    # Find unheard songs
    all_songs = df['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    # Give users suggestions from unheard songs
    recommendations = [(song, model.predict(user_id, song).est) for song in unheard_songs]

    # Sort and get top k suggestions
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return recommendations

# Test
user_id = '59bad0aea569a12d785e401d185f8814d128333e'

# Collaborative Filtering
collab_recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)

# In ra kết quả
print("Collaborative Filtering Recommendations:")
for recommendation in collab_recommendations:
    print(f"Song: {recommendation[0]} - Estimate: {recommendation[1]}")

# Save the model
model_path = "model/collab_model.pkl"
dump.dump(model_path, algo=collab_model)
