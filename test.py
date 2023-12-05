from flask import Flask, request, jsonify, make_response, render_template
import pandas as pd
from surprise import SVD, KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Load cleaned dataset
song_df = pd.read_csv("cleaned_song_dataset.csv")

# Create a temporary user id for new users
temp_user_id = 'temp_user'

# Collaborative Filtering Model
reader = Reader(rating_scale=(0, song_df['play_count'].max()))
data = Dataset.load_from_df(song_df[['user', 'title', 'play_count']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
collab_model = KNNBasic(sim_options={'user_based': True})
collab_model.fit(trainset)

# Content-Based Model (using TF-IDF on song titles)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(song_df['title'])
# content_model = linear_kernel(tfidf_matrix, tfidf_matrix)

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', songs=song_df['title'])


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get selected songs from the request
    # selected_songs = request.form.getlist('selected_songs')
    data = request.get_json()
    selected_songs = data.get('selected_songs', [])
    print(selected_songs)
    # Collaborative Filtering Recommendations
    collab_recommendations = collaborative_filtering_recommendation(temp_user_id, song_df, selected_songs)
    print(collab_recommendations)

    # Content-Based Recommendations
    ##content_recommendations = content_based_recommendation(selected_songs, song_df, tfidf_vectorizer)

    # Combine recommendations (you can customize the merging strategy)
    ##combined_recommendations = combine_recommendations(collab_recommendations, content_recommendations)

    return make_response(jsonify({"recommendations": collab_recommendations}))


def collaborative_filtering_recommendation(user_id, df, listened_songs):
    # Add selected songs to the dataset for the temporary user
    temp_user_df = pd.DataFrame({'user': [temp_user_id] * len(listened_songs), 'title': listened_songs,
                                 'play_count': [1] * len(listened_songs)})
    song_df1 = pd.concat([df, temp_user_df])

    # Collaborative Filtering Model
    reader = Reader(rating_scale=(0, song_df['play_count'].max()))

    # Load the combined data into Surprise
    data = Dataset.load_from_df(song_df1[['user', 'title', 'play_count']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Build and train the updated model
    collab_model = KNNBasic(sim_options={'user_based': True})
    collab_model.fit(trainset)

    # get unique songs
    all_songs = song_df1['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    recommendations = [(song, collab_model.predict(user_id, song).est) for song in unheard_songs]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return recommendations


def content_based_recommendation(selected_songs, df, tfidf_vectorizer):
    # Combine selected songs into a single string for TF-IDF vectorization
    selected_songs_str = ' '.join(selected_songs)

    # Transform the selected songs using the pre-trained TF-IDF vectorizer
    selected_songs_tfidf = tfidf_vectorizer.transform([selected_songs_str])

    # Calculate cosine similarities between the selected_songs_tfidf and similarity_matrix
    cosine_similarities = linear_kernel(selected_songs_tfidf, tfidf_matrix)

    # Get indices of top-N recommendations based on cosine similarities
    top_n_recommendations_indices = cosine_similarities.argsort()[:-11:-1]
    print(top_n_recommendations_indices)

    return top_n_recommendations_indices


def combine_recommendations(collab_recommendations, content_recommendations):
    # Combine recommendations using a simple strategy (e.g., averaging scores)
    combined_recommendations = []
    for collab_rec, content_rec in zip(collab_recommendations, content_recommendations):
        combined_score = (collab_rec[1] + content_rec[1]) / 2
        combined_recommendations.append((collab_rec[0], combined_score))

    # Sort combined recommendations by score
    combined_recommendations = sorted(combined_recommendations, key=lambda x: x[1], reverse=True)[:10]

    return combined_recommendations


# Create an API endpoint to return a list of users
@app.route('/get_songs')
def get_songs():
    songs = song_df[['song', 'title']].drop_duplicates()
    return jsonify({
        'songs': songs['song'].tolist(),
        'title': songs['title'].tolist()
    })


if __name__ == '__main__':
    app.run(debug=True)