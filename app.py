from flask import Flask, request, jsonify, render_template
import pandas as pd
from surprise import dump, Reader, Dataset, SVD
from flask import make_response
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
from surprise.model_selection import train_test_split

# load cleaned dataset
song_df = pd.read_csv("Data/cleaned_song_dataset.csv")

# Create a temporary user id for new users
temp_user_id = 'temp_user'

# Collaborative Filtering Model
reader = Reader(rating_scale=(0, song_df['play_count'].max()))
data = Dataset.load_from_df(song_df[['user', 'title', 'play_count']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
collab_model = SVD()
collab_model.fit(trainset)

# Content-Based Model (using TF-IDF on song titles)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(song_df['title'])
content_model = linear_kernel(tfidf_matrix, tfidf_matrix)

# Load the model
# collab_model = dump.load("model/collab_model.pkl")[1]

# Flask app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


# API endpoint for recommendations
# @app.route('/recommendations', methods=['GET'])
# def get_recommendations():
#     user_id = str(request.args.get('user_id'))
#
#     if user_id is None:
#         return jsonify({"error": "Missing 'user_id' parameter"}), 400
#
#     listened_songs, recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)
#
#     return make_response(jsonify({
#         'user_songs': listened_songs,
#         'recommendations': recommendations
#     }))
#
#
# def collaborative_filtering_recommendation(selected_songs, model, df):
#     listened_songs = df[df['song'].isin(selected_songs)]['title'].tolist()
#
#     all_songs = df['title'].unique()
#     unheard_songs = [song for song in all_songs if song not in listened_songs]
#
#     recommendations = [(song, str(model.predict(selected_songs, song).est)) for song in unheard_songs]
#
#     recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]
#
#     return listened_songs, recommendations

# Create an API endpoint to return a list of users
# @app.route('/get_songs')
# def get_songs():
#     songs = song_df['title'].unique().tolist()
#     return jsonify(songs)


# @app.route('/get_songs')
# def get_songs():
#     songs = song_df[['song', 'title']].drop_duplicates()
#     return jsonify({
#         'songs': songs['song'].tolist(),
#         'title': songs['title'].tolist()
#     })
#
#
# # Create an API endpoint to return the user's playlist and recommendations
# @app.route('/get_user_and_recommendations', methods=['POST'])
# def get_user_and_recommendations():
#     user_id = request.json['user_id']
#     user_songs, recommendations = collaborative_filtering_recommendation(user_id, collab_model, song_df)
#
#     return jsonify({
#         'user_songs': user_songs,
#         'recommendations': recommendations
#     })


# Create an API endpoint to return recommendations
# @app.route('/get_recommendations', methods=['POST'])
# def get_recommendations():
#     song = request.json['song']
#     user_songs, recommendations = collaborative_filtering_recommendation(song, collab_model, song_df)
#
#     return jsonify({
#         'user_songs': user_songs,
#         'recommendations': recommendations
#     })


# @app.route('/get_recommendations_by_song', methods=['POST'])
# def get_recommendations_by_song():
#     data = request.get_json()
#     # selected_songs = data.get('selectedSongs', [])
#     # song = song_df['title'].unique()
#     selected_songs = data.get('selectedSongs', [])
#
#     listened_songs, recommendations = collaborative_filtering_recommendation(selected_songs, collab_model, song_df)
#     print(data)
#     return jsonify({
#         'listenedSongs': listened_songs,
#         'recommendations': recommendations
#     })


@app.route('/')
def index():
    return render_template('index.html', songs=song_df['title'])


@app.route('/recommend', methods=['POST'])
def get_recommendations():
    # Get selected songs from the request
    # selected_songs = request.form.getlist('selected_songs')
    data = request.get_json()
    selected_songs = data.get('selected_songs', [])
    # Collaborative Filtering Recommendations
    collab_recommendations = collaborative_filtering_recommendation(temp_user_id, collab_model, song_df, selected_songs)

    # Content-Based Recommendations
    content_recommendations = content_based_recommendation(selected_songs, song_df, tfidf_vectorizer, content_model)

    # Combine recommendations (you can customize the merging strategy)
    combined_recommendations = combine_recommendations(collab_recommendations, content_recommendations)

    return render_template("index.html",  recommendations=combined_recommendations)


def collaborative_filtering_recommendation(user_id, model, df, listened_songs):
    # Add selected songs to the dataset for the temporary user
    temp_user_df = pd.DataFrame({'user': [temp_user_id] * len(listened_songs), 'title': listened_songs,
                                 'play_count': [1] * len(listened_songs)})
    song_df1 = pd.concat([song_df, temp_user_df])

    # get unique songs
    all_songs = song_df1['title'].unique()
    unheard_songs = [song for song in all_songs if song not in listened_songs]

    recommendations = [(song, model.predict(user_id, song).est) for song in unheard_songs]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:10]

    return recommendations


def content_based_recommendation(selected_songs, df, tfidf_vectorizer, similarity_matrix):
    # Combine selected songs into a single string for TF-IDF vectorization
    selected_songs_str = ' '.join(selected_songs)

    # Transform the selected songs using the pre-trained TF-IDF vectorizer
    selected_songs_tfidf = tfidf_vectorizer.transform([selected_songs_str])

    # Calculate cosine similarities between the selected_songs_tfidf and similarity_matrix
    cosine_similarities = linear_kernel(selected_songs_tfidf, tfidf_matrix)

    # Get indices of top-N recommendations based on cosine similarities
    top_n_recommendations_indices = cosine_similarities.argsort()[:-11:-1]

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


if __name__ == '__main__':
    app.run(debug=True)
