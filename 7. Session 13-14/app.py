# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

json_path = "Kaggle/good.json"
yes_json_path = "Kaggle/yes.json"

with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

if "audio_features" in json_data:
    df = pd.DataFrame(json_data["audio_features"])
else:
    st.error("Key 'audio_features' tidak ditemukan dalam JSON.")

with open(yes_json_path, 'r', encoding='utf-8') as f:
    yes_data = json.load(f)

yes_tracks = {track["track"]["id"]: track["track"]["name"] for track in yes_data["items"]}

df["track_name"] = df["id"].map(yes_tracks)
df["song_id"] = df["id"]

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 
            'speechiness', 'acousticness', 'instrumentalness', 'liveness']

scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

song_features = df[['song_id'] + features].drop_duplicates().set_index('song_id')
song_similarity = cosine_similarity(song_features)
song_similarity_df = pd.DataFrame(song_similarity, index=song_features.index, columns=song_features.index)

def recommend_songs(song_id, song_similarity_df, df, top_n=5):
    if song_id not in song_similarity_df.index:
        return pd.DataFrame()
    
    similar_songs = song_similarity_df[song_id].sort_values(ascending=False).iloc[1:top_n*2]
    
    selected_songs = np.random.choice(similar_songs.index, size=top_n, replace=False)
    
    recommended_songs = df[df['song_id'].isin(selected_songs)][['track_name', 'song_id', 'tempo', 'danceability', 'energy', 
                                                                 'valence', 'instrumentalness', 'track_href']].drop_duplicates()
    
    return recommended_songs

def find_closest_songs(user_input, df, top_n=5):
    
    input_df = pd.DataFrame([user_input], columns=features)
    input_df = scaler.transform(input_df)
    
    similarity_scores = cosine_similarity(input_df, df[features])
    
    top_indices = np.argsort(similarity_scores[0])[::-1][:top_n*2]
    
    selected_indices = np.random.choice(top_indices, size=top_n, replace=False)
    
    return df.iloc[selected_indices][['track_name', 'song_id', 'tempo', 'danceability', 'energy', 'valence', 
                                       'instrumentalness', 'track_href']]

# STREAMLIT APP (User Interface)
st.title("🎵 Spotify Music Recommender System")
st.write("Pilih bagaimana Anda ingin mendapatkan rekomendasi lagu.")

option = st.selectbox("Pilih Metode Rekomendasi", ["Berdasarkan Lagu", "Berdasarkan Kriteria Fitur"])

if option == "Berdasarkan Lagu":
    st.subheader("🎵 Rekomendasi Lagu Berdasarkan Lagu yang Dipilih")
    
    selected_song = st.selectbox("Pilih lagu:", df['track_name'].dropna().unique())
    
    song_id = df[df['track_name'] == selected_song]['song_id'].values[0]
    
    if st.button("Dapatkan Rekomendasi 🎶"):
        recommendations = recommend_songs(song_id, song_similarity_df, df)
        
        if recommendations.empty:
            st.error("Tidak ada rekomendasi lagu ditemukan.")
        else:
            st.write("🎶 Berikut lagu-lagu yang mirip dengan:", selected_song)
            st.dataframe(recommendations)

elif option == "Berdasarkan Kriteria Fitur":
    st.subheader("🎛️ Rekomendasi Berdasarkan Fitur Lagu yang Dipilih")
    
    user_input = {}
    user_input['danceability'] = st.slider("Danceability", 0.0, 1.0, 0.5)
    user_input['energy'] = st.slider("Energy", 0.0, 1.0, 0.5)
    user_input['valence'] = st.slider("Valence (Mood)", 0.0, 1.0, 0.5)
    user_input['tempo'] = st.slider("Tempo (BPM)", float(df['tempo'].min()), float(df['tempo'].max()), 120.0)
    user_input['loudness'] = st.slider("Loudness (dB)", float(df['loudness'].min()), float(df['loudness'].max()), -5.0)
    user_input['speechiness'] = st.slider("Speechiness", 0.0, 1.0, 0.5)
    user_input['acousticness'] = st.slider("Acousticness", 0.0, 1.0, 0.5)
    user_input['instrumentalness'] = st.slider("Instrumentalness", 0.0, 1.0, 0.5)
    user_input['liveness'] = st.slider("Liveness", 0.0, 1.0, 0.5)

    if st.button("Cari Lagu 🎼"):
        recommendations = find_closest_songs(user_input, df)
        
        if recommendations.empty:
            st.error("Tidak ada lagu yang cocok ditemukan.")
        else:
            st.write("🎵 Berikut adalah lagu-lagu yang sesuai dengan kriteria yang Anda pilih:")
            st.dataframe(recommendations)


# To run this copy the command line below:
# streamlit run app.py