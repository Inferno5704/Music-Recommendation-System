# 🎵 Music Recommendation System using Annoy

## 📌 Overview
This **Music Recommendation System** utilizes **Annoy (Approximate Nearest Neighbors Oh Yeah)** to perform fast and efficient nearest-neighbor searches. Given a dataset of songs with feature embeddings, the system recommends similar tracks based on user input.

## 🛠 Tech Stack
- **Notebook Environment:** Jupyter Notebook
- **Backend:** Python
- **Key Python Libraries:**
  - `pandas` - For handling dataset operations
  - `sklearn.preprocessing.StandardScaler` - For scaling song features
  - `annoy.AnnoyIndex` - For building and searching nearest-neighbor indices
  - `collections.defaultdict` - For managing recommendation frequency

---

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/music-recommendation-annoy.git
cd music-recommendation-annoy
```

### 2️⃣ Install Dependencies
Ensure Python is installed, then install the required packages:
```sh
pip install -r requirements.txt
```

### 3️⃣ Download the Dataset
Download the dataset from Kaggle:
1. Go to [Kaggle Music Dataset](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs/data) 
2. Download the dataset and place it in the project folder.
3. Ensure the dataset file name matches what is used in the code.

---

## ⚙️ How It Works
### 📌 Step 1: Load & Scale Dataset
- Load the dataset using `pandas`.
- Normalize the features using `StandardScaler` to improve performance.

### 📌 Step 2: Create Annoy Index
- Construct an Annoy index with an appropriate metric (e.g., angular distance).
- Store song embeddings in the Annoy index for quick lookups.

### 📌 Step 3: Recommendation Methods
- **Single Song Recommendation:** Given a song, find the top N most similar songs.
- **Most Common Song Recommendation:** Given a list of songs, find recommendations for each and prioritize the most commonly suggested track.

---

## 🏃 Usage in Jupyter Notebook
### 1️⃣ Open the Jupyter Notebook
Run the following command in the terminal to start Jupyter Notebook:
```sh
jupyter notebook
```
Open the `.ipynb` file in the project directory.

### 2️⃣ Get Song Recommendations
#### 🔹 Single Song Recommendation
```python
def recommend_song_annoy(song_title, df, X_scaled, annoy_index, top_n=5):
    """Returns a list of recommended song names based on Annoy search."""
    indices = df.index[df['name'] == song_title].tolist()
    if not indices:
        return []  # Return an empty list if the song is not found
    idx = indices[0]

    # Get the top_n + 1 nearest neighbors (including the song itself)
    nearest_indices = annoy_index.get_nns_by_item(idx, top_n + 1)

    # Exclude the query song itself (usually the first one)
    nearest_indices = [i for i in nearest_indices if i != idx][:top_n]

    return df.iloc[nearest_indices]['name'].tolist()  # Extract only song names
query_song = 'Dancing With Your Ghost'
recommended_songs = recommend_song_annoy(query_song, df, X_scaled, annoy_index, top_n=5)
print("\nRecommended Songs:")
print(recommended_songs)
```

#### 🔹 Most Common Song Recommendation
```python
from collections import defaultdict
#Getting the list of most common songs in list 
def common_songs_priority(*lists):
    song_counts = defaultdict(int)
    song_positions = {}

    # Count occurrences and track first positions
    for i, song_list in enumerate(lists):
        song_set = set(song_list)  # Remove duplicates in a single list
        for song in song_set:
            song_counts[song] += 1
            if song not in song_positions:
                song_positions[song] = i  # First occurrence index

    # Find common songs in all lists
    num_lists = len(lists)
    common_songs = [song for song, count in song_counts.items() if count == num_lists]

    # Sort based on first appearance
    common_songs.sort(key=lambda song: song_positions[song])

    return common_songs
#Recommendating Songs based Playlist
li = ['Levitating', 'Psycho (feat. Ty Dolla $ign)', 'Blank Space', 'You Belong with Me', 'Look What You Made Me Do']

# Get recommendations for each song in the playlist
recommend = [recommend_song_annoy(song, df, X_scaled, annoy_index, top_n=10000) for song in li]

# Find common songs across all recommended lists
t = common_songs_priority(*recommend)

print(t)
```

---

## 📈 Future Enhancements
🔹 Improve recommendations using **deep learning models** 🧠
🔹 Add **genre, mood, and tempo-based filtering** 🎶
🔹 Deploy as a web service using **Flask or FastAPI** 🌍



