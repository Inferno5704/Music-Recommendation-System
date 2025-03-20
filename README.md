# 🎵 Music Recommendation System using Annoy

## 📌 Overview
This **Music Recommendation System** utilizes **Annoy (Approximate Nearest Neighbors Oh Yeah)** to perform fast and efficient nearest-neighbor searches. Given a dataset of songs with feature embeddings, the system recommends similar tracks based on user input.

## 🛠 Tech Stack
- **Language:** Python
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

## 🏃 Usage
### 1️⃣ Build the Annoy Index
Run the following script to generate the Annoy index:
```sh
python build_index.py
```
This will:
✔ Load the dataset from Kaggle
✔ Scale the dataset
✔ Create and save an Annoy index as `music.ann`

### 2️⃣ Get Song Recommendations
#### 🔹 Single Song Recommendation
```python
from annoy import AnnoyIndex
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
songs = pd.read_csv("music_dataset.csv")  # Ensure filename matches Kaggle dataset
scaler = StandardScaler()
songs["scaled_embedding"] = scaler.fit_transform(songs["embedding"].tolist())

# Load Annoy index
dim = len(songs["scaled_embedding"][0])
annoy_index = AnnoyIndex(dim, 'angular')
annoy_index.load("music.ann")

# Get top N recommendations
def get_recommendations(song_id, top_n=5):
    return annoy_index.get_nns_by_item(song_id, top_n, include_distances=True)

song_id = 1
print(get_recommendations(song_id, 5))
```

#### 🔹 Most Common Song Recommendation
```python
from collections import defaultdict

def get_most_common_song(song_lists):
    song_count = defaultdict(int)
    for song_list in song_lists:
        for song in song_list:
            song_count[song] += 1
    return max(song_count, key=song_count.get)

# Example Usage
song_recommendations = [
    ["Song A", "Song B", "Song C"],
    ["Song B", "Song C", "Song D"],
    ["Song A", "Song B", "Song D"]
]

print(get_most_common_song(song_recommendations))
```

---

## 📈 Future Enhancements
🔹 Improve recommendations using **deep learning models** 🧠
🔹 Add **genre, mood, and tempo-based filtering** 🎶


