# Predicting Spotify Track Popularity

A machine learning project that predicts whether a Spotify track is *popular* based on
its features, built as part of Summer Research 2026 (ASSURE-US) at CSU Fullerton.

## Overview

Using a dataset of 8,582 Spotify tracks, we frame popularity prediction as a **binary
classification** problem: a track is labeled "popular" if its `track_popularity` score is
65 or higher (roughly 38% of tracks). We then train and compare two models to see how well
a track's features predict that label, and to understand *which* features matter most.

## Research Question

**Can we predict whether a Spotify track is popular from its features, and what drives popularity?**

## Dataset

- **Source:** Kaggle (Spotify track metadata)
- **Size:** 8,582 tracks, 15 columns
- **Time span:** Releases from 1952 to 2025
- **Target:** `track_popularity` (0–99), converted to a binary `popular` label (≥ 65)
- **Key features used:** `artist_popularity`, `artist_followers`, `explicit`,
  `track_duration_min`, `album_total_tracks`, `track_number`, `album_type`, and an
  engineered `release_year`

## Approach

1. **Pre-processing** — remove duplicates, engineer `release_year` from the release date,
   create the binary target, one-hot encode `album_type`, and scale numeric features.
2. **Train/test split** — 80/20, stratified to preserve the class balance.
3. **Modeling** — train and compare:
   - Logistic Regression (interpretable baseline)
   - Random Forest (stronger, non-linear model)
4. **Evaluation** — accuracy, precision, recall, ROC-AUC, and average precision, plus a
   confusion matrix, ROC curve, and feature-importance chart.

## Results

| Model | Accuracy | Precision | Recall | ROC-AUC | Avg Precision |
|-------|----------|-----------|--------|---------|---------------|
| Logistic Regression | 0.689 | 0.592 | 0.576 | 0.769 | 0.620 |
| Random Forest | **0.739** | **0.663** | **0.630** | **0.816** | **0.699** |

The **Random Forest** was the stronger model across every metric, reaching a ROC-AUC of
0.816 on the held-out test set.

**Key finding:** The most important predictors were `artist_followers` and
`artist_popularity`, followed by `track_duration_min`. In other words, a track's
popularity is driven largely by how established the artist already is, rather than by
properties of the individual track. Because the dataset contains no audio features
(e.g. tempo, danceability), there is a natural ceiling on how well track-level popularity
can be predicted from this data alone.

## How to Run

1. Install [Anaconda](https://www.anaconda.com/download) (includes Python + Jupyter Notebook).
2. Install dependencies if needed:
   ```
   pip install scikit-learn pandas numpy matplotlib
   ```
3. Place `spotify_data_clean.csv` in the same folder as the notebook.
4. Open the notebook in Jupyter and run the cells top to bottom.

## Project Structure

```
.
├── spotify_data_clean.csv       # dataset
├── spotify_data_science.ipynb   # analysis notebook
└── README.md                    # this file
```

## Team

- James Ubiadas
- Ava Saltzman

**Faculty Mentor:** Dr. Doina Bein, CSU Fullerton

## Acknowledgments

Built for the Summer Research 2026 / ASSURE-US program under the mentorship of Dr. Doina Bein.
