#!/usr/bin/env python
# coding: utf-8

# nfl-over-under.py  (Streamlit app)

# --- Make CPU math predictable (set before NumPy/Sklearn imports) ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances

st.set_page_config(page_title="NFL Totals KNN Predictor", layout="wide")

# Sidebar env info
st.sidebar.markdown("**Environment**")
st.sidebar.text(f"pandas: {pd.__version__}")
st.sidebar.text(f"sklearn: {sklearn.__version__}")

# -------------------------------
# Deterministic KNN with RECENCY tie-break
# -------------------------------
def knn_predict_stable(X_train, y_train, X_query, seasons, weeks, k=7):
    """
    Deterministic KNN with *recency* tie-break:
      primary:   distance (ascending)
      secondary: Season  (descending -> newer first)
      tertiary:  Week    (descending -> newer first)
      fallback:  row index (ascending)
    Returns: preds, distances[(nq,k)], indices[(nq,k)]
    """
    Xt = np.ascontiguousarray(X_train, dtype=np.float64)
    Xq = np.ascontiguousarray(X_query, dtype=np.float64)
    y  = np.asarray(y_train)

    n_train = Xt.shape[0]
    row_idx = np.arange(n_train)
    pr_season = np.asarray(seasons, dtype=np.int64)
    pr_week   = np.asarray(weeks,   dtype=np.int64)

    # Distances: (n_train, n_query)
    D = pairwise_distances(Xt, Xq, metric="euclidean", n_jobs=1)

    inds, dists = [], []
    # np.lexsort: last key is primary
    # order by: (row_idx ↑) after (-week ↓) after (-season ↓) after (distance ↑)
    for j in range(D.shape[1]):
        order = np.lexsort((row_idx, -pr_week, -pr_season, D[:, j]))
        take = order[:k]
        inds.append(take)
        dists.append(D[take, j])

    ind  = np.stack(inds, axis=0)   # (n_query, k)
    dist = np.stack(dists, axis=0)  # (n_query, k)

    # sklearn-style uniform voting; ties -> lowest class label
    max_label = int(np.max(y)) if y.size else 1
    preds = []
    for nbr_idx in ind:
        votes = np.bincount(y[nbr_idx].astype(int), minlength=max_label + 1)
        preds.append(np.flatnonzero(votes == votes.max())[0])
    return np.array(preds), dist, ind


@st.cache_data
def load_data():
    df = pd.read_csv("nfl_cleaned_for_modeling_2015-2024-Copy1.csv")
    df = df.sort_values(by=['Season', 'Week']).reset_index(drop=True)
    df['True_Total'] = df['Tm_Pts'] + df['Opp_Pts']
    df['Over'] = np.where(df['True_Total'] > df['Total'], 1, 0)
    df['Under'] = np.where(df['True_Total'] < df['Total'], 1, 0)
    df['Push'] = np.where(df['True_Total'] == df['Total'], 1, 0)
    return df


def predict_week(df, season=2025, week=1):
    features = ['Spread', 'Total']
    target = 'Under'

    # Training set up to the chosen week (home-team rows only if that's your convention upstream)
    train_df = df.query('Season < @season or (Season == @season and Week < @week)')
    # Deterministic order just for reproducible fallback when *all* keys tie
    train_df = train_df.sort_values(['Season', 'Week', 'Tm_Name']).reset_index(drop=True)

    # Full precision for the model (do NOT round before fit)
    X_train_np = np.ascontiguousarray(train_df[features].to_numpy(dtype="float64"))
    y_train_np = train_df[target].to_numpy()
    seasons_np = train_df['Season'].to_numpy()
    weeks_np   = train_df['Week'].to_numpy()

    # Incoming games (home-team perspective)
    week1 = [
        ['Cowboys @ Eagles', -6.5, 46.5], ['Chiefs @ Chargers', +2.5, 45.5],
        ['Giants @ Commanders', -6.5, 45.5], ['Panthers @ Jaguars', -2.5, 46.5],
        ['Steelers @ Jets', +3.0, 38.5], ['Raiders @ Patriots', -2.5, 42.5],
        ['Cardinals @ Saints', +5.5, 41.5], ['Bengals @ Browns', +5.5, 45.5],
        ['Dolphins @ Colts', -1.5, 46.5], ['Buccaneers @ Falcons', +1.5, 48.5],
        ['Titans @ Broncos', -7.5, 41.5], ['49ers @ Seahawks', +2.5, 45.5],
        ['Lions @ Packers', -1.5, 49.5], ['Texans @ Rams', +2.5, 45.5],
        ['Ravens @ Bills', -1.5, 51.5], ['Vikings @ Bears', -1.5, 43.5]
    ]
    X_new = pd.DataFrame(week1, columns=['Game', 'Spread', 'Total'])
    X_new_np = np.ascontiguousarray(X_new[['Spread', 'Total']].to_numpy(dtype="float64"))

    # --- Diagnostics (use the first query like before) ---
    try:
        k = 7
        xq = X_new_np[0:1]
        D = pairwise_distances(X_train_np, xq, metric='euclidean', n_jobs=1).ravel()
        cut = np.partition(D, k-1)[k-1]
        ties = int(np.sum(np.isclose(D, cut, rtol=0, atol=0)))
        st.sidebar.text(f"kth dist: {cut:.10f} | ties@k: {ties}")
        order = np.argsort(D, kind="mergesort")
        st.sidebar.text("nearest 10 dists: " + ", ".join(f"{D[i]:.6f}" for i in order[:10]))
    except Exception as e:
        st.sidebar.text(f"diag error: {e}")

    # --- Deterministic neighbors + predictions with RECENCY tie-break ---
    raw_preds, distances, indices = knn_predict_stable(
        X_train_np, y_train_np, X_new_np,
        seasons=seasons_np, weeks=weeks_np, k=7
    )
    X_new['Prediction'] = ['Under' if p == 1 else 'Over' for p in raw_preds]

    # --- Confidence/neighbor analysis (same as before) ---
    confidence_percents = []
    avg_distances = []
    confidence_scores = []
    neighbors_info = []

    for i in range(len(X_new)):
        neighbor_idxs = indices[i]
        neighbor_dists = distances[i]
        neighbor_labels = y_train_np[neighbor_idxs]

        prediction_label = 1 if X_new.loc[i, 'Prediction'] == 'Under' else 0
        agreeing = int(np.sum(neighbor_labels == prediction_label))
        confidence_percent = agreeing / len(neighbor_labels)
        avg_distance = float(np.mean(neighbor_dists))
        confidence_score = (confidence_percent * 100) * (1 - avg_distance)

        confidence_percents.append(round(confidence_percent, 3))
        avg_distances.append(round(avg_distance, 3))
        confidence_scores.append(round(confidence_score, 1))

        neighbors = train_df.iloc[neighbor_idxs][['Season', 'Week', 'Spread', 'Total', 'Under']].copy()
        neighbors['Distance'] = neighbor_dists
        neighbors_info.append(neighbors)

    X_new['ConfidencePercent'] = confidence_percents
    X_new['AvgDistance'] = avg_distances
    X_new['ConfidenceScore'] = confidence_scores
    X_new['Neighbors'] = neighbors_info

    return X_new


# === Streamlit UI ===
st.title("NFL KNN Week 1 Totals Predictor")

df = load_data()
pred_df = predict_week(df)

st.subheader("Predictions for Week 1 of the 2025 NFL Season")

for _, row in pred_df.iterrows():
    with st.expander(f"{row['Game']} — Prediction: {row['Prediction']}"):
        st.write(f"**Spread:** {row['Spread']} | **Total:** {row['Total']}")
        st.write(f"**Prediction:** {row['Prediction']}")
        st.write(f"**Confidence %:** {row['ConfidencePercent']*100:.1f}%")
        st.write(f"**Avg Distance:** {row['AvgDistance']} | **Score:** {row['ConfidenceScore']:.1f}")
        st.dataframe(row['Neighbors'])

