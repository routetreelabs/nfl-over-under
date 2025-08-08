#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# nfl_knn_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="NFL Totals KNN Predictor", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("nfl_cleaned_for_modeling_2015-2024-Copy1.csv")
    df = df.sort_values(by=['Season','Week']).reset_index(drop=True)
    df['True_Total'] = df['Tm_Pts'] + df['Opp_Pts']
    df['Over'] = np.where(df['True_Total'] > df['Total'], 1, 0)
    df['Under'] = np.where(df['True_Total'] < df['Total'], 1, 0)
    df['Push'] = np.where(df['True_Total'] == df['Total'], 1, 0)
    return df

def predict_week(df, season=2025, week=1):
    features = ['Spread', 'Total']
    target = 'Under'
    df = df.query('Home == 1').reset_index(drop=True)

    train_df = df.query('Season < @season or (Season == @season and Week < @week)')
    train_df = train_df.sort_values(['Season', 'Week', 'Tm_Name']).reset_index(drop=True)
    X_train = train_df[features].astype(float).round(4)
    y_train = train_df[target]

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
    X_new[['Spread', 'Total']] = X_new[['Spread', 'Total']].astype(float).round(4)

    model = KNeighborsClassifier(n_neighbors=7)
    clf = model.fit(X_train, y_train)

    X_new_features = X_new[['Spread', 'Total']]
    raw_preds = clf.predict(X_new_features)
    X_new['Prediction'] = ['Under' if p == 1 else 'Over' for p in raw_preds]

    distances, indices = clf.kneighbors(X_new_features)

    confidence_percents = []
    avg_distances = []
    confidence_scores = []
    neighbors_info = []

    for i in range(len(X_new)):
        neighbor_idxs = indices[i]
        neighbor_dists = distances[i]
        neighbor_labels = y_train.iloc[neighbor_idxs].values

        prediction_label = 1 if X_new.loc[i, 'Prediction'] == 'Under' else 0
        agreeing = np.sum(neighbor_labels == prediction_label)
        confidence_percent = agreeing / len(neighbor_labels)
        avg_distance = np.mean(neighbor_dists)
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

for idx, row in pred_df.iterrows():
    with st.expander(f"{row['Game']} — Prediction: {row['Prediction']}"):
        st.write(f"**Spread:** {row['Spread']} | **Total:** {row['Total']}")
        st.write(f"**Prediction:** {row['Prediction']}")
        st.write(f"**Confidence %:** {row['ConfidencePercent']*100:.1f}%")
        st.write(f"**Avg Distance:** {row['AvgDistance']} | **Score:** {row['ConfidenceScore']:.1f}")
        st.dataframe(row['Neighbors'])

