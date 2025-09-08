import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
st.set_page_config(page_title="Pokémon Type Predictor", layout="centered")
st.title("Pokémon Type Prediction 🐉")
st.write("Predict if a Pokémon is Legendary based on its stats.")

# -----------------------------
# Load preprocessing objects
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load trained CatBoost model
with open("catboost_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Sidebar Inputs
st.sidebar.header("Pokémon Stats Input")
Type_1 = st.sidebar.selectbox("Type 1", label_encoders['Type 1'].classes_)
Type_2 = st.sidebar.selectbox("Type 2", label_encoders['Type 2'].classes_)

Total = st.sidebar.slider("Total", 100, 800, 400)
HP = st.sidebar.slider("HP", 10, 250, 70)
Attack = st.sidebar.slider("Attack", 10, 200, 80)
Defense = st.sidebar.slider("Defense", 10, 200, 80)
Sp_Atk = st.sidebar.slider("Sp. Atk", 10, 200, 80)
Sp_Def = st.sidebar.slider("Sp. Def", 10, 200, 80)
Speed = st.sidebar.slider("Speed", 10, 180, 60)
Generation = st.sidebar.selectbox("Generation", [1, 2, 3, 4, 5, 6])

# -----------------------------
# Prepare input for model
input_df = pd.DataFrame({
    'Type 1': [Type_1],
    'Type 2': [Type_2],
    'Total': [Total],
    'HP': [HP],
    'Attack': [Attack],
    'Defense': [Defense],
    'Sp. Atk': [Sp_Atk],
    'Sp. Def': [Sp_Def],
    'Speed': [Speed],
    'Generation': [Generation]
})

# Encode categorical variables
for col in ['Type 1', 'Type 2']:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
num_cols = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# -----------------------------
# Predict
if st.button("Predict Legendary Status"):
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[:,1][0]
    
    if pred_class == 1:
        st.success(f"This Pokémon is likely LEGENDARY! 🏆 (Probability: {pred_prob:.2f})")
    else:
        st.info(f"This Pokémon is likely NOT Legendary. (Probability: {pred_prob:.2f})")
