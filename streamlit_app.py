# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
st.set_page_config(page_title="Pokémon Legendary Predictor", layout="centered")
st.title("Pokémon Legendary Predictor 🐉")
st.write("Predict if a Pokémon is Legendary based on its stats.")

# -----------------------------
# Load Preprocessed dataset (before encoding/scaling)
preprocessed_df = pd.read_csv("Preprocessed.csv")

# Load preprocessing objects
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load trained CatBoost model
with open("catboost_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Sidebar Inputs - dynamically from dataset
st.sidebar.header("Enter Pokémon Stats")

Name = st.sidebar.selectbox("Pokémon Name", sorted(preprocessed_df['Name'].unique()))

Type_1 = st.sidebar.selectbox("Type 1", sorted(preprocessed_df['Type 1'].unique()))
Type_2 = st.sidebar.selectbox("Type 2", sorted(preprocessed_df['Type 2'].dropna().unique()))

# For numeric columns, get min/max from dataset
Total_min, Total_max = int(preprocessed_df['Total'].min()), int(preprocessed_df['Total'].max())
HP_min, HP_max = int(preprocessed_df['HP'].min()), int(preprocessed_df['HP'].max())
Attack_min, Attack_max = int(preprocessed_df['Attack'].min()), int(preprocessed_df['Attack'].max())
Defense_min, Defense_max = int(preprocessed_df['Defense'].min()), int(preprocessed_df['Defense'].max())
Sp_Atk_min, Sp_Atk_max = int(preprocessed_df['Sp. Atk'].min()), int(preprocessed_df['Sp. Atk'].max())
Sp_Def_min, Sp_Def_max = int(preprocessed_df['Sp. Def'].min()), int(preprocessed_df['Sp. Def'].max())
Speed_min, Speed_max = int(preprocessed_df['Speed'].min()), int(preprocessed_df['Speed'].max())
Generation_min, Generation_max = int(preprocessed_df['Generation'].min()), int(preprocessed_df['Generation'].max())

Total = st.sidebar.slider("Total", Total_min, Total_max, int(preprocessed_df['Total'].mean()))
HP = st.sidebar.slider("HP", HP_min, HP_max, int(preprocessed_df['HP'].mean()))
Attack = st.sidebar.slider("Attack", Attack_min, Attack_max, int(preprocessed_df['Attack'].mean()))
Defense = st.sidebar.slider("Defense", Defense_min, Defense_max, int(preprocessed_df['Defense'].mean()))
Sp_Atk = st.sidebar.slider("Sp. Atk", Sp_Atk_min, Sp_Atk_max, int(preprocessed_df['Sp. Atk'].mean()))
Sp_Def = st.sidebar.slider("Sp. Def", Sp_Def_min, Sp_Def_max, int(preprocessed_df['Sp. Def'].mean()))
Speed = st.sidebar.slider("Speed", Speed_min, Speed_max, int(preprocessed_df['Speed'].mean()))
Generation = st.sidebar.selectbox("Generation", sorted(preprocessed_df['Generation'].unique()))

# -----------------------------
# Prepare input DataFrame
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

# Encode categorical columns
for col in ['Type 1', 'Type 2']:
    le = label_encoders[col]
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
num_cols = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Ensure column order
columns_order = ['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
                 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
input_df = input_df[columns_order]

# -----------------------------
# Prediction
if st.button("Predict Legendary Status"):
    pred_class = model.predict(input_df)[0]
    pred_prob = model.predict_proba(input_df)[:, 1][0]

    if pred_class == 1:
        st.success(f"{Name} is likely LEGENDARY! 🏆 (Probability: {pred_prob:.2f})")
    else:
        st.info(f"{Name} is likely NOT Legendary. (Probability: {pred_prob:.2f})")
