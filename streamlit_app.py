# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle

# -----------------------------
st.set_page_config(page_title="Pokémon Legendary Predictor", layout="centered")
st.title("Pokémon Legendary Predictor 🐉")
st.write("Predict if a Pokémon is Legendary based on its stats.")

# -----------------------------
# Load Preprocessed dataset
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
# Sidebar Inputs
st.sidebar.header("Enter Pokémon Stats")

Name = st.sidebar.selectbox("Pokémon Name", sorted(preprocessed_df['Name'].unique()))

Type_1_options = sorted(preprocessed_df['Type 1'].unique())
Type_1 = st.sidebar.selectbox("Type 1", Type_1_options)

Type_2_options = sorted(preprocessed_df['Type 2'].dropna().unique())
Type_2 = st.sidebar.selectbox("Type 2 (Optional)", ["None"] + Type_2_options)
if Type_2 == "None":
    Type_2 = "None"

def slider_range(col):
    return int(preprocessed_df[col].min()), int(preprocessed_df[col].max()), int(preprocessed_df[col].mean())

Total = st.sidebar.slider("Total", *slider_range("Total"))
HP = st.sidebar.slider("HP", *slider_range("HP"))
Attack = st.sidebar.slider("Attack", *slider_range("Attack"))
Defense = st.sidebar.slider("Defense", *slider_range("Defense"))
Sp_Atk = st.sidebar.slider("Sp. Atk", *slider_range("Sp. Atk"))
Sp_Def = st.sidebar.slider("Sp. Def", *slider_range("Sp. Def"))
Speed = st.sidebar.slider("Speed", *slider_range("Speed"))
Generation = st.sidebar.selectbox("Generation", sorted(preprocessed_df['Generation'].unique()))

# -----------------------------
# Prepare input DataFrame
input_df = pd.DataFrame({
    'Name': [Name],
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

# -----------------------------
# Encode categorical columns
for col in ['Name', 'Type 1', 'Type 2']:
    le = label_encoders[col]
    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Scale numeric columns
num_cols = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Ensure correct column order (must match training)
columns_order = ['Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
                 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
input_df = input_df[columns_order]

# Ensure proper dtypes
input_df[num_cols] = input_df[num_cols].astype(float)
input_df[['Name', 'Type 1', 'Type 2']] = input_df[['Name', 'Type 1', 'Type 2']].astype(int)

# -----------------------------
# Prediction
if st.button("Predict Legendary Status"):
    try:
        pred_class = model.predict(input_df)[0]
        pred_prob = model.predict_proba(input_df)[:, 1][0]

        if pred_class == 1:
            st.success(f"{Name} is likely LEGENDARY! 🏆 (Probability: {pred_prob:.2f})")
        else:
            st.info(f"{Name} is likely NOT Legendary. (Probability: {pred_prob:.2f})")
    except Exception as e:
        st.error("Prediction failed! Check that all inputs are valid and match training data.")
        st.write(str(e))
