# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
import re

# -----------------------------
# Page Config & Styling
st.set_page_config(page_title="Pokémon Legendary Predictor", page_icon="🐉", layout="wide")

st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f6f9fc, #e9f5ff);
        }
        .stButton>button {
            background-color: #ffcb05;
            color: black;
            border-radius: 12px;
            font-size: 18px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #3b4cca;
            color: white;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
st.title("🐉 Pokémon Legendary Predictor")
st.write("Use Pokémon stats to predict whether it is **Legendary or not**!")

# -----------------------------
# Load Data & Models
preprocessed_df = pd.read_csv("Preprocessed.csv")

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("catboost_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Sidebar Inputs
st.sidebar.header("⚙️ Input Pokémon Stats")

Name = st.sidebar.selectbox("Pokémon Name", sorted(preprocessed_df['Name'].unique()))

Type_1 = st.sidebar.selectbox("Type 1", sorted(preprocessed_df['Type 1'].unique()))

Type_2 = st.sidebar.selectbox(
    "Type 2 (Optional)", ["None"] + sorted(preprocessed_df['Type 2'].dropna().unique())
)
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

shiny_mode = st.sidebar.checkbox("✨ Show Shiny Sprite", value=False)

# -----------------------------
# Input Data
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
# Encode categorical features
for col in ['Name', 'Type 1', 'Type 2']:
    le = label_encoders[col]
    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

# Scale numeric features
num_cols = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Generation"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# Column order
columns_order = ['Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
                 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']
input_df = input_df[columns_order]

# -----------------------------
# Prediction & Layout
col1, col2 = st.columns([2, 3])

with col1:
    try:
        # Normalize Pokémon name for URL
        poke_name = Name.lower()
        poke_name = re.sub(r"[^\w\s-]", "", poke_name)   # remove special chars
        poke_name = poke_name.replace(" ", "-")

        # Sprite URL
        sprite_type = "shiny" if shiny_mode else "normal"
        sprite_url = f"https://img.pokemondb.net/sprites/home/{sprite_type}/{poke_name}.png"

        st.image(sprite_url, width=250, caption=f"✨ {Name} ({'Shiny' if shiny_mode else 'Normal'})")
    except Exception:
        st.image("https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png",
                 width=200, caption="❓ Unknown Pokémon")

with col2:
    if st.button("🔮 Predict Legendary Status"):
        try:
            pred_class = model.predict(input_df)[0]
            pred_prob = model.predict_proba(input_df)[:, 1][0]

            if pred_class == 1:
                st.markdown(
                    f"<div class='prediction-box' style='background-color:#ffd700; color:black;'>"
                    f"🏆 {Name} is **LEGENDARY!** <br>(Probability: {pred_prob:.2f})"
                    f"</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='prediction-box' style='background-color:#87ceeb; color:black;'>"
                    f"✨ {Name} is **NOT Legendary** <br>(Probability: {pred_prob:.2f})"
                    f"</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error("❌ Prediction failed! Please check inputs.")
            st.write(str(e))
