# Pokémon Legendary Predictor - Streamlit App

This project predicts if a Pokémon is Legendary based on its stats using a CatBoost machine learning model.

## Features
- User-friendly interface with sliders and dropdowns for Pokémon stats.
- Predicts Legendary status (Yes/No) with probability.
- Uses saved preprocessing pipelines (LabelEncoding + Scaling).
- Fully deployable on Streamlit Cloud.

## Files
- `streamlit_app.py` : Main Streamlit application.
- `Preprocessed.csv` : Dataset before encoding/scaling.
- `Processed.csv` : Dataset after encoding/scaling.
- `catboost_best_model.pkl` : Trained CatBoost model.
- `label_encoders.pkl` : Saved LabelEncoders for Type 1 and Type 2.
- `scaler.pkl` : Saved scaler for numeric columns.
- `requirements.txt` : Required Python packages.
- `.gitignore` : Git ignore file.

## Deployment
1. Push all files to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New App" → Connect your GitHub repo → Deploy.

## Usage
1. Use sidebar to input Pokémon stats.
2. Click **Predict Legendary Status**.
3. View the prediction and probability in the main panel.
