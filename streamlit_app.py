import streamlit as st
import pandas as pd
import numpy as np
import pickle


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pokémon Legendary Predictor",
    page_icon="🐉",
    layout="centered"
)


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "Type 1",
    "Type 2",
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed",
    "Generation"
]

CATEGORICAL_COLUMNS = [
    "Type 1",
    "Type 2"
]

NUMERICAL_COLUMNS = [
    "Total",
    "HP",
    "Attack",
    "Defense",
    "Sp. Atk",
    "Sp. Def",
    "Speed",
    "Generation"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    with open(
        "catboost_best_model.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD ENCODERS
# ============================================================

@st.cache_resource
def load_encoders():

    with open(
        "label_encoders.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD SCALER
# ============================================================

@st.cache_resource
def load_scaler():

    with open(
        "scaler.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# LOAD OBJECTS
# ============================================================

try:

    model = load_model()
    label_encoders = load_encoders()
    scaler = load_scaler()

except Exception as error:

    st.error(
        "Could not load model files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# VERIFY MODEL
# ============================================================

if not hasattr(
    model,
    "feature_names_"
):

    st.error(
        "The loaded CatBoost model does not contain feature names."
    )

    st.stop()


model_features = list(
    model.feature_names_
)


# ============================================================
# IMPORTANT SAFETY CHECK
# ============================================================

if model_features != FEATURE_COLUMNS:

    st.error(
        "❌ OLD / INCOMPATIBLE MODEL DETECTED"
    )

    st.write(
        "The application expects:"
    )

    st.code(
        "\n".join(FEATURE_COLUMNS)
    )

    st.write(
        "But the loaded model expects:"
    )

    st.code(
        "\n".join(model_features)
    )

    st.warning(
        "Run train_model.py again and restart Streamlit."
    )

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🐉 Pokémon Legendary Predictor"
)

st.write(
    "Enter the Pokémon's statistics to predict "
    "whether it is Legendary."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎮 Pokémon Details"
)


# ============================================================
# TYPE 1
# ============================================================

type1_options = (
    label_encoders[
        "Type 1"
    ].classes_
)

type_1 = st.sidebar.selectbox(
    "Type 1",
    type1_options
)


# ============================================================
# TYPE 2
# ============================================================

type2_options = (
    label_encoders[
        "Type 2"
    ].classes_
)

type_2 = st.sidebar.selectbox(
    "Type 2",
    type2_options
)


# ============================================================
# STATS
# ============================================================

total = st.sidebar.slider(
    "Total",
    100,
    800,
    400
)

hp = st.sidebar.slider(
    "HP",
    1,
    255,
    70
)

attack = st.sidebar.slider(
    "Attack",
    1,
    200,
    80
)

defense = st.sidebar.slider(
    "Defense",
    1,
    250,
    80
)

sp_atk = st.sidebar.slider(
    "Sp. Atk",
    1,
    250,
    80
)

sp_def = st.sidebar.slider(
    "Sp. Def",
    1,
    250,
    80
)

speed = st.sidebar.slider(
    "Speed",
    1,
    200,
    60
)

generation = st.sidebar.selectbox(
    "Generation",
    [1, 2, 3, 4, 5, 6]
)


# ============================================================
# CREATE RAW INPUT
# ============================================================

input_df = pd.DataFrame({

    "Type 1": [type_1],

    "Type 2": [type_2],

    "Total": [total],

    "HP": [hp],

    "Attack": [attack],

    "Defense": [defense],

    "Sp. Atk": [sp_atk],

    "Sp. Def": [sp_def],

    "Speed": [speed],

    "Generation": [generation]
})


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

try:

    for column in CATEGORICAL_COLUMNS:

        encoder = label_encoders[column]

        input_df[column] = encoder.transform(
            input_df[column].astype(str)
        )

except Exception as error:

    st.error(
        "Error encoding Type 1 / Type 2."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SCALE NUMERICAL FEATURES
# ============================================================

try:

    input_df[NUMERICAL_COLUMNS] = scaler.transform(
        input_df[NUMERICAL_COLUMNS]
    )

except Exception as error:

    st.error(
        "Error scaling numerical features."
    )

    st.exception(error)

    st.stop()


# ============================================================
# FORCE EXACT FEATURE ORDER
# ============================================================

input_df = input_df[
    FEATURE_COLUMNS
]


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Legendary Status",
    use_container_width=True
):

    try:

        # ================================================
        # FINAL DEBUG CHECK
        # ================================================

        if list(
            input_df.columns
        ) != model_features:

            st.error(
                "Feature order mismatch."
            )

            st.write(
                "Input:"
            )

            st.write(
                list(input_df.columns)
            )

            st.write(
                "Model:"
            )

            st.write(
                model_features
            )

            st.stop()


        # ================================================
        # PREDICT
        # ================================================

        prediction = model.predict(
            input_df
        )

        pred_class = int(
            np.asarray(
                prediction
            ).flatten()[0]
        )


        # ================================================
        # PROBABILITY
        # ================================================

        probabilities = model.predict_proba(
            input_df
        )[0]

        legendary_probability = float(
            probabilities[1]
        )

        not_legendary_probability = float(
            probabilities[0]
        )


        # ================================================
        # RESULT
        # ================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )


        if pred_class == 1:

            st.success(
                "🏆 This Pokémon is likely LEGENDARY!"
            )

            st.metric(
                "Legendary Confidence",
                f"{legendary_probability:.2%}"
            )

        else:

            st.info(
                "⚪ This Pokémon is likely NOT Legendary."
            )

            st.metric(
                "Not Legendary Confidence",
                f"{not_legendary_probability:.2%}"
            )


        # ================================================
        # PROBABILITY
        # ================================================

        st.write(
            "### Probability"
        )

        st.write(
            f"Legendary: "
            f"{legendary_probability:.2%}"
        )

        st.progress(
            legendary_probability
        )


        # ================================================
        # INPUT DATA
        # ================================================

        with st.expander(
            "View processed input"
        ):

            st.dataframe(
                input_df
            )


    except Exception as error:

        st.error(
            "❌ Prediction Error"
        )

        st.exception(error)