import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from catboost import CatBoostClassifier


# ============================================================
# FILES
# ============================================================

DATA_FILE = "Pokemon.csv"

MODEL_FILE = "catboost_best_model.pkl"
ENCODER_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"
PROCESSED_FILE = "Processed.csv"


# ============================================================
# FEATURES
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

TARGET_COLUMN = "Legendary"

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
# LOAD DATA
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# REMOVE ID COLUMNS
# ============================================================

if "#" in df.columns:
    df = df.drop(columns=["#"])

if "Name" in df.columns:
    df = df.drop(columns=["Name"])

print("\nRemoved identifier columns: # and Name")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

df["Type 1"] = (
    df["Type 1"]
    .fillna("None")
    .astype(str)
)

df["Type 2"] = (
    df["Type 2"]
    .fillna("None")
    .astype(str)
)


# ============================================================
# CREATE X AND Y
# ============================================================

X = df[FEATURE_COLUMNS].copy()

y = df[TARGET_COLUMN].astype(int)


print("\nFeatures used for training:")
print(FEATURE_COLUMNS)

print("\nTarget:")
print(TARGET_COLUMN)


# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

label_encoders = {}

for column in CATEGORICAL_COLUMNS:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column]
    )

    label_encoders[column] = encoder

    print(
        f"\n{column} encoding completed."
    )


# ============================================================
# SCALE NUMERICAL FEATURES
# ============================================================

scaler = StandardScaler()

X[NUMERICAL_COLUMNS] = scaler.fit_transform(
    X[NUMERICAL_COLUMNS]
)

print("\nNumerical scaling completed.")


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# CATBOOST MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CATBOOST MODEL")
print("=" * 60)

model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100,
    allow_writing_files=False
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

y_pred = (
    np.asarray(y_pred)
    .astype(int)
    .flatten()
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy:.2%}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Legendary",
            "Legendary"
        ]
    )
)


# ============================================================
# VERIFY MODEL FEATURES
# ============================================================

print("\n" + "=" * 60)
print("MODEL FEATURES")
print("=" * 60)

print(model.feature_names_)


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    MODEL_FILE,
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print(
    f"\nSaved: {MODEL_FILE}"
)


# ============================================================
# SAVE ENCODERS
# ============================================================

with open(
    ENCODER_FILE,
    "wb"
) as file:

    pickle.dump(
        label_encoders,
        file
    )

print(
    f"Saved: {ENCODER_FILE}"
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_FILE,
    "wb"
) as file:

    pickle.dump(
        scaler,
        file
    )

print(
    f"Saved: {SCALER_FILE}"
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

processed_df = X.copy()

processed_df[TARGET_COLUMN] = y

processed_df.to_csv(
    PROCESSED_FILE,
    index=False
)

print(
    f"Saved: {PROCESSED_FILE}"
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal model features:")

for index, feature in enumerate(
    model.feature_names_,
    start=1
):
    print(
        f"{index}. {feature}"
    )

print("\nTarget:", TARGET_COLUMN)

print("\nYou can now run:")
print("streamlit run streamlit_app.py")