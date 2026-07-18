import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_FILE = "Loan_data - Sheet1.csv"
TARGET_COL = "Loan _Status"
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Credit_History",
]

NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Credit_History",
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
]

MODEL_PATH = "model.pkl"


def load_data(path: str = DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[NUMERIC_FEATURES] = SimpleImputer(strategy="median").fit_transform(df[NUMERIC_FEATURES])
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].fillna("missing")
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_save(dataset_path: str = DATA_FILE, output_path: str = MODEL_PATH) -> None:
    df = load_data(dataset_path)
    X = df[FEATURE_COLUMNS]
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df[TARGET_COL].astype(str))

    model_pipeline = build_pipeline()
    use_full_training = False

    label_counts = pd.Series(y).value_counts()
    if len(df) < 10 or label_counts.min() < 2:
        print("Warning: dataset is too small or has a rare class; training on all available data.")
        model_pipeline.fit(X, y)
        print("Trained model using the full dataset.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Validation accuracy: {accuracy:.4f}")

    package = {
        "pipeline": model_pipeline,
        "target_encoder": target_encoder,
        "feature_columns": FEATURE_COLUMNS,
    }
    joblib.dump(package, output_path)
    print(f"Saved trained model to {output_path}")


if __name__ == "__main__":
    train_and_save()
