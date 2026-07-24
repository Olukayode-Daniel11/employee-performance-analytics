
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE

from app.utils import save_object


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(
    "data/processed/emp_perf_analytics_processed_data.csv"
)


# ==========================================================
# Define Features and Target
# ==========================================================

X = df.drop(columns="PerformanceRating")
y = df["PerformanceRating"]


# ==========================================================
# Detect Feature Types
# ==========================================================

numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()


# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=3,
    stratify=y
)


# ==========================================================
# Numerical Pipeline
# ==========================================================

num_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ==========================================================
# Categorical Pipeline
# ==========================================================

cat_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# ==========================================================
# Column Transformer
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ]
)


# ==========================================================
# Fit ONLY on Training Data
# ==========================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ==========================================================
# Save Production Preprocessor
# ==========================================================

save_object(
    preprocessor,
    "app/employee_performance_analytics_preprocessor.pkl"
)


# ==========================================================
# Handle Class Imbalance
# ==========================================================

smote = SMOTE(random_state=3)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed,
    y_train
)


# ==========================================================
# Train Model
# ==========================================================

model = RandomForestClassifier(random_state=3)

model.fit(
    X_train_resampled,
    y_train_resampled
)


# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test_processed)


# ==========================================================
# Evaluate Model
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("=" * 60)
print("Model Evaluation")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Macro F1 : {f1:.4f}")


# ==========================================================
# Save Production Model
# ==========================================================

save_object(
    model,
    "app/employee_performance_analytics_model.pkl"
)

print("=" * 60)
print("Production model and preprocessor saved successfully.")
print("=" * 60)