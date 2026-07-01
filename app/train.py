
import pandas as pd
import joblib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from imblearn.over_sampling import SMOTE

from utils import save_object

# Load Dataset

df = pd.read_csv(
    "data/processed/emp_performance_processed_data.csv"
)


# Define Features and Target

X = df.drop("PerformanceRating", axis=1)
y = df["PerformanceRating"]


# Separate Numerical and Categorical Columns

numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "string"]).columns
# Numerical Pipeline

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical Pipeline

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Column Transformer

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])


# Transform Features

X_processed = preprocessor.fit_transform(X)


# Save Preprocessor

save_object(preprocessor, "app/preprocessor.pkl")

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=3,
    stratify=y
)


# Handle Class Imbalance

smote = SMOTE(random_state=3)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)

# Train Model

model = RandomForestClassifier(random_state=3)
model.fit(X_train_resampled, y_train_resampled)


# Predictions

y_pred = model.predict(X_test)


# Evaluation

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Save Model

save_object(model, "app/model.pkl")

print("Model and preprocessor saved successfully.")

