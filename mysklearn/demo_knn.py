import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# 1. Load Dataset
# =====================================
filename = "../input_data/bitcoin_sentiment_discretized_cleaned.csv"

rows = []
with open(filename, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Extract column names
all_columns = list(rows[0].keys())
target_column = "price_direction"

# Create X and y manually
X = [{col: row[col] for col in all_columns if col != target_column} for row in rows]
y = [row[target_column] for row in rows]

# Convert to list-of-lists for sklearn
X_cols = list(X[0].keys())
X_matrix = [[row[col] for col in X_cols] for row in X]

# Convert y to numpy array for sklearn
y = np.array(y)

# =====================================
# 2. Detect categorical features
# =====================================
categorical_features = list(range(len(X_cols)))  # All columns are categorical

# =====================================
# 3. Preprocessor: One-Hot Encode
# =====================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =====================================
# 4. Pipeline = Preprocess + KNN
# =====================================
seed = 9

clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("knn", KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        metric="minkowski",
        p=2  # Euclidean distance
    ))
])

# =====================================
# 5. Train/Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y, test_size=0.33, stratify=y, random_state=seed
)

# =====================================
# 6. Fit the Model
# =====================================
clf.fit(X_train, y_train)

# =====================================
# 7. Evaluate
# =====================================
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
