import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mysklearn.myclassifiers import MyRandomForestClassifier
import random

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

# Convert y to list (MyRandomForestClassifier expects list, not numpy array)
y = [row[target_column] for row in rows]

# =====================================
# 2. Train/Test Split (same as other algorithms)
# =====================================
seed = 9

X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y, test_size=0.33, stratify=y, random_state=seed
)

# =====================================
# 3. Random Forest Classifier
# =====================================
# Using test_size=0.0 to disable internal splitting since we already split
# Using consistent parameters: n_estimators=20 (reasonable for comparison)
# max_features will be sqrt(n_features) by default (similar to sklearn)
rf = MyRandomForestClassifier(
    n_estimators=20,
    n_best_trees=None,  # Use all trees for this demo
    max_features=None,  # Will use sqrt(n_features) by default
    bootstrap=True,
    random_state=seed,
    test_size=0.0  # No internal split, use our pre-split data
)

# =====================================
# 4. Fit the Model
# =====================================
rf.fit(X_train, y_train)

# =====================================
# 5. Evaluate
# =====================================
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
