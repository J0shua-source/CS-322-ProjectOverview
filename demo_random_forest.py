from mysklearn.myclassifiers import MyRandomForestClassifier
from mysklearn.myevaluation import train_test_split
import random

# Set random seed for reproducibility
random.seed(0)

# Interview dataset (14 instances) from test_myclassifiers.py
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"]
]
y_train_interview = ["False", "False", "True", "True", "True", "False", "True", 
                     "False", "True", "True", "True", "True", "True", "False"]

print("=" * 70)
print("MyRandomForestClassifier Demo - Interview Dataset")
print("=" * 70)
print("Dataset: Job Interview Predictions")
print(f"Total instances: {len(X_train_interview)}")
print(f"Attributes: {header_interview}")
print()

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_train_interview, y_train_interview, test_size=0.33, random_state=0
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print()

# Create and train random forest with different configurations
print("Testing Random Forest with N=20, M=7, F=2")
print("-" * 70)
rf = MyRandomForestClassifier(n_estimators=20, max_features=2, 
                              bootstrap=True, random_state=0)

print("Training the forest...")
rf.fit(X_train, y_train)

print("\nForest Information:")
rf.print_forest_info()

# Make predictions
print("\nMaking predictions on test set...")
predictions = rf.predict(X_test)

print("\nTest Instances and Predictions:")
for i, (x, pred, actual) in enumerate(zip(X_test, predictions, y_test)):
    match = "✓" if pred == actual else "✗"
    print(f"  {i+1}. {dict(zip(header_interview[:-1], x))}")
    print(f"     Predicted: {pred}, Actual: {actual} {match}")

# Calculate accuracy
correct = sum(1 for pred, actual in zip(predictions, y_test) if pred == actual)
accuracy = correct / len(y_test)
print()
print(f"Test Accuracy: {accuracy:.2%} ({correct}/{len(y_test)})")

# Show feature subsets used
print("\nFeature subsets used by each tree:")
for i, subset in enumerate(rf.feature_subsets):
    feature_names = [header_interview[idx] for idx in subset]
    print(f"  Tree {i+1}: {feature_names}")

print()
print("=" * 70)
print("Testing with entire dataset (N=20, M=7, F=2)")
print("-" * 70)

# Train on full dataset
rf_full = MyRandomForestClassifier(n_estimators=20, max_features=2, 
                                   bootstrap=True, random_state=42)
rf_full.fit(X_train_interview, y_train_interview)

# Predict on same data
predictions_full = rf_full.predict(X_train_interview)
correct_full = sum(1 for pred, actual in zip(predictions_full, y_train_interview) 
                  if pred == actual)
accuracy_full = correct_full / len(y_train_interview)

print(f"\nTraining Accuracy: {accuracy_full:.2%} ({correct_full}/{len(y_train_interview)})")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
