from mysklearn.myclassifiers import MyRandomForestClassifier
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
print("With Internal Stratified Train/Test Split")
print("=" * 70)
print("Dataset: Job Interview Predictions")
print(f"Total instances: {len(X_train_interview)}")
print(f"Attributes: {header_interview}")
print()

# Create and train random forest with N=20, M=7, F=2
# The classifier will internally create a stratified 1/3 test set
# and train on the remaining 2/3 (remainder set)
print("Testing Random Forest with N=20, M=7, F=2")
print("(Classifier performs internal stratified split: 2/3 train, 1/3 test)")
print("-" * 70)
rf = MyRandomForestClassifier(n_estimators=20, max_features=2, 
                              bootstrap=True, random_state=0, test_size=0.33)

print("\nTraining the forest...")
print("(Internal stratified split will create remainder and test sets)")
rf.fit(X_train_interview, y_train_interview)

print("\nForest Information:")
rf.print_forest_info()

# Make predictions on the internal test set
print("\n" + "=" * 70)
print("Predictions on Internal Stratified Test Set")
print("-" * 70)
predictions = rf.predict(rf.X_test_internal)

print("\nTest Instances and Predictions:")
for i, (x, pred, actual) in enumerate(zip(rf.X_test_internal, predictions, rf.y_test_internal)):
    match = "✓" if pred == actual else "✗"
    print(f"  {i+1}. {dict(zip(header_interview[:-1], x))}")
    print(f"     Predicted: {pred}, Actual: {actual} {match}")

# Calculate accuracy
correct = sum(1 for pred, actual in zip(predictions, rf.y_test_internal) if pred == actual)
accuracy = correct / len(rf.y_test_internal)
print()
print(f"Test Accuracy: {accuracy:.2%} ({correct}/{len(rf.y_test_internal)})")

print()
print("=" * 70)
print("Additional Test: Training on Full Dataset")
print("(For comparison - no internal split)")
print("-" * 70)

# Train on full dataset with test_size=0 to disable splitting
rf_full = MyRandomForestClassifier(n_estimators=20, max_features=2, 
                                   bootstrap=True, random_state=42, test_size=0.0)
rf_full.fit(X_train_interview, y_train_interview)

# Predict on same data
predictions_full = rf_full.predict(rf_full.X_remainder)
correct_full = sum(1 for pred, actual in zip(predictions_full, rf_full.y_remainder) 
                  if pred == actual)
accuracy_full = correct_full / len(rf_full.y_remainder)

print(f"\nTraining Accuracy (no test split): {accuracy_full:.2%} ({correct_full}/{len(rf_full.y_remainder)})")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
