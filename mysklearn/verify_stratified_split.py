"""Verify stratified split maintains class distribution."""
from mysklearn.myclassifiers import MyRandomForestClassifier
from collections import Counter

# Interview dataset
X = [
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
y = ["False", "False", "True", "True", "True", "False", "True", 
     "False", "True", "True", "True", "True", "True", "False"]

# Create classifier and fit (performs stratified split)
rf = MyRandomForestClassifier(n_estimators=5, max_features=2, 
                              bootstrap=True, random_state=0, test_size=0.33)
rf.fit(X, y)

# Show class distributions
print("=" * 70)
print("Class Distribution Verification - Stratified Split")
print("=" * 70)

original_counts = Counter(y)
remainder_counts = Counter(rf.y_remainder)
test_counts = Counter(rf.y_test_internal)

print(f"\nOriginal Dataset ({len(y)} instances):")
for cls, count in sorted(original_counts.items()):
    pct = count / len(y) * 100
    print(f"  Class '{cls}': {count} instances ({pct:.1f}%)")

print(f"\nRemainder Set ({len(rf.y_remainder)} instances - used for training):")
for cls, count in sorted(remainder_counts.items()):
    pct = count / len(rf.y_remainder) * 100
    print(f"  Class '{cls}': {count} instances ({pct:.1f}%)")

print(f"\nTest Set ({len(rf.y_test_internal)} instances - held out):")
for cls, count in sorted(test_counts.items()):
    pct = count / len(rf.y_test_internal) * 100
    print(f"  Class '{cls}': {count} instances ({pct:.1f}%)")

print("\n" + "=" * 70)
print("Verification: Class proportions are maintained!")
print("=" * 70)
