# MyRandomForestClassifier - Implementation Details

## Overview

`MyRandomForestClassifier` implements a **true Random Forest** algorithm with:

1. **Internal stratified train/test splitting** (1/3 test, 2/3 train)
2. **Bootstrap aggregating (bagging)** for each tree
3. **Random feature selection AT EACH SPLIT** (not per tree!)

## Key Implementation Detail: Random Feature Selection

### ✓ Correct Implementation (Current)

**At each node/split during tree construction**, randomly select **F attributes** from the remaining available attributes as candidates to partition on.

- Each decision tree can use ALL features
- At every split point in the tree, randomly choose F features to consider
- Different splits in the same tree may use different feature subsets
- This is how sklearn's RandomForestClassifier works!

Example with 4 total features and F=2:
```
Tree 1:
  Root split: randomly selects [level, tweets] from [level, lang, tweets, phd]
    → Splits on level (best of the 2)
  Left child split: randomly selects [lang, phd] from [lang, tweets, phd]
    → Splits on phd (best of the 2)
  Right child split: randomly selects [tweets, phd] from [lang, tweets, phd]
    → Splits on tweets (best of the 2)
```

### ✗ Previous (Incorrect) Implementation

Selected F features once per tree and used only those features for all splits in that tree.

## Stratified Splitting

The `stratified_train_test_split()` function in `myevaluation.py`:
- Groups instances by their class label
- Samples proportionally from each class
- Ensures both train and test sets have similar class distributions
- Default: 33% test, 67% remainder (training)

### Example with Interview Dataset

**Original Dataset (14 instances):**
- Class "True": 9 instances (64.3%)
- Class "False": 5 instances (35.7%)

**After Stratified Split:**

**Remainder Set (9 instances - used for training):**
- Class "True": 6 instances (66.7%)
- Class "False": 3 instances (33.3%)

**Test Set (5 instances - held out for evaluation):**
- Class "True": 3 instances (60%)
- Class "False": 2 instances (40%)

The class distribution is approximately maintained in both sets!

## Usage

### Basic Usage (with internal split)
```python
from mysklearn.myclassifiers import MyRandomForestClassifier

# Full dataset
X = [...]  # 14 instances
y = [...]  # 14 labels

# Create classifier with N=20 trees, F=2 features, test_size=0.33
rf = MyRandomForestClassifier(n_estimators=20, max_features=2, 
                              bootstrap=True, random_state=0, test_size=0.33)

# Fit performs stratified split internally
# Trains on 2/3 (remainder set), holds out 1/3 (test set)
rf.fit(X, y)

# Access the internal sets
print(f"Remainder set size: {len(rf.X_remainder)}")  # 9
print(f"Test set size: {len(rf.X_test_internal)}")   # 5

# Get test accuracy on the stratified test set
test_acc = rf.get_test_accuracy()
print(f"Test Accuracy: {test_acc:.2%}")

# Make predictions on new data
predictions = rf.predict(new_data)
```

### Training on Full Dataset (no split)
```python
# Set test_size=0.0 to disable internal splitting
rf_full = MyRandomForestClassifier(n_estimators=20, max_features=2,
                                   test_size=0.0)
rf_full.fit(X, y)  # Uses all data for training
```

## Key Attributes

- `X_remainder`: Training data (2/3 of original, stratified)
- `y_remainder`: Training labels
- `X_test_internal`: Stratified test set (1/3 of original)
- `y_test_internal`: Test labels
- `test_size`: Proportion used for test set (default=0.33)

## Methods

### `fit(X, y)`
- Performs stratified split
- Trains N trees on remainder set
- Each tree uses bootstrap sampling and random features

### `predict(X_test)`
- Makes predictions using majority voting
- Works on any data (internal test set or external data)

### `get_test_accuracy()`
- Returns accuracy on internal stratified test set
- Returns `None` if no test set exists

### `get_oob_score()`
- Returns out-of-bag accuracy on remainder set
- Only available when `bootstrap=True`

### `print_forest_info()`
- Shows dataset split information
- Displays feature importances
- Shows OOB score and test accuracy

## Parameters

- **N (n_estimators)**: Number of decision trees (default=10)
- **F (max_features)**: Number of random features to consider AT EACH SPLIT (default=sqrt(n_features))
- **test_size**: Proportion for stratified test set (default=0.33)
- **bootstrap**: Whether to use bootstrap sampling (default=True)
- **random_state**: Seed for reproducibility (default=None)

## Demo Results (N=20, F=2)

```
Dataset split:
  Remainder set (training): 9 instances
  Test set (stratified): 5 instances

Feature Importances: Uniform (0.25 each for 4 features)

Out-of-Bag Score (on remainder set): 100.00%
Test Set Accuracy (stratified): 80.00%
```

**Note**: Each tree randomly selects F=2 features **at every decision node**, not once per tree. This provides maximum diversity in the ensemble.
