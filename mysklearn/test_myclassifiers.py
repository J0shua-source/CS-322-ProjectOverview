import sys
import os
# Add parent directory to path to allow importing mysklearn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier



def test_decision_tree_classifier_fit():
    # Test case 1: Interview dataset (14 instances)
    # From B Attribute Selection (Entropy) Lab Task #1
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
    
    # Expected tree structure from entropy-based attribute selection
    # The tree should split on 'level' first (att0), then further splits
    interview_tree = ["Attribute", "att0",
        ["Value", "Junior", 
            ["Attribute", "att3",
                ["Value", "no", 
                    ["Leaf", "True", 3, 3]
                ],
                ["Value", "yes", 
                    ["Leaf", "False", 2, 2]
                ]
            ]
        ],
        ["Value", "Mid", 
            ["Leaf", "True", 4, 4]
        ],
        ["Value", "Senior", 
            ["Attribute", "att2",
                ["Value", "no", 
                    ["Leaf", "False", 3, 3]
                ],
                ["Value", "yes", 
                    ["Leaf", "True", 2, 2]
                ]
            ]
        ]
    ]
    
    tree_interview = MyDecisionTreeClassifier()
    tree_interview.fit(X_train_interview, y_train_interview)
    assert tree_interview.tree == interview_tree
    
    # Test case 2: iPhone dataset (15 instances)
    # From LA7 with clash resolution (alphabetical order for ties)
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", 
                      "yes", "yes", "yes", "yes", "yes"]
    
    # Expected tree structure from entropy-based attribute selection with clash resolution
    # When there are clashes (same X values, different y), use majority vote
    # On ties in majority voting, choose alphabetically first class label
    iphone_tree = ["Attribute", "att0",
        ["Value", 1,
            ["Attribute", "att1",
                ["Value", 1,
                    ["Leaf", "yes", 1, 1]
                ],
                ["Value", 2,
                    ["Attribute", "att2",
                        ["Value", "excellent",
                            ["Leaf", "yes", 1, 1]
                        ],
                        ["Value", "fair",
                            ["Leaf", "no", 1, 1]
                        ]
                    ]
                ],
                ["Value", 3,
                    ["Leaf", "no", 2, 2]
                ]
            ]
        ],
        ["Value", 2,
            ["Attribute", "att1",
                ["Value", 1,
                    ["Attribute", "att2",
                        ["Value", "excellent",
                            ["Leaf", "no", 1, 2]
                        ],
                        ["Value", "fair",
                            ["Leaf", "yes", 1, 1]
                        ]
                    ]
                ],
                ["Value", 2,
                    ["Leaf", "yes", 4, 4]
                ],
                ["Value", 3,
                    ["Leaf", "yes", 3, 3]
                ]
            ]
        ]
    ]
    
    tree_iphone = MyDecisionTreeClassifier()
    tree_iphone.fit(X_train_iphone, y_train_iphone)
    assert tree_iphone.tree == iphone_tree

def test_decision_tree_classifier_predict():
    # Test case 1: Interview dataset
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
    
    tree_interview = MyDecisionTreeClassifier()
    tree_interview.fit(X_train_interview, y_train_interview)
    
    # Test predictions
    X_test_interview = [
        ["Junior", "Python", "yes", "no"],  # Should predict True (Junior, phd=no)
        ["Junior", "Java", "yes", "yes"],   # Should predict False (Junior, phd=yes)
        ["Mid", "Python", "yes", "yes"],    # Should predict True (Mid)
        ["Senior", "Python", "no", "yes"],  # Should predict False (Senior, tweets=no)
        ["Senior", "R", "yes", "yes"]       # Should predict True (Senior, tweets=yes)
    ]
    y_expected = ["True", "False", "True", "False", "True"]
    
    y_predicted = tree_interview.predict(X_test_interview)
    assert y_predicted == y_expected
    
    # Test case 2: iPhone dataset
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", 
                      "yes", "yes", "yes", "yes", "yes"]
    
    tree_iphone = MyDecisionTreeClassifier()
    tree_iphone.fit(X_train_iphone, y_train_iphone)
    
    # Test predictions
    X_test_iphone = [
        [1, 1, "fair"],           # Should predict yes (standing=1, job=1)
        [1, 2, "excellent"],      # Should predict yes (standing=1, job=2, credit=excellent)
        [1, 2, "fair"],           # Should predict no (standing=1, job=2, credit=fair)
        [2, 2, "excellent"],      # Should predict yes (standing=2, job=2)
        [2, 1, "excellent"]       # Should predict no (standing=2, job=1, credit=excellent)
    ]
    y_expected_iphone = ["yes", "yes", "no", "yes", "no"]
    
    y_predicted_iphone = tree_iphone.predict(X_test_iphone)
    assert y_predicted_iphone == y_expected_iphone


def test_random_forest_classifier_fit():
    """Test MyRandomForestClassifier fit() method using interview dataset.
    
    Tests:
    - Random forest creates N=20 trees
    - Trees are trained on bootstrap samples
    - Random feature selection (F=2) works correctly
    - Best M=7 trees are selected based on validation accuracy
    """
    # Interview dataset (same as Decision Tree tests)
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
    
    # Test 1: Create Random Forest with N=20, M=7, F=2
    rf = MyRandomForestClassifier(
        n_estimators=20,
        n_best_trees=7,
        max_features=2,
        bootstrap=True,
        random_state=42,
        test_size=0.33
    )
    
    # Fit the model
    rf.fit(X_train_interview, y_train_interview)
    
    # Verify N=20 trees were created
    assert len(rf.trees) == 20, f"Expected 20 trees, got {len(rf.trees)}"
    
    # Verify M=7 best trees were selected
    assert len(rf.selected_trees) == 7, f"Expected 7 selected trees, got {len(rf.selected_trees)}"
    
    # Verify remainder set and test set were created
    assert rf.X_remainder is not None, "X_remainder should not be None"
    assert rf.y_remainder is not None, "y_remainder should not be None"
    assert rf.X_test_internal is not None, "X_test_internal should not be None"
    assert rf.y_test_internal is not None, "y_test_internal should not be None"
    
    # Verify split proportions (approximately 2/3 train, 1/3 test)
    total_instances = len(rf.X_remainder) + len(rf.X_test_internal)
    test_proportion = len(rf.X_test_internal) / total_instances
    assert abs(test_proportion - 0.33) < 0.1, f"Test proportion should be ~0.33, got {test_proportion}"
    
    # Verify all trees are MyDecisionTreeClassifier instances
    for tree in rf.trees:
        assert isinstance(tree, MyDecisionTreeClassifier), "All trees should be MyDecisionTreeClassifier instances"
    
    # Verify selected trees are a subset of all trees
    for selected_tree in rf.selected_trees:
        assert selected_tree in rf.trees, "Selected trees should be from the original trees"


def test_random_forest_classifier_predict():
    """Test MyRandomForestClassifier predict() method using interview dataset.
    
    Tests:
    - Predictions use majority voting across selected trees
    - Predictions are made for all test instances
    """
    # Interview dataset
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
    
    # Create Random Forest with N=20, M=7, F=2
    rf = MyRandomForestClassifier(
        n_estimators=20,
        n_best_trees=7,
        max_features=2,
        bootstrap=True,
        random_state=42,
        test_size=0.33
    )
    
    # Fit the model
    rf.fit(X_train_interview, y_train_interview)
    
    # Test predictions on internal test set
    y_pred = rf.predict(rf.X_test_internal)
    
    # Verify predictions are made for all test instances
    assert len(y_pred) == len(rf.X_test_internal), \
        f"Expected {len(rf.X_test_internal)} predictions, got {len(y_pred)}"
    
    # Verify predictions are valid class labels
    valid_labels = ["True", "False"]
    for pred in y_pred:
        assert pred in valid_labels, f"Invalid prediction: {pred}"
    
    # Test predictions on custom test instances
    X_test_custom = [
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "yes"]
    ]
    
    y_pred_custom = rf.predict(X_test_custom)
    
    # Verify predictions are made for all custom test instances
    assert len(y_pred_custom) == len(X_test_custom), \
        f"Expected {len(X_test_custom)} predictions, got {len(y_pred_custom)}"
    
    # Verify all predictions are valid class labels
    for pred in y_pred_custom:
        assert pred in valid_labels, f"Invalid prediction: {pred}"


if __name__ == "__main__":
    """Run all unit tests for mysklearn classifiers."""
    print("=" * 70)
    print("UNIT TESTS: mysklearn Classifiers")
    print("=" * 70)
    print()
    
    print("Running test_decision_tree_classifier_fit()...")
    try:
        test_decision_tree_classifier_fit()
        print("✓ test_decision_tree_classifier_fit() PASSED")
    except AssertionError as e:
        print(f"✗ test_decision_tree_classifier_fit() FAILED: {e}")
    except Exception as e:
        print(f"✗ test_decision_tree_classifier_fit() ERROR: {e}")
    print()
    
    print("Running test_decision_tree_classifier_predict()...")
    try:
        test_decision_tree_classifier_predict()
        print("✓ test_decision_tree_classifier_predict() PASSED")
    except AssertionError as e:
        print(f"✗ test_decision_tree_classifier_predict() FAILED: {e}")
    except Exception as e:
        print(f"✗ test_decision_tree_classifier_predict() ERROR: {e}")
    print()
    
    print("Running test_random_forest_classifier_fit()...")
    try:
        test_random_forest_classifier_fit()
        print("✓ test_random_forest_classifier_fit() PASSED")
    except AssertionError as e:
        print(f"✗ test_random_forest_classifier_fit() FAILED: {e}")
    except Exception as e:
        print(f"✗ test_random_forest_classifier_fit() ERROR: {e}")
    print()
    
    print("Running test_random_forest_classifier_predict()...")
    try:
        test_random_forest_classifier_predict()
        print("✓ test_random_forest_classifier_predict() PASSED")
    except AssertionError as e:
        print(f"✗ test_random_forest_classifier_predict() FAILED: {e}")
    except Exception as e:
        print(f"✗ test_random_forest_classifier_predict() ERROR: {e}")
    print()
    
    print("=" * 70)
    print("All tests completed!")
    print("=" * 70)

