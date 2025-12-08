import numpy as np
from scipy import stats

from mysklearn.myclassifiers import MyDecisionTreeClassifier



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


