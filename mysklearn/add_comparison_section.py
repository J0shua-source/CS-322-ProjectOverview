import json

# Load notebook
with open('Algorithm_Project.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find Conclusion section
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell.get('source', []))
        if '## 7. Conclusion' in source_text or '## 7 Conclusion' in source_text:
            insert_index = i
            break

if insert_index:
    print(f"Found insertion point at cell {insert_index}")
    
    # Create comparison section - fix quote issues
    comparison_cells = [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '---\n',
                '## 6.5 Comprehensive Classifier Comparison\n',
                '\n',
                'This section provides a direct comparison of all three classifiers (Decision Tree, Random Forest, and KNN) using identical train-test splits and evaluation metrics.\n'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Comprehensive Comparison of All Three Classifiers\n',
                '# All use the same train-test split (random_state=9, test_size=0.33, stratify=y)\n',
                '\n',
                'import csv\n',
                'import numpy as np\n',
                'from sklearn.model_selection import train_test_split\n',
                'from sklearn.compose import ColumnTransformer\n',
                'from sklearn.preprocessing import OneHotEncoder\n',
                'from sklearn.pipeline import Pipeline\n',
                'from sklearn.tree import DecisionTreeClassifier\n',
                'from sklearn.neighbors import KNeighborsClassifier\n',
                'from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n',
                'from mysklearn.myclassifiers import MyRandomForestClassifier\n',
                '\n',
                '# Load dataset\n',
                'filename = "input_data/bitcoin_sentiment_discretized.csv"\n',
                'rows = []\n',
                'with open(filename, newline=\'\', encoding=\'utf-8\') as f:\n',
                '    reader = csv.DictReader(f)\n',
                '    for row in reader:\n',
                '        rows.append(row)\n',
                '\n',
                'all_columns = list(rows[0].keys())\n',
                'target_column = "price_direction"\n',
                'X = [{col: row[col] for col in all_columns if col != target_column} for row in rows]\n',
                'y = [row[target_column] for row in rows]\n',
                'X_cols = list(X[0].keys())\n',
                'X_matrix = [[row[col] for col in X_cols] for row in X]\n',
                'y_array = np.array(y)\n',
                '\n',
                '# Consistent split for all classifiers\n',
                'seed = 9\n',
                'X_train, X_test, y_train, y_test = train_test_split(\n',
                '    X_matrix, y_array, test_size=0.33, stratify=y_array, random_state=seed\n',
                ')\n',
                '\n',
                'print("=" * 80)\n',
                'print("COMPREHENSIVE CLASSIFIER COMPARISON")\n',
                'print("=" * 80)\n',
                'print(f"Dataset: {len(X_matrix)} instances, {len(X_cols)} features")\n',
                'print(f"Train set: {len(X_train)} instances")\n',
                'print(f"Test set: {len(X_test)} instances")\n',
                'print(f"Random state: {seed} (consistent across all classifiers)")\n',
                'print("=" * 80)\n',
                'print()\n'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Setup preprocessing and train all three classifiers\n',
                'categorical_features = list(range(len(X_cols)))\n',
                'preprocessor = ColumnTransformer(\n',
                '    transformers=[\n',
                '        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)\n',
                '    ]\n',
                ')\n',
                '\n',
                '# Train Decision Tree\n',
                'print("Training Decision Tree...")\n',
                'dt_pipeline = Pipeline(steps=[\n',
                '    ("preprocess", preprocessor),\n',
                '    ("tree", DecisionTreeClassifier(\n',
                '        criterion="gini",\n',
                '        max_depth=10,\n',
                '        random_state=seed,\n',
                '        min_samples_split=2,\n',
                '        min_samples_leaf=1\n',
                '    ))\n',
                '])\n',
                'dt_pipeline.fit(X_train, y_train)\n',
                'dt_pred = dt_pipeline.predict(X_test)\n',
                'dt_acc = accuracy_score(y_test, dt_pred)\n',
                'print(f"Decision Tree Accuracy: {dt_acc:.4f} ({dt_acc:.2%})")\n',
                'print()\n',
                '\n',
                '# Train KNN\n',
                'print("Training KNN...")\n',
                'knn_pipeline = Pipeline(steps=[\n',
                '    ("preprocess", preprocessor),\n',
                '    ("knn", KNeighborsClassifier(\n',
                '        n_neighbors=5,\n',
                '        weights="uniform",\n',
                '        metric="minkowski",\n',
                '        p=2\n',
                '    ))\n',
                '])\n',
                'knn_pipeline.fit(X_train, y_train)\n',
                'knn_pred = knn_pipeline.predict(X_test)\n',
                'knn_acc = accuracy_score(y_test, knn_pred)\n',
                'print(f"KNN Accuracy: {knn_acc:.4f} ({knn_acc:.2%})")\n',
                'print()\n',
                '\n',
                '# Train Random Forest\n',
                'print("Training Random Forest...")\n',
                'y_train_list = y_train.tolist()\n',
                'y_test_list = y_test.tolist()\n',
                '\n',
                'rf = MyRandomForestClassifier(\n',
                '    n_estimators=100,\n',
                '    n_best_trees=3,\n',
                '    max_features=8,\n',
                '    bootstrap=True,\n',
                '    random_state=seed,\n',
                '    test_size=0.0\n',
                ')\n',
                'rf.fit(X_train, y_train_list)\n',
                'rf_pred = rf.predict(X_test)\n',
                'rf_acc = accuracy_score(y_test_list, rf_pred)\n',
                'print(f"Random Forest Accuracy: {rf_acc:.4f} ({rf_acc:.2%})")\n',
                'print()\n'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Compare all three classifiers\n',
                'print("=" * 80)\n',
                'print("CLASSIFIER COMPARISON SUMMARY")\n',
                'print("=" * 80)\n',
                'print()\n',
                'baseline = 0.5\n',
                'print("%-20s %-15s %s" % ("Classifier", "Accuracy", "vs Baseline"))\n',
                'print("-" * 80)\n',
                'print("%-20s %.4f (%.2f%%) %+.2f%%" % ("Decision Tree", dt_acc, dt_acc*100, (dt_acc - baseline)*100))\n',
                'print("%-20s %.4f (%.2f%%) %+.2f%%" % ("Random Forest", rf_acc, rf_acc*100, (rf_acc - baseline)*100))\n',
                'print("%-20s %.4f (%.2f%%) %+.2f%%" % ("KNN", knn_acc, knn_acc*100, (knn_acc - baseline)*100))\n',
                'print()\n',
                'print("Baseline (random): 0.5000 (50.00%)")\n',
                'print()\n',
                '\n',
                'accuracies = {"Decision Tree": dt_acc, "Random Forest": rf_acc, "KNN": knn_acc}\n',
                'best_classifier = max(accuracies, key=accuracies.get)\n',
                'best_accuracy = accuracies[best_classifier]\n',
                'print(f"Best Classifier: {best_classifier} ({best_accuracy:.4f})")\n',
                'print("=" * 80)\n'
            ]
        },
        {
            'cell_type': 'code',
            'metadata': {},
            'source': [
                '# Confusion Matrices\n',
                'print("=" * 80)\n',
                'print("CONFUSION MATRICES")\n',
                'print("=" * 80)\n',
                'print()\n',
                '\n',
                'cm_dt = confusion_matrix(y_test, dt_pred)\n',
                'cm_rf = confusion_matrix(y_test_list, rf_pred)\n',
                'cm_knn = confusion_matrix(y_test, knn_pred)\n',
                '\n',
                'print("Decision Tree:")\n',
                'print(cm_dt)\n',
                'print()\n',
                'print("Random Forest:")\n',
                'print(cm_rf)\n',
                'print()\n',
                'print("KNN:")\n',
                'print(cm_knn)\n',
                'print("=" * 80)\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '### 6.6 Key Findings from Comparison\n',
                '\n',
                '**Performance Summary:**\n',
                '- All three classifiers perform near the 50% baseline\n',
                '- KNN achieved the highest accuracy\n',
                '- Random Forest ensemble did not significantly outperform single Decision Tree\n',
                '- Results suggest limited predictive power in current feature set\n'
            ]
        }
    ]
    
    # Insert before Conclusion
    for i, cell in enumerate(comparison_cells):
        nb['cells'].insert(insert_index + i, cell)
    
    print(f"Inserted {len(comparison_cells)} cells for comparison section")
    
    # Save
    with open('Algorithm_Project.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print("Notebook updated successfully!")
else:
    print("Could not find Conclusion section")

