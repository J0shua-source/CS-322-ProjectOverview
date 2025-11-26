from mysklearn import myutils
import math
from collections import Counter
import copy
import random


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Create list of available attributes (indices)
        available_attributes = list(range(len(X_train[0])))
        
        # Build the tree using TDIDT
        self.tree = self._tdidt(X_train, y_train, available_attributes)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        predictions = []
        for instance in X_test:
            prediction = self._predict_instance(self.tree, instance)
            predictions.append(prediction)
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            # Generate default attribute names
            num_attributes = len(self.X_train[0]) if self.X_train else 0
            attribute_names = [f"att{i}" for i in range(num_attributes)]
        
        rules = []
        self._extract_rules(self.tree, [], rules, attribute_names, class_name)
        
        for rule in rules:
            print(rule)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        if attribute_names is None:
            # Generate default attribute names
            num_attributes = len(self.X_train[0]) if self.X_train else 0
            attribute_names = [f"att{i}" for i in range(num_attributes)]
        
        # Create DOT file
        with open(dot_fname, 'w') as f:
            f.write('digraph Tree {\n')
            f.write('    node [shape=box];\n')
            self.node_counter = 0
            self._write_tree_dot(self.tree, f, attribute_names, None)
            f.write('}\n')
        
        # Generate PDF using graphviz
        import subprocess
        try:
            subprocess.run(['dot', '-Tpdf', dot_fname, '-o', pdf_fname], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error generating PDF: {e}")
            print("Make sure graphviz is installed: apt-get install graphviz")
    
    def _tdidt(self, X, y, available_attributes):
        """Recursively builds a decision tree using the TDIDT algorithm.
        
        Args:
            X: Current partition of instances
            y: Current partition of labels
            available_attributes: List of attribute indices that can still be used for splitting
            
        Returns:
            A subtree (nested list representation)
        """
        # Base case 1: All labels are the same (pure partition)
        unique_labels = list(set(y))
        if len(unique_labels) == 1:
            return ["Leaf", unique_labels[0], len(y), len(y)]
        
        # Base case 2: No more attributes to partition on (clash)
        if len(available_attributes) == 0:
            # Use majority vote, with alphabetical tie-breaking
            label_counts = Counter(y)
            majority_label = sorted(label_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]
            return ["Leaf", majority_label, label_counts[majority_label], len(y)]
        
        # Base case 3: All instances have same attribute values (clash)
        all_same = True
        if len(X) > 0:
            first = X[0]
            for instance in X[1:]:
                if instance != first:
                    all_same = False
                    break
        
        if all_same:
            label_counts = Counter(y)
            majority_label = sorted(label_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]
            return ["Leaf", majority_label, label_counts[majority_label], len(y)]
        
        # Select attribute with highest information gain
        best_attribute = self._select_attribute(X, y, available_attributes)
        
        # Create attribute node
        tree = ["Attribute", f"att{best_attribute}"]
        
        # Partition data by attribute values
        partitions = {}
        for i, instance in enumerate(X):
            value = instance[best_attribute]
            if value not in partitions:
                partitions[value] = {'X': [], 'y': []}
            partitions[value]['X'].append(instance)
            partitions[value]['y'].append(y[i])
        
        # Sort partition keys for consistent ordering
        sorted_values = sorted(partitions.keys(), key=lambda x: str(x))
        
        # Recursively build subtrees
        remaining_attributes = [att for att in available_attributes if att != best_attribute]
        
        for value in sorted_values:
            partition_X = partitions[value]['X']
            partition_y = partitions[value]['y']
            
            subtree = self._tdidt(partition_X, partition_y, remaining_attributes)
            tree.append(["Value", value, subtree])
        
        return tree
    
    def _select_attribute(self, X, y, available_attributes):
        """Selects the attribute with the highest information gain.
        
        Args:
            X: Current partition of instances
            y: Current partition of labels
            available_attributes: List of attribute indices to consider
            
        Returns:
            Index of the best attribute
        """
        best_attribute = None
        best_info_gain = -1
        
        # Calculate entropy of current partition
        current_entropy = self._calculate_entropy(y)
        
        for attribute in available_attributes:
            # Calculate weighted entropy after split
            partitions = {}
            for i, instance in enumerate(X):
                value = instance[attribute]
                if value not in partitions:
                    partitions[value] = []
                partitions[value].append(y[i])
            
            weighted_entropy = 0
            for value, labels in partitions.items():
                weight = len(labels) / len(y)
                weighted_entropy += weight * self._calculate_entropy(labels)
            
            # Calculate information gain
            info_gain = current_entropy - weighted_entropy
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_attribute = attribute
        
        return best_attribute
    
    def _calculate_entropy(self, labels):
        """Calculates the entropy of a list of labels.
        
        Args:
            labels: List of class labels
            
        Returns:
            Entropy value
        """
        if len(labels) == 0:
            return 0
        
        label_counts = Counter(labels)
        entropy = 0
        
        for count in label_counts.values():
            probability = count / len(labels)
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _predict_instance(self, tree, instance):
        """Recursively traverses the tree to make a prediction for a single instance.
        
        Args:
            tree: Current node in the tree
            instance: The instance to classify
            
        Returns:
            Predicted class label
        """
        if tree[0] == "Leaf":
            return tree[1]
        
        # Get the attribute to split on
        attribute_name = tree[1]
        attribute_index = int(attribute_name[3:])  # Extract index from "attX"
        instance_value = instance[attribute_index]
        
        # Find the matching branch
        for i in range(2, len(tree)):
            value_branch = tree[i]
            if value_branch[1] == instance_value:
                return self._predict_instance(value_branch[2], instance)
        
        # If no matching branch found (unseen value), use majority vote from available branches
        # Collect all leaf labels from this subtree
        leaf_labels = self._collect_leaf_labels(tree)
        if leaf_labels:
            # Use majority vote with alphabetical tie-breaking
            label_counts = Counter(leaf_labels)
            majority_label = sorted(label_counts.items(), key=lambda x: (-x[1], str(x[0])))[0][0]
            return majority_label
        
        # Fallback (should rarely happen)
        return None
    
    def _collect_leaf_labels(self, tree):
        """Collect all leaf labels from a tree/subtree.
        
        Args:
            tree: A tree node
            
        Returns:
            List of all leaf labels in the subtree
        """
        if tree[0] == "Leaf":
            # Return the label repeated by its count
            return [tree[1]] * tree[2]
        
        labels = []
        # Collect from all branches
        for i in range(2, len(tree)):
            value_branch = tree[i]
            labels.extend(self._collect_leaf_labels(value_branch[2]))
        
        return labels
    
    def _extract_rules(self, tree, conditions, rules, attribute_names, class_name):
        """Recursively extracts decision rules from the tree.
        
        Args:
            tree: Current node in the tree
            conditions: List of conditions accumulated so far
            rules: List to store complete rules
            attribute_names: List of attribute names
            class_name: Name for the class variable
        """
        if tree[0] == "Leaf":
            # Build rule string
            if len(conditions) > 0:
                rule = "IF " + " AND ".join(conditions) + f" THEN {class_name} = {tree[1]}"
            else:
                rule = f"IF True THEN {class_name} = {tree[1]}"
            rules.append(rule)
            return
        
        # Get attribute name
        attribute_name = tree[1]
        attribute_index = int(attribute_name[3:])
        actual_name = attribute_names[attribute_index]
        
        # Recurse on each branch
        for i in range(2, len(tree)):
            value_branch = tree[i]
            value = value_branch[1]
            new_conditions = conditions + [f"{actual_name} == {value}"]
            self._extract_rules(value_branch[2], new_conditions, rules, attribute_names, class_name)
    
    def _write_tree_dot(self, tree, f, attribute_names, parent_id):
        """Recursively writes tree structure to DOT file.
        
        Args:
            tree: Current node in the tree
            f: File object to write to
            attribute_names: List of attribute names
            parent_id: ID of parent node (for edge creation)
        """
        current_id = self.node_counter
        self.node_counter += 1
        
        if tree[0] == "Leaf":
            # Leaf node
            label = f"{tree[1]}\\n({tree[2]}/{tree[3]})"
            f.write(f'    node{current_id} [label="{label}", shape=ellipse];\n')
        else:
            # Attribute node
            attribute_name = tree[1]
            attribute_index = int(attribute_name[3:])
            actual_name = attribute_names[attribute_index]
            f.write(f'    node{current_id} [label="{actual_name}"];\n')
            
            # Process branches
            for i in range(2, len(tree)):
                value_branch = tree[i]
                value = value_branch[1]
                child_id = self.node_counter
                
                # Recursively write child
                self._write_tree_dot(value_branch[2], f, attribute_names, current_id)
                
                # Create edge
                f.write(f'    node{current_id} -> node{child_id} [label="{value}"];\n')
        
        # Create edge from parent if exists
        if parent_id is not None and current_id != 0:
            # Edge already created in parent's recursive call
            pass


class MyRandomForestClassifier:
    """Represents a random forest classifier using bootstrap aggregating (bagging)
    and random attribute subset selection.

    Attributes:
        n_estimators(int): The number of decision trees in the forest.
        max_features(int): The maximum number of features to consider when looking 
            for the best split. If None, uses sqrt(n_features).
        bootstrap(bool): Whether bootstrap samples are used when building trees.
        random_state(int): Seed for random number generator for reproducibility.
        trees(list of MyDecisionTreeClassifier): The collection of decision trees.
        feature_subsets(list of list of int): The feature subsets used for each tree.

    Notes:
        Loosely based on sklearn's RandomForestClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        Implements ensemble learning using bagging and random feature selection.
    """
    
    def __init__(self, n_estimators=10, max_features=None, bootstrap=True, random_state=None):
        """Initializer for MyRandomForestClassifier.
        
        Args:
            n_estimators(int): The number of trees in the forest (default=10).
            max_features(int): The number of features to consider when looking for the best split.
                If None, uses sqrt(n_features) (default=None).
            bootstrap(bool): Whether to use bootstrap samples (default=True).
            random_state(int): Random seed for reproducibility (default=None).
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_subsets = []
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a random forest classifier to X_train and y_train using bagging
        and random feature selection.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Creates n_estimators decision trees, each trained on:
            - A bootstrap sample of the training data (if bootstrap=True)
            - A random subset of features (size determined by max_features)
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Set random seed if provided
        if self.random_state is not None:
            random.seed(self.random_state)
        
        n_samples = len(X_train)
        n_features = len(X_train[0]) if n_samples > 0 else 0
        
        # Determine max_features if not set
        if self.max_features is None:
            self.max_features = int(math.sqrt(n_features))
        
        # Ensure max_features doesn't exceed total features
        self.max_features = min(self.max_features, n_features)
        
        # Build each tree in the forest
        self.trees = []
        self.feature_subsets = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample or use full dataset
            if self.bootstrap:
                X_sample, y_sample = self._create_bootstrap_sample(X_train, y_train)
            else:
                X_sample, y_sample = X_train, y_train
            
            # Select random feature subset
            feature_subset = self._select_random_features(n_features)
            self.feature_subsets.append(feature_subset)
            
            # Create feature mapping for this subset
            X_sample_subset = self._apply_feature_subset(X_sample, feature_subset)
            
            # Train a decision tree on the subset
            tree = MyDecisionTreeClassifier()
            tree.fit(X_sample_subset, y_sample)
            self.trees.append(tree)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test using majority voting
        across all trees in the forest.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # Collect predictions from all trees
        all_predictions = []
        
        for i, tree in enumerate(self.trees):
            # Apply feature subset for this tree
            X_test_subset = self._apply_feature_subset(X_test, self.feature_subsets[i])
            predictions = tree.predict(X_test_subset)
            all_predictions.append(predictions)
        
        # Majority vote for each instance
        y_predicted = []
        n_test_samples = len(X_test)
        
        for sample_idx in range(n_test_samples):
            # Collect votes from all trees for this sample
            votes = [all_predictions[tree_idx][sample_idx] 
                    for tree_idx in range(self.n_estimators)]
            
            # Majority vote with tie-breaking
            vote_counts = Counter(votes)
            majority_vote = sorted(vote_counts.items(), 
                                 key=lambda x: (-x[1], str(x[0])))[0][0]
            y_predicted.append(majority_vote)
        
        return y_predicted

    def _create_bootstrap_sample(self, X, y):
        """Creates a bootstrap sample from X and y.
        
        Args:
            X(list of list of obj): The list of samples
            y(list of obj): The target y values (parallel to X)
            
        Returns:
            X_sample(list of list of obj): Bootstrap sample of X
            y_sample(list of obj): Bootstrap sample of y
        """
        n_samples = len(X)
        indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
        
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        
        return X_sample, y_sample

    def _select_random_features(self, n_features):
        """Selects a random subset of features.
        
        Args:
            n_features(int): Total number of features available
            
        Returns:
            feature_subset(list of int): Randomly selected feature indices
        """
        all_features = list(range(n_features))
        feature_subset = random.sample(all_features, self.max_features)
        return sorted(feature_subset)

    def _apply_feature_subset(self, X, feature_subset):
        """Applies a feature subset to the data.
        
        Args:
            X(list of list of obj): The list of samples
            feature_subset(list of int): The feature indices to keep
            
        Returns:
            X_subset(list of list of obj): X with only the selected features
        """
        X_subset = []
        for instance in X:
            subset_instance = [instance[i] for i in feature_subset]
            X_subset.append(subset_instance)
        return X_subset

    def get_feature_importances(self):
        """Calculates feature importances based on how often each feature
        is used across all trees in the forest.
        
        Returns:
            importances(list of float): Feature importance scores (sum to 1.0)
        """
        if not self.trees or not self.X_train:
            return []
        
        n_features = len(self.X_train[0])
        feature_counts = [0] * n_features
        
        # Count how many times each feature appears in feature subsets
        for feature_subset in self.feature_subsets:
            for feature_idx in feature_subset:
                feature_counts[feature_idx] += 1
        
        # Normalize to get importances
        total_count = sum(feature_counts)
        if total_count == 0:
            return [0.0] * n_features
        
        importances = [count / total_count for count in feature_counts]
        return importances

    def get_oob_score(self):
        """Calculates the out-of-bag (OOB) score if bootstrap=True.
        
        The OOB score is the accuracy of predictions on instances that were
        not included in the bootstrap sample for each tree.
        
        Returns:
            oob_score(float): The OOB accuracy score, or None if bootstrap=False
        """
        if not self.bootstrap or not self.trees:
            return None
        
        n_samples = len(self.X_train)
        oob_predictions = [[] for _ in range(n_samples)]
        
        # For each tree, track which samples were OOB
        for tree_idx in range(self.n_estimators):
            # Re-create the bootstrap sample to identify OOB samples
            if self.random_state is not None:
                random.seed(self.random_state + tree_idx)
            
            bootstrap_indices = set([random.randint(0, n_samples - 1) 
                                    for _ in range(n_samples)])
            oob_indices = [i for i in range(n_samples) if i not in bootstrap_indices]
            
            # Make predictions for OOB samples
            if oob_indices:
                X_oob = [self.X_train[i] for i in oob_indices]
                X_oob_subset = self._apply_feature_subset(X_oob, 
                                                         self.feature_subsets[tree_idx])
                predictions = self.trees[tree_idx].predict(X_oob_subset)
                
                for i, idx in enumerate(oob_indices):
                    oob_predictions[idx].append(predictions[i])
        
        # Calculate accuracy on OOB predictions
        correct = 0
        total = 0
        
        for i in range(n_samples):
            if len(oob_predictions[i]) > 0:
                # Majority vote
                vote_counts = Counter(oob_predictions[i])
                majority_vote = sorted(vote_counts.items(), 
                                     key=lambda x: (-x[1], str(x[0])))[0][0]
                
                if majority_vote == self.y_train[i]:
                    correct += 1
                total += 1
        
        if total == 0:
            return None
        
        return correct / total

    def print_forest_info(self):
        """Prints information about the random forest including number of trees,
        feature subsets used, and feature importances.
        """
        print(f"Random Forest with {self.n_estimators} trees")
        print(f"Max features per tree: {self.max_features}")
        print(f"Bootstrap: {self.bootstrap}")
        
        if self.trees:
            importances = self.get_feature_importances()
            print("\nFeature Importances:")
            for i, importance in enumerate(importances):
                print(f"  Feature att{i}: {importance:.4f}")
            
            if self.bootstrap:
                oob_score = self.get_oob_score()
                if oob_score is not None:
                    print(f"\nOut-of-Bag Score: {oob_score:.4f}")

