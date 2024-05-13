from collections import Counter

import numpy as np
import pandas as pd

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""

# 1. Load datasets
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    # Load iris dataset
    iris = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        header=None,
    )
    iris.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "class",
    ]

    # Load Boston housing dataset
    boston = pd.read_csv(
        "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    )

    return iris, boston

# 2. Preprocessing functions
def train_test_split(
    df: pd.DataFrame, target: str, test_size: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Shuffle and split dataset into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Split target and features
    X_train = train.drop(target, axis=1).values
    y_train = train[target].values
    X_test = test.drop(target, axis=1).values
    y_test = test[target].values

    return X_train, X_test, y_train, y_test


def normalize(X: np.ndarray) -> np.ndarray:
    # Normalize features to [0, 1]
    # You can try other normalization methods, e.g., z-score, etc.
    # TODO: 1%
    # raise NotImplementedError
    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    range = max_values - min_values
    normalized_X = (X - min_values) / range
    return normalized_X


def encode_labels(y: np.ndarray) -> np.ndarray:
    """
    Encode labels to integers.
    """
    # TODO: 1%
    # raise NotImplementedError
    unique_labels = np.unique(y)
    print(unique_labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_int[label] for label in y])
    print(encoded_labels)
    return encoded_labels

# 3. Models
class LinearModel:
    def __init__(
        self, learning_rate=0.01, iterations=1000, model_type="linear"
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        # You can try different learning rate and iterations
        self.model_type = model_type

        assert model_type in [
            "linear",
            "logistic",
        ], "model_type must be either 'linear' or 'logistic'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Add the intercept term to every sample in X
        X = np.insert(X, 0, 1, axis=1)
        n_classes = len(np.unique(y))
        n_samples, n_features = X.shape
        
        # Initialize weights based on the model type
        if self.model_type == "logistic":
            self.W = np.zeros((n_features, n_classes))
        else:
            self.W = np.zeros(n_features)
        
        # Perform gradient descent
        for _ in range(self.iterations):
            gradients = self._compute_gradients(X, y)
            self.W -= self.learning_rate * gradients
        # TODO: 2%
        # Hint: Initialize the weights based on the model type (logistic or linear).
        # Then, update the weights using gradient descent within a loop for the
        # specified number of iterations.

    def predict(self, X: np.ndarray) -> np.ndarray:
        # insert a column of 1 into the index=0 -> constant intercept
        print(X.shape) # (45, 4)
        X = np.insert(X, 0, 1, axis=1)
        print(X.shape) # (45, 5)

        if self.model_type == "linear":
            logits = X @ self.W
            return logits
        
            # TODO: 2%
            # Hint: Perform a matrix multiplication between the input features (X)
            # and the learned weights.
            # Predict using linear model, @ in np means matrix multiplication. -> got the logits
        else:
            logits = X @ self.W
            return np.argmax(self._softmax(logits), axis=1)
        
            # TODO: 2%
            # Hint: Perform a matrix multiplication between the input features (X)
            # and the learned weights, then apply the softmax function to the result.
            # Predict using logistic model by applying softmax to logits, @ in np means matrix multiplication. -> got the max logit one.

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        # n_samples, n_features = X.shape
        # print(type(X.shape))
        # print(n_samples, n_features)
        n_samples = X.shape[0]
        
        if self.model_type == "linear":
            # Calculate predictions for linear regression
            predictions = X @ self.W

            # Compute error as difference between predictions and true values
            error = predictions - y

            # Calculate gradients as average of products of errors and features
            gradients = X.T @ error / n_samples
            return gradients
        
            # TODO: 3%
            # Hint: Calculate the gradients for linear regression by computing
            # the dot product of X transposed and the difference between the
            # predicted values and the true values, then normalize by the number of samples.
        
        elif self.model_type == "logistic":
            # Compute the softmax probabilities for the current weights
            probabilities = self._softmax(X @ self.W)

            # Create one-hot encoded labels matrix
            y_encoded = np.eye(np.max(y) + 1)[y]

            # Compute error as difference between encoded labels and probabilities
            error = y_encoded - probabilities

            # Calculate gradients as average of products of errors and features, negative since it needs to be minimize the loss
            gradients = -X.T @ error / n_samples
            return gradients
        
            # TODO: 3%
            # Hint: Calculate the gradients for logistic regression by computing
            # the dot product of X transposed and the difference between the one-hot
            # encoded true values and the softmax of the predicted values,
            # then normalize by the number of samples.

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        # Shift the input z by subtracting its max value to improve numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)

        # Compute the exponentials of the shifted inputs
        exp_z = np.exp(z_shifted)
        
        # Calculate the softmax by dividing the exponentials by their sum across classes
        softmax = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return softmax
    
class DecisionTree:
    def __init__(self, max_depth: int = 5, model_type: str = "classifier"):
        self.max_depth = max_depth
        self.model_type = model_type

        assert model_type in [
            "classifier",
            "regressor",
        ], "model_type must be either 'classifier' or 'regressor'"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build_tree(X, y, 0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        if depth >= self.max_depth or self._is_pure(y):
            return self._create_leaf(y)

        feature, threshold = self._find_best_split(X, y)
        # TODO: 4%
        # Hint: Create a mask based on the best feature and threshold that
        # separates the samples into two groups. Then, recursively build
        # the left and right child nodes of the current node.

        # Determine the mask for splitting the data based on the given feature and threshold. mask is element-wise bool array.
        mask = X[:, feature] <= threshold

        # Use the mask to split the data into left and right children directly
        left_X, left_y = X[mask], y[mask]
        right_X, right_y = X[~mask], y[~mask]
            
        # Recursively build the tree for the left and right children
        left_child = self._build_tree(left_X, left_y, depth + 1)
        right_child = self._build_tree(right_X, right_y, depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def _is_pure(self, y: np.ndarray) -> bool:
        return len(set(y)) == 1

    def _create_leaf(self, y: np.ndarray):
        if self.model_type == "classifier":
            # TODO: 1%
            # Hint: For classification, return the most common class in the given samples.
            values, counts = np.unique(y, return_counts=True)
            # print(values, counts, sep='\n')
            # print()
            # print(np.argmax(counts))
            most_common = values[np.argmax(counts)]
            return most_common
        else:
            # TODO: 1%
            # Hint: For regression, return the mean of the given samples.
            return y.mean()

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        best_gini = float("inf")
        best_mse = float("inf")
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            sorted_indices = np.argsort(X[:, feature])
            for i in range(1, len(X)):
                if X[sorted_indices[i - 1], feature] != X[sorted_indices[i], feature]:
                    threshold = (
                        X[sorted_indices[i - 1], feature]
                        + X[sorted_indices[i], feature]
                    ) / 2
                    mask = X[:, feature] <= threshold
                    left_y, right_y = y[mask], y[~mask]

                    if self.model_type == "classifier":
                        gini = self._gini_index(left_y, right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold
                    else:
                        mse = self._mse(left_y, right_y)
                        if mse < best_mse:
                            best_mse = mse
                            best_feature = feature
                            best_threshold = threshold

        return best_feature, best_threshold

    def _gini_index(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Hint: Calculate the Gini index for the left and right samples,
        # then compute the weighted average based on the number of samples in each group.
        def gini_calc(y: np.ndarray) -> float:
            if y.size == 0:
                return 0
            _, counts = np.unique(y, return_counts=True)
            p = counts / y.size
            return 1 - np.sum(p**2)

        total_size = left_y.size + right_y.size
        if total_size == 0:
            return 0
        return (left_y.size * gini_calc(left_y) + right_y.size * gini_calc(right_y)) / total_size
    
    def _mse(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        # TODO: 4%
        # Hint: Calculate the mean squared error for the left and right samples,
        # then compute the weighted average based on the number of samples in each group.
        def mse_calc(y: np.ndarray) -> float:
            if y.size == 0:
                return 0
            return np.square(y - y.mean()).mean()

        total_samples = left_y.size + right_y.size
        if total_samples == 0:
            return 0

        weighted_mse = (left_y.size * mse_calc(left_y) + right_y.size * mse_calc(right_y)) / total_samples
        return weighted_mse
    
    def _traverse_tree(self, x: np.ndarray, node: dict):
        if isinstance(node, dict):
            feature, threshold = node["feature"], node["threshold"]
            if x[feature] <= threshold:
                return self._traverse_tree(x, node["left"])
            else:
                return self._traverse_tree(x, node["right"])
        else:
            return node
        
class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        model_type: str = "classifier",
    ):
        # TODO: 1%
        # Hint: Initialize a list of DecisionTree instances based on the
        # specified number of estimators, max depth, and model type.
        self.model_type = model_type
        self.trees = [DecisionTree(max_depth=max_depth, model_type=model_type) for _ in range(n_estimators)]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples = X.shape[0]
        for tree in self.trees:
            # TODO: 2%
            # Hint: Generate bootstrap indices by random sampling with replacement,
            # then fit each tree with the corresponding samples from X and y.
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Sample the data points and labels according to the bootstrap indices
            sampled_X = X[bootstrap_indices]
            sampled_y = y[bootstrap_indices]

            tree.fit(sampled_X, sampled_y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        # Hint: Predict the output for each tree and combine the predictions
        # based on the model type (majority voting for classification or averaging
        # for regression).
        n_samples = X.shape[0]

        if self.model_type == "classifier":
            predictions = []
            for tree in self.trees:
                predictions.append(tree.predict(X))
            
            matrix = np.array(predictions).T
            res = np.zeros(n_samples)
            for i in range(n_samples):
                values, counts = np.unique(matrix[i], return_counts=True)
                res[i] = values[np.argmax(counts)]
            return res

        else:
            res = np.zeros(n_samples)
            for tree in self.trees:
                res += tree.predict(X)
            res /= len(self.trees)
            return res
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: 2%
        # Hint: Predict the output for each tree and combine the predictions
        # based on the model type (majority voting for classification or averaging
        # for regression).
        # Collect predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])

        if self.model_type == "classifier":
            # Use majority voting for classification
            n_samples = X.shape[0]
            res = np.zeros(n_samples)
            for i in range(n_samples):
                # Count the frequency of each class in the predictions for sample i
                values, counts = np.unique(predictions[:, i], return_counts=True)
                # Find the most frequent class, break ties by choosing the smallest class label
                res[i] = values[np.argmax(counts)]
            return res
        else:
            # Use averaging for regression
            mean_predictions = np.mean(predictions, axis=0)
            return mean_predictions


# 4. Evaluation metrics
def accuracy(y_true, y_pred):
    # TODO: 1%
    # Hint: Calculate the percentage of correct predictions by comparing
    # the true and predicted labels.
    print(y_true)
    print(y_pred)
    return (y_true == y_pred).sum() / y_true.size


def mean_squared_error(y_true, y_pred):
    # TODO: 1%
    # Hint: Calculate the mean squared error between the true and predicted values.
    return np.square(np.subtract(y_true, y_pred)).mean()


# 5. Main function
def main():
    iris, boston = load_data()

    # Iris dataset - Classification
    X_train, X_test, y_train, y_test = train_test_split(iris, "class")
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train, y_test = encode_labels(y_train), encode_labels(y_test)

    # acc = 0.6666
    logistic_regression = LinearModel(model_type="logistic")

    # acc = 0.9777
    # logistic_regression = LinearModel(learning_rate=0.03, iterations=20000, model_type="logistic")
    
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy(y_test, y_pred))

    decision_tree_classifier = DecisionTree(model_type="classifier")
    decision_tree_classifier.fit(X_train, y_train)
    y_pred = decision_tree_classifier.predict(X_test)
    print("Decision Tree Classifier Accuracy:", accuracy(y_test, y_pred))

    random_forest_classifier = RandomForest(model_type="classifier")
    random_forest_classifier.fit(X_train, y_train)
    y_pred = random_forest_classifier.predict(X_test)
    print("Random Forest Classifier Accuracy:", accuracy(y_test, y_pred))

    # Boston dataset - Regression
    X_train, X_test, y_train, y_test = train_test_split(boston, "medv")
    X_train, X_test = normalize(X_train), normalize(X_test)

    linear_regression = LinearModel(model_type="linear")
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))

    decision_tree_regressor = DecisionTree(model_type="regressor")
    decision_tree_regressor.fit(X_train, y_train)
    y_pred = decision_tree_regressor.predict(X_test)
    print("Decision Tree Regressor MSE:", mean_squared_error(y_test, y_pred))

    random_forest_regressor = RandomForest(model_type="regressor")
    random_forest_regressor.fit(X_train, y_train)
    y_pred = random_forest_regressor.predict(X_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred))


if __name__ == "__main__":
    main()
