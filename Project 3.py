import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
data = pd.read_csv('wdbc.data.mb.csv', header=None)

# Step 2: Preprocess the data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Step 3: Split the dataset into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Implement a probability calculation module
# For simplicity, let's assume Gaussian distribution for continuous attributes
def calculate_probabilities(X_train, y_train):
    n_samples, n_features = X_train.shape
    class_labels = np.unique(y_train)
    probabilities = {}

    for label in class_labels:
        X_class = X_train[y_train == label]
        means = X_class.mean(axis=0)
        stds = X_class.std(axis=0)
        probabilities[label] = [(mean, std) for mean, std in zip(means, stds)]

    return probabilities

# Step 5: Implement a classifying module
def classify_sample(sample, probabilities):
    class_labels = list(probabilities.keys())
    best_label = None
    best_prob = -1

    for label in class_labels:
        class_prob = 1
        for i in range(len(sample)):
            mean, std = probabilities[label][i]
            class_prob *= (1 / (std * np.sqrt(2 * np.pi))) * \
                          np.exp(-((sample[i] - mean) ** 2) / (2 * (std ** 2)))
        if class_prob > best_prob:
            best_prob = class_prob
            best_label = label

    return best_label

# Step 6: Train and test the Naïve Bayesian Classifier
def train_and_test(X_train, y_train, X_test):
    probabilities = calculate_probabilities(X_train, y_train)
    y_pred = []

    for sample in X_test:
        predicted_label = classify_sample(sample, probabilities)
        y_pred.append(predicted_label)

    return y_pred

# Step 7: Calculate accuracy and confusion matrix
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    return accuracy, confusion

# Calculate accuracy and confusion matrix on the test set
y_pred_test = train_and_test(X_train, y_train, X_test)
accuracy_test, confusion_test = evaluate(y_test, y_pred_test)

print("\nTesting Results:")
print(f"Accuracy: {accuracy_test:.2f}")
print("Confusion Matrix:")
print(confusion_test)
print("\n")

# Step 8: Perform k-fold cross-validation (K=5)
def k_fold_cross_validation(X, y, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    confusions = []

    for train_idx, test_idx in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        y_pred = train_and_test(X_train_fold, y_train_fold, X_test_fold)
        accuracy, confusion = evaluate(y_test_fold, y_pred)
        accuracies.append(accuracy)
        confusions.append(confusion)

    return accuracies, confusions

# Step 9: Run the classifier and display results
if __name__ == "__main__":
    accuracies, confusions = k_fold_cross_validation(X, y, k=5)

    for i, (accuracy, confusion) in enumerate(zip(accuracies, confusions)):
        print(f"Fold {i + 1}:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Confusion Matrix:")
        print(confusion)
        print("\n")

    mean_accuracy = np.mean(accuracies)
    print(f"Mean Accuracy: {mean_accuracy:.2f}")
