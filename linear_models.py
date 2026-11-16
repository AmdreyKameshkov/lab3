import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TREE_MAX_DEPTH = 4
KNN_NEIGHBORS = 51
GERMAN_CATEGORICAL = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
]


def _as_matrix(values) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    return matrix


def _as_vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _standardization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def _to_original_scale(weights: np.ndarray, bias: float, mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, float]:
    original_weights = weights / std
    original_bias = float(bias - np.dot(original_weights, mean))
    return original_weights, original_bias


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _binary_cross_entropy(target: np.ndarray, probabilities: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probabilities, eps, 1 - eps)
    return float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    x_train = _as_matrix(data.train_features)
    y_train = _as_vector(data.train_target)
    x_test = _as_matrix(data.test_features)

    mean, std = _standardization(x_train)
    x = (x_train - mean) / std
    n_samples, n_features = x.shape

    weights = np.zeros(n_features)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        error = x @ weights + bias - y_train
        loss_history.append(float(np.mean(error**2)))
        weights -= data.learning_rate * (2.0 / n_samples) * (x.T @ error)
        bias -= data.learning_rate * 2.0 * float(error.mean())

    original_weights, original_bias = _to_original_scale(weights, bias, mean, std)
    predictions = x_test @ original_weights + original_bias

    return RegressionResult(
        weights=original_weights,
        bias=original_bias,
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    x_train = _as_matrix(data.train_features)
    x_test = _as_matrix(data.test_features)
    raw_target = np.asarray(data.train_target).reshape(-1)

    classes = np.unique(raw_target)
    if len(classes) != 2:
        raise ValueError(f"Нужна бинарная целевая переменная, получено классов: {len(classes)}")
    y_train = (raw_target == classes[1]).astype(float)

    mean, std = _standardization(x_train)
    x = (x_train - mean) / std
    n_samples, n_features = x.shape

    weights = np.zeros(n_features)
    bias = 0.0
    loss_history: list[float] = []

    for _ in range(data.epochs):
        probabilities = _sigmoid(x @ weights + bias)
        loss_history.append(_binary_cross_entropy(y_train, probabilities))
        error = probabilities - y_train
        weights -= data.learning_rate * (x.T @ error) / n_samples
        bias -= data.learning_rate * float(error.mean())

    original_weights, original_bias = _to_original_scale(weights, bias, mean, std)
    test_probabilities = _sigmoid(x_test @ original_weights + original_bias)
    predictions = np.where(test_probabilities >= 0.5, classes[1], classes[0])

    return ClassificationResult(
        weights=original_weights,
        bias=original_bias,
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=predictions,
    )


def classification_metrics(target, probabilities, predictions, positive_label=1) -> ClassificationMetrics:
    target = np.asarray(target).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    positive = (target == positive_label).astype(int)
    predicted_positive = (predictions == positive_label).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(positive, np.asarray(probabilities, dtype=float).reshape(-1))),
        accuracy=float(accuracy_score(positive, predicted_positive)),
        precision=float(precision_score(positive, predicted_positive, zero_division=0)),
        recall=float(recall_score(positive, predicted_positive, zero_division=0)),
        f1=float(f1_score(positive, predicted_positive, zero_division=0)),
    )


def _sklearn_metrics(model, data: ClassificationInput) -> ClassificationMetrics:
    x_train = _as_matrix(data.train_features)
    x_test = _as_matrix(data.test_features)
    y_train = np.asarray(data.train_target).reshape(-1)
    model.fit(x_train, y_train)
    positive_label = model.classes_[1]
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = model.predict(x_test)
    return classification_metrics(data.test_target, probabilities, predictions, positive_label)


def own_logistic_metrics(data: ClassificationInput) -> ClassificationMetrics:
    result = train_logistic_classifier(data)
    positive_label = np.unique(np.asarray(data.train_target).reshape(-1))[1]
    return classification_metrics(data.test_target, result.probabilities, result.predictions, positive_label)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    logistic = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    tree = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))
    return BaselineComparisonResult(
        logistic_regression=_sklearn_metrics(logistic, data),
        decision_tree=_sklearn_metrics(tree, data),
        knn=_sklearn_metrics(knn, data),
    )


def load_student_regression_input(path: str = "Student_Performance.csv", **params) -> RegressionInput:
    df = pd.read_csv(path)
    df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    features = df.drop(columns=["Performance Index"])
    target = df["Performance Index"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return RegressionInput(x_train, y_train, x_test, y_test, **params)


def load_german_classification_input(path: str = "german.csv", one_hot: bool = False, **params) -> ClassificationInput:
    df = pd.read_csv(path, sep=";")
    if one_hot:
        df = pd.get_dummies(df, columns=GERMAN_CATEGORICAL, drop_first=True, dtype=float)
    features = df.drop(columns=["Creditability"])
    target = df["Creditability"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )
    return ClassificationInput(x_train, y_train, x_test, y_test, **params)
