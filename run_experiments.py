import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import linear_models as lm

FIGURES = Path("figures")


def rounded(metrics) -> dict[str, float]:
    return {name: round(value, 4) for name, value in asdict(metrics).items()}


def regression_experiments() -> dict:
    data = lm.load_student_regression_input()
    result = lm.train_linear_regression(data)
    test_mse = float(np.mean((result.predictions - np.asarray(data.test_target)) ** 2))

    lr_grid = {}
    for learning_rate in (0.001, 0.01, 0.05, 0.1):
        run = lm.train_linear_regression(replace(data, learning_rate=learning_rate))
        lr_grid[learning_rate] = {
            "train_mse_last": round(run.loss_history[-1], 4),
            "test_mse": round(float(np.mean((run.predictions - np.asarray(data.test_target)) ** 2)), 4),
        }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.loss_history)
    ax.set_yscale("log")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE на train (log)")
    ax.set_title("Линейная регрессия: ход MSE")
    fig.tight_layout()
    fig.savefig(FIGURES / "regression_mse.png", dpi=120)
    plt.close(fig)

    history = result.loss_history
    return {
        "n_train": len(data.train_target),
        "n_test": len(data.test_target),
        "test_mse": round(test_mse, 4),
        "test_rmse": round(test_mse**0.5, 4),
        "mse_checkpoints": {epoch: round(history[epoch], 4) for epoch in (0, 10, 50, 100, 200, 400, len(history) - 1)},
        "weights": dict(zip(data.train_features.columns, np.round(result.weights, 4).tolist())),
        "bias": round(result.bias, 4),
        "learning_rate_grid": lr_grid,
    }


def tune_baselines(data) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=lm.RANDOM_STATE)
    depth = {
        str(d): round(
            float(
                cross_val_score(
                    DecisionTreeClassifier(max_depth=d, random_state=lm.RANDOM_STATE),
                    data.train_features,
                    data.train_target,
                    cv=cv,
                    scoring="roc_auc",
                ).mean()
            ),
            4,
        )
        for d in (2, 3, 4, 5, 6, 8, None)
    }
    neighbors = {
        str(k): round(
            float(
                cross_val_score(
                    make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)),
                    data.train_features,
                    data.train_target,
                    cv=cv,
                    scoring="roc_auc",
                ).mean()
            ),
            4,
        )
        for k in (5, 15, 25, 35, 51, 75, 101)
    }
    return {"tree_max_depth_cv_roc_auc": depth, "knn_neighbors_cv_roc_auc": neighbors}


def one_hot_experiment() -> dict[str, float]:
    return rounded(lm.own_logistic_metrics(lm.load_german_classification_input(one_hot=True)))


def classification_experiments() -> tuple[dict, dict]:
    data = lm.load_german_classification_input()
    result = lm.train_logistic_classifier(data)
    own = lm.own_logistic_metrics(data)
    baselines = lm.compare_classical_models(data)

    lr_grid = {}
    for learning_rate in (0.01, 0.1, 0.5):
        run = lm.train_logistic_classifier(replace(data, learning_rate=learning_rate))
        lr_grid[learning_rate] = {
            "train_bce_last": round(run.loss_history[-1], 4),
            "roc_auc": rounded(lm.own_logistic_metrics(replace(data, learning_rate=learning_rate)))["roc_auc"],
        }

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.loss_history)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("BCE на train")
    ax.set_title("Логистическая регрессия: ход кросс-энтропии")
    fig.tight_layout()
    fig.savefig(FIGURES / "logistic_bce.png", dpi=120)
    plt.close(fig)

    y_test = np.asarray(data.test_target)
    curves = {"Собственная логистическая": result.probabilities}
    models = {
        "LogisticRegression (sklearn)": lm.make_pipeline(lm.StandardScaler(), lm.LogisticRegression(max_iter=5000)),
        f"DecisionTree (max_depth={lm.TREE_MAX_DEPTH})": DecisionTreeClassifier(
            max_depth=lm.TREE_MAX_DEPTH, random_state=lm.RANDOM_STATE
        ),
        f"KNN (k={lm.KNN_NEIGHBORS})": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=lm.KNN_NEIGHBORS)),
    }
    for name, model in models.items():
        model.fit(data.train_features, data.train_target)
        curves[name] = model.predict_proba(data.test_features)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, scores in curves.items():
        fpr, tpr, _ = roc_curve(y_test, scores)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые на тестовой выборке")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curves.png", dpi=120)
    plt.close(fig)

    classification = {
        "own_logistic": rounded(own),
        "logistic_regression": rounded(baselines.logistic_regression),
        "decision_tree": rounded(baselines.decision_tree),
        "knn": rounded(baselines.knn),
    }
    details = {
        "n_train": len(data.train_target),
        "n_test": len(data.test_target),
        "test_class_balance": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().items()},
        "bce_first": round(result.loss_history[0], 4),
        "bce_last": round(result.loss_history[-1], 4),
        "learning_rate_grid": lr_grid,
        "one_hot_own_logistic": one_hot_experiment(),
        "baseline_tuning": tune_baselines(data),
    }
    return classification, details


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    regression = regression_experiments()
    classification, details = classification_experiments()

    results = {
        "linear_regression": {"test_mse": regression["test_mse"]},
        "classification": classification,
    }
    Path("experiment_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"results": results, "regression": regression, "classification_details": details}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
