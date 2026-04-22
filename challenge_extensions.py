import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.inspection import permutation_importance

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_support_calls", "senior_citizen",
    "has_partner", "has_dependents", "contract_months"
]

MODEL_REGISTRY = {
    "DummyClassifier": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data(filepath="data/telecom_churn.csv", random_state=42):
    df = pd.read_csv(filepath)
    X = df[NUMERIC_FEATURES].copy()
    y = df["churned"].copy()

    return train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )


def threshold_sweep(model, X_test, y_test, output_path="results/threshold_sweep.png"):
    thresholds = np.arange(0.1, 0.95, 0.05)
    y_proba = model.predict_proba(X_test)[:, 1]

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        alerts_per_1000 = np.mean(y_pred) * 1000

        rows.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "alerts_per_1000": alerts_per_1000,
        })

    df = pd.DataFrame(rows)

    valid = df[df["alerts_per_1000"] <= 15]
    if len(valid) > 0:
        best_row = valid.sort_values("recall", ascending=False).iloc[0]
    else:
        best_row = df.sort_values("alerts_per_1000").iloc[0]

    print("\n=== Threshold Recommendation ===")
    print(best_row.to_string())

    ensure_dir("results")
    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.axvline(best_row["threshold"], linestyle="--", label="Chosen Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return df, best_row


def permutation_analysis(models, X_test, y_test, output_path="results/permutation_importance.png"):
    importances = {}

    for name, model in models.items():
        print(f"Running permutation importance for {name}...")
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        importances[name] = result.importances_mean

    df = pd.DataFrame(importances, index=X_test.columns)
    df["mean"] = df.mean(axis=1)
    top_df = df.sort_values("mean", ascending=False).head(8).drop(columns="mean")

    print("\nTop features:")
    print(top_df)

    ensure_dir("results")
    top_df.plot(kind="bar", figsize=(10, 5))
    plt.title("Permutation Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return top_df


class ModelSelector:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
        self.output_dir = self._make_output_dir()
        self.models = self._build_models()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _make_output_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.config.get("output_base_dir", "challenge_runs")
        experiment_name = self.config.get("experiment_name", "model_selection")
        output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        ensure_dir(output_dir)
        return output_dir

    def _build_models(self):
        models = {}

        for model_name, spec in self.config["models"].items():
            model_type = spec["type"]
            params = spec.get("params", {})
            scale = spec.get("scale", False)

            model_class = MODEL_REGISTRY[model_type]
            scaler = StandardScaler() if scale else "passthrough"

            pipeline = Pipeline([
                ("scaler", scaler),
                ("model", model_class(**params))
            ])

            models[model_name] = pipeline

        return models

    def run_cv_comparison(self, X, y):
        n_splits = self.config.get("cv_folds", 5)
        random_state = self.config.get("random_state", 42)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        rows = []

        for name, pipeline in self.models.items():
            acc_scores = []
            prec_scores = []
            rec_scores = []
            f1_scores = []
            pr_auc_scores = []

            for train_idx, val_idx in cv.split(X, y):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                pipeline.fit(X_train_fold, y_train_fold)

                y_pred = pipeline.predict(X_val_fold)
                y_proba = pipeline.predict_proba(X_val_fold)[:, 1]

                acc_scores.append(accuracy_score(y_val_fold, y_pred))
                prec_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
                rec_scores.append(recall_score(y_val_fold, y_pred, zero_division=0))
                f1_scores.append(f1_score(y_val_fold, y_pred, zero_division=0))
                pr_auc_scores.append(average_precision_score(y_val_fold, y_proba))

            rows.append({
                "model": name,
                "accuracy_mean": np.mean(acc_scores),
                "accuracy_std": np.std(acc_scores),
                "precision_mean": np.mean(prec_scores),
                "precision_std": np.std(prec_scores),
                "recall_mean": np.mean(rec_scores),
                "recall_std": np.std(rec_scores),
                "f1_mean": np.mean(f1_scores),
                "f1_std": np.std(f1_scores),
                "pr_auc_mean": np.mean(pr_auc_scores),
                "pr_auc_std": np.std(pr_auc_scores),
            })

        results_df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).reset_index(drop=True)
        results_path = os.path.join(self.output_dir, "comparison_table.csv")
        results_df.to_csv(results_path, index=False)
        return results_df

    def fit_all(self, X_train, y_train):
        fitted_models = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            fitted_models[name] = model
        return fitted_models

    def save_experiment_log(self, results_df):
        timestamp = datetime.now().isoformat()
        log_df = pd.DataFrame({
            "model_name": results_df["model"],
            "accuracy": results_df["accuracy_mean"],
            "precision": results_df["precision_mean"],
            "recall": results_df["recall_mean"],
            "f1": results_df["f1_mean"],
            "pr_auc": results_df["pr_auc_mean"],
            "timestamp": timestamp
        })
        log_df.to_csv(os.path.join(self.output_dir, "experiment_log.csv"), index=False)

    def run(self):
        X_train, X_test, y_train, y_test = load_data(
            filepath=self.config.get("data_path", "data/telecom_churn.csv"),
            random_state=self.config.get("random_state", 42)
        )

        print(f"Output directory: {self.output_dir}")
        print(f"Models: {list(self.models.keys())}")

        results_df = self.run_cv_comparison(X_train, y_train)
        print("\n=== Comparison Results ===")
        print(results_df.to_string(index=False))

        fitted_models = self.fit_all(X_train, y_train)
        self.save_experiment_log(results_df)

        best_name = results_df.iloc[0]["model"]
        best_model = fitted_models[best_name]

        threshold_sweep(
            best_model,
            X_test,
            y_test,
            output_path=os.path.join(self.output_dir, "threshold_sweep.png")
        )

        top3_names = results_df["model"].head(3).tolist()
        top3_models = {name: fitted_models[name] for name in top3_names}

        permutation_analysis(
            top3_models,
            X_test,
            y_test,
            output_path=os.path.join(self.output_dir, "permutation_importance.png")
        )

        print("\nDone. Files saved in:")
        print(self.output_dir)


def run_simple_demo():
    print("Running simple Tier 1 + Tier 2 demo...\n")

    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LR_default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RF_default": Pipeline([
            ("scaler", "passthrough"),
            ("model", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
        ]),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    scores = {
        name: average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
        for name, model in models.items()
    }

    best_name = max(scores, key=scores.get)
    print("Best model:", best_name)

    threshold_sweep(models[best_name], X_test, y_test)
    permutation_analysis(models, X_test, y_test)

    print("\nDone! Check the results folder.")


if __name__ == "__main__":
    CONFIG_PATH = "challenge_config.json"

    if os.path.exists(CONFIG_PATH):
        selector = ModelSelector(CONFIG_PATH)
        selector.run()
    else:
        run_simple_demo()