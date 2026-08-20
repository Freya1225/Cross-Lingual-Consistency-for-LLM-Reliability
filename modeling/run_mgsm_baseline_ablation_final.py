import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None


ANSWER_ONLY_FEATURES = [
    "is_en",
    "is_zh",
    "is_es",
    "self_consistency_number",
    "n_unique_number_answers",
    "number_cross_lingual_agreement_ratio",
    "number_cross_lingual_num_unique_answers",
    "number_en_zh_answer_match",
    "number_en_es_answer_match",
    "number_zh_es_answer_match",
    "current_number_matches_cross_lingual_majority",
    "current_number_matches_en",
    "current_number_matches_zh",
    "current_number_matches_es",
]

SEMANTIC_FEATURES = [
    "semantic_cross_lingual_mean_pairwise",
    "semantic_current_to_others_mean",
    "semantic_sim_en_zh",
    "semantic_sim_en_es",
    "semantic_sim_zh_es",
    "semantic_sim_current_to_en",
    "semantic_sim_current_to_zh",
    "semantic_sim_current_to_es",
]

NLI_FEATURES = [
    "nli_cross_lingual_mean_entailment",
    "nli_cross_lingual_max_contradiction",
    "nli_current_to_others_mean_entailment",
    "nli_current_to_others_max_contradiction",
    "nli_en_zh_bidirectional_entailment",
    "nli_en_zh_max_contradiction",
    "nli_en_es_bidirectional_entailment",
    "nli_en_es_max_contradiction",
    "nli_zh_es_bidirectional_entailment",
    "nli_zh_es_max_contradiction",
    "nli_current_to_en_bidirectional_entailment",
    "nli_current_to_en_max_contradiction",
    "nli_current_to_zh_bidirectional_entailment",
    "nli_current_to_zh_max_contradiction",
    "nli_current_to_es_bidirectional_entailment",
    "nli_current_to_es_max_contradiction",
]


FEATURE_SETS: Dict[str, List[str]] = {
    "language_only_control": [
        "is_en",
        "is_zh",
        "is_es",
    ],
    "monolingual_self_consistency": [
        "self_consistency_number",
        "n_unique_number_answers",
    ],
    "cross_lingual_answer_agreement": [
        "number_cross_lingual_agreement_ratio",
        "number_cross_lingual_num_unique_answers",
        "current_number_matches_cross_lingual_majority",
    ],
    "pairwise_language_agreement": [
        "number_en_zh_answer_match",
        "number_en_es_answer_match",
        "number_zh_es_answer_match",
        "current_number_matches_en",
        "current_number_matches_zh",
        "current_number_matches_es",
    ],
    "self_plus_cross_lingual": [
        "self_consistency_number",
        "n_unique_number_answers",
        "number_cross_lingual_agreement_ratio",
        "number_cross_lingual_num_unique_answers",
        "current_number_matches_cross_lingual_majority",
    ],
    "all_answer_level_features": [
        *ANSWER_ONLY_FEATURES,
    ],
    "increment_answer_only": [
        *ANSWER_ONLY_FEATURES,
    ],
    "increment_answer_plus_semantic": [
        *ANSWER_ONLY_FEATURES,
        *SEMANTIC_FEATURES,
    ],
    "increment_answer_plus_semantic_plus_nli": [
        *ANSWER_ONLY_FEATURES,
        *SEMANTIC_FEATURES,
        *NLI_FEATURES,
    ],
}


RULES = {
    "self_consistency_is_1": lambda df: df["self_consistency_number"] >= 1.0,
    "cross_lingual_agreement_is_1": lambda df: df["number_cross_lingual_agreement_ratio"] >= 1.0,
    "matches_cross_lingual_majority": lambda df: df[
        "current_number_matches_cross_lingual_majority"
    ]
    >= 1.0,
    "self_1_and_cross_lingual_1": lambda df: (df["self_consistency_number"] >= 1.0)
    & (df["number_cross_lingual_agreement_ratio"] >= 1.0),
}


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
    )


def load_task(path: Path, task: str, label_column: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["type"] == task].copy()
    if df.empty:
        raise ValueError(f"No {task} rows found in {path}")

    if label_column is None:
      if "final_label" in df.columns:
        label_column = "final_label"
      elif "evaluation_label_any_correct" in df.columns:
        label_column = "evaluation_label_any_correct"
      else:
        label_column = "provided_is_correct"
#修改为language_final数据
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    df["label_correct"] = parse_bool_series(df[label_column])
    df = df[df["label_correct"].notna()].copy()
    df["label_correct"] = df["label_correct"].astype(int)
    if df.empty:
        raise ValueError(f"No scorable {task} rows found in {path} using {label_column}")

    for lang in ("en", "zh", "es"):
        df[f"is_{lang}"] = (df["lang"] == lang).astype(float)

    available_feature_sets = {
        name: features
        for name, features in FEATURE_SETS.items()
        if all(col in df.columns or col.startswith("is_") for col in features)
    }
    skipped = sorted(set(FEATURE_SETS) - set(available_feature_sets))
    if skipped:
        print(f"Skipping feature sets with missing columns: {', '.join(skipped)}")

    required_cols = set().union(*available_feature_sets.values())
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        if col.startswith("is_"):
            continue
        if df[col].dtype == object:
            parsed_bool = parse_bool_series(df[col])
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.fillna(parsed_bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.attrs["available_feature_sets"] = available_feature_sets
    return df


def get_cv(y: pd.Series, groups: pd.Series, n_splits: int, seed: int):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(y)), y, groups
        )
    return GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)


def score_predictions(y_true, y_score, y_pred) -> Dict[str, float]:
    out = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return out


def summarize_fold_scores(name: str, fold_scores: List[Dict[str, float]]) -> Dict[str, float]:
    row = {"model": name}
    metrics = fold_scores[0].keys()
    for metric in metrics:
        values = np.array([score[metric] for score in fold_scores], dtype=float)
        row[f"{metric}_mean"] = values.mean()
        row[f"{metric}_std"] = values.std(ddof=0)
    return row


def evaluate_feature_set(
    df: pd.DataFrame,
    name: str,
    features: List[str],
    n_splits: int,
    seed: int,
) -> Dict[str, float]:
    X = df[features]
    y = df["label_correct"]
    groups = df["id"]

    fold_scores = []
    for train_idx, test_idx in get_cv(y, groups, n_splits=n_splits, seed=seed):
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_score = model.predict_proba(X.iloc[test_idx])[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        fold_scores.append(score_predictions(y.iloc[test_idx], y_score, y_pred))

    row = summarize_fold_scores(name, fold_scores)
    row["features"] = ", ".join(features)
    return row


def evaluate_dummy(df: pd.DataFrame, n_splits: int, seed: int) -> Dict[str, float]:
    X = np.zeros((len(df), 1))
    y = df["label_correct"]
    groups = df["id"]
    fold_scores = []
    for train_idx, test_idx in get_cv(y, groups, n_splits=n_splits, seed=seed):
        model = DummyClassifier(strategy="prior")
        model.fit(X[train_idx], y.iloc[train_idx])
        y_score = model.predict_proba(X[test_idx])[:, 1]
        y_pred = model.predict(X[test_idx])
        fold_scores.append(score_predictions(y.iloc[test_idx], y_score, y_pred))
    row = summarize_fold_scores("majority_prior_control", fold_scores)
    row["features"] = "none"
    return row


def evaluate_rules(df: pd.DataFrame) -> pd.DataFrame:
    y = df["label_correct"].to_numpy()
    rows = []
    for name, rule_fn in RULES.items():
        y_pred = rule_fn(df).fillna(False).astype(int).to_numpy()
        y_score = y_pred.astype(float)
        row = {"rule": name}
        row.update(score_predictions(y, y_score, y_pred))
        row["coverage_predicted_reliable"] = y_pred.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run answer-level baseline and ablation experiments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_outputs/language_level_feature_table.csv"),
    )
    parser.add_argument("--task", type=str, default="MGSM", choices=["MGSM", "XQuAD", "MKQA"])
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Defaults to evaluation_label_any_correct when available, otherwise provided_is_correct.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_ablation_outputs"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_task(args.input, args.task, args.label_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [evaluate_dummy(df, args.n_splits, args.seed)]
    available_feature_sets = df.attrs.get("available_feature_sets", FEATURE_SETS)
    for name, features in available_feature_sets.items():
        rows.append(evaluate_feature_set(df, name, features, args.n_splits, args.seed))

    ablation = pd.DataFrame(rows)
    metric_cols = [c for c in ablation.columns if c.endswith("_mean") or c.endswith("_std")]
    ablation[metric_cols] = ablation[metric_cols].round(4)
    ablation = ablation.sort_values("roc_auc_mean", ascending=False)

    rules = evaluate_rules(df).round(4)

    task_slug = args.task.lower()
    ablation_path = args.output_dir / f"{task_slug}_baseline_ablation.csv"
    rules_path = args.output_dir / f"{task_slug}_threshold_rules.csv"
    ablation.to_csv(ablation_path, index=False)
    rules.to_csv(rules_path, index=False)

    print(f"Loaded {len(df)} {args.task} language-level rows from {df['id'].nunique()} questions.")
    print(f"Positive label rate: {df['label_correct'].mean():.3f}")
    print("\nAblation results:")
    print(
        ablation[
            [
                "model",
                "roc_auc_mean",
                "average_precision_mean",
                "accuracy_mean",
                "f1_mean",
            ]
        ].to_string(index=False)
    )
    print("\nThreshold-rule results:")
    print(rules.to_string(index=False))
    print(f"\nWrote {ablation_path}")
    print(f"Wrote {rules_path}")


if __name__ == "__main__":
    main()
