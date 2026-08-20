import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LANGS = ("en", "zh", "es")
PAIR_COLUMNS = {
    ("en", "zh"): "en_zh",
    ("en", "es"): "en_es",
    ("zh", "es"): "zh_es",
}


def clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text if text else "[EMPTY]"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


def mean_or_nan(values: Iterable[float]) -> float:
    clean = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(clean)) if clean else np.nan


def load_sentence_transformer(model_name: str, device: Optional[str]):
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def compute_embeddings(texts: List[str], model_name: str, batch_size: int, device: Optional[str]):
    model = load_sentence_transformer(model_name, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings


def add_semantic_features(df: pd.DataFrame, model_name: str, batch_size: int, device: Optional[str]):
    texts = [clean_text(text) for text in df["majority_short_answer"].tolist()]
    embeddings = compute_embeddings(texts, model_name, batch_size, device)
    row_embedding = {idx: embeddings[pos] for pos, idx in enumerate(df.index)}

    feature_rows = []
    for qid, group in df.groupby("id", sort=False):
        by_lang = {row["lang"]: idx for idx, row in group.iterrows()}
        pair_sims: Dict[str, float] = {}
        for (left, right), suffix in PAIR_COLUMNS.items():
            if left in by_lang and right in by_lang:
                pair_sims[f"semantic_sim_{suffix}"] = cosine(
                    row_embedding[by_lang[left]], row_embedding[by_lang[right]]
                )
            else:
                pair_sims[f"semantic_sim_{suffix}"] = np.nan

        all_pair_mean = mean_or_nan(pair_sims.values())
        for idx, row in group.iterrows():
            current_lang = row["lang"]
            current_embedding = row_embedding[idx]
            other_sims = []
            sims_to_langs = {}
            for other_lang, other_idx in by_lang.items():
                if other_lang == current_lang:
                    continue
                sim = cosine(current_embedding, row_embedding[other_idx])
                other_sims.append(sim)
                sims_to_langs[f"semantic_sim_current_to_{other_lang}"] = sim

            feature_rows.append(
                {
                    "qkey": row["qkey"],
                    "semantic_model": model_name,
                    "semantic_cross_lingual_mean_pairwise": all_pair_mean,
                    "semantic_current_to_others_mean": mean_or_nan(other_sims),
                    **pair_sims,
                    **sims_to_langs,
                }
            )

    features = pd.DataFrame(feature_rows)
    return df.merge(features, on="qkey", how="left")


def load_nli_model(model_name: str, device_name: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def normalize_nli_scores(scores) -> Dict[str, float]:
    normalized = {"entailment": np.nan, "neutral": np.nan, "contradiction": np.nan}
    for item in scores:
        label = item["label"].lower()
        score = float(item["score"])
        if "entail" in label:
            normalized["entailment"] = score
        elif "contrad" in label:
            normalized["contradiction"] = score
        elif "neutral" in label:
            normalized["neutral"] = score
    return normalized


def score_nli_pairs(
    pairs: List[Tuple[str, str]],
    tokenizer,
    model,
    device,
    batch_size: int,
) -> List[Dict[str, float]]:
    import torch

    id2label = model.config.id2label
    outputs: List[Dict[str, float]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        premises = [premise for premise, _ in batch]
        hypotheses = [hypothesis for _, hypothesis in batch]
        encoded = tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        for row in probs:
            scores = [
                {"label": id2label[i], "score": float(score)}
                for i, score in enumerate(row)
            ]
            outputs.append(normalize_nli_scores(scores))
    return outputs


def add_nli_features(
    df: pd.DataFrame,
    model_name: str,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    tokenizer, model, device = load_nli_model(model_name, device_name)

    pair_requests = []
    for qid, group in df.groupby("id", sort=False):
        by_lang = {row["lang"]: row for _, row in group.iterrows()}
        for left, right in itertools.permutations(by_lang.keys(), 2):
            pair_requests.append(
                (
                    qid,
                    left,
                    right,
                    clean_text(by_lang[left]["majority_short_answer"]),
                    clean_text(by_lang[right]["majority_short_answer"]),
                )
            )

    print(f"Scoring {len(pair_requests)} directional NLI pairs")
    scored_pairs = score_nli_pairs(
        [(premise, hypothesis) for _, _, _, premise, hypothesis in pair_requests],
        tokenizer,
        model,
        device,
        batch_size,
    )
    directional_scores = {
        (qid, left, right): scores
        for (qid, left, right, _, _), scores in zip(pair_requests, scored_pairs)
    }

    feature_rows = []
    for qid, group in df.groupby("id", sort=False):
        by_lang = {row["lang"]: row for _, row in group.iterrows()}
        pair_features: Dict[str, float] = {}

        for (left, right), suffix in PAIR_COLUMNS.items():
            if left not in by_lang or right not in by_lang:
                pair_features[f"nli_{suffix}_bidirectional_entailment"] = np.nan
                pair_features[f"nli_{suffix}_max_contradiction"] = np.nan
                continue

            left_to_right = directional_scores[(qid, left, right)]
            right_to_left = directional_scores[(qid, right, left)]
            pair_features[f"nli_{suffix}_bidirectional_entailment"] = mean_or_nan(
                [left_to_right["entailment"], right_to_left["entailment"]]
            )
            pair_features[f"nli_{suffix}_max_contradiction"] = max(
                left_to_right["contradiction"], right_to_left["contradiction"]
            )

        entail_cols = [
            value for key, value in pair_features.items() if key.endswith("bidirectional_entailment")
        ]
        contra_cols = [value for key, value in pair_features.items() if key.endswith("max_contradiction")]
        question_entailment_mean = mean_or_nan(entail_cols)
        question_contradiction_max = max([v for v in contra_cols if not pd.isna(v)], default=np.nan)

        for _, row in group.iterrows():
            current_lang = row["lang"]
            current_entailments = []
            current_contradictions = []
            current_to_langs = {}

            for other_lang, other_row in by_lang.items():
                if other_lang == current_lang:
                    continue
                forward = directional_scores[(qid, current_lang, other_lang)]
                backward = directional_scores[(qid, other_lang, current_lang)]
                entailment = mean_or_nan([forward["entailment"], backward["entailment"]])
                contradiction = max(forward["contradiction"], backward["contradiction"])
                current_entailments.append(entailment)
                current_contradictions.append(contradiction)
                current_to_langs[f"nli_current_to_{other_lang}_bidirectional_entailment"] = entailment
                current_to_langs[f"nli_current_to_{other_lang}_max_contradiction"] = contradiction

            feature_rows.append(
                {
                    "qkey": row["qkey"],
                    "nli_model": model_name,
                    "nli_cross_lingual_mean_entailment": question_entailment_mean,
                    "nli_cross_lingual_max_contradiction": question_contradiction_max,
                    "nli_current_to_others_mean_entailment": mean_or_nan(current_entailments),
                    "nli_current_to_others_max_contradiction": max(
                        [v for v in current_contradictions if not pd.isna(v)], default=np.nan
                    ),
                    **pair_features,
                    **current_to_langs,
                }
            )

    features = pd.DataFrame(feature_rows)
    return df.merge(features, on="qkey", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add semantic embedding similarity and multilingual NLI features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("scored_outputs/language_level_feature_table_scored.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scored_outputs/language_level_feature_table_semantic_nli.csv"),
    )
    parser.add_argument("--semantic-model", default="sentence-transformers/LaBSE")
    parser.add_argument("--nli-model", default="MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cpu/mps/cuda.")
    parser.add_argument(
        "--nli-device",
        default="cpu",
        help="Torch device for NLI, e.g. cpu/mps/cuda.",
    )
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-nli", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "majority_short_answer" not in df.columns:
        raise ValueError("Input must contain majority_short_answer.")

    out = df.copy()
    if not args.skip_semantic:
        print(f"Computing semantic features with {args.semantic_model}")
        out = add_semantic_features(out, args.semantic_model, args.batch_size, args.device)
    if not args.skip_nli:
        print(f"Computing NLI features with {args.nli_model}")
        out = add_nli_features(out, args.nli_model, args.nli_device, args.batch_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Rows: {len(out)}, columns: {len(out.columns)}")


if __name__ == "__main__":
    main()
