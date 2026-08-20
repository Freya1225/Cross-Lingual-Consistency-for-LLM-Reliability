import argparse
import ast
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


EN_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

ES_NUMBER_WORDS = {
    "cero": 0,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "dieciséis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
}

ZH_NUMBER_WORDS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}

ARTICLES = {"a", "an", "the", "el", "la", "los", "las", "un", "una", "unos", "unas"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if str(value).strip().lower() in {"", "none", "nan", "null"}:
        return True
    return False


def normalize_text(value: Any) -> str:
    if is_missing(value):
        return ""
    text = str(value).lower().strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_for_match(value: Any) -> str:
    text = normalize_text(value)
    text = text.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    tokens = [tok for tok in text.split() if tok not in ARTICLES]
    return " ".join(tokens)


def tokenize_for_f1(value: Any) -> List[str]:
    text = normalize_for_match(value)
    if re.search(r"[\u4e00-\u9fff]", text):
        cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
        latin_tokens = re.findall(r"[a-z0-9]+", text)
        return cjk_chars + latin_tokens
    return re.findall(r"[a-z0-9]+", text)


def token_f1(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = tokenize_for_f1(prediction)
    gt_tokens = tokenize_for_f1(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def english_number_phrase_to_value(text: str) -> Optional[float]:
    text = normalize_for_match(text)
    if text in EN_NUMBER_WORDS:
        return float(EN_NUMBER_WORDS[text])
    if text in ES_NUMBER_WORDS:
        return float(ES_NUMBER_WORDS[text])
    if text in ZH_NUMBER_WORDS:
        return float(ZH_NUMBER_WORDS[text])
    return None


def extract_numeric_values(value: Any) -> List[float]:
    text = normalize_text(value).replace(",", "")
    values: List[float] = []

    for match in re.findall(r"[-+]?\d+(?:\.\d+)?", text):
        try:
            values.append(float(match))
        except ValueError:
            pass

    for word, number in EN_NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            values.append(float(number))
    for word, number in ES_NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            values.append(float(number))
    for char, number in ZH_NUMBER_WORDS.items():
        if char in text:
            values.append(float(number))

    phrase_value = english_number_phrase_to_value(text)
    if phrase_value is not None:
        values.append(phrase_value)

    return values


def numeric_targets(ground_truth: Any) -> List[float]:
    if is_missing(ground_truth):
        return []
    text = normalize_text(ground_truth)
    values = extract_numeric_values(text)
    if values:
        return values
    phrase_value = english_number_phrase_to_value(text)
    return [phrase_value] if phrase_value is not None else []


def nearly_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def has_numeric_answer(output: Any, ground_truth: Any) -> bool:
    targets = numeric_targets(ground_truth)
    if not targets:
        return False
    observed = extract_numeric_values(output)
    return any(nearly_equal(obs, target) for obs in observed for target in targets)


def parse_iso_date(value: Any) -> Optional[Tuple[int, int, int]]:
    text = normalize_text(value)
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def has_date_answer(output: Any, ground_truth: Any) -> bool:
    iso = parse_iso_date(ground_truth)
    if iso is None:
        return False
    year, month, day = iso
    text = normalize_text(output)
    numeric_values = extract_numeric_values(text)
    has_year = any(nearly_equal(v, year) for v in numeric_values)
    has_day = any(nearly_equal(v, day) for v in numeric_values)
    has_month = any(nearly_equal(v, month) for v in numeric_values)

    month_names = {
        1: ["january", "jan", "enero"],
        2: ["february", "feb", "febrero"],
        3: ["march", "mar", "marzo"],
        4: ["april", "apr", "abril"],
        5: ["may", "mayo"],
        6: ["june", "jun", "junio"],
        7: ["july", "jul", "julio"],
        8: ["august", "aug", "agosto"],
        9: ["september", "sep", "septiembre"],
        10: ["october", "oct", "octubre"],
        11: ["november", "nov", "noviembre"],
        12: ["december", "dec", "diciembre"],
    }
    has_month_name = any(name in text for name in month_names.get(month, []))
    return has_year and has_day and (has_month or has_month_name)


def string_answer_match(output: Any, ground_truth: Any, f1_threshold: float = 0.8) -> bool:
    pred = normalize_for_match(output)
    gt = normalize_for_match(ground_truth)
    if not pred or not gt:
        return False
    if gt in pred:
        return True
    return token_f1(pred, gt) >= f1_threshold


def score_output(output: Any, ground_truth: Any) -> Tuple[Optional[bool], str]:
    if is_missing(ground_truth):
        return None, "unscorable_missing_ground_truth"
    if has_date_answer(output, ground_truth):
        return True, "date_match"
    if parse_iso_date(ground_truth) is not None:
        return False, "date_mismatch"
    if numeric_targets(ground_truth):
        return has_numeric_answer(output, ground_truth), "numeric_match"
    return string_answer_match(output, ground_truth), "text_match"


def parse_list(value: Any) -> List[str]:
    if is_missing(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [str(value)]


def majority_bool(values: Iterable[Optional[bool]]) -> Optional[bool]:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]


def score_sample_table(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in sample_df.iterrows():
        scored, method = score_output(row["raw_output"], row["ground_truth_raw"])
        new_row = row.copy()
        new_row["auto_sample_is_correct"] = scored
        new_row["auto_score_method"] = method
        rows.append(new_row)
    return pd.DataFrame(rows)


def score_language_table(language_df: pd.DataFrame, scored_samples: pd.DataFrame) -> pd.DataFrame:
    grouped = scored_samples.groupby("qkey")["auto_sample_is_correct"].agg(list).reset_index()
    grouped["auto_any_sample_correct"] = grouped["auto_sample_is_correct"].apply(
        lambda values: any(value is True for value in values)
        if any(value is not None for value in values)
        else None
    )
    grouped["auto_majority_sample_correct"] = grouped["auto_sample_is_correct"].apply(majority_bool)
    grouped["auto_n_scored_samples"] = grouped["auto_sample_is_correct"].apply(
        lambda values: sum(value is not None for value in values)
    )

    scored = language_df.merge(
        grouped[
            [
                "qkey",
                "auto_any_sample_correct",
                "auto_majority_sample_correct",
                "auto_n_scored_samples",
            ]
        ],
        on="qkey",
        how="left",
    )

    scored["evaluation_label_any_correct"] = scored["provided_is_correct"]
    needs_auto = scored["evaluation_label_any_correct"].isna()
    scored.loc[needs_auto, "evaluation_label_any_correct"] = scored.loc[
        needs_auto, "auto_any_sample_correct"
    ]
    scored["evaluation_label_majority_correct"] = scored["auto_majority_sample_correct"]
    mgsm_mask = scored["type"] == "MGSM"
    scored.loc[mgsm_mask, "evaluation_label_majority_correct"] = scored.loc[
        mgsm_mask, "majority_number_is_correct"
    ]
    return scored


def summarize(scored_language: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, lang), group in scored_language.groupby(["type", "lang"]):
        labels = group["evaluation_label_any_correct"].dropna()
        majority_labels = group["evaluation_label_majority_correct"].dropna()
        rows.append(
            {
                "type": task,
                "lang": lang,
                "n_rows": len(group),
                "n_any_labels": len(labels),
                "any_correct_accuracy": labels.astype(bool).mean() if len(labels) else None,
                "n_majority_labels": len(majority_labels),
                "majority_correct_accuracy": majority_labels.astype(bool).mean()
                if len(majority_labels)
                else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatically score XQuAD and MKQA outputs.")
    parser.add_argument(
        "--language-level",
        type=Path,
        default=Path("processed_outputs/language_level_feature_table.csv"),
    )
    parser.add_argument(
        "--sample-level",
        type=Path,
        default=Path("processed_outputs/sample_level_outputs.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("scored_outputs"))
    args = parser.parse_args()

    language_df = pd.read_csv(args.language_level)
    sample_df = pd.read_csv(args.sample_level)

    scored_sample = score_sample_table(sample_df)
    scored_language = score_language_table(language_df, scored_sample)
    summary = summarize(scored_language)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "sample_level_outputs_scored.csv"
    language_path = args.output_dir / "language_level_feature_table_scored.csv"
    summary_path = args.output_dir / "auto_scoring_summary.csv"

    scored_sample.to_csv(sample_path, index=False)
    scored_language.to_csv(language_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {sample_path}")
    print(f"Wrote {language_path}")
    print(f"Wrote {summary_path}")
    print("\nScoring summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
