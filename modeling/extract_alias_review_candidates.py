import argparse
import re
from pathlib import Path

import pandas as pd


def has_digit_or_numeric_word(value) -> bool:
    text = "" if pd.isna(value) else str(value).lower()
    if re.search(r"\d", text):
        return True
    numeric_words = {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    }
    return text.strip(" .") in numeric_words


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract XQuAD/MKQA false-negative candidates for alias review."
    )
    parser.add_argument(
        "--language-level",
        type=Path,
        default=Path("scored_outputs/language_level_feature_table_scored.csv"),
    )
    parser.add_argument(
        "--sample-level",
        type=Path,
        default=Path("scored_outputs/sample_level_outputs_scored.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("manual_review_outputs"))
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lang_df = pd.read_csv(args.language_level)
    sample_df = pd.read_csv(args.sample_level)

    scored_samples = sample_df[
        sample_df["type"].isin(["XQuAD", "MKQA"])
        & (sample_df["auto_sample_is_correct"].astype(str).str.lower() == "false")
        & (sample_df["auto_score_method"] == "text_match")
    ].copy()

    sample_summary = (
        scored_samples.groupby("qkey")
        .agg(
            auto_failed_outputs=("raw_output", lambda values: " ||| ".join(dict.fromkeys(map(str, values)))),
            n_failed_samples=("raw_output", "size"),
        )
        .reset_index()
    )

    candidates = lang_df[
        lang_df["type"].isin(["XQuAD", "MKQA"])
        & (lang_df["auto_majority_sample_correct"].astype(str).str.lower() == "false")
    ].copy()
    candidates = candidates.merge(sample_summary, on="qkey", how="inner")
    candidates["ground_truth_has_digit_or_number_word"] = candidates["ground_truth_raw"].apply(
        has_digit_or_numeric_word
    )

    # Entity alias review is most useful for text answers rather than numeric/date answers.
    candidates = candidates[~candidates["ground_truth_has_digit_or_number_word"]].copy()

    lang_priority = {"zh": 0, "es": 1, "en": 2}
    type_priority = {"XQuAD": 0, "MKQA": 1}
    candidates["lang_priority"] = candidates["lang"].map(lang_priority).fillna(9)
    candidates["type_priority"] = candidates["type"].map(type_priority).fillna(9)
    candidates = candidates.sort_values(
        ["lang_priority", "type_priority", "id", "qkey"], kind="stable"
    )

    review_cols = [
        "needs_alias",
        "suggested_alias",
        "human_label",
        "notes",
    ]
    for col in review_cols:
        candidates[col] = ""

    output_cols = [
        "needs_alias",
        "suggested_alias",
        "human_label",
        "notes",
        "type",
        "id",
        "lang",
        "qkey",
        "question",
        "ground_truth_raw",
        "majority_short_answer",
        "auto_failed_outputs",
        "n_failed_samples",
        "self_consistency_short_answer",
        "short_cross_lingual_context",
    ]

    context = (
        lang_df[lang_df["type"].isin(["XQuAD", "MKQA"])]
        .pivot_table(
            index="id",
            columns="lang",
            values="majority_short_answer",
            aggfunc="first",
        )
        .reset_index()
    )
    context["short_cross_lingual_context"] = context.apply(
        lambda row: " | ".join(
            f"{lang}: {row[lang]}"
            for lang in ["en", "zh", "es"]
            if lang in row and not pd.isna(row[lang])
        ),
        axis=1,
    )

    candidates = candidates.merge(context[["id", "short_cross_lingual_context"]], on="id", how="left")
    candidates = candidates[output_cols].copy()

    sampled_parts = []
    if len(candidates) > 0:
        per_bucket = max(1, args.sample_size // 6)
        for (_, _), group in candidates.groupby(["type", "lang"], sort=False):
            n = min(len(group), per_bucket)
            sampled_parts.append(group.sample(n=n, random_state=args.seed))
        sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else candidates.head(0)
        if len(sampled) < args.sample_size:
            remaining = candidates.drop(sampled.index, errors="ignore")
            n_more = min(args.sample_size - len(sampled), len(remaining))
            if n_more > 0:
                sampled = pd.concat(
                    [sampled, remaining.sample(n=n_more, random_state=args.seed)],
                    ignore_index=True,
                )
        sampled = sampled.sort_values(["type", "lang", "id"], kind="stable")
    else:
        sampled = candidates

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "alias_review_candidates_all.csv"
    sample_path = args.output_dir / "alias_review_candidates_sample.csv"
    priority_path = args.output_dir / "alias_review_candidates_priority.csv"
    candidates.to_csv(all_path, index=False)
    sampled.to_csv(sample_path, index=False)
    priority = candidates.head(args.sample_size).copy()
    priority.to_csv(priority_path, index=False)

    print(f"Wrote {all_path} ({len(candidates)} rows)")
    print(f"Wrote {sample_path} ({len(sampled)} rows)")
    print(f"Wrote {priority_path} ({len(priority)} rows)")
    print("\nCandidate counts:")
    if len(candidates):
        print(candidates.groupby(["type", "lang"]).size().to_string())
    else:
        print("No candidates found.")


if __name__ == "__main__":
    main()
