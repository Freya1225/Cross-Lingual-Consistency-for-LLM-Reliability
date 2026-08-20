# Experiment Results

This directory contains the consolidated reliability-prediction results for MGSM, XQuAD, and MKQA.

## Files

- `ablation_results.csv`: Complete ablation results using the primary evaluation labels. MGSM uses provided numeric correctness labels; XQuAD and MKQA use automatic answer scoring.
- `main_results.csv`: A compact selection of the most informative baselines and feature combinations for the project README.
- `ablation_results_manual_mkqa.csv`: A separate MKQA experiment using only manually reviewed alias candidates. It is not directly comparable with the primary MKQA experiment because the evaluation subset and label source differ.

## Metrics

- **ROC-AUC** measures how often the classifier ranks a correct response above an incorrect response. A value of 0.5 corresponds to random ranking and 1.0 corresponds to perfect ranking.
- **Average precision** summarizes precision-recall performance and is particularly useful when correct and incorrect examples are imbalanced.
- **Accuracy** is the fraction of predictions classified correctly at a probability threshold of 0.5.
- **F1** is the harmonic mean of precision and recall.

All learned models use logistic regression with balanced class weights, median imputation, feature standardization, and five-fold stratified group cross-validation. Rows from the same question are assigned to the same fold to prevent question-level leakage across languages.

## Primary Findings

| Dataset | Feature set | ROC-AUC | Accuracy | F1 |
|---|---|---:|---:|---:|
| MGSM | Monolingual self-consistency | 0.6470 | 0.6289 | 0.6912 |
| MGSM | Cross-lingual answer agreement | 0.8392 | 0.7556 | 0.7613 |
| MGSM | Answer + semantic features | **0.8563** | 0.7689 | 0.7540 |
| MGSM | Answer + semantic + NLI | 0.8534 | **0.7956** | **0.7806** |
| XQuAD | Answer features | 0.8856 | 0.7992 | 0.8422 |
| XQuAD | Answer + semantic features | **0.9015** | 0.8015 | 0.8451 |
| MKQA | Answer features | 0.7031 | 0.6590 | 0.6267 |
| MKQA | Answer + semantic + NLI | **0.7168** | **0.6692** | **0.6287** |

On MGSM, cross-lingual answer agreement substantially outperforms monolingual self-consistency, increasing ROC-AUC from 0.6470 to 0.8392. Semantic and NLI features provide smaller additional gains.

## Interpretation Cautions

XQuAD and MKQA correctness labels rely partly on automatic text, numeric, and date matching. Translations, transliterations, and entity aliases can therefore produce false-negative labels. The separate manually reviewed MKQA file should be treated as an error-analysis experiment, not as a replacement result from the same evaluation population.

The XQuAD language-only control reaches a ROC-AUC of 0.7866, indicating substantial language-dependent label or performance differences. Consequently, the full XQuAD score should not be interpreted as evidence that all predictive power comes from cross-lingual consistency.
