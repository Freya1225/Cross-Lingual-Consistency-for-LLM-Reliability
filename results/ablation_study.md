# Ablation Study

## Research Question

This study evaluates whether cross-lingual response consistency can predict the correctness of large language model answers. The central comparison is between monolingual self-consistency, which measures agreement across repeated samples in one language, and cross-lingual consistency, which measures agreement across English, Chinese, and Spanish responses to the same question.

## Prediction Unit

Each classification row represents one question-language pair. The target label indicates whether the answer for that language is correct. Features summarize repeated outputs within the current language and relationships among the three language-level majority answers.

## Feature Groups

| Feature group | Description |
|---|---|
| Language control | One-hot indicators for English, Chinese, and Spanish |
| Monolingual self-consistency | Agreement and number of unique answers across repeated samples in one language |
| Cross-lingual answer agreement | Majority-answer agreement across languages and whether the current response matches the cross-lingual majority |
| Pairwise agreement | English-Chinese, English-Spanish, and Chinese-Spanish answer matches |
| Semantic similarity | LaBSE cosine similarity between language-level majority answers |
| Natural language inference | Bidirectional entailment and contradiction scores from a multilingual NLI model |

## Evaluation Design

The classifier is a logistic regression model with balanced class weights. Missing feature values are median-imputed and numeric features are standardized. Evaluation uses five-fold stratified group cross-validation, with question ID as the group. This ensures that different language versions of the same question do not appear in both training and test folds.

## MGSM

The MGSM experiment provides the clearest test of the main hypothesis because its numeric answers can be evaluated without multilingual entity-alias matching.

| Model | ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|
| Majority-prior control | 0.5000 | 0.5289 | 0.0000 |
| Monolingual self-consistency | 0.6470 | 0.6289 | 0.6912 |
| Cross-lingual answer agreement | 0.8392 | 0.7556 | 0.7613 |
| Self + cross-lingual agreement | 0.8522 | 0.7667 | 0.7602 |
| Answer + semantic features | **0.8563** | 0.7689 | 0.7540 |
| Answer + semantic + NLI | 0.8534 | **0.7956** | **0.7806** |

Cross-lingual answer agreement improves ROC-AUC by 0.1922 over monolingual self-consistency. Adding semantic similarity produces a modest further improvement, while adding NLI improves threshold-based accuracy and F1 but not ROC-AUC.

## XQuAD

Adding semantic similarity improves ROC-AUC from 0.8856 to 0.9015. However, the language-only control obtains 0.7866 ROC-AUC, suggesting that language-specific correctness rates or automatic-scoring artifacts explain a meaningful portion of performance. Automatic text matching is particularly vulnerable to translated names and aliases.

## MKQA

The answer-only feature set obtains 0.7031 ROC-AUC. Adding semantic features alone does not improve the result, while the full semantic and NLI feature set reaches 0.7168. A separate experiment using only manually reviewed alias candidates obtains different results because it evaluates a smaller, deliberately selected subset; it should therefore be reported separately.

## Conclusion

The strongest evidence for the project hypothesis comes from MGSM: cross-lingual answer agreement is substantially more predictive of correctness than monolingual self-consistency. Results on open-domain and extractive question answering are promising but more sensitive to multilingual answer normalization, alias handling, language imbalance, and label quality.
