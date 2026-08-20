# Data Processing and Multilingual Response Generation

This directory contains the data preparation and model-generation pipeline for the **Cross-Lingual Consistency for LLM Reliability** project.

The pipeline combines multilingual examples from MGSM, XQuAD, and MKQA into a shared JSON structure, generates three responses per language with a multilingual instruction-tuned language model, and assigns automatic correctness labels to MGSM outputs.


## Data sources

| Dataset | Task | Languages used | How it is loaded in this code | Original source |
| --- | --- | --- | --- | --- |
| **MGSM** | Multilingual mathematical reasoning | English, Chinese, Spanish | Local TSV files stored in Google Drive | [Google Research MGSM](https://github.com/google-research/url-nlp/tree/main/mgsm) |
| **XQuAD** | Cross-lingual extractive question answering | English, Chinese, Spanish | `load_dataset("google/xquad", ...)` | [Hugging Face: google/xquad](https://huggingface.co/datasets/google/xquad) |
| **MKQA** | Multilingual open-domain knowledge QA | English, Chinese, Spanish | Local compressed JSONL file stored in Google Drive | [Apple MKQA repository](https://github.com/apple/ml-mkqa) |



## Unified data format

All datasets are converted into a common structure:

```json
{
  "id": "mgsm_000",
  "type": "MGSM",
  "ground_truth": "42",
  "languages": {
    "en": {
      "question": "...",
      "prompt": "...",
      "model_output": [],
      "is_correct": null
    },
    "zh": {
      "question": "...",
      "prompt": "...",
      "model_output": [],
      "is_correct": null
    },
    "es": {
      "question": "...",
      "prompt": "...",
      "model_output": [],
      "is_correct": null
    }
  }
}
```

The combined dataset is written to:

```text
g24_full_multilingual_data.json
```

## Model response generation

The generation stage uses the Groq Python client and the following model configuration:

```python
MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
temperature = 0.7
max_tokens = 512
```

For each question, the pipeline generates:

- Three English responses
- Three Chinese responses
- Three Spanish responses

With 450 questions, this produces up to 4,050 model responses.

The generation loop supports:

- Checkpointing after every processed question
- Resuming from an existing output file
- Skipping question-language pairs that already have three valid responses
- Retrying after rate-limit and temporary service errors
- Filtering failed outputs before requesting replacements

The checkpointed results are written to a JSON file such as:

```text
g24_final_results_groq.json
```


## Automatic correctness labels

MGSM answers are evaluated automatically by extracting the final number from the first generated response and comparing it with the English ground-truth answer:

```python
entry["languages"][lang]["is_correct"] = (pred == gt)
```

XQuAD and MKQA retain `is_correct = null` at this stage. Their free-form answers require a later scoring pipeline that handles normalized strings, dates, numeric answers, token overlap, aliases, translations, and transliterations.


### Ground-truth alignment

MGSM examples are aligned by row index across the three TSV files, and the English answer is used as the shared ground truth. This assumes that all three files preserve identical example ordering.

### Reproducibility

Generation uses `temperature=0.7` and does not currently record a random seed or full API response metadata. Exact text outputs may vary across runs or model revisions.



## Data and model licenses

This repository does not relicense MGSM, XQuAD, MKQA, the generated model outputs, or the pretrained model. Users should review the terms of the original dataset sources, the selected model, and the Groq service before redistribution or reuse.


