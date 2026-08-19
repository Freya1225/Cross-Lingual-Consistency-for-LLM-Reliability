# Cross-Lingual-Consistency-for-LLM-Reliability

I compare agreement across languages with monolingual self-consistency and enrich the analysis with multilingual semantic-similarity and natural language inference (NLI) features. Lightweight classifiers then use these signals to distinguish more reliable answers from less reliable ones.
# Why this matters

Large language models can produce confident but incorrect answers. Standard confidence signals are often unavailable to end users and may not reflect whether an answer is factually correct. Cross-lingual prompting offers another perspective: if a model reaches semantically consistent answers when the same question is expressed in different languages, that agreement may provide evidence of reliability.
Cross-lingual agreement is not a guarantee of correctness—a model can repeat the same mistake in multiple languages—but it can serve as a useful warning or ranking signal.
# Research questions

1.Is cross-lingual consistency associated with answer correctness?

2.Does it provide a stronger reliability signal than monolingual self-consistency?

3.Do semantic-similarity and NLI features improve reliability prediction?

4.How well do these signals generalize across multilingual reasoning and question-answering tasks?

# Method
flowchart TD

    A["Multilingual benchmark questions"] --> B["Generate answers across languages"]
    
    B --> C["Compute consistency signals"]
    
    C --> D["Build similarity and NLI features"]
    
    D --> E["Train reliability classifiers"]
    
    E --> F["Evaluate with ROC-AUC and related metrics"]
