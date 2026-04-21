# Sample Inputs & Outputs

This document provides realistic test cases for each detection scenario
PlagiarismCheck is designed to handle.

---

## Case 1 — Verbatim Copy (Expected: ~95%)

**Document A**
```
The water cycle, also known as the hydrological cycle, describes the continuous
movement of water on, above, and below Earth's surface. It involves processes
such as evaporation, condensation, precipitation, and runoff.
```

**Document B**
```
The water cycle, also known as the hydrological cycle, describes the continuous
movement of water on, above, and below Earth's surface. It involves processes
such as evaporation, condensation, precipitation, and runoff.
```

**Expected API response (approximate)**
```json
{
  "similarity_percent": 97.2,
  "jaccard_score": 0.96,
  "tfidf_score": 1.00,
  "semantic_score": 0.99,
  "matched_pairs": [
    {
      "sentence_a": "The water cycle, also known as the hydrological cycle...",
      "sentence_b": "The water cycle, also known as the hydrological cycle...",
      "score": 0.99,
      "method": "tfidf"
    }
  ]
}
```

---

## Case 2 — Paraphrased Content (Expected: 50–70%)

**Document A**
```
Artificial intelligence is transforming every sector of the modern economy.
Machine learning algorithms can now process vast amounts of data to identify
patterns that would be invisible to human analysts. Companies that invest in
AI research early are likely to gain significant competitive advantages.
```

**Document B**
```
The modern economy is being reshaped by artificial intelligence in profound ways.
Advanced learning algorithms are capable of analysing enormous datasets, uncovering
patterns beyond human perception. Organisations that adopt AI technology ahead of
their competitors stand to benefit greatly.
```

**Expected API response (approximate)**
```json
{
  "similarity_percent": 61.3,
  "jaccard_score": 0.38,
  "tfidf_score": 0.65,
  "semantic_score": 0.79,
  "matched_pairs": [
    {
      "sentence_a": "Artificial intelligence is transforming every sector...",
      "sentence_b": "The modern economy is being reshaped by artificial intelligence...",
      "score": 0.79,
      "method": "semantic"
    },
    {
      "sentence_a": "Machine learning algorithms can now process vast amounts...",
      "sentence_b": "Advanced learning algorithms are capable of analysing enormous datasets...",
      "score": 0.74,
      "method": "semantic"
    }
  ]
}
```

---

## Case 3 — Unrelated Texts (Expected: <15%)

**Document A**
```
The Maillard reaction is a chemical reaction between amino acids and reducing
sugars that gives browned food its distinctive flavour. It is responsible for
the colour and taste of browned meats, bread crusts, and roasted coffee.
```

**Document B**
```
The Treaty of Westphalia, signed in 1648, ended the Thirty Years War in the Holy
Roman Empire. It established the principle of state sovereignty and is considered
the foundation of the modern international order.
```

**Expected API response (approximate)**
```json
{
  "similarity_percent": 4.1,
  "jaccard_score": 0.00,
  "tfidf_score": 0.03,
  "semantic_score": 0.08,
  "matched_pairs": []
}
```

---

## Case 4 — Synonym Substitution (Semantic detection critical)

**Document A**
```
The patient exhibited symptoms of severe dehydration following prolonged exposure
to high temperatures. Medical staff administered intravenous fluids immediately.
```

**Document B**
```
The individual showed signs of acute water deficiency after extended time in
hot conditions. Healthcare workers gave IV drips without delay.
```

**Expected API response (approximate)**
```json
{
  "similarity_percent": 44.7,
  "jaccard_score": 0.04,
  "tfidf_score": 0.18,
  "semantic_score": 0.82,
  "matched_pairs": [
    {
      "sentence_a": "The patient exhibited symptoms of severe dehydration...",
      "sentence_b": "The individual showed signs of acute water deficiency...",
      "score": 0.82,
      "method": "semantic"
    }
  ]
}
```
*Note: Jaccard and TF-IDF scores are low because almost no words overlap.
Only semantic embeddings catch the paraphrase — demonstrating why the
multi-strategy approach is essential.*

---

## Case 5 — File Upload (.docx)

```bash
curl -X POST http://localhost:8000/api/check-files \
  -F "file_a=@essay_draft.docx" \
  -F "file_b=@wikipedia_excerpt.pdf" \
  | python3 -m json.tool
```

---

## Verdict Thresholds

| Similarity % | Verdict | Colour |
|---|---|---|
| 0 – 19 | Likely Original | Green |
| 20 – 49 | Moderate Similarity | Amber |
| 50 – 100 | High Similarity | Red |
