# Bias Evaluation Datasets

Datasets for the project: "Do Internal Debiasing Edits Actually Change Behavior? A Causal Audit of Erasure and Steering"

## Summary

| Dataset | Examples | Bias Categories | Source | Format |
|---------|----------|-----------------|--------|--------|
| BBQ | 10,864 | 11 (age, disability, gender, nationality, appearance, race, race x SES, race x gender, religion, SES, sexual orientation) | `lighteval/bbq_helm` (HuggingFace) | JSONL |
| CrowS-Pairs | 1,508 | 9 (age, disability, gender, nationality, appearance, race-color, religion, sexual orientation, socioeconomic) | GitHub: nyu-mll/crows-pairs | CSV/JSONL |
| StereoSet | 4,229 (2,123 intersentence + 2,106 intrasentence) | 4 (gender, profession, race, religion) | `McGill-NLP/stereoset` (HuggingFace) | JSONL |
| WinoBias | 1,584 | 1 (gender in coreference) | GitHub: uclanlp/corefBias | JSONL + raw text |
| WinoGender | 720 | 1 (gender in coreference) | GitHub: rudinger/winogender-schemas | TSV |

**Total: ~18,905 evaluation examples**

## Dataset Descriptions

### BBQ (Bias Benchmark for QA) -- PRIMARY BENCHMARK
- **Citation**: Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering"
- **Task**: Multiple-choice QA testing whether models rely on social biases
- **Format**: Context + question + 3 answer choices (correct answer is typically "Not enough info" for ambiguous contexts)
- **Files**:
  - `bbq/all_categories.jsonl` -- Full dataset with category labels (10,864 examples)
  - `bbq/test.jsonl` -- 1,000 sampled examples (from the 'all' config)
- **Columns**: `context`, `question`, `references`, `choices`, `gold_index`, `category`
- **Categories**: Age, Disability_status, Gender_identity, Nationality, Physical_appearance, Race_ethnicity, Race_x_SES, Race_x_gender, Religion, SES, Sexual_orientation

### CrowS-Pairs
- **Citation**: Nangia et al. (2020). "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models"
- **Task**: Paired sentences differing only in a social group mention; models should not prefer the stereotypical sentence
- **Format**: Sentence pairs (more stereotypical vs. less stereotypical)
- **Files**:
  - `crows_pairs/test.jsonl` -- 1,508 sentence pairs
  - `crows_pairs/crows_pairs_anonymized.csv` -- Original CSV format
- **Columns**: `sent_more`, `sent_less`, `stereo_antistereo`, `bias_type`
- **Bias types**: age, disability, gender, nationality, physical-appearance, race-color, religion, sexual-orientation, socioeconomic

### StereoSet
- **Citation**: Nadeem et al. (2021). "StereoSet: Measuring stereotypical bias in pretrained language models"
- **Task**: Choose between stereotypical, anti-stereotypical, and unrelated sentence continuations
- **Format**: Context + 3 candidate sentences (stereotype, anti-stereotype, unrelated)
- **Files**:
  - `stereoset/intersentence_validation.jsonl` -- 2,123 examples (sentence-level context)
  - `stereoset/intrasentence_validation.jsonl` -- 2,106 examples (word-level fill-in)
- **Columns**: `id`, `target`, `bias_type`, `context`, `sentences`
- **Bias types**: gender, profession, race, religion

### WinoBias
- **Citation**: Zhao et al. (2018). "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods"
- **Task**: Coreference resolution with gendered occupations
- **Format**: Sentences with bracketed entities and pronouns
- **Files**:
  - `winobias/all.jsonl` -- All 1,584 examples in structured format
  - `winobias/*.txt.test` -- Original raw text files (pro/anti stereotype, type1/type2)
- **Columns**: `sentence`, `stereotype` (pro/anti), `coreference_type` (type1/type2)
- **Bias focus**: Gender stereotypes in occupations

### WinoGender
- **Citation**: Rudinger et al. (2018). "Gender Bias in Coreference Resolution"
- **Task**: Gender bias in coreference resolution via Winograd-style schemas
- **Files**:
  - `winogender/all_sentences.tsv` -- 720 sentences
- **Columns**: `sentid`, `sentence`

## Download Instructions

To re-download all datasets, activate the project venv and run:

```python
from datasets import load_dataset
import urllib.request

# BBQ (all categories)
categories = ['Age', 'Disability_status', 'Gender_identity', 'Nationality',
              'Physical_appearance', 'Race_ethnicity', 'Race_x_SES',
              'Race_x_gender', 'Religion', 'SES', 'Sexual_orientation']
for cat in categories:
    ds = load_dataset("lighteval/bbq_helm", cat)

# StereoSet
for config in ["intersentence", "intrasentence"]:
    ds = load_dataset("McGill-NLP/stereoset", config)

# CrowS-Pairs (from GitHub)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    "crows_pairs/crows_pairs_anonymized.csv"
)

# WinoBias (from GitHub)
base = "https://raw.githubusercontent.com/uclanlp/corefBias/master/WinoBias/wino/data"
for f in ["pro_stereotyped_type1.txt.test", "anti_stereotyped_type1.txt.test",
          "pro_stereotyped_type2.txt.test", "anti_stereotyped_type2.txt.test"]:
    urllib.request.urlretrieve(f"{base}/{f}", f"winobias/{f}")

# WinoGender (from GitHub)
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/rudinger/winogender-schemas/master/data/all_sentences.tsv",
    "winogender/all_sentences.tsv"
)
```

## Notes

- Data files (*.jsonl, *.csv, *.tsv) are excluded from git via `.gitignore`.
- BBQ is the primary behavioral benchmark for this project.
- All datasets were downloaded on 2026-06-15.
