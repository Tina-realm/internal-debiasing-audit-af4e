# Code Dependencies for Internal Debiasing Audit

This directory contains cloned repositories used in our research project:
"Do Internal Debiasing Edits Actually Change Behavior? A Causal Audit of Erasure and Steering."

---

## 1. concept-erasure (LEACE)

- **URL:** https://github.com/EleutherAI/concept-erasure
- **Paper:** "LEAst-squares Concept Erasure" (Belrose et al., 2023) — https://arxiv.org/abs/2306.03819
- **Purpose:** Closed-form linear concept erasure. Provably prevents all linear classifiers from detecting a concept while minimizing damage to representations. This is the primary erasure baseline for our audit.
- **Key files:**
  - `concept_erasure/leace.py` — Core `LeaceFitter` and `LeaceEraser` classes
  - `concept_erasure/concept_scrubber.py` — Applies erasure across model layers
  - `concept_erasure/scrubbing/llama.py` — LLaMA-specific scrubbing implementation
  - `concept_erasure/scrubbing/neox.py` — GPT-NeoX-specific scrubbing implementation
- **Dependencies:** Python >=3.10, PyTorch. Optional: transformers, datasets, scikit-learn
- **Install:** `pip install concept-erasure` or `pip install -e .`
- **Relevance:** We use LEACE to erase gender/race concepts from intermediate representations and then measure whether downstream behavior (BBQ, generation) actually changes.

---

## 2. nullspace_projection (INLP)

- **URL:** https://github.com/shauli-ravfogel/nullspace_projection
- **Paper:** "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection" (Ravfogel et al., ACL 2020) — https://www.aclweb.org/anthology/2020.acl-main.647/
- **Purpose:** Iteratively trains linear classifiers to predict a protected attribute, then projects representations into the classifier's nullspace. Repeats until no linear classifier can recover the attribute. This is the classic linear-erasure baseline.
- **Key files:**
  - `src/debias.py` — Core INLP algorithm (bare-bones classification-based version)
  - `src/inlp-oop/` — OOP implementation supporting classification and metric-learning objectives
  - `src/classifier.py` — Linear classifiers used in the iterative loop
  - `notebooks/` — Jupyter notebooks for word vector debiasing, bias-in-bios experiments
  - `run_deepmoji_debiasing.sh`, `run_bias_bios.sh` — Experiment scripts
- **Dependencies:** Python 3.7+, scikit-learn, NumPy, PyTorch (for some experiments)
- **Relevance:** INLP is the predecessor to LEACE. We compare both erasure methods to test whether iterative projection vs. closed-form erasure differ in their downstream behavioral effects.

---

## 3. representation-engineering (RepE)

- **URL:** https://github.com/andyzoujm/representation-engineering
- **Paper:** "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., 2023) — https://arxiv.org/abs/2310.01405
- **Purpose:** Population-level representation reading and control. Provides pipelines for (a) reading concepts from representations (RepReading) and (b) steering model behavior by adding/subtracting representation directions (RepControl). Covers honesty, fairness, emotions, harmlessness, memorization, and more.
- **Key files:**
  - `repe/rep_readers.py` — RepReading: extract concept directions from representations
  - `repe/rep_control_pipeline.py` — RepControl: steer generation by modifying activations
  - `repe/rep_control_contrast_vec.py` — Contrastive vector computation
  - `repe/pipelines.py` — HuggingFace pipeline integration
  - `examples/honesty/` — Honesty steering experiments
  - `examples/fairness/` — Fairness-related representation engineering
  - `examples/harmless_harmful/` — Harmlessness steering
- **Dependencies:** Python >=3.9, transformers, accelerate, scikit-learn
- **Install:** `pip install -e .`
- **Relevance:** RepE provides the representation reading framework we use to (a) identify concept directions and (b) test steering-based debiasing. The fairness examples are directly relevant.

---

## 4. CAA (Contrastive Activation Addition)

- **URL:** https://github.com/nrimsky/CAA
- **Paper:** "Steering Llama 2 with Contrastive Activation Addition" (Rimsky et al., 2023)
- **Purpose:** Generates steering vectors from contrastive pairs (positive vs. negative examples of a behavior) and adds them to model activations during inference. Targets sycophancy, corrigibility, hallucination, survival instinct, myopia, and refusal in Llama 2.
- **Key files:**
  - `generate_vectors.py` — Generate steering vectors from contrast pairs for each layer
  - `prompting_with_steering.py` — Run inference with steering vectors applied
  - `llama_wrapper.py` — Wrapper that hooks into Llama forward pass to add steering vectors
  - `normalize_vectors.py` — Normalize vectors across behaviors for fair comparison
  - `behaviors.py` — Behavior definitions and dataset paths
  - `datasets/` — Contrast pair datasets for each behavior
  - `vectors/` — Pre-computed (unnormalized) steering vectors
  - `scoring.py` — GPT-4 based scoring of open-ended responses
- **Dependencies:** PyTorch, transformers, datasets, openai, python-dotenv, scikit-learn
- **Relevance:** CAA provides the activation steering methodology. We adapt it to steer for/against demographic biases and measure whether the steering actually changes bias metrics on BBQ and other evaluations.

---

## 5. BBQ (Bias Benchmark for QA)

- **URL:** https://github.com/nyu-mll/BBQ
- **Paper:** "BBQ: A Hand-Built Bias Benchmark for Question Answering" (Parrish et al., Findings of ACL 2022) — https://aclanthology.org/2022.findings-acl.165/
- **Purpose:** Bias evaluation benchmark with template-generated QA examples across 9 social dimensions (age, disability, gender identity, nationality, physical appearance, race/ethnicity, religion, SES, sexual orientation) plus 2 intersectional categories (race x SES, race x gender). Tests whether models rely on stereotypes in ambiguous contexts and whether biases override correct answers in disambiguated contexts.
- **Key files:**
  - `data/*.jsonl` — 11 category files with templated QA examples (the evaluation data)
  - `supplemental/additional_metadata.csv` — Metadata including target locations for bias scoring
  - `templates/` — Templates and vocabulary used to generate BBQ
  - `results/` — Baseline results for UnifiedQA, RoBERTa, DeBERTaV3
- **Data format:** Each JSONL line contains: context, question, 3 answer options (ans0/ans1/ans2), correct label, bias-related metadata (context_condition: ambig/disambig, question_polarity: neg/nonneg)
- **Dependencies:** None (data-only repository)
- **Relevance:** BBQ is our primary downstream behavioral evaluation. After applying erasure (LEACE/INLP) or steering (CAA/RepE), we measure whether the model's BBQ bias scores actually change. The ambiguous-context condition is especially important — it reveals whether the model still relies on stereotypes after internal edits.

---

## How These Fit Together

Our audit pipeline:

1. **Baseline:** Run a target LLM on BBQ to measure pre-intervention bias scores
2. **Erasure methods:** Apply INLP or LEACE to erase gender/race concepts from intermediate representations
3. **Steering methods:** Use CAA or RepE to steer the model away from biased behavior
4. **Post-intervention evaluation:** Re-run BBQ (and other metrics) to measure whether bias actually decreased
5. **Causal analysis:** Compare representation-level metrics (probe accuracy, concept detectability) with behavioral metrics (BBQ bias score, generation quality) to determine if internal edits causally affect behavior or merely change linear probes without changing outputs
