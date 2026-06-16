# Literature Review: Do Internal Debiasing Edits Actually Change Behavior?

## Research Area Overview

This research sits at the intersection of **representation engineering**, **concept erasure**, and **bias evaluation** in language models. The central question is whether internal debiasing interventions—methods that modify model representations to remove or reduce bias as measured by linear probes—actually produce proportional reductions in behavioral bias on downstream tasks. A growing body of evidence suggests a significant disconnect between intrinsic (probe-based) and extrinsic (behavioral) measures of bias, raising concerns that current debiasing methods may be "cosmetic" rather than substantive.

---

## Key Papers

### 1. Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases (Gonen & Goldberg, 2019)
- **arXiv**: 1903.03862
- **Key Contribution**: Demonstrated that standard debiasing methods (Bolukbasi et al. 2016) only superficially remove bias from word embeddings. After debiasing, biased information persists and can be recovered.
- **Methods tested**: Hard-Debiasing (Bolukbasi et al.) and GN-GloVe (Zhao et al.)
- **Four key experiments**:
  1. **Clustering**: k-means on 1,000 most biased words still achieves 92.5% (Hard-Debiased) / 85.6% (GN-GloVe) cluster purity after debiasing
  2. **Neighbor-based bias**: Bias measured by gender ratio of k-nearest neighbors correlates strongly with original bias (r=0.686–0.736)
  3. **WEAT tests**: All gender association tests remain statistically significant after debiasing
  4. **SVM classification**: RBF-kernel SVM recovers gender bias from "debiased" embeddings with 88.9% (Hard-Debiased) / 96.5% (GN-GloVe) accuracy
- **Core insight**: The gender direction *measures* bias but does not *constitute* it—removing the indicator doesn't remove the phenomenon
- **Code**: https://github.com/gonenhila/gender_bias_lipstick
- **Relevance**: Foundational motivation for our work—established that probe-level debiasing ≠ true debiasing. We extend this from static embeddings to LLM internal representations with modern methods (INLP, LEACE, steering).

### 2. Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (Ravfogel et al., 2020)
- **arXiv**: 2004.07667
- **Key Contribution**: Introduced INLP, which iteratively trains linear classifiers to predict protected attributes, then projects representations onto their nullspace. Each iteration removes one dimension (for binary Z).
- **Algorithm**: Train linear SVM → compute nullspace via SVD → project data → repeat. Final projection is intersection of all nullspaces, computed efficiently via rowspace accumulation.
- **Key results**:
  - Linear gender classification: 100% → 49.3% (near random) ✓
  - **MLP still recovers gender at 85%** — nonlinear information persists despite linear guardedness
  - Bias-by-neighbors only drops from 0.852 → 0.734 (substantial bias remains in neighborhood structure)
  - WEAT associations become non-significant (p > 0.3)
  - TPR-GAP on biography classification reduced 39–52% across representations
  - Main-task accuracy drops moderately (−5.5% worst case on BERT)
  - Word similarity actually improves (SimLex-999: 0.373 → 0.489), suggesting gender was noise for semantics
- **Critical limitation**: No evaluation on standard bias benchmarks (BBQ, CrowS-Pairs, WinoBias). BERT evaluation uses frozen representations, not generation. Does not test whether INLP changes stereotypical text generation.
- **Code**: https://github.com/shauli-ravfogel/nullspace_projection
- **Relevance**: Primary intervention method. Own results already demonstrate the probe-behavior disconnect: linear probes fooled while nonlinear probes succeed and neighborhood bias persists. Ideal test case for our audit.

### 3. LEACE: Perfect Linear Concept Erasure in Closed Form (Belrose et al., 2023)
- **arXiv**: 2306.03819
- **Key Contribution**: Introduced LEACE, a closed-form solution for concept erasure that provably prevents ANY linear classifier from recovering the erased concept—stronger guarantee than INLP.
- **Method**: Computes the least-squares optimal oblique projection via `r(x) = x - W⁺ P_{W·Σ_XZ} W(x - E[X])`, using the cross-covariance Σ_XZ between representations and concept labels. Uses oblique (not orthogonal) projections, proven to be least-squares optimal.
- **Advantages over INLP**: Closed-form (~100x faster), provably optimal, minimal distortion (removes only 17 dims for 18-class POS vs INLP's 360), enables multi-layer "concept scrubbing."
- **Key results**:
  - Gender erasure: TPR-GAP drops from 0.198→0.084 on profession classification; probe accuracy drops to majority baseline
  - Concept scrubbing (POS): Causes large perplexity increases (LLaMA-7B: 0.69→1.73 bits/byte), showing real downstream impact
  - **Critical finding for our work**: Probing accuracy and causal importance DIVERGE—POS probing peaks at early layers, but causal effect (via erasure) is concentrated at layer 11
- **Authors explicitly acknowledge** in Limitations: "Much work remains to be done... specifically, we'd like to see experiments that use **behavioral metrics** to determine whether scrubbing changes the network in the ways we'd intuitively expect."
- **Code**: https://github.com/EleutherAI/concept-erasure
- **Relevance**: Our second primary intervention. LEACE's "perfect" erasure guarantee makes it an ideal test case. The authors' own finding that probes and causal effects diverge directly motivates our audit.

### 4. Steering Llama 2 via Contrastive Activation Addition (Panickssery et al., 2023)
- **arXiv**: 2312.06681
- **Key Contribution**: Introduced CAA, which computes steering vectors from mean activation differences between contrastive prompt pairs (290–1000 pairs per behavior), then adds these to residual stream during inference.
- **Method**: Mean difference of residual stream activations at answer token position for positive vs negative behavior examples. Applied at all token positions after prompt, scaled by multiplier coefficient.
- **Practical details**: Optimal layer is **layer 13** for Llama-2-7B (≈1/3 through model). Vectors transfer across layers 10–15. Multiplier provides continuous dose-response dial (-2 to +2). Generation takes <5 min on single GPU.
- **Seven behaviors tested**: Sycophancy, corrigibility, hallucination, myopic reward, survival instinct, AI coordination, refusal. All steerable in both directions.
- **Key results**: CAA generalizes to open-ended generation (unlike finetuning for sycophancy). Stacks with system prompting. MMLU scores largely preserved (0.63 baseline, 0.57–0.65 with steering). TruthfulQA slightly improves with anti-sycophancy steering.
- **Side effects**: Text quality degrades at high multipliers. Counter-intuitive interactions with finetuning. **No systematic cross-behavior contamination measurement** — a gap for our work.
- **Critical gap**: They **never tested bias-direction steering on standard bias benchmarks** (BBQ, CrowS-Pairs). All behaviors tested are alignment-relevant, not demographic bias. This is precisely our contribution.
- **Code**: https://github.com/nrimsky/CAA (MIT license, includes pre-computed vectors)
- **Relevance**: Our third intervention. Inference-time, no weight changes, continuous multiplier dial enables dose-response analysis. We compute "bias" steering vectors and test whether subtracting them actually reduces BBQ bias scores.

### 5. Representation Engineering: A Top-Down Approach to AI Transparency (Zou et al., 2023)
- **arXiv**: 2310.01405
- **Key Contribution**: Introduced RepE, a top-down framework (inspired by cognitive neuroscience) for reading and controlling model representations. Uses Linear Artificial Tomography (LAT) to extract concept directions via PCA on contrastive activation differences.
- **Two pillars**: (1) Representation Reading — extract "reading vectors" via PCA on paired stimulus differences; (2) Representation Control — steer behavior via adding/subtracting vectors, contrast vectors, or LoRRA fine-tuning.
- **Four-level evaluation framework**: Correlation (does probe predict concept?) → Manipulation (does adding/subtracting change behavior?) → Termination (is direction necessary?) → Recovery (is direction sufficient?). Inspired by neuroscience lesion studies.
- **Critical finding for our work**: Logistic regression probes yield highest correlation accuracy but **fail at manipulation** — they find neural correlates, not causal directions. PCA-based directions perform robustly across all levels.
- **Key results**: On TruthfulQA, contrast vector control improves Llama-2-13B from 35.9% to 54.0% (genuine behavioral change). Section 6.3 addresses bias/fairness with "unified bias representations."
- **Limitations**: Behavioral evaluations are deep for honesty but lighter for bias/fairness concepts; most bias demonstrations are qualitative.
- **Code**: https://github.com/andyzoujm/representation-engineering
- **Relevance**: Provides both the theoretical framework and practical tools. The finding that high-probe-accuracy directions can have zero causal effect is a critical warning — reducing probe accuracy ≠ removing bias. Our audit must test manipulation and termination, not just correlation.

### 6. BBQ: A Hand-Built Bias Benchmark for Question Answering (Parrish et al., 2022)
- **arXiv**: 2110.08193
- **Key Contribution**: Created a behavioral benchmark for measuring social bias in QA via actual model predictions (not likelihoods). 325 hand-written templates, each linked to a specific attested bias with sourced documentation.
- **Structure**: Each item comes in sets of 4 (ambiguous/disambiguated × negative/non-negative question). Identity labels swapped and answer order permuted for balance.
  - **Ambiguous context**: Not enough info to answer → correct answer is always "Unknown" (sampled from 10 equivalent phrasings)
  - **Disambiguated context**: Answer explicitly stated in text
- **Bias score formulas** (critical for implementation):
  - Disambiguated: `s_DIS = 2 × (n_biased_ans / n_non_unknown_outputs) - 1`
  - Ambiguous: `s_AMB = (1 - accuracy) × s_DIS` (scaled by error rate — bias matters more when model answers more often)
  - Range: -100% (counter-stereotypical) to 0% (no bias) to +100% (fully biased)
- **Categories**: 9 bias categories + 2 intersectional (race×gender, race×SES). **58,492 total examples.**
- **Key results**: More capable models show worse bias in ambiguous contexts (~77% of errors align with stereotypes for UnifiedQA). Physical appearance shows highest bias. Models treat "Black" vs "African American" labels differently.
- **Download**: https://github.com/nyu-mll/BBQ (CC-BY 4.0)
- **Relevance**: PRIMARY behavioral benchmark. Per-category and per-stereotype granularity enables fine-grained mapping between debiasing interventions and specific bias changes. The ambiguous/disambiguated split gives two complementary signals.

---

## Critical Background: The Intrinsic-Extrinsic Gap

### Goldfarb-Tarrant et al. (2021) - "Intrinsic Bias Metrics Do Not Correlate with Application Bias"
- Found that intrinsic bias metrics (WEAT, SEAT) do not correlate with extrinsic bias in downstream applications
- This is the foundational negative result motivating our research

### Cao et al. (2022) - "On the Intrinsic and Extrinsic Fairness Evaluation Metrics"
- **19 models tested** across BERT, GPT-2, RoBERTa, etc.
- Found **poor correlations** between intrinsic (CEAT, ILPS, StereoSet) and extrinsic metrics (toxicity, sentiment, stereotype in BOLD)
- Even after correcting for metric misalignment, noise, and confounding factors, correlations remain weak
- **Key takeaway**: "We cannot assume that an improvement in language model fairness will fix bias in downstream systems"
- Code: https://github.com/pruksmhc/fairness-metrics-correlations

### Orgad & Belinkov (2022) - "Choose Your Lenses: Flaws in Gender Bias Evaluation"
- Position paper arguing intrinsic metrics create an "illusion of success"
- Found that ~1/3 of papers only measure intrinsic metrics
- Demonstrated that dataset composition and metric choice dramatically affect measured bias
- **Guidelines**: Always measure extrinsic metrics; decouple datasets from metrics; use balanced test sets

### Iluz et al. (2024) - "Applying Intrinsic Debiasing on Downstream Tasks"
- Tested Hard-Debiasing, INLP, LEACE on machine translation
- **Critical finding**: Debiasing all tokens produces garbled output (representational distortion)
- Only 11% of translations changed; of those, only 32% corrected gender
- INLP hurts BLEU scores (destroys useful information); LEACE preserves quality better
- Where you apply debiasing matters enormously (encoder vs decoder)

### Kaneko et al. (2024) - "The Gaps between Pre-train and Downstream Settings"
- FT-based debiasing: **low correlation** (0.10–0.25) between intrinsic and extrinsic bias scores
- ICL-based debiasing: higher correlation (0.29–0.44) because it doesn't modify parameters
- **Representational damage measured**: cosine similarity between original and debiased outputs is 0.51–0.66 for FT (severe distortion) vs 0.73–0.87 for ICL
- Directly supports our hypothesis that parameter-modifying debiasing causes "generic representational distortion"

---

## Additional Relevant Papers

### Akyürek et al. (2022) - "Challenges in Measuring Bias via Open-Ended Language Generation"
- Identified difficulties in measuring bias through text generation
- Highlighted that generation-based metrics are noisy and inconsistent

### Basu Roy Chowdhury et al. (2025) - "Fundamental Limits of Perfect Concept Erasure"
- Theoretical analysis of what concept erasure can and cannot achieve
- Shows fundamental trade-offs between erasure completeness and utility preservation

### Turner et al. (2023) - "Activation Addition: Steering Language Models Without Optimization"
- Precursor to CAA; demonstrated that adding activation vectors can steer model behavior
- Simpler approach than full representation engineering

### Li et al. (2023) - "Inference-Time Intervention"
- Showed that targeted interventions at specific attention heads during inference can change model behavior
- Related approach to steering but more targeted

---

## Common Methodologies

1. **Linear Concept Erasure**: INLP (iterative), LEACE (closed-form) — remove concept from representations so linear probes can't detect it
2. **Activation Steering**: CAA, RepE control vectors — add/subtract direction vectors during inference
3. **Linear Probing**: Train linear classifiers on representations to detect concepts — the standard "intrinsic" measurement
4. **Behavioral Benchmarks**: BBQ, CrowS-Pairs, StereoSet, WinoBias — measure bias through model outputs on specific tasks

## Standard Baselines
- **No intervention**: Unmodified model as baseline
- **Random projection**: Project to random subspace of same rank (controls for dimensionality reduction)
- **INLP with random labels**: INLP trained on shuffled labels (controls for the erasure procedure itself)

## Evaluation Metrics
- **Probe accuracy**: Can a linear classifier recover the concept? (intrinsic)
- **BBQ bias score**: Stereotypical answer rate in ambiguous contexts (extrinsic/behavioral)
- **CrowS-Pairs score**: Preference for stereotypical vs anti-stereotypical sentences
- **StereoSet ICAT**: Combined language modeling + stereotype score
- **Perplexity change**: Measures representational distortion / model degradation
- **KL divergence**: Between original and debiased output distributions

## Datasets in the Literature
- **BBQ** (Parrish et al., 2022): QA bias benchmark — most comprehensive behavioral measure
- **CrowS-Pairs** (Nangia et al., 2020): Stereotype preference via likelihood comparison
- **StereoSet** (Nadeem et al., 2021): Stereotype scoring with language model quality control
- **WinoBias** (Zhao et al., 2018): Gender bias in coreference resolution
- **Bias in Bios** (De-Arteaga et al., 2019): Occupation prediction from biographies

---

## Gaps and Opportunities

1. **No systematic causal audit**: Papers either propose debiasing methods (measuring probes) OR evaluate behavioral bias, but rarely connect the two causally. Our work bridges this gap.

2. **Distortion not measured**: Most debiasing papers don't quantify how much "generic distortion" their intervention causes. We will measure this via perplexity change and KL divergence.

3. **No dose-response analysis**: If you equalize probe reduction across methods, do they equally reduce behavioral bias? This controlled comparison hasn't been done.

4. **Missing causal controls**: Random-direction erasure and label-shuffled controls are rarely used to distinguish genuine concept removal from generic representational damage.

5. **LLM-scale evaluation gap**: Most intrinsic/extrinsic correlation studies use smaller models (BERT, GPT-2). Our audit on modern LLMs (Llama-2 scale) would be novel.

---

## Recommendations for Our Experiment

### Recommended datasets:
- **BBQ**: Primary behavioral benchmark (most comprehensive, covers 9 bias categories)
- **CrowS-Pairs**: Secondary benchmark (simpler, likelihood-based)
- **StereoSet**: Tertiary benchmark (includes language model quality via ICAT score)

### Recommended interventions (independent variable):
1. **INLP** — iterative linear erasure (using nullspace_projection code)
2. **LEACE** — closed-form linear erasure (using concept-erasure code)
3. **CAA/Steering** — activation addition with bias-direction vectors (using CAA code)
4. **Random projection control** — project away random direction of same rank
5. **No intervention baseline**

### Recommended metrics:
- **Intrinsic**: Linear probe accuracy for detecting bias concepts (gender, race)
- **Behavioral**: BBQ bias score, CrowS-Pairs score
- **Distortion**: Perplexity on held-out text, KL divergence from original model
- **Causal attribution**: Compare behavioral change from targeted vs random erasure

### Methodological considerations:
- Use a single model family (e.g., Llama-2 7B) to control for architecture
- Apply interventions at multiple layers to understand layer effects
- Equalize probe reduction across methods before comparing behavioral effects
- Include random-direction controls to measure generic distortion
- Report confidence intervals via bootstrap resampling
