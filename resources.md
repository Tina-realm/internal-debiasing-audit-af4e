# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project: **"Do Internal Debiasing Edits Actually Change Behavior? A Causal Audit of Erasure and Steering."**

---

## Papers
Total papers downloaded: **16**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Lipstick on a Pig | Gonen, Goldberg | 2019 | papers/gonen_goldberg_2019_lipstick_on_a_pig.pdf | Foundational: debiasing hides but doesn't remove bias |
| 2 | Null It Out (INLP) | Ravfogel et al. | 2020 | papers/ravfogel_2020_INLP_nullspace_projection.pdf | Primary intervention: iterative nullspace projection |
| 3 | LEACE | Belrose et al. | 2023 | papers/belrose_2023_LEACE_concept_erasure.pdf | Primary intervention: closed-form concept erasure |
| 4 | CAA: Steering Llama 2 | Panickssery et al. | 2023 | papers/panickssery_2023_CAA_steering_llama2.pdf | Primary intervention: contrastive activation addition |
| 5 | Representation Engineering | Zou et al. | 2023 | papers/zou_2023_representation_engineering.pdf | Framework for reading/controlling representations |
| 6 | BBQ Benchmark | Parrish et al. | 2022 | papers/parrish_2022_BBQ_bias_benchmark_QA.pdf | Primary behavioral evaluation benchmark |
| 7 | Choose Your Lenses | Orgad, Belinkov | 2022 | papers/orgad_2022_choose_your_lenses_gender_bias_eval.pdf | Flaws in gender bias evaluation paradigm |
| 8 | Intrinsic & Extrinsic Fairness | Cao et al. | 2022 | papers/cao_2022_intrinsic_extrinsic_fairness_metrics.pdf | 19-model study: intrinsic/extrinsic metrics don't correlate |
| 9 | Intrinsic Debiasing on Downstream | Iluz et al. | 2024 | papers/iluz_2024_intrinsic_debiasing_downstream.pdf | INLP/LEACE on MT: limited behavioral impact |
| 10 | Gaps: Pre-train vs Downstream | Kaneko et al. | 2024 | papers/kaneko_2024_gaps_pretrain_downstream_bias.pdf | FT debiasing causes severe representational damage |
| 11 | Challenges Measuring Bias in Generation | Akyürek et al. | 2022 | papers/akyurek_2022_challenges_measuring_bias_generation.pdf | Difficulties with generation-based bias metrics |
| 12 | Fundamental Limits of Concept Erasure | Basu Roy Chowdhury et al. | 2025 | papers/basu_2025_fundamental_limits_concept_erasure.pdf | Theoretical limits of erasure |
| 13 | Birth of Bias | van der Wal et al. | 2022 | papers/vanderwal_2022_birth_of_bias.pdf | How bias develops during training |
| 14 | Multiclass Debiasing Evaluation | Schlender, Spanakis | 2020 | papers/schlender_2020_multiclass_debiasing_eval.pdf | Evaluation of debiasing on word embeddings |
| 15 | Activation Addition | Turner et al. | 2023 | papers/turner_2023_activation_addition_steering.pdf | Precursor to CAA |
| 16 | Inference-Time Intervention | Li et al. | 2023 | papers/li_2023_inference_time_intervention.pdf | Targeted attention head interventions |

See [papers/README.md](papers/README.md) for detailed descriptions.

---

## Datasets
Total datasets downloaded: **5**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| BBQ | HuggingFace (lighteval/bbq_helm) | 10,864 | QA bias benchmark | datasets/bbq/ | PRIMARY benchmark, 11 bias categories |
| CrowS-Pairs | GitHub (nyu-mll) | 1,508 | Stereotype preference | datasets/crows_pairs/ | 9 bias types, likelihood comparison |
| StereoSet | HuggingFace (McGill-NLP) | 4,229 | Stereotype scoring | datasets/stereoset/ | Intra- and inter-sentence, ICAT metric |
| WinoBias | GitHub (uclanlp) | 1,584 | Coreference gender bias | datasets/winobias/ | Pro/anti stereotype sentence pairs |
| WinoGender | GitHub (rudinger) | 720 | Coreference gender bias | datasets/winogender/ | Occupation-based gender schemas |

See [datasets/README.md](datasets/README.md) for detailed descriptions and download instructions.

---

## Code Repositories
Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| concept-erasure | github.com/EleutherAI/concept-erasure | LEACE implementation | code/concept-erasure/ | Has LLaMA scrubbing support |
| nullspace_projection | github.com/shauli-ravfogel/nullspace_projection | INLP implementation | code/nullspace_projection/ | OOP and functional APIs |
| representation-engineering | github.com/andyzoujm/representation-engineering | RepE framework | code/representation-engineering/ | Reading + control vectors, fairness examples |
| CAA | github.com/nrimsky/CAA | Contrastive Activation Addition | code/CAA/ | Pre-computed vectors, Llama 2 wrapper |
| BBQ | github.com/nyu-mll/BBQ | BBQ evaluation data + templates | code/BBQ/ | Templates, baseline results, analysis scripts |

See [code/README.md](code/README.md) for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Started with the 5 papers specified in the research topic (arXiv IDs)
2. Used arXiv API to search for related work on debiasing evaluation, intrinsic/extrinsic gap, concept erasure
3. Downloaded 11 additional papers covering the intrinsic-extrinsic gap literature
4. Located datasets from paper references and HuggingFace
5. Cloned code repositories from paper links and GitHub

### Selection Criteria
- **Papers**: Prioritized work that (a) proposes debiasing methods we'll test, (b) evaluates the intrinsic-extrinsic gap, or (c) provides behavioral benchmarks
- **Datasets**: Selected established behavioral bias benchmarks used across the literature
- **Code**: Cloned implementations of all three intervention methods plus the primary evaluation benchmark

### Challenges Encountered
- Paper-finder service was unavailable; used arXiv API directly
- Some downloads required retry due to connection issues
- StereoSet only has validation split publicly available (test set held out by authors)

---

## Recommendations for Experiment Design

Based on gathered resources:

### 1. Primary dataset: **BBQ**
- Most comprehensive behavioral benchmark (11 categories, 10K+ examples)
- Well-established in the literature for measuring social bias in QA
- Clear scoring methodology for ambiguous vs disambiguated contexts

### 2. Secondary datasets: **CrowS-Pairs**, **StereoSet**
- CrowS-Pairs: Likelihood-based measure, simpler but complementary
- StereoSet: Includes language modeling quality via ICAT score (measures utility preservation)

### 3. Intervention methods to compare:
| Method | Type | Code | Guarantee |
|--------|------|------|-----------|
| INLP | Iterative linear erasure | code/nullspace_projection/ | No specific linear classifier recovers concept |
| LEACE | Closed-form linear erasure | code/concept-erasure/ | NO linear classifier recovers concept |
| CAA | Activation steering | code/CAA/ | None (heuristic) |
| Random projection | Control | (implement) | None (baseline for distortion) |

### 4. Evaluation metrics:
- **Intrinsic**: Linear probe accuracy (concept detectability after intervention)
- **Behavioral**: BBQ bias score, CrowS-Pairs preference, StereoSet ICAT
- **Distortion**: Perplexity change, KL divergence from original output distribution
- **Causal**: Behavioral change from targeted erasure vs random-direction erasure of same rank

### 5. Key experimental design principles:
- **Equalize probe reduction**: Compare methods at equal levels of probe accuracy reduction
- **Include random controls**: Random-direction erasure distinguishes concept removal from generic distortion
- **Measure distortion**: Track perplexity and KL divergence as indicators of representational damage
- **Layer sweep**: Apply interventions at different layers to understand where bias is causally encoded
- **Single model**: Use Llama-2-7B (or similar) to control for architecture effects
