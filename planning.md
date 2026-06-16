# Research Plan: Do Internal Debiasing Edits Actually Change Behavior?

## Motivation & Novelty Assessment

### Why This Research Matters
Current debiasing methods for LLMs are evaluated primarily by whether linear probes can still detect bias after intervention. But Gonen & Goldberg (2019) showed this approach is fundamentally flawed for static embeddings—bias information persists even when probes fail. The field has moved to activation-level interventions (INLP, LEACE, CAA) but the same validity concern applies: reducing probe accuracy may not reduce actual biased behavior. If debiasing methods are cosmetic, deployed "debiased" models may give a false sense of safety.

### Gap in Existing Work
From the literature review:
- Ravfogel et al. (2020) showed INLP fools linear probes but MLPs still recover gender at 85%
- Belrose et al. (2023) explicitly noted that probing accuracy and causal importance diverge
- Cao et al. (2022) found poor correlations between intrinsic and extrinsic metrics across 19 models
- Iluz et al. (2024) found INLP/LEACE have minimal behavioral impact on translation
- **No study has systematically compared erasure and steering methods with random-direction controls on behavioral benchmarks while equalizing probe reduction**

### Our Novel Contribution
We provide a controlled causal audit that:
1. Equalizes probe-metric reduction across methods before comparing behavioral effects
2. Uses random-direction and random-magnitude controls (per Dobrzeniecka et al. 2025) to distinguish concept-specific editing from generic perturbation
3. Measures the "causal specificity ratio"—how much behavioral change is attributable to the target concept vs. generic distortion
4. Tests across multiple behavioral benchmarks (BBQ, CrowS-Pairs)

### Experiment Justification
- **Exp 1 (Probe Training)**: Establishes where gender information is encoded and provides the baseline for intervention strength calibration
- **Exp 2 (Interventions)**: Tests whether INLP, LEACE, and steering produce equal behavioral changes when calibrated to equal probe reduction
- **Exp 3 (Random Controls)**: Tests whether behavioral changes are causally specific to gender or arise from generic representational distortion
- **Exp 4 (Distortion Measurement)**: Quantifies the capability cost of each intervention

## Research Question
Among interventions that equally reduce a linear-probe measure of gender bias, do they proportionally reduce behavioral bias? And is observed behavioral change causally attributable to gender-concept editing or to generic representational distortion?

## Hypothesis Decomposition
H1: Methods that achieve equal probe-accuracy reduction will show unequal behavioral bias reduction on BBQ and CrowS-Pairs.
H2: A significant fraction of behavioral change from each intervention will also occur with a random-direction control of equal magnitude—indicating generic distortion rather than concept-specific editing.
H3: LEACE (provably optimal linear erasure) will show the largest gap between probe reduction and behavioral reduction, because its mathematical guarantee only covers linear detectability.

## Proposed Methodology

### Model
GPT-2 (124M parameters) — feasible on CPU, well-studied for bias, decoder-only architecture representative of modern LLMs.

### Approach
1. Extract activations from GPT-2 on gender-marked sentence pairs
2. Train linear probes at each layer to detect gender
3. Apply INLP, LEACE, and steering interventions at the most informative layer
4. Calibrate each intervention to achieve ~equal probe accuracy reduction
5. Evaluate behavioral bias on BBQ (gender subset) and CrowS-Pairs (gender subset)
6. Run random-direction controls of matched magnitude
7. Compute causal specificity ratios

### Experimental Steps
1. **Activation extraction** (15 min): Run gender-paired sentences through GPT-2, collect residual stream activations at all layers
2. **Probe training** (10 min): Train logistic regression probes per layer, identify optimal intervention layer
3. **INLP implementation** (15 min): Iterative nullspace projection at target layer
4. **LEACE implementation** (15 min): Closed-form concept erasure at target layer
5. **Steering vector** (15 min): Compute mean-difference steering vector, apply during inference
6. **Random controls** (10 min): Random-direction projection/steering of matched magnitude
7. **BBQ evaluation** (30 min): Log-likelihood scoring on gender subset with each intervention
8. **CrowS-Pairs evaluation** (20 min): Likelihood comparison on gender subset
9. **Distortion measurement** (10 min): Perplexity on held-out text
10. **Analysis & visualization** (30 min): Statistical tests, figures, tables

### Baselines
1. No intervention (unmodified GPT-2)
2. Random-direction projection (same rank as INLP/LEACE)
3. Random-direction steering (same magnitude as concept steering vector)

### Evaluation Metrics
- **Probe accuracy**: Linear classifier accuracy for gender detection (pre/post intervention)
- **BBQ bias score**: Per BBQ formula, proportion of stereotypical answers in ambiguous contexts
- **CrowS-Pairs**: Proportion of stereotype-consistent likelihood preferences
- **Perplexity**: On held-out text (WikiText-2 subset), measures distortion
- **Causal specificity ratio**: (behavioral_change_targeted - behavioral_change_random) / behavioral_change_targeted

### Statistical Analysis Plan
- Bootstrap confidence intervals (1000 resamples) for all metrics
- Paired permutation tests for method comparisons
- Pearson correlation between probe reduction and behavioral reduction
- Effect sizes (Cohen's d) for intervention vs. control comparisons
- Significance level: α = 0.05 with Bonferroni correction for multiple comparisons

## Expected Outcomes
- **Supports H1**: INLP, LEACE, and steering will show different behavioral bias reduction even when probe reduction is equalized
- **Supports H2**: Random-direction controls will produce 30-60% of the behavioral change seen from targeted interventions
- **Supports H3**: LEACE will show the smallest behavioral change relative to its probe reduction

## Timeline and Milestones
- Phase 2 (Setup): 15 min
- Phase 3 (Implementation): 90 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 30 min

## Potential Challenges
- GPT-2 may show weaker gender bias than larger models → use larger evaluation sets
- CPU inference may be slow → batch efficiently, subsample if needed
- Steering vectors designed for decoder models may need adaptation → follow CAA methodology
- BBQ format may not work well with GPT-2's completion-style → use log-likelihood scoring

## Success Criteria
1. Clear probe accuracy results showing interventions work at the probe level
2. Behavioral evaluation completed on ≥500 BBQ examples and ≥200 CrowS-Pairs examples
3. Random-direction controls run and compared
4. Statistical tests with confidence intervals reported
5. At least 2 publication-quality figures produced
