"""Main experiment runner: probes, interventions, and behavioral evaluation."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import json
import time
from datetime import datetime

from config import *
from activation_extraction import load_model, generate_gendered_sentences, extract_activations
from probes import train_probes, save_probe_results
from interventions import (
    INLPProjection, LEACEProjection, SteeringVector,
    RandomDirectionControl, RandomMagnitudeControl,
    evaluate_probe_after_intervention
)
from behavioral_eval import (
    load_bbq_gender, load_crows_pairs_gender,
    IntervenedGPT2, evaluate_bbq, evaluate_crows_pairs, compute_perplexity
)

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_perplexity_texts():
    """Get a small set of held-out texts for perplexity measurement."""
    return [
        "The cat sat on the mat and watched the birds fly by.",
        "Scientists have discovered a new species of frog in the Amazon rainforest.",
        "The stock market experienced significant volatility today due to rising interest rates.",
        "A group of students organized a protest against climate change outside city hall.",
        "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
        "After years of research, the team finally published their findings in Nature.",
        "The old library on Main Street has been serving the community for over a century.",
        "Artificial intelligence is transforming how businesses operate across industries.",
        "The marathon runner crossed the finish line after four hours of grueling effort.",
        "Local farmers are adopting sustainable practices to protect the environment.",
        "The symphony orchestra performed Beethoven's Ninth Symphony to a sold-out audience.",
        "Researchers at the university developed a new algorithm for protein folding.",
        "The small town celebrated its annual harvest festival with music and dancing.",
        "Engineers designed a bridge that can withstand earthquakes of magnitude seven.",
        "The documentary explored the history of space exploration from the 1960s to today.",
        "A new study suggests that regular exercise can improve cognitive function in older adults.",
        "The chef prepared a five-course meal using locally sourced ingredients.",
        "Political leaders gathered at the summit to discuss trade agreements.",
        "The novelist spent three years writing her latest book about life in rural America.",
        "Volunteers cleaned up the beach, collecting over two tons of plastic waste.",
    ]


def run_full_experiment():
    """Run the complete experiment pipeline."""
    start_time = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_NAME,
        'seed': SEED,
        'device': DEVICE,
    }

    # =========================================================================
    # PHASE 1: Activation Extraction
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Extracting activations from GPT-2")
    print("=" * 70)

    sentences, labels = generate_gendered_sentences(N_PROBE_SENTENCES)
    model, tokenizer = load_model()

    print(f"\nExtracting activations for {len(sentences)} sentences...")
    activations = extract_activations(model, tokenizer, sentences)

    np.save(os.path.join(RESULTS_DIR, "labels.npy"), labels)
    print(f"Activations extracted: {len(activations)} layers, shape: {activations[0].shape}")

    # =========================================================================
    # PHASE 2: Probe Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Training gender probes per layer")
    print("=" * 70)

    probe_results = train_probes(activations, labels)
    save_probe_results(probe_results)

    best_layer = probe_results['best_layer']
    best_acc = probe_results['best_acc']
    print(f"\nBest layer for gender detection: {best_layer} (acc={best_acc:.3f})")

    all_results['probe_results'] = {
        str(k): {kk: vv for kk, vv in v.items() if kk != 'weights'}
        if isinstance(v, dict) else v
        for k, v in probe_results.items()
    }
    all_results['intervention_layer'] = best_layer

    X = activations[best_layer]

    # =========================================================================
    # PHASE 3: Apply Interventions
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Applying interventions at layer", best_layer)
    print("=" * 70)

    intervention_results = {}

    # --- INLP ---
    print("\n--- INLP ---")
    inlp = INLPProjection(max_iters=INLP_MAX_ITERS)
    inlp.fit(X, labels)
    X_inlp = inlp.transform(X)
    inlp_probe = evaluate_probe_after_intervention(X_inlp, labels)
    print(f"  INLP: {inlp.n_iters_used} iterations, "
          f"probe acc: {inlp_probe['test_acc']:.3f}")
    intervention_results['inlp'] = {
        'probe': inlp_probe,
        'n_iters': inlp.n_iters_used,
        'accuracy_history': inlp.accuracy_history,
    }

    # --- LEACE ---
    print("\n--- LEACE ---")
    leace = LEACEProjection()
    leace.fit(X, labels)
    X_leace = leace.transform(X)
    leace_probe = evaluate_probe_after_intervention(X_leace, labels)
    print(f"  LEACE: probe acc: {leace_probe['test_acc']:.3f}")
    intervention_results['leace'] = {
        'probe': leace_probe,
    }

    # --- Steering Vector ---
    print("\n--- Steering Vector ---")
    sv = SteeringVector()
    sv.fit(X, labels)
    X_steered = sv.transform(X, alpha=1.0)
    sv_probe = evaluate_probe_after_intervention(X_steered, labels)
    print(f"  Steering (alpha=1.0): probe acc: {sv_probe['test_acc']:.3f}")

    # Try different alpha values to find one matching LEACE's probe reduction
    print("  Calibrating steering strength...")
    best_alpha = 1.0
    best_diff = abs(sv_probe['test_acc'] - leace_probe['test_acc'])
    for alpha in np.arange(0.5, 5.0, 0.25):
        X_s = sv.transform(X, alpha=alpha)
        p = evaluate_probe_after_intervention(X_s, labels)
        diff = abs(p['test_acc'] - leace_probe['test_acc'])
        if diff < best_diff:
            best_diff = diff
            best_alpha = alpha
            best_sv_probe = p

    X_steered = sv.transform(X, alpha=best_alpha)
    sv_probe_calibrated = evaluate_probe_after_intervention(X_steered, labels)
    print(f"  Steering (alpha={best_alpha:.2f}): probe acc: {sv_probe_calibrated['test_acc']:.3f}")
    intervention_results['steering'] = {
        'probe': sv_probe_calibrated,
        'alpha': float(best_alpha),
        'magnitude': float(sv.magnitude),
    }

    # --- Random Direction Control ---
    print("\n--- Random Direction Control ---")
    n_dirs = max(1, inlp.n_iters_used)  # Match INLP's rank reduction
    rand_ctrl = RandomDirectionControl(n_directions=n_dirs, seed=SEED + 200)
    rand_ctrl.fit(X)
    X_random = rand_ctrl.transform(X)
    random_probe = evaluate_probe_after_intervention(X_random, labels)
    print(f"  Random direction ({n_dirs} dirs): probe acc: {random_probe['test_acc']:.3f}")
    intervention_results['random_direction'] = {
        'probe': random_probe,
        'n_directions': n_dirs,
    }

    # --- Random Direction (rank-1, matching LEACE) ---
    print("\n--- Random Direction Control (rank-1, matching LEACE) ---")
    rand_ctrl_1 = RandomDirectionControl(n_directions=1, seed=SEED + 300)
    rand_ctrl_1.fit(X)
    X_random_1 = rand_ctrl_1.transform(X)
    random1_probe = evaluate_probe_after_intervention(X_random_1, labels)
    print(f"  Random direction (1 dir): probe acc: {random1_probe['test_acc']:.3f}")
    intervention_results['random_direction_rank1'] = {
        'probe': random1_probe,
    }

    # --- Random Magnitude Control ---
    print("\n--- Random Magnitude Control ---")
    rand_mag = RandomMagnitudeControl(seed=SEED + 400)
    rand_mag.fit(X, labels, sv)
    X_rand_mag = rand_mag.transform(X)
    rand_mag_probe = evaluate_probe_after_intervention(X_rand_mag, labels)
    print(f"  Random magnitude: probe acc: {rand_mag_probe['test_acc']:.3f}")
    intervention_results['random_magnitude'] = {
        'probe': rand_mag_probe,
    }

    all_results['intervention_results'] = intervention_results

    # =========================================================================
    # PHASE 4: Behavioral Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Behavioral Evaluation")
    print("=" * 70)

    # Load evaluation data
    bbq_examples = load_bbq_gender()
    crows_examples = load_crows_pairs_gender()

    perplexity_texts = get_perplexity_texts()

    behavioral_results = {}

    # Define intervention configurations
    interventions = {
        'baseline': {
            'projection_matrix': None,
            'steering_vector': None,
            'desc': 'No intervention (baseline)',
        },
        'inlp': {
            'projection_matrix': inlp.projection_matrix,
            'steering_vector': None,
            'desc': f'INLP ({inlp.n_iters_used} iterations)',
        },
        'leace': {
            'projection_matrix': leace.projection_matrix,
            'steering_vector': None,
            'desc': 'LEACE (closed-form)',
        },
        'steering': {
            'projection_matrix': None,
            'steering_vector': sv.direction_unit,
            'steering_alpha': best_alpha,
            'desc': f'Steering (alpha={best_alpha:.2f})',
        },
        'random_direction': {
            'projection_matrix': rand_ctrl.projection_matrix,
            'steering_vector': None,
            'desc': f'Random direction ({n_dirs} dirs)',
        },
        'random_direction_rank1': {
            'projection_matrix': rand_ctrl_1.projection_matrix,
            'steering_vector': None,
            'desc': 'Random direction (rank-1)',
        },
        'random_magnitude': {
            # For random magnitude, we need special handling since it's per-sample
            'projection_matrix': None,
            'steering_vector': None,
            'desc': 'Random magnitude (correct dir)',
            'skip_behavioral': True,  # Can't easily apply per-sample in hook
        },
    }

    for name, config in interventions.items():
        if config.get('skip_behavioral'):
            print(f"\n--- {config['desc']} (probe-only, skipping behavioral) ---")
            behavioral_results[name] = {'skipped': True}
            continue

        print(f"\n--- Evaluating: {config['desc']} ---")

        # Create intervened model wrapper
        wrapper = IntervenedGPT2(
            model=model,
            tokenizer=tokenizer,
            intervention_layer=best_layer,
            projection_matrix=config.get('projection_matrix'),
            steering_vector=config.get('steering_vector'),
            steering_alpha=config.get('steering_alpha', 1.0),
        )

        if config['projection_matrix'] is not None or config['steering_vector'] is not None:
            wrapper.register_hooks()

        # Perplexity
        print(f"  Computing perplexity...")
        ppl = compute_perplexity(wrapper, perplexity_texts)
        print(f"  Perplexity: {ppl:.2f}")

        # CrowS-Pairs (faster, do first)
        print(f"  Evaluating CrowS-Pairs...")
        crows_result = evaluate_crows_pairs(wrapper, crows_examples)
        print(f"  CrowS-Pairs stereo score: {crows_result['stereo_score']:.3f}")

        # BBQ
        print(f"  Evaluating BBQ...")
        bbq_result = evaluate_bbq(wrapper, bbq_examples)
        print(f"  BBQ ambiguous accuracy: {bbq_result.get('ambiguous_accuracy', 'N/A')}")

        behavioral_results[name] = {
            'perplexity': ppl,
            'crows_pairs': crows_result,
            'bbq': bbq_result,
        }

        wrapper.remove_hooks()

    all_results['behavioral_results'] = behavioral_results

    # =========================================================================
    # PHASE 5: Compute Causal Specificity Ratios
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Computing causal specificity ratios")
    print("=" * 70)

    causal_results = {}

    baseline_crows = behavioral_results.get('baseline', {}).get('crows_pairs', {}).get('stereo_score', 0.5)
    baseline_ppl = behavioral_results.get('baseline', {}).get('perplexity', 0)

    for method in ['inlp', 'leace', 'steering']:
        method_crows = behavioral_results.get(method, {}).get('crows_pairs', {}).get('stereo_score', 0.5)
        method_ppl = behavioral_results.get(method, {}).get('perplexity', 0)

        # Use the appropriate random control
        if method == 'steering':
            rand_key = 'random_direction_rank1'
        else:
            rand_key = 'random_direction'

        rand_crows = behavioral_results.get(rand_key, {}).get('crows_pairs', {}).get('stereo_score', 0.5)
        rand_ppl = behavioral_results.get(rand_key, {}).get('perplexity', 0)

        # Behavioral change from intervention
        delta_method = abs(method_crows - baseline_crows)
        delta_random = abs(rand_crows - baseline_crows)

        # Causal specificity ratio: how much of the behavioral change is specific
        # to the concept vs generic perturbation
        if delta_method > 0:
            specificity_ratio = max(0, (delta_method - delta_random) / delta_method)
        else:
            specificity_ratio = 0.0

        # Perplexity increase (distortion)
        ppl_increase_method = method_ppl - baseline_ppl
        ppl_increase_random = rand_ppl - baseline_ppl

        causal_results[method] = {
            'behavioral_change': float(delta_method),
            'random_change': float(delta_random),
            'specificity_ratio': float(specificity_ratio),
            'perplexity_increase': float(ppl_increase_method),
            'random_ppl_increase': float(ppl_increase_random),
        }

        print(f"\n  {method}:")
        print(f"    Behavioral Δ (CrowS): {delta_method:.4f}")
        print(f"    Random Δ (CrowS):     {delta_random:.4f}")
        print(f"    Specificity ratio:     {specificity_ratio:.3f}")
        print(f"    Perplexity increase:   {ppl_increase_method:.2f}")

    all_results['causal_analysis'] = causal_results

    # =========================================================================
    # Save All Results
    # =========================================================================
    elapsed = time.time() - start_time
    all_results['elapsed_seconds'] = elapsed
    print(f"\n{'=' * 70}")
    print(f"Experiment completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'=' * 70}")

    # Save results (convert numpy types for JSON serialization)
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    results_clean = convert_numpy(all_results)

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/all_results.json")

    return all_results


if __name__ == "__main__":
    results = run_full_experiment()
