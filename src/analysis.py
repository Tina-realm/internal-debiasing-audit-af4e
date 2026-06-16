"""Statistical analysis and visualization for the debiasing audit."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import SEED, RESULTS_DIR, FIGURES_DIR, N_BOOTSTRAP, ALPHA

np.random.seed(SEED)


def load_results():
    """Load experiment results."""
    with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
        return json.load(f)


def bootstrap_ci(data, statistic_fn=np.mean, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA):
    """Compute bootstrap confidence interval."""
    data = np.array(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic_fn(sample))
    boot_stats = np.array(boot_stats)
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return float(np.mean(boot_stats)), float(lower), float(upper)


def create_results_table(results: dict) -> pd.DataFrame:
    """Create comprehensive results table."""
    rows = []
    behavioral = results.get('behavioral_results', {})
    interventions = results.get('intervention_results', {})

    methods = ['baseline', 'inlp', 'leace', 'steering',
               'random_direction', 'random_direction_rank1']

    for method in methods:
        row = {'method': method}

        # Probe accuracy
        if method in interventions:
            probe = interventions[method].get('probe', {})
            row['probe_acc'] = probe.get('test_acc', None)
        elif method == 'baseline':
            # Get baseline probe accuracy from probe_results
            probe_res = results.get('probe_results', {})
            best_layer = str(results.get('intervention_layer', 0))
            if best_layer in probe_res:
                row['probe_acc'] = probe_res[best_layer].get('test_acc', None)
        else:
            row['probe_acc'] = None

        # Behavioral metrics
        beh = behavioral.get(method, {})
        if not beh.get('skipped'):
            row['perplexity'] = beh.get('perplexity', None)
            crows = beh.get('crows_pairs', {})
            row['crows_stereo'] = crows.get('stereo_score', None)
            bbq = beh.get('bbq', {})
            row['bbq_amb_acc'] = bbq.get('ambiguous_accuracy', None)
            row['bbq_amb_bias'] = bbq.get('ambiguous_bias_rate', None)
            row['bbq_dis_acc'] = bbq.get('disambiguated_accuracy', None)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def create_causal_table(results: dict) -> pd.DataFrame:
    """Create causal specificity analysis table."""
    causal = results.get('causal_analysis', {})
    rows = []
    for method, data in causal.items():
        rows.append({
            'method': method,
            'behavioral_change': data.get('behavioral_change', 0),
            'random_change': data.get('random_change', 0),
            'specificity_ratio': data.get('specificity_ratio', 0),
            'perplexity_increase': data.get('perplexity_increase', 0),
            'random_ppl_increase': data.get('random_ppl_increase', 0),
        })
    return pd.DataFrame(rows)


def plot_probe_accuracy_per_layer(results: dict):
    """Figure 1: Probe accuracy across layers."""
    probe_res = results.get('probe_results', {})
    layers = []
    accs = []
    for k, v in probe_res.items():
        if k not in ('best_layer', 'best_acc') and isinstance(v, dict):
            layers.append(int(k))
            accs.append(v['cv_mean'])

    layers, accs = zip(*sorted(zip(layers, accs)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, accs, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance (50%)')

    best_layer = results.get('intervention_layer', 0)
    best_idx = list(layers).index(best_layer) if best_layer in layers else 0
    ax.plot(best_layer, accs[best_idx], 's', color='red', markersize=14,
            zorder=5, label=f'Best layer ({best_layer})')

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Gender Probe Accuracy (5-fold CV)', fontsize=14)
    ax.set_title('Gender Detection by Linear Probes Across GPT-2 Layers', fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_probe_accuracy_layers.png'), dpi=150)
    plt.close()
    print("Saved: fig1_probe_accuracy_layers.png")


def plot_probe_vs_behavioral(results: dict):
    """Figure 2: Probe reduction vs behavioral reduction scatter."""
    behavioral = results.get('behavioral_results', {})
    interventions = results.get('intervention_results', {})
    probe_res = results.get('probe_results', {})

    # Get baseline probe accuracy
    best_layer = str(results.get('intervention_layer', 0))
    baseline_probe = probe_res.get(best_layer, {}).get('test_acc', 1.0)
    baseline_crows = behavioral.get('baseline', {}).get('crows_pairs', {}).get('stereo_score', 0.5)

    methods = ['inlp', 'leace', 'steering', 'random_direction', 'random_direction_rank1']
    labels_map = {
        'inlp': 'INLP', 'leace': 'LEACE', 'steering': 'Steering',
        'random_direction': 'Random Dir.', 'random_direction_rank1': 'Random Dir. (r=1)'
    }
    colors = {
        'inlp': '#e74c3c', 'leace': '#3498db', 'steering': '#2ecc71',
        'random_direction': '#95a5a6', 'random_direction_rank1': '#bdc3c7'
    }
    markers = {
        'inlp': 'o', 'leace': 's', 'steering': '^',
        'random_direction': 'x', 'random_direction_rank1': '+'
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in methods:
        probe_acc = interventions.get(method, {}).get('probe', {}).get('test_acc', None)
        crows_score = behavioral.get(method, {}).get('crows_pairs', {}).get('stereo_score', None)

        if probe_acc is None or crows_score is None:
            continue

        # Probe reduction (how much probe accuracy dropped)
        probe_reduction = baseline_probe - probe_acc
        # Behavioral reduction (how much closer to 0.5 = unbiased)
        behavioral_reduction = abs(baseline_crows - 0.5) - abs(crows_score - 0.5)

        ax.scatter(probe_reduction, behavioral_reduction,
                  s=200, c=colors[method], marker=markers[method],
                  label=labels_map[method], zorder=5, edgecolors='black', linewidth=1)

    # Perfect alignment line
    xlim = ax.get_xlim()
    ax.plot([0, 0.5], [0, 0.5], 'k--', alpha=0.3, label='Perfect alignment')

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel('Probe Accuracy Reduction (Δ from baseline)', fontsize=14)
    ax.set_ylabel('Behavioral Bias Reduction (CrowS-Pairs, Δ toward 0.5)', fontsize=14)
    ax.set_title('Probe-Level vs. Behavioral Debiasing:\nThe Signal-Hiding Diagnostic', fontsize=15)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_probe_vs_behavioral.png'), dpi=150)
    plt.close()
    print("Saved: fig2_probe_vs_behavioral.png")


def plot_intervention_vs_random(results: dict):
    """Figure 3: Targeted intervention vs random control behavioral effect."""
    causal = results.get('causal_analysis', {})

    methods = list(causal.keys())
    method_labels = {'inlp': 'INLP', 'leace': 'LEACE', 'steering': 'Steering'}

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.35

    targeted_changes = [causal[m]['behavioral_change'] for m in methods]
    random_changes = [causal[m]['random_change'] for m in methods]
    specificity = [causal[m]['specificity_ratio'] for m in methods]

    bars1 = ax.bar(x - width/2, targeted_changes, width,
                   label='Targeted intervention', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, random_changes, width,
                   label='Random-direction control', color='#e74c3c', alpha=0.7, edgecolor='black')

    # Add specificity ratio labels
    for i, (t, r, s) in enumerate(zip(targeted_changes, random_changes, specificity)):
        ax.text(i, max(t, r) + 0.005, f'Spec: {s:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Intervention Method', fontsize=14)
    ax.set_ylabel('|Δ CrowS-Pairs Score| from Baseline', fontsize=14)
    ax.set_title('Causal Specificity: Targeted vs. Random-Direction Controls', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels.get(m, m) for m in methods], fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_causal_specificity.png'), dpi=150)
    plt.close()
    print("Saved: fig3_causal_specificity.png")


def plot_capability_retention(results: dict):
    """Figure 4: Capability retention (perplexity) vs bias reduction Pareto frontier."""
    behavioral = results.get('behavioral_results', {})

    methods = ['baseline', 'inlp', 'leace', 'steering',
               'random_direction', 'random_direction_rank1']
    labels_map = {
        'baseline': 'Baseline', 'inlp': 'INLP', 'leace': 'LEACE',
        'steering': 'Steering', 'random_direction': 'Random Dir.',
        'random_direction_rank1': 'Random Dir. (r=1)'
    }
    colors = {
        'baseline': '#2c3e50', 'inlp': '#e74c3c', 'leace': '#3498db',
        'steering': '#2ecc71', 'random_direction': '#95a5a6',
        'random_direction_rank1': '#bdc3c7'
    }

    baseline_crows = behavioral.get('baseline', {}).get('crows_pairs', {}).get('stereo_score', 0.5)

    fig, ax = plt.subplots(figsize=(10, 8))

    for method in methods:
        beh = behavioral.get(method, {})
        if beh.get('skipped') or not beh:
            continue

        ppl = beh.get('perplexity', None)
        crows = beh.get('crows_pairs', {}).get('stereo_score', None)

        if ppl is None or crows is None:
            continue

        # Bias = distance from 0.5 (higher = more biased)
        bias_remaining = abs(crows - 0.5)

        ax.scatter(ppl, bias_remaining, s=200, c=colors.get(method, 'gray'),
                  label=labels_map.get(method, method), zorder=5,
                  edgecolors='black', linewidth=1)

    ax.set_xlabel('Perplexity (Lower = Better Capability)', fontsize=14)
    ax.set_ylabel('Remaining Bias |CrowS Score - 0.5| (Lower = Less Bias)', fontsize=14)
    ax.set_title('Capability Retention vs. Bias Reduction\n(Pareto Frontier Analysis)', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add arrow showing ideal direction
    ax.annotate('← Better', xy=(0.02, 0.02), xycoords='axes fraction',
               fontsize=12, color='green', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig4_pareto_frontier.png'), dpi=150)
    plt.close()
    print("Saved: fig4_pareto_frontier.png")


def plot_per_benchmark_breakdown(results: dict):
    """Figure 5: Per-benchmark breakdown across methods."""
    behavioral = results.get('behavioral_results', {})

    methods = ['baseline', 'inlp', 'leace', 'steering',
               'random_direction', 'random_direction_rank1']
    method_labels = {
        'baseline': 'Baseline', 'inlp': 'INLP', 'leace': 'LEACE',
        'steering': 'Steering', 'random_direction': 'Rand. Dir.',
        'random_direction_rank1': 'Rand. Dir.(r=1)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: CrowS-Pairs stereo score
    crows_scores = []
    valid_methods = []
    for m in methods:
        beh = behavioral.get(m, {})
        if not beh.get('skipped') and beh:
            score = beh.get('crows_pairs', {}).get('stereo_score', None)
            if score is not None:
                crows_scores.append(score)
                valid_methods.append(method_labels.get(m, m))

    colors_list = ['#2c3e50', '#e74c3c', '#3498db', '#2ecc71', '#95a5a6', '#bdc3c7']
    axes[0].bar(range(len(crows_scores)), crows_scores,
                color=colors_list[:len(crows_scores)], edgecolor='black')
    axes[0].set_xticks(range(len(valid_methods)))
    axes[0].set_xticklabels(valid_methods, rotation=45, ha='right')
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Unbiased (0.5)')
    axes[0].set_ylabel('Stereotype Score')
    axes[0].set_title('CrowS-Pairs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Panel 2: BBQ ambiguous accuracy
    bbq_accs = []
    valid_methods2 = []
    for m in methods:
        beh = behavioral.get(m, {})
        if not beh.get('skipped') and beh:
            acc = beh.get('bbq', {}).get('ambiguous_accuracy', None)
            if acc is not None:
                bbq_accs.append(acc)
                valid_methods2.append(method_labels.get(m, m))

    axes[1].bar(range(len(bbq_accs)), bbq_accs,
                color=colors_list[:len(bbq_accs)], edgecolor='black')
    axes[1].set_xticks(range(len(valid_methods2)))
    axes[1].set_xticklabels(valid_methods2, rotation=45, ha='right')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('BBQ (Ambiguous)')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Panel 3: Perplexity
    ppls = []
    valid_methods3 = []
    for m in methods:
        beh = behavioral.get(m, {})
        if not beh.get('skipped') and beh:
            ppl = beh.get('perplexity', None)
            if ppl is not None:
                ppls.append(ppl)
                valid_methods3.append(method_labels.get(m, m))

    axes[2].bar(range(len(ppls)), ppls,
                color=colors_list[:len(ppls)], edgecolor='black')
    axes[2].set_xticks(range(len(valid_methods3)))
    axes[2].set_xticklabels(valid_methods3, rotation=45, ha='right')
    axes[2].set_ylabel('Perplexity')
    axes[2].set_title('Model Quality (Perplexity)')
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Per-Benchmark Breakdown Across Intervention Methods', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig5_benchmark_breakdown.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: fig5_benchmark_breakdown.png")


def save_csv_tables(results: dict):
    """Save results as CSV files."""
    # Results table
    df = create_results_table(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'results_table.csv'), index=False)
    print(f"Saved: results_table.csv")
    print(df.to_string(index=False))

    # Causal analysis table
    df_causal = create_causal_table(results)
    df_causal.to_csv(os.path.join(RESULTS_DIR, 'causal_analysis.csv'), index=False)
    print(f"\nSaved: causal_analysis.csv")
    print(df_causal.to_string(index=False))


def run_analysis():
    """Run complete analysis and generate all outputs."""
    print("Loading results...")
    results = load_results()

    print("\n=== Creating CSV Tables ===")
    save_csv_tables(results)

    print("\n=== Generating Figures ===")
    plot_probe_accuracy_per_layer(results)
    plot_probe_vs_behavioral(results)
    plot_intervention_vs_random(results)
    plot_capability_retention(results)
    plot_per_benchmark_breakdown(results)

    print("\n=== Analysis Complete ===")
    print(f"CSV files saved to: {RESULTS_DIR}/")
    print(f"Figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    run_analysis()
