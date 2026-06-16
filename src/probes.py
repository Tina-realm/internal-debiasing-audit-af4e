"""Train linear probes to detect gender in GPT-2 activations."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import json
import os
from config import SEED, RESULTS_DIR, PROBE_TEST_RATIO

np.random.seed(SEED)


def train_probes(activations: dict, labels: np.ndarray) -> dict:
    """Train logistic regression probes at each layer.

    Returns dict with per-layer results: accuracy, weights, best_layer.
    """
    results = {}
    best_acc = 0
    best_layer = 0

    for layer_idx in sorted(activations.keys()):
        X = activations[layer_idx]
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=PROBE_TEST_RATIO, random_state=SEED, stratify=labels
        )

        probe = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
        probe.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, probe.predict(X_train))
        test_acc = accuracy_score(y_test, probe.predict(X_test))

        # Cross-validation for robustness
        cv_scores = cross_val_score(
            LogisticRegression(max_iter=1000, random_state=SEED, C=1.0),
            X, labels, cv=5, scoring='accuracy'
        )

        results[layer_idx] = {
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'weights': probe.coef_[0],  # (hidden_dim,)
            'bias': float(probe.intercept_[0]),
        }

        if test_acc > best_acc:
            best_acc = test_acc
            best_layer = layer_idx

        print(f"  Layer {layer_idx:2d}: train={train_acc:.3f}, test={test_acc:.3f}, "
              f"cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    results['best_layer'] = best_layer
    results['best_acc'] = float(best_acc)

    return results


def save_probe_results(results: dict):
    """Save probe results (without numpy arrays) to JSON."""
    save_dict = {}
    for k, v in results.items():
        if isinstance(v, dict):
            save_dict[str(k)] = {
                kk: vv for kk, vv in v.items() if kk != 'weights'
            }
        else:
            save_dict[str(k)] = v

    with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
        json.dump(save_dict, f, indent=2)


if __name__ == "__main__":
    # Load saved activations
    activations = np.load(os.path.join(RESULTS_DIR, "activations.npy"),
                          allow_pickle=True).item()
    labels = np.load(os.path.join(RESULTS_DIR, "labels.npy"))

    print("Training probes per layer...")
    results = train_probes(activations, labels)
    save_probe_results(results)
    print(f"\nBest layer: {results['best_layer']} "
          f"(test acc: {results['best_acc']:.3f})")
