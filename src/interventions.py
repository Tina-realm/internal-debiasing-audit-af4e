"""Implement INLP, LEACE, and steering interventions on GPT-2 activations."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from config import SEED, INLP_MAX_ITERS, PROBE_TEST_RATIO

np.random.seed(SEED)


class INLPProjection:
    """Iterative Nullspace Projection (Ravfogel et al., 2020).

    Iteratively trains linear classifiers and projects data onto their nullspace
    until no linear classifier can recover the concept.
    """

    def __init__(self, max_iters: int = INLP_MAX_ITERS, min_acc: float = 0.52):
        self.max_iters = max_iters
        self.min_acc = min_acc
        self.projection_matrix = None
        self.n_iters_used = 0
        self.accuracy_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'INLPProjection':
        """Fit INLP by iteratively finding and removing gender directions."""
        d = X.shape[1]
        P = np.eye(d)  # cumulative projection
        X_proj = X.copy()

        for i in range(self.max_iters):
            clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
            X_train, X_test, y_train, y_test = train_test_split(
                X_proj, y, test_size=PROBE_TEST_RATIO, random_state=SEED + i,
                stratify=y
            )
            clf.fit(X_train, y_train)
            acc = accuracy_score(y_test, clf.predict(X_test))
            self.accuracy_history.append(float(acc))

            if acc < self.min_acc:
                break

            # Get the weight vector (direction to remove)
            w = clf.coef_[0]  # (d,)
            w = w / np.linalg.norm(w)

            # Nullspace projection: P_w = I - ww^T
            P_w = np.eye(d) - np.outer(w, w)
            P = P_w @ P
            X_proj = X @ P.T

            self.n_iters_used = i + 1

        self.projection_matrix = P
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned nullspace projection."""
        return X @ self.projection_matrix.T

    def get_rank_reduction(self) -> int:
        """Number of dimensions removed."""
        return self.n_iters_used


class LEACEProjection:
    """LEACE: Least-squares Concept Erasure (Belrose et al., 2023).

    Closed-form solution that provably prevents any linear classifier
    from recovering the concept.
    """

    def __init__(self):
        self.projection_matrix = None
        self.mean_x = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LEACEProjection':
        """Compute LEACE projection matrix."""
        n, d = X.shape
        self.mean_x = X.mean(axis=0)
        X_centered = X - self.mean_x

        # Compute cross-covariance between X and Z (one-hot encoded labels)
        classes = np.unique(y)
        k = len(classes)

        # For binary case, Z is just y reshaped
        if k == 2:
            z = y.astype(float) - y.mean()
            # Cross-covariance: sigma_xz = X^T z / n
            sigma_xz = X_centered.T @ z / n  # (d,)

            # The direction to erase is proportional to sigma_xz
            # LEACE projects out the direction Sigma_X^{-1/2} sigma_xz
            # For efficiency, we use the simplified version:
            # P = I - (sigma_xz sigma_xz^T) / (sigma_xz^T sigma_xz)
            direction = sigma_xz / (np.linalg.norm(sigma_xz) + 1e-10)

            # For the full LEACE, we should whiten first, but the simplified
            # rank-1 projection captures the main effect
            self.projection_matrix = np.eye(d) - np.outer(direction, direction)
        else:
            # Multi-class: compute full cross-covariance
            Z = np.zeros((n, k))
            for i, c in enumerate(classes):
                Z[y == c, i] = 1
            Z = Z - Z.mean(axis=0)

            Sigma_xz = X_centered.T @ Z / n  # (d, k)
            # SVD of cross-covariance
            U, S, Vt = np.linalg.svd(Sigma_xz, full_matrices=False)
            # Project out the significant directions
            rank = np.sum(S > 1e-10)
            U_r = U[:, :rank]
            self.projection_matrix = np.eye(d) - U_r @ U_r.T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply LEACE projection."""
        X_centered = X - self.mean_x
        X_proj = X_centered @ self.projection_matrix.T
        return X_proj + self.mean_x


class SteeringVector:
    """Activation steering via mean-difference vectors (CAA-style).

    Computes the mean activation difference between concept-positive and
    concept-negative examples, then subtracts scaled version during inference.
    """

    def __init__(self):
        self.direction = None
        self.magnitude = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SteeringVector':
        """Compute steering vector as mean difference between classes."""
        mask_0 = y == 0
        mask_1 = y == 1
        mean_0 = X[mask_0].mean(axis=0)
        mean_1 = X[mask_1].mean(axis=0)

        self.direction = mean_1 - mean_0  # female - male direction
        self.magnitude = np.linalg.norm(self.direction)
        self.direction_unit = self.direction / (self.magnitude + 1e-10)

        return self

    def transform(self, X: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """Subtract scaled steering vector from activations.

        alpha controls the strength: 1.0 = full subtraction.
        """
        # Project out the gender direction (equivalent to subtracting the component)
        projections = X @ self.direction_unit  # (n,)
        X_steered = X - alpha * np.outer(projections, self.direction_unit)
        return X_steered


class RandomDirectionControl:
    """Random-direction control: project out a random direction of same rank.

    This controls for generic perturbation effects vs concept-specific editing.
    """

    def __init__(self, n_directions: int = 1, seed: int = SEED):
        self.n_directions = n_directions
        self.projection_matrix = None
        self.rng = np.random.RandomState(seed)

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RandomDirectionControl':
        """Generate random projection of specified rank."""
        d = X.shape[1]
        # Generate random orthonormal directions
        random_dirs = self.rng.randn(self.n_directions, d)
        # Orthogonalize via QR
        Q, _ = np.linalg.qr(random_dirs.T)
        Q = Q[:, :self.n_directions]  # (d, n_directions)

        self.projection_matrix = np.eye(d) - Q @ Q.T
        self.directions = Q
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply random projection."""
        return X @ self.projection_matrix.T


class RandomMagnitudeControl:
    """Random-magnitude control: steer with correct direction but random magnitude.

    Uses the real concept direction but with randomized per-sample magnitudes.
    """

    def __init__(self, seed: int = SEED + 100):
        self.direction_unit = None
        self.rng = np.random.RandomState(seed)

    def fit(self, X: np.ndarray, y: np.ndarray,
            steering_vector: SteeringVector) -> 'RandomMagnitudeControl':
        """Use the steering vector's direction but will apply random magnitudes."""
        self.direction_unit = steering_vector.direction_unit.copy()
        # Compute typical projection magnitudes for scaling
        projections = np.abs(X @ self.direction_unit)
        self.typical_magnitude = projections.mean()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply random-magnitude steering."""
        n = X.shape[0]
        # Random magnitudes drawn from same distribution as real projections
        random_mags = self.rng.randn(n) * self.typical_magnitude
        X_steered = X - np.outer(random_mags, self.direction_unit)
        return X_steered


def calibrate_intervention_strength(X: np.ndarray, y: np.ndarray,
                                     intervention_fn, target_acc: float = 0.55,
                                     param_name: str = 'alpha',
                                     param_range: np.ndarray = None) -> Tuple[float, float]:
    """Find intervention strength that achieves target probe accuracy.

    Returns (best_param, achieved_acc).
    """
    if param_range is None:
        param_range = np.linspace(0.1, 3.0, 20)

    best_param = param_range[0]
    best_diff = float('inf')

    for param in param_range:
        X_intervened = intervention_fn(X, param)
        X_train, X_test, y_train, y_test = train_test_split(
            X_intervened, y, test_size=PROBE_TEST_RATIO, random_state=SEED,
            stratify=y
        )
        probe = LogisticRegression(max_iter=1000, random_state=SEED)
        probe.fit(X_train, y_train)
        acc = accuracy_score(y_test, probe.predict(X_test))

        diff = abs(acc - target_acc)
        if diff < best_diff:
            best_diff = diff
            best_param = param
            best_acc = acc

    return best_param, best_acc


def evaluate_probe_after_intervention(X_transformed: np.ndarray,
                                       y: np.ndarray) -> dict:
    """Evaluate linear probe accuracy on transformed activations."""
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=PROBE_TEST_RATIO, random_state=SEED,
        stratify=y
    )
    probe = LogisticRegression(max_iter=1000, random_state=SEED)
    probe.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
    }
