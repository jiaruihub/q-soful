import numpy as np
from typing import Tuple, Optional, Literal

# Try fast Lasso solver first
try:
    from sklearn.linear_model import Lasso
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False


PerturbationType = Literal["uniform", "trunc_gaussian"]


class QSOFULAgent:
    """
    Q-SOFUL agent with the SAME API your experiment expects:
      - prepare_epoch() -> (epsilon_k, delta_k, estimated_n_k)
      - select_action() -> (x_base, x_played)
      - update_model(x_played, y_hat) -> updates hat_theta and beta_k

    Key practical changes vs your previous version:
      - gamma and M default to constants (not 1/Q_total)
      - perturbation defaults to uniform in [-M, M]^d (zero-mean, bounded)
      - alpha_scale knob to avoid "all-zeros" Lasso in finite simulations
    """

    def __init__(
        self,
        d: int,
        s_star: int,
        delta_tot: float,
        Q_total,
        seed: int = 42,
        # confidence constants (tunable in sims)
        C_l1: float = 24.0,
        kappa_sq: float = 0.1,  # RE constant squared (simulation knob)
        # smoothing parameters (IMPORTANT for learning)
        gamma: float = 0.02,    # safe-set shrink (paper requires 0 < gamma < 1)
        M: Optional[float] = None,  # perturbation bound (must satisfy 0 < M <= gamma)

        perturbation: PerturbationType = "trunc_gaussian",
        sigma_p: float = 0.05,  # only used for trunc_gaussian; choose comparable to M
        # schedules / numerics
        eps_clip: float = 0.999999,   # env requires epsilon < 1
        theta_threshold: float = 1e-6,
        alpha_scale: float = 0.25,    # set to 1.0 for strict paper scaling
        # solver choices
        prefer_sklearn: bool = True,
        lasso_max_iter: int = 20000,
        lasso_tol: float = 1e-5,
        cvxpy_solver: str = "SCS",
        cvxpy_eps: float = 1e-4,
    ):
        self.d = int(d)
        self.s_star = int(s_star)

        if not (0 < delta_tot < 1):
            raise ValueError("delta_tot must be in (0,1).")
        self.delta_tot = float(delta_tot)

        # Q_total sometimes passed as float (e.g., 1e8). Normalize.
        self.Q_total = int(Q_total)

        self.rng = np.random.default_rng(seed)

        # Smoothed exploration: safe set A'=(1-gamma)A and truncation M <= gamma
        self.gamma = float(gamma)
        if not (0 < self.gamma < 1):
            raise ValueError("gamma must be in (0,1).")

        self.M = float(M) if M is not None else self.gamma
        if not (0 < self.M <= self.gamma):
            raise ValueError("Need 0 < M <= gamma to guarantee feasibility after perturbation.")

        self.perturbation: PerturbationType = perturbation
        self.sigma_p = float(sigma_p)

        # Confidence / RE knobs
        self.C_l1 = float(C_l1)
        self.kappa_sq = float(kappa_sq)

        # Numerics
        self.eps_clip = float(eps_clip)
        self.theta_threshold = float(theta_threshold)
        self.alpha_scale = float(alpha_scale)

        # Solver
        self.prefer_sklearn = bool(prefer_sklearn)
        self.lasso_max_iter = int(lasso_max_iter)
        self.lasso_tol = float(lasso_tol)
        self.cvxpy_solver = str(cvxpy_solver)
        self.cvxpy_eps = float(cvxpy_eps)

        self.reset()

    def reset(self) -> None:
        # Algorithm init: k=0, W0=1, theta_hat=0
        self.k = 0
        self.W_k = 1.0
        self.hat_theta = np.zeros(self.d, dtype=float)

        # beta_0 just affects the very first base action; not critical
        self.beta_k = 1.0

        # per-epoch cached values set by prepare_epoch()
        self.epsilon_k: Optional[float] = None
        self.delta_k: Optional[float] = None
        self.w_k: Optional[float] = None

        # history buffers for regression
        self.X_history = []
        self.y_history = []
        self.w_history = []

    # ---------------------------
    # Step 1: schedules
    # ---------------------------
    def prepare_epoch(self) -> Tuple[float, float, int]:
        """
        Computes epsilon_k, delta_k, w_k.
        Returns (epsilon_k, delta_k, estimated_n_k).

        Note: the experiment should charge budget using the actual cost returned
        from the oracle/env, not estimated_n_k.
        """
        self.k += 1

        # delta_k = 6 delta_tot / (pi^2 k^2)
        self.delta_k = (6.0 * self.delta_tot) / (np.pi ** 2 * (self.k ** 2))

        # epsilon_k = 1/sqrt(W_{k-1})  (we store W_{k-1} in self.W_k before update_model)
        raw_eps = 1.0 / np.sqrt(max(self.W_k, 1.0))
        self.epsilon_k = min(raw_eps, self.eps_clip)  # enforce < 1 for env validation

        # weight w_k = 1/epsilon_k^2 (consistent even if clipped)
        self.w_k = 1.0 / (self.epsilon_k ** 2)

        # estimated quantum query cost (your env/oracle returns the true cost anyway)
        estimated_n_k = int(np.ceil(20.0 * (1.0 / self.epsilon_k) * np.log(1.0 / self.delta_k)))

        return self.epsilon_k, self.delta_k, estimated_n_k

    # ---------------------------
    # Step 2: action selection
    # ---------------------------
    def select_action(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Smoothed OFU:
          - base action x_base in safe set A'=(1-gamma)[-1,1]^d
          - played action x_played = x_base + xi, xi in [-M,M]^d (zero-mean)
        """
        safe_radius = 1.0 - self.gamma

        # Over a hypercube, maximizing x^T hat_theta + beta||x||_inf picks a corner;
        # beta term is effectively constant at the maximizer, so sign(hat_theta) is fine.
        signs = np.sign(self.hat_theta)
        zero_idx = np.where(signs == 0.0)[0]
        if zero_idx.size > 0:
            signs[zero_idx] = self.rng.choice([-1.0, 1.0], size=zero_idx.size)

        x_base = safe_radius * signs

        if self.perturbation == "uniform":
            noise = self.rng.uniform(-self.M, self.M, size=self.d)
        elif self.perturbation == "trunc_gaussian":
            noise = self.rng.normal(0.0, self.sigma_p, size=self.d)
            noise = np.clip(noise, -self.M, self.M)
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation}")

        x_played = x_base + noise

        # By construction (M <= gamma), x_played is feasible; keep an assert as a bug-catcher.
        if np.max(np.abs(x_played)) > 1.0 + 1e-9:
            # If this ever triggers, something is inconsistent; clip to keep env happy.
            x_played = np.clip(x_played, -1.0, 1.0)

        return x_base.astype(float), x_played.astype(float)

    # ---------------------------
    # Step 3: weighted Lasso update
    # ---------------------------
    def update_model(self, x_played: np.ndarray, y_hat: float) -> None:
        """
        Append (x_played, y_hat, w_k) and solve weighted Lasso:
          min_theta (1/(2W)) sum_i w_i (y_i - x_i^T theta)^2 + alpha_k ||theta||_1
        then update beta_k.

        IMPORTANT: update_model must be called with the *played* action x_played.
        """
        if self.epsilon_k is None or self.delta_k is None or self.w_k is None:
            raise RuntimeError("Call prepare_epoch() before update_model().")

        x_played = np.asarray(x_played, dtype=float).reshape(-1)
        if x_played.shape != (self.d,):
            raise ValueError(f"x_played must have shape ({self.d},), got {x_played.shape}.")

        self.X_history.append(x_played)
        self.y_history.append(float(y_hat))
        self.w_history.append(float(self.w_k))

        # total weight update: W_k = W_{k-1} + w_k
        self.W_k += float(self.w_k)

        # alpha_k: score bound scaling ~ sqrt(log(d/delta)/W_k)
        # Keep a practical alpha_scale knob for finite simulations.
        log_term = np.log((2.0 * self.d) / max(self.delta_k, 1e-300))
        alpha_k = 2.0 * np.sqrt((2.0 * log_term) / max(self.W_k, 1e-12))
        alpha_k *= max(self.alpha_scale, 0.0)

        theta_hat = self._solve_weighted_lasso(alpha_k)
        if theta_hat is not None:
            theta_hat = np.asarray(theta_hat, dtype=float).reshape(-1)
            theta_hat[np.abs(theta_hat) < self.theta_threshold] = 0.0
            self.hat_theta = theta_hat

        # beta_k ∝ (s*/kappa^2) * alpha_k
        self.beta_k = self.C_l1 * (self.s_star / max(self.kappa_sq, 1e-12)) * alpha_k

    def _solve_weighted_lasso(self, alpha_k: float) -> Optional[np.ndarray]:
        """
        Solve weighted Lasso via rescaling to unweighted form:
          (1/(2W)) sum_i w_i (y_i - x_i^T theta)^2 + alpha||theta||_1
        becomes
          0.5 ||yw - Xw theta||_2^2 + alpha||theta||_1
        where rows are scaled by sqrt(w_i/W).
        """
        X = np.vstack(self.X_history)               # (n, d)
        y = np.asarray(self.y_history, dtype=float) # (n,)
        w = np.asarray(self.w_history, dtype=float) # (n,)
        n = X.shape[0]
        W = float(self.W_k)

        scale = np.sqrt(w / max(W, 1e-12))
        Xw = X * scale[:, None]
        yw = y * scale

        # Fast path: sklearn
        if self.prefer_sklearn and _HAS_SKLEARN:
            # sklearn objective: (1/(2n))||yw - Xw theta||^2 + alpha_sklearn||theta||_1
            # Our objective: 0.5||...||^2 + alpha_k||theta||_1
            # => alpha_sklearn = alpha_k / n
            alpha_sklearn = alpha_k / max(n, 1)
            model = Lasso(
                alpha=alpha_sklearn,
                fit_intercept=False,
                max_iter=self.lasso_max_iter,
                tol=self.lasso_tol,
            )
            model.fit(Xw, yw)
            return model.coef_.copy()

        # Fallback: cvxpy
        if _HAS_CVXPY:
            theta = cp.Variable(self.d)
            resid = yw - Xw @ theta
            obj = 0.5 * cp.sum_squares(resid) + alpha_k * cp.norm1(theta)
            prob = cp.Problem(cp.Minimize(obj))
            try:
                prob.solve(solver=getattr(cp, self.cvxpy_solver), eps=self.cvxpy_eps, warm_start=True)
            except Exception:
                return None
            if theta.value is None:
                return None
            return np.array(theta.value).reshape(-1)

        raise RuntimeError("No Lasso solver available. Install scikit-learn or cvxpy.")
