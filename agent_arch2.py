import numpy as np
from sklearn.linear_model import Lasso
from typing import Tuple

# Paper Constants
C_L1 = 1.0  
KAPPA_SQ = 0.1 

class QSOFULAgent:
    """
    Q-SOFUL Agent (Optimized with Sklearn & Scaled Regularization)
    """
    def __init__(self, d: int, s_star: int, delta_tot: float, Q_total: float, seed: int = 42):
        self.d = d
        self.s_star = s_star
        self.delta_tot = delta_tot
        self.Q_total = Q_total
        self.rng = np.random.default_rng(seed)
        
        # Smoothed Exploration Params
        self.gamma = 1.0 / Q_total
        self.M = self.gamma
        self.sigma_p = 0.5
        
        # State
        self.k = 0
        self.W_k = 1.0
        self.hat_theta = np.zeros(d)
        self.beta_k = 1.0
        
        # History
        self.X_history = []
        self.y_history = []
        self.w_history = []
        
        # --- TUNING PARAMETER ---
        # Theoretical alpha is too conservative for simulation.
        # We scale it down to allow signal detection.
        self.REG_SCALING = 0.1 
        
    def prepare_epoch(self) -> Tuple[float, float, int]:
        self.k += 1
        
        # Schedule
        self.delta_k = (6.0 * self.delta_tot) / (np.pi**2 * self.k**2)
        
        # Accuracy & Weights
        raw_epsilon = 1.0 / np.sqrt(self.W_k)
        self.epsilon_k = min(raw_epsilon, 0.99)
        self.w_k = 1.0 / (self.epsilon_k ** 2)
        
        estimated_n_k = int(20.0 * (1/self.epsilon_k) * np.log(1/self.delta_k))
        return self.epsilon_k, self.delta_k, estimated_n_k

    def select_action(self) -> Tuple[np.ndarray, np.ndarray]:
        # 1. OFU on Safe Set
        safe_radius = 1.0 - self.gamma
        signs = np.sign(self.hat_theta)
        
        # Explore random corners if model is empty (handle zero gradients)
        zero_indices = np.where(signs == 0)[0]
        if len(zero_indices) > 0:
            signs[zero_indices] = self.rng.choice([-1, 1], size=len(zero_indices))
            
        x_base = safe_radius * signs
        
        # 2. Perturbation
        noise = self.rng.normal(0, self.sigma_p, size=self.d)
        noise = np.clip(noise, -self.M, self.M)
        x_played = x_base + noise
        
        return x_base, x_played

    def update_model(self, x_played: np.ndarray, y_hat: float):
        # Update History
        self.X_history.append(x_played)
        self.y_history.append(y_hat)
        self.w_history.append(self.w_k)
        self.W_k += self.w_k
        
        # --- SKLEARN LASSO IMPLEMENTATION ---
        
        # 1. Data Transformation for Weighted Lasso
        # Sklearn minimizes sum((y - Xw)^2) + alpha*|w|.
        # We need sum(weight * (y - Xw)^2).
        # Transform: X_tilde = sqrt(weight) * X, y_tilde = sqrt(weight) * y
        
        weights = np.sqrt(np.array(self.w_history))
        X_arr = np.array(self.X_history)
        y_arr = np.array(self.y_history)
        
        X_tilde = X_arr * weights[:, np.newaxis]
        y_tilde = y_arr * weights
        
        # 2. Calculate Scaled Alpha
        log_term = np.log(2 * self.d / self.delta_k)
        alpha_theory = 2.0 * np.sqrt(2.0 * log_term / self.W_k)
        
        # APPLY SCALING HERE
        alpha_final = self.REG_SCALING * alpha_theory
        
        # 3. Map to Sklearn Alpha
        # Sklearn objective divides by 2*N_samples.
        # Our objective divides by 2*W_k.
        # Equivalence: alpha_sklearn = alpha_final * (N_samples / W_k)
        n_samples = len(y_arr)
        alpha_sklearn = alpha_final * (n_samples / self.W_k)
        
        # 4. Solve
        # warm_start=True speeds up iterative updates
        clf = Lasso(alpha=alpha_sklearn, fit_intercept=False, warm_start=False, max_iter=5000)
        clf.fit(X_tilde, y_tilde)
        
        self.hat_theta = clf.coef_
        
        # Update Confidence Radius
        self.beta_k = C_L1 * (self.s_star / KAPPA_SQ) * alpha_final