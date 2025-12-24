import numpy as np
import cvxpy as cp
from typing import Tuple, List

# Paper Constants (Eq 14, 15)
# In practice, C_l1 and kappa are often tuned or set to 1.0 for experiments
C_L1 = 1.0  
KAPPA_SQ = 0.1 

class QSOFULAgent:
    """
    Q-SOFUL Algorithm Implementation.
    """
    def __init__(self, d: int, s_star: int, delta_tot: float, Q_total: float, seed: int = 42):
        self.d = d
        self.s_star = s_star
        self.delta_tot = delta_tot
        self.Q_total = Q_total
        self.rng = np.random.default_rng(seed)
        
        # Parameters (Smoothed Exploration)
        self.gamma = 1.0 / Q_total  # Safe set shrinkage
        self.M = self.gamma         # Truncation bound
        self.sigma_p = 0.5          # Perturbation scale
        
        # State Initialization (Step 2 of Algo 1)
        self.k = 0                  # Epoch index
        self.W_k = 1.0              # Total weight W_0 = 1
        self.hat_theta = np.zeros(d)
        self.beta_k = 1.0           # Initial confidence radius
        
        # History
        self.X_history = []
        self.y_history = []
        self.w_history = []
        
    def prepare_epoch(self) -> Tuple[float, float, int]:
            """
            Calculates epsilon_k, delta_k, and n_k (Eq 6, 10, 11).
            Increments epoch counter k.
            """
            self.k += 1
            
            # 1. Failure Schedule (Eq 6)
            self.delta_k = (6.0 * self.delta_tot) / (np.pi**2 * self.k**2)
            
            # 2. Accuracy Schedule (Eq 10)
            # We enforce strictly epsilon < 1 to satisfy Assumption 1 (Eq 3)
            raw_epsilon = 1.0 / np.sqrt(self.W_k)
            self.epsilon_k = min(raw_epsilon, 0.99)
            
            # 3. Weight (Eq 10)
            # Recalculate weight based on the *actual* epsilon used
            self.w_k = 1.0 / (self.epsilon_k ** 2)
            
            # 4. Estimated Queries (Eq 11)
            # This is for internal tracking; the environment calculates the true cost.
            estimated_n_k = int(20.0 * (1/self.epsilon_k) * np.log(1/self.delta_k))
            
            return self.epsilon_k, self.delta_k, estimated_n_k  

    def select_action(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements Smoothed Exploration (Sec 3.1 & 3.3).
        1. Select x_k = argmax_{A'} U_{k-1}(x)
        2. Perturb tilde_x_k = x_k + xi_k
        """
        # 1. Base Action Selection (OFU)
        # U(x) = x^T hat_theta + beta * ||x||_inf
        # Safe set A' = (1-gamma)[-1, 1]^d
        # Since ||x||_inf is constant on the boundary of A', 
        # we maximize x^T hat_theta.
        # Max is at corner: (1-gamma) * sign(hat_theta)
        
        safe_radius = 1.0 - self.gamma
        signs = np.sign(self.hat_theta)
        
        # Handle zero gradients: Pick random sign to encourage exploration
        zero_indices = np.where(signs == 0)[0]
        if len(zero_indices) > 0:
            random_signs = self.rng.choice([-1, 1], size=len(zero_indices))
            signs[zero_indices] = random_signs
            
        x_base = safe_radius * signs
        
        # 2. Perturbation (Eq 78)
        # xi ~ TruncatedGaussian(0, sigma, [-M, M])
        noise = self.rng.normal(0, self.sigma_p, size=self.d)
        noise = np.clip(noise, -self.M, self.M)
        
        x_played = x_base + noise
        
        # Sanity Check
        assert np.max(np.abs(x_played)) <= 1.0 + 1e-9
        
        return x_base, x_played

    def update_model(self, x_played: np.ndarray, y_hat: float):
        """
        Solves Weighted Lasso (Eq 8) and updates Confidence Radius.
        """
        # Store Data
        self.X_history.append(x_played)
        self.y_history.append(y_hat)
        self.w_history.append(self.w_k)
        
        # Update Total Weight W_k = W_{k-1} + w_k
        self.W_k += self.w_k
        
        # --- Solve Weighted Lasso ---
        # Regularization alpha_k (Eq 17)
        # alpha_k = 2 * sqrt(2 log(2d/delta_k) / W_k)
        log_term = np.log(2 * self.d / self.delta_k)
        alpha_k = 2.0 * np.sqrt(2.0 * log_term / self.W_k)
        
        # CVXPY Setup
        theta_var = cp.Variable(self.d)
        
        # Weights matrix (diagonal)
        # Constructing full diag matrix is O(d^2), inefficient.
        # We use elementwise multiplication in the loss sum.
        X_arr = np.array(self.X_history)
        y_arr = np.array(self.y_history)
        w_arr = np.array(self.w_history)
        
        # Loss = (1/2W) * sum w_i (y_i - x_i^T theta)^2
        residuals = y_arr - X_arr @ theta_var
        weighted_sse = cp.sum(cp.multiply(w_arr, cp.square(residuals)))
        objective = (1.0 / (2.0 * self.W_k)) * weighted_sse + alpha_k * cp.norm(theta_var, 1)
        
        problem = cp.Problem(cp.Minimize(objective))
        
        try:
            # Use SCS or ECOS. 
            # Note: For d > 500, this becomes slow.
            problem.solve(solver=cp.SCS, eps=1e-4)
    
        except Exception as e:
            print(f"Solver failed at k={self.k}: {e}")
            return # Keep old estimate
            
        if theta_var.value is not None:
            self.hat_theta = theta_var.value
            # Threshold small values to strictly enforce sparsity if desired
            self.hat_theta[np.abs(self.hat_theta) < 1e-5] = 0.0
        
        # --- Update Confidence Radius (Eq 14) ---
        # beta_k = C * (s* / kappa^2) * alpha_k
        self.beta_k = C_L1 * (self.s_star / KAPPA_SQ) * alpha_k