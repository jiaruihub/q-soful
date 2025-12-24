import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple

# Constants from the paper
C_QME = 20.0  # Universal constant for QME queries (Eq 4) [cite: 46]

class QuantumOracle(ABC):
    """
    Abstract Base Class for the Quantum Mean Estimation Oracle.
    Focuses purely on the quantum estimation mechanics (noise and cost).
    """
    @abstractmethod
    def query(self, true_mean: float, epsilon: float, delta: float) -> Tuple[float, int]:
        """
        Performs the QMeanEstimate procedure (Eq 3) given a ground truth mean.
        
        Args:
            true_mean (float): The actual value x^T theta*.
            epsilon (float): Target accuracy (must be in (0, 1)).
            delta (float): Failure probability (must be in (0, 1)).
            
        Returns:
            estimate (float): y_hat such that |y_hat - true_mean| <= epsilon.
            cost (int): Number of oracle queries consumed.
        """
        pass

class SimulatedQuantumOracle(QuantumOracle):
    """. . 
    Fast statistical proxy for the Quantum Oracle.
    
    Simulation Mode: 'Good Event' Regime
    Instead of simulating rare failures, we simulate the conditional distribution 
    where the algorithm succeeds. This ensures the data strictly respects the 
    bounded noise assumption (|y - mean| <= epsilon) required by the Lasso proof.
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def query(self, true_mean: float, epsilon: float, delta: float) -> Tuple[float, int]:
        # 1. Cost Model (Eq 4) [cite: 46]
        # n <= C_QME * (1/eps) * log(1/delta)
        queries_needed = int(np.ceil(C_QME * (1.0 / epsilon) * np.log(1.0 / delta)))
        
        #  2. Estimate Model (Conditioned on Good Event) [cite: 43, 85]
        # We model the noise zeta as Uniform(-1, 1), scaled by epsilon.
        # This guarantees |estimate - true_mean| <= epsilon.
        noise = self.rng.uniform(-epsilon, epsilon)
        estimate = true_mean + noise
        
        # Clip to standard bounded reward range [-1, 1] for physical realism
        estimate = np.clip(estimate, -1.0, 1.0)
        
        return estimate, queries_needed

class LinearBanditEnv:
    """
    The High-Dimensional Sparse Linear Bandit Environment.
    Owns the ground truth theta* and enforces interaction rules.
    """
    def __init__(self, d: int, s_star: int, oracle: QuantumOracle, random_seed: int = 42):
        self.d = d
        self.s_star = s_star
        self.oracle = oracle
        
        # Initialize RNG for the environment (distinct from Oracle if needed)
        self.rng = np.random.default_rng(random_seed)
        
        #  Initialize Sparse Theta* (Ground Truth) [cite: 7, 32]
        self.theta_star = np.zeros(d)
        if s_star > d:
            raise ValueError(f"Sparsity s* ({s_star}) cannot be greater than dimension d ({d})")
            
        support = self.rng.choice(d, s_star, replace=False)
        
        #  Uniform coefficients ensuring ||theta*||_1 <= 1 [cite: 33]
        # We sample in [-1/s*, 1/s*] so the sum of absolute values is <= 1.
        self.theta_star[support] = self.rng.uniform(-1.0/s_star, 1.0/s_star, size=s_star)
        
        # Sanity check
        assert np.linalg.norm(self.theta_star, 1) <= 1.0 + 1e-9

    def get_reward_query(self, action: np.ndarray, epsilon: float, delta: float) -> Tuple[float, int]:
        """
        Interface for the agent to query the environment.
        """
        # Validate Inputs
        if not (0 < epsilon < 1):
             raise ValueError(f"Epsilon must be in (0, 1), got {epsilon}")
        if not (0 < delta < 1):
             raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if action.shape != (self.d,):
            raise ValueError(f"Action shape mismatch. Expected ({self.d},), got {action.shape}")

        #  Validate action bounds (Action set A is [-1, 1]^d) [cite: 32]
        if np.max(np.abs(action)) > 1.0 + 1e-9:
             raise ValueError(f"Action violates infinity norm constraint: ||x||_inf = {np.max(np.abs(action))}")

        #  Compute True Mean (Hidden from agent) [cite: 7]
        # E[r|x] = x^T theta*
        true_mean = np.dot(action, self.theta_star)

        # Delegate physics to Oracle (Option B: pass mean, not theta)
        estimate, cost = self.oracle.query(true_mean, epsilon, delta)
        
        return estimate, cost

    def true_reward(self, action: np.ndarray) -> float:
        """Helper for calculating Regret (Eq 5)[cite: 56]."""
        return np.dot(action, self.theta_star)