import numpy as np
from typing import Tuple
from environment import QuantumOracle

# Check for Qiskit (Optional)
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

class StatisticalAEOracle(QuantumOracle):
    """
    Physics-Faithful Simulator of Quantum Amplitude Estimation (QAE).
    
    Implements 'Median-of-Means' estimation to strictly respect the 
    (epsilon, delta) requirements using the correct query scaling.
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def query(self, true_mean: float, epsilon: float, delta: float) -> Tuple[float, int]:
        # 1. Mapping: Value [-1, 1] -> Probability [0, 1]
        # v = 2p - 1  =>  p = (v + 1) / 2
        true_prob = (true_mean + 1.0) / 2.0
        
        # 2. Precision Adjustment
        # Error in v is 2 * error in p. To bound v-error by epsilon,
        # we need p-error <= epsilon / 2.
        eps_prob = epsilon / 2.0
        
        # 3. Grover Depth (m)
        # Canonical AE: Error ~ pi / (2m + 1).
        # We set m to ensure the main probability mass is within eps_prob.
        m = int(np.ceil(np.pi / eps_prob))
        queries_per_trial = 2 * m + 1
        
        # 4. Success Boosting (Median of Means)
        # Standard QAE has const success prob (approx 0.81).
        # To reach 1-delta, we need k = O(log(1/delta)) trials.
        # C_boost=2.0 is a safe heuristic for the Chernoff bound.
        k_trials = int(np.ceil(2.0 * np.log(1.0 / delta)))
        
        total_cost = k_trials * queries_per_trial
        
        # 5. Simulate Outcomes
        estimates = []
        for _ in range(k_trials):
            # QAE Simulation:
            # 81% chance: Success (Error <= eps_prob)
            # 19% chance: Failure (Heavy tail / Bad measurement)
            is_good_event = self.rng.random() < 0.81
            
            if is_good_event:
                # Bounded error within the main lobe
                noise = self.rng.uniform(-eps_prob, eps_prob)
                p_hat = true_prob + noise
            else:
                # Failure: Return random garbage in [0, 1]
                p_hat = self.rng.uniform(0.0, 1.0)
            
            estimates.append(p_hat)
            
        # 6. Aggregation
        p_median = np.median(estimates)
        
        # Map back to Value [-1, 1]
        v_hat = 2.0 * p_median - 1.0
        
        # Clip to safe bounds (Physics implies probability is never <0 or >1)
        v_hat = np.clip(v_hat, -1.0, 1.0)
        
        return v_hat, total_cost

class HardwareQiskitOracle(QuantumOracle):
    """
    Hardware-Correct Stub.
    Calculates correct circuit parameters but blocks execution 
    to prevent exponential simulation times.
    """
    def __init__(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed.")
            
    def query(self, true_mean: float, epsilon: float, delta: float) -> Tuple[float, int]:
        # Calculate parameters correctly
        eps_prob = epsilon / 2.0
        m = int(np.ceil(np.pi / eps_prob))
        
        # The 'Real' cost of running the circuit
        total_cost = 2 * m + 1
        
        # Block execution
        raise NotImplementedError(
            f"Hardware simulation for epsilon={epsilon:.1e} requires "
            f"simulating a circuit of depth ~{m}. "
            "Use StatisticalAEOracle for tractable experiments."
        )
    

class ClassicalOracle(QuantumOracle):
    """
    True classical Monte Carlo mean estimation.

    Model: we sample a bounded RV V in {-1,+1} with mean true_mean.
    This matches the same p=(v+1)/2 mapping used by AE.
    Cost: N samples, where Hoeffding gives P(|mu_hat-mu|>eps) <= delta.
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def query(self, true_mean: float, epsilon: float, delta: float) -> Tuple[float, int]:
        # Map value mean in [-1,1] to probability in [0,1]
        p = (true_mean + 1.0) / 2.0
        p = float(np.clip(p, 0.0, 1.0))

        # Hoeffding on V in [-1,1]:
        # P(|mu_hat - mu| >= eps) <= 2 exp(- N eps^2 / 2)
        # => N >= (2/eps^2) * log(2/delta)
        N = int(np.ceil((2.0 / (epsilon**2)) * np.log(2.0 / delta)))

        # Safety cap to avoid absurd runtimes in experiments
        N = min(N, int(2e9))

        # Draw Bernoulli samples then map to {-1,+1}
        # V = 2*B - 1 has mean 2p-1 = true_mean
        B = self.rng.binomial(n=1, p=p, size=N)
        V = 2.0 * B - 1.0

        mu_hat = float(np.mean(V))
        mu_hat = float(np.clip(mu_hat, -1.0, 1.0))

        return mu_hat, N
