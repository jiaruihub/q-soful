# Q-SOFUL: Empirical Validation of Exponential Quantum Speedups

## 1. Project Objective
This repository contains the reference implementation of **Q-SOFUL** (Quantum Sparse Optimism in the Face of Uncertainty for Linear bandits).

[cite_start]The primary goal is to empirically verify the theoretical bound derived in *Hu (2025)*[cite: 1, 2]:
$$R(Q_{total}) \le \tilde{\mathcal{O}}\left(\frac{s^*}{\kappa^2}\sqrt{\log d} \log Q_{total}\right)$$

Specifically, we aim to demonstrate:
1.  **Dimension Independence:** Regret scales with $\sqrt{\log d}$ rather than $\sqrt{d}$ (as seen in classical LinUCB).
2.  **Quantum Speedup:** Regret scales logarithmically with the query budget ($\log Q_{total}$), implying exponential speedup over classical limits.
3.  **Sparsity Recovery:** The Weighted Lasso correctly identifies the support $S$ of $\theta^*$ despite the heterogeneous noise $\epsilon_k$ induced by the quantum oracle.

## 2. Theoretical Mapping (Paper $\to$ Code)

| Concept | Paper Reference | Code Implementation |
| :--- | :--- | :--- |
| **Interaction Model** | Eq (1) & (5) | `Environment.query_oracle(action, epsilon)` |
| **Quantum Oracle** | Assumption 1 (Eq 3-4) | Simulates noise $\zeta \sim U[-1, 1]$, scaled by $\epsilon$. Cost $n \propto 1/\epsilon$. |
| **Smoothing** | Sec 3.1, Eq (78) | `Agent.perturb_action()` adds truncated $\mathcal{N}(0, \sigma_p^2)$. |
| **Weighted Lasso** | Eq (8) | `cvxpy` optimization with weights $w_i = 1/\epsilon_i^2$. |
| **$l_1$ Confidence** | Lemma 1, Eq (14) | `Agent.beta_k` calculated dynamically based on effective sample size $W_k$. |
| **Schedule** | Eq (10) | Epoch schedule $\epsilon_k = 1/\sqrt{W_{k-1}}$. |

## 3. Implementation Details & Assumptions

### 3.1 The Simulated Quantum Oracle
[cite_start]Since we do not have a physical QPU, we simulate `QMeanEstimate` [cite: 42] classically:
* **Input:** Action $x$, Accuracy $\epsilon$, Failure Prob $\delta$.
* **Process:**
    1. Calculate true mean $\mu = x^\top \theta^*$.
    2. Sample noise $\zeta$ such that $|\zeta| \le 1$. In this repo, we use uniform noise $\zeta \sim \text{Uniform}(-1, 1)$.
    3. Return $\hat{y} = \mu + \epsilon \cdot \zeta$.
    4. Increment global counter `Q_total` by $\lceil \frac{C_{QME}}{\epsilon} \log(1/\delta) \rceil$.

### 3.2 Optimization Solvers
The Weighted Lasso problem (Eq 8) is convex. We use `cvxpy` for rigorous correctness to ensure the KKT conditions (and thus the RE condition logic) are respected.
* **Note:** For very high dimensions ($d > 1000$), we may switch to `sklearn.linear_model.Lasso` (coordinate descent) for speed, ensuring `sample_weight` is set to $1/\epsilon_i^2$.

### 3.3 The Restricted Eigenvalue (RE) Condition
[cite_start]The paper relies on the design matrix satisfying the RE condition on the cone $C(S, 3)$[cite: 134].
* **Sanity Check:** The code includes a `monitor_eigenvalues` flag. If enabled, it computes the minimum eigenvalue of the weighted design matrix restricted to the support $S$ to explicitly verify $\kappa > 0$.

## 4. Running Experiments

### Prerequisites
* Python 3.9+
* `numpy`, `scipy`, `matplotlib`, `cvxpy`, `seaborn`

### Reproduction Steps
1.  **Sanity Check (Low Dimensions):**
    Run `test_lasso_convergence.py` to confirm the Weighted Lasso recovers $\theta^*$ as $Q_{total} \to \infty$ with the geometric schedule.

2.  **Experiment 1: The Dimension Dependence ($d$ scaling)**
    * Fix $s^*=5$, $Q_{total}=10^7$.
    * Vary $d \in [20, 50, 100, 200, 500, 1000]$.
    * **Hypothesis:** Regret should stay roughly constant (or grow very slowly as $\sqrt{\log d}$).
    * **Baseline:** Compare against standard `LinUCB` (Ridge + $l_2$ confidence), which should degrade rapidly ($\sqrt{d}$).

3.  **Experiment 2: Horizon Scaling ($Q_{total}$ scaling)**
    * Fix $d=100$, $s^*=5$.
    * Vary $Q_{total}$ from $10^4$ to $10^8$ (log scale).
    * **Hypothesis:** Plotting Regret vs $\log(Q_{total})$ should yield a linear relationship.

## 5. Numerical Limitations
* **Solver Precision:** As $\epsilon_k \to 0$, weights $w_i \to \infty$. We use log-space calculations where possible, but numerical instability in the Lasso solver may occur at extremely high query counts ($> 10^{10}$).
* **Action Maximization:** Finding $\text{argmax}_{x \in \mathcal{A}'} (x^\top \hat{\theta} + \beta \|x\|_\infty)$ over the hypercube $[-1, 1]^d$ is a convex maximization problem (max of convex function), which is generally NP-hard.
    * *Approximation:* Since the objective is convex, the maximum lies at a vertex. We iterate over the $2^d$ vertices? No, that is exponential.
    * *Relaxation:* In this implementation, we simply check the corners aligned with the sign of $\hat{\theta}$ and random samples, or use a linear programming relaxation if valid. *Correction based on standard literature:* For $l_1$ balls and box constraints, the max is often at a corner. We will implement a heuristic corner search.

## 6. Citation
If you use this code, please cite the original paper:
> Hu, H. (2025). Q-SOFUL: Exponential Quantum Speedups for High-Dimensional Sparse Linear Bandits.
