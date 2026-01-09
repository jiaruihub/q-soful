# Q-SOFUL: Exponential Quantum Speedups for High-Dimensional Sparse Linear Bandits

## 1. Project Overview
This repository contains the reference implementation of **Q-SOFUL** (Quantum Sparse Optimism in the Face of Uncertainty for Linear bandits), as proposed in *Hu (2025)*.

### The Problem
Classical high-dimensional bandit algorithms (like LinUCB) suffer from a "Curse of Dimensionality," with regret scaling as $\tilde{\mathcal{O}}(d\sqrt{T})$. In high-dimensional settings (large $d$), this is prohibitively expensive.

### The Solution
We propose a **quantum-assisted algorithm** that leverages:
1.  **Quantum Mean Estimation (QME):** A subroutine that trades queries for precision ($n \propto 1/\varepsilon$).
2.  **Sparsity:** The assumption that the true parameter $\theta^\star$ is $s^\star$-sparse.
3.  **Smoothed Exploration:** Adding perturbations to actions to guarantee identifying information (Restricted Eigenvalue condition).

### The Main Result
We empirically verify the theoretical query-regret bound derived in the paper:
$$R(Q_{\mathrm{total}}) \;\le\; \tilde{\mathcal{O}}\!\left(\frac{s^\star}{\kappa^2}\sqrt{\log d}\,\log Q_{\mathrm{total}}\right)$$
This result demonstrates **dimension independence** (scaling with $\log d$ instead of $d$) and **exponential quantum speedup** (logarithmic regret in total queries).

## 2. Theoretical Mapping (Paper $\to$ Code)

The codebase is structured to strictly follow the definitions in the paper.

| Concept | Paper Equation | Implementation Details |
| :--- | :--- | :--- |
| **Smoothed Action** | Sec 3.1 | `agent.py`: `select_action` adds truncated Gaussian noise $\xi \in [-M, M]^d$. |
| **Safe Set** | Sec 3.1 | Actions are constrained to $\mathcal{A}' = (1-\gamma)\mathcal{A}$ to keep $x+\xi$ feasible. |
| **Quantum Oracle** | Eq (3-4) | `oracles.py`: Simulates `QMeanEstimate` with cost $n_k \propto \varepsilon_k^{-1} \log(1/\delta_k)$. |
| **Weighted Lasso** | Eq (8) | `agent.py`: Solves $\min \frac{1}{2W_k}\sum w_i(y_i - x_i^\top \theta)^2 + \alpha \|\theta\|_1$ with $w_i = \varepsilon_i^{-2}$. |
| **$L_\infty$ Bonus** | Eq (11) | `agent.py`: UCB Index $= x^\top \hat{\theta} + \beta \|x\|_\infty$. |
| **Geometric Schedule** | Eq (12) | `agent.py`: Updates accuracy $\varepsilon_k = 1/\sqrt{W_{k-1}}$ at each epoch. |

## 3. Experimental Validation

The `experiments_new.ipynb` notebook contains five key experiments that validate the theoretical claims.

### Experiment 1: The "Dimension Kill" (Validation of $\sqrt{\log d}$)
* **Hypothesis:** Unlike classical methods, Q-SOFUL's regret should not explode as dimension $d$ grows.
* **Result:** Comparing Q-SOFUL vs. LinUCB for $d \in [50, 100, 200, 500]$ shows that Q-SOFUL's regret remains nearly flat, validating the $\sqrt{\log d}$ scaling.

### Experiment 2: Horizon Scaling (Validation of $\log Q$)
* **Hypothesis:** Regret should scale logarithmically with the query budget.
* **Result:** A plot of $Regret$ vs $\log(Q_{total})$ yields a linear relationship, confirming the exponential speedup provided by the quantum oracle.

### Experiment 3: Sparsity Robustness ($s^\star$)
* **Hypothesis:** The algorithm should handle varying levels of sparsity.
* **Result:** We varied $s^\star \in [3, 20]$. Regret increases only slightly, showing that the cost of learning the "null space" dominates, making the algorithm robust to small changes in sparsity.

### Experiment 4: Support Recovery Analysis
* **Hypothesis:** The Weighted Lasso should identify the non-zero indices of $\theta^\star$.
* **Result:** Using a threshold of $0.01$ (to filter solver noise), we track the Jaccard Index (IoU). The agent successfully recovers the support set as $Q \to \infty$, though perfect recovery is not strictly required for low regret.

### Experiment 5: Ablation Study (Weighted vs. Unweighted)
* **Hypothesis:** The inverse-variance weighting $w_i = 1/\varepsilon_i^2$ is crucial for handling heterogeneous quantum noise.
* **Result:** We compare **Q-SOFUL** against a **Naive Unweighted** baseline. The Unweighted agent fails to converge efficiently, proving that the weighting scheme is essential for the theoretical guarantee.

## 4. Usage

### Prerequisites
* Python 3.9+
* Dependencies: `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`.

### Running the Code
1.  **Clone the repository.**
2.  **Run the experiments:**
    ```bash
    jupyter notebook experiments_new.ipynb
    ```
3.  **Inspect Source:**
    * `agent.py`: Contains the `QSOFULAgent` and the Weighted Lasso logic.
    * `environment.py`: Manages the high-dimensional linear bandit simulation.
    * `oracles.py`: Contains the `StatisticalAEOracle` (Physics-faithful) and `SimulatedQuantumOracle` (Fast proxy).

## 5. Citation

If you use this code or the theoretical results, please cite:

```bibtex
@article{Hu2025qsoful,
  title={Q-SOFUL: Exponential Quantum Speedups for High-Dimensional Sparse Linear Bandits via $\ell_1$ Confidence and Smoothed Exploration},
  author={Hu, Hubery},
  year={2025}
}