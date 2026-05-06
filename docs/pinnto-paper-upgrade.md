# Implementation Plan: PINNTO Research Paper — Journal-Ready Upgrade

## Task Type
- [x] Research Paper (Code + Writing)
- [x] Computational Experiments (New code in notebook)
- [x] Document Restructure (Journal formatting)

## Current State Assessment

**Paper**: `docs/PINNTO_FINAL_JOURNAL.md` — ~220 lines, missing critical sections, broken numbering, informal tone
**Code**: `major.ipynb` — 18 cells, functional but covers only one parameter set, no ablation, no Greeks computation

**Core Weakness**: Trains on Black-Scholes analytical prices, tests against Black-Scholes analytical prices. No harder problem. No ablation. No Greeks validation. Reviewers will reject on novelty grounds.

---

## Technical Solution

Transform PINNTO from a single-scenario BS surrogate into a **multi-problem PINN framework** with:
1. **American put option pricing with free boundary** — the killer novelty (no closed-form, genuinely hard)
2. Greeks via autograd (not just SHAP) — the finance-native explainability
3. Comprehensive ablation proving each loss component matters
4. Self-implemented baselines for fair comparison (binomial tree, FD, Longstaff-Schwartz MC)
5. Parametric inputs (sigma, r as network inputs) so one model prices across market conditions

**Key Novelty Shift**: The paper goes from "we reproduced a known formula" to "we solve the American option free boundary problem — where no closed-form exists — using PINNs with complementarity constraints, and validate against three classical numerical methods"

---

## Implementation Steps

### Phase 1: Code Upgrades (Notebook) — HIGH PRIORITY

#### Step 1.1 — Parametric PINN Architecture
**File**: `major.ipynb` (new cells after cell 3)
**Deliverable**: Modified PINN that takes (tau, S, sigma, r) as 4D input

- Modify `PINN.__init__` to accept `in_dim=4`
- Modify `forward(tau, S, sigma, r)` to concatenate all 4 inputs
- Normalize inputs: tau/tau_max, S/S_max, sigma/sigma_ref, r/r_ref
- Update all loss functions to pass sigma, r tensors
- Update `make_data_points` to sample sigma in [0.1, 0.5] and r in [0.01, 0.10]
- Train on varied (sigma, r) — one model prices everything

**Why**: Transforms the contribution from "we solved one known problem" to "we built a parametric surrogate that generalizes across market regimes"

#### Step 1.2 — Greeks via Autograd
**File**: `major.ipynb` (new cell after evaluation)
**Deliverable**: Delta, Gamma, Theta, Vega computed from PINN vs analytical

```python
# Pseudocode
def compute_greeks(model, tau, S, sigma, r):
    C = model(tau, S, sigma, r)
    Delta = autograd(C, S)          # dC/dS
    Gamma = autograd(Delta, S)      # d2C/dS2
    Theta = -autograd(C, tau)       # -dC/dtau
    Vega  = autograd(C, sigma)      # dC/dsigma
    return Delta, Gamma, Theta, Vega
```

- Compare each Greek against analytical BS Greek formulas
- Plot: PINN Greek vs BS Greek for each, across S range at fixed tau
- Compute MAE for each Greek
- Table: Greek | MAE | Max Error | R2

**Why**: This is what a finance audience actually cares about. SHAP shows importance; Greeks show the model learned correct derivatives. Far stronger than SHAP alone.

#### Step 1.3 — Ablation Study
**File**: `major.ipynb` (new section)
**Deliverable**: Table with ~10 experiment rows

Run the following experiments (each full train + eval cycle):

| # | Experiment | What Changes |
|---|-----------|-------------|
| 1 | Baseline | 4 layers, 64 neurons, all losses, current config |
| 2 | Shallow | 2 hidden layers |
| 3 | Deep | 6 hidden layers |
| 4 | Narrow | 32 neurons per layer |
| 5 | Wide | 128 neurons per layer |
| 6 | No PDE loss | lambda_pde = 0 (pure data-driven) |
| 7 | No data loss | lambda_data = 0 (pure physics) |
| 8 | No boundary loss | lambda_bc = 0 |
| 9 | No terminal loss | lambda_term = 0 |
| 10 | Few collocation pts | n_pde = 500 instead of 5000 |
| 11 | Many collocation pts | n_pde = 20000 |
| 12 | High PDE weight | lambda_pde = 10 |

For each: record MAE, RMSE, R2, training time.

**Critical row**: #6 (No PDE loss) — if removing PDE loss barely hurts, the "physics-informed" claim is weak. Must show it matters.

#### Step 1.4 — Loss Weight Sweep
**File**: `major.ipynb` (subsection of ablation)
**Deliverable**: Heatmap or table of lambda_pde vs lambda_data effect on MAE

- Fix lambda_bc = lambda_term = 1.0
- Sweep lambda_data in {0.1, 1, 10} x lambda_pde in {0.1, 1, 10}
- 9 experiments, report MAE for each
- Optionally: implement adaptive loss balancing from Wang et al. [10] and compare

#### Step 1.5 — Self-Implemented Baselines
**File**: `major.ipynb` (new section)
**Deliverable**: 5 baselines evaluated on same hold-out sets

**For European Call (BS validation):**

| Baseline | Implementation |
|----------|---------------|
| Vanilla NN | Same architecture, MSE loss only (no PDE/BC/terminal) |
| Finite Difference | Crank-Nicolson on 200x200 grid |
| Monte Carlo | Already in code — use 100k paths per point |
| DGM-style | Single large network with PDE loss only (no supervised data) |

**For American Put (main contribution):**

| Baseline | Implementation |
|----------|---------------|
| Binomial Tree (CRR) | 10,000+ steps — converges to true price |
| Finite Difference | Implicit FD with Brennan-Schwartz algorithm, 500x1000 grid |
| Longstaff-Schwartz MC | 500k paths, polynomial basis degree 3-5 |

All evaluated on the SAME hold-out sets. Same metrics tables.

#### Step 1.6 — American Put Option with Free Boundary (THE KILLER CONTRIBUTION)
**File**: `major.ipynb` (new major section — ~4-5 new cells)
**Deliverable**: PINNTO pricing American put options where NO closed-form exists

**Mathematical Formulation:**

The American put V(S, t) satisfies the Linear Complementarity Problem (LCP):

```
max[ ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV,  V − max(K−S, 0) ] = 0
```

Three simultaneous conditions (complementarity):
```
(1) ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV  ≥  0   (PDE inequality)
(2) V(S,t) − max(K−S, 0)                        ≥  0   (value ≥ payoff)
(3) [PDE residual] × [V − payoff]                =  0   (complementarity)
```

This defines a **free boundary** S_f(t) — the optimal exercise boundary separating "hold" and "exercise" regions. Below S_f(t), immediate exercise is optimal.

**Implementation approach — Penalty Method:**

```python
# Pseudocode for American PINN loss

def loss_american_complementarity(model, tau, S, params):
    V_pred = model(tau, S)
    payoff = torch.clamp(params.K - S, min=0)  # put payoff
    
    # PDE residual (same autograd machinery as European)
    pde_res = compute_pde_residual(model, tau, S, params)
    
    # Penalty: V must be >= payoff
    exercise_violation = torch.clamp(payoff - V_pred, min=0)
    
    # Penalty: PDE residual must be >= 0 in hold region
    pde_violation = torch.clamp(-pde_res, min=0)
    
    # Complementarity: pde_res * (V - payoff) = 0
    compl_violation = pde_res * (V_pred - payoff)
    
    return (lambda_exercise * torch.mean(exercise_violation**2) +
            lambda_pde_ineq * torch.mean(pde_violation**2) +
            lambda_compl * torch.mean(compl_violation**2))
```

**New loss function for American PINN:**
```
L_american = λ_pde * L_pde_inequality
           + λ_bc  * L_boundary
           + λ_ic  * L_terminal
           + λ_ex  * L_exercise_constraint  (V ≥ payoff)
           + λ_cp  * L_complementarity      (PDE_res × (V-payoff) = 0)
```

**Benchmarking (no closed-form — use three reference methods):**
1. Binomial tree (CRR) with 50,000 steps → reference price
2. Implicit finite difference (Brennan-Schwartz) with 500×2000 grid → reference price
3. Longstaff-Schwartz MC with 500k paths → reference price
4. Verify all three agree within 0.01 → establishes "ground truth"
5. Compare PINNTO against this consensus

**Free boundary extraction:**
- For each time step τ, find S where V_PINN(τ, S) ≈ max(K-S, 0) (exercise boundary)
- Plot extracted boundary vs FD-derived boundary
- Report boundary MAE across time

**Parameters:**
- K=100, r=0.05, σ=0.20 (same as European for comparability)
- S ∈ [1, 200], τ ∈ [0, 1]
- Additional tests: σ ∈ {0.1, 0.2, 0.3, 0.4} to show boundary sensitivity

**Deliverable figures:**
- Fig A: American put price surface (PINN vs FD reference)
- Fig B: Absolute error heatmap
- Fig C: Free boundary S_f(τ) — PINN vs FD vs binomial
- Fig D: Price slices at selected maturities

**Deliverable tables:**
- Table: MAE, RMSE, R², Max Error vs each baseline
- Table: Free boundary accuracy (MAE of S_f across τ)
- Table: Error by moneyness region

**Why this is the game-changer**: 
- No closed-form exists → PINN has genuine utility as a solver, not just a surrogate
- The free boundary problem is actively researched (2023-2025 papers) but NOT saturated
- Most existing PINN-for-American papers use penalty OR variational inequality — we use penalty with complementarity AND extract the free boundary AND validate Greeks, which is a more complete treatment
- The European BS section becomes the "validation" and the American section becomes the "contribution"

#### Step 1.7 — Computational Benchmark Table
**File**: `major.ipynb` (extend cell 17)
**Deliverable**: Wall-clock comparison table for BOTH European and American

**European Call:**

| Method | Offline Cost | Per-Query (5000 pts) | MAE |
|--------|-------------|---------------------|-----|
| BS Analytical | 0 | X ms | 0 (ground truth) |
| PINNTO | Y min (training) | Z ms | 0.113 |
| Finite Difference | 0 | W ms | ? |
| Monte Carlo (100k) | 0 | V ms | ? |

**American Put (this is where PINN shines):**

| Method | Offline Cost | Per-Query (5000 pts) | MAE vs consensus |
|--------|-------------|---------------------|-----|
| PINNTO | Y min (training) | Z ms | ? |
| Binomial (50k steps) | 0 | W ms | baseline |
| Implicit FD (500×2000) | 0 | V ms | baseline |
| Longstaff-Schwartz (500k) | 0 | U ms | baseline |

Show that PINNTO's amortized inference is 10-100x faster than MC/FD for repeated pricing queries (the realistic use case for a trading desk).

---

### Phase 2: Paper Restructure — MEDIUM PRIORITY

#### Step 2.1 — Fix Document Structure
**File**: `docs/PINNTO_FINAL_JOURNAL.md`

New structure:

```
1. Introduction
   - Motivation (2 paragraphs)
   - Contributions list (numbered, 5 items)
   - Paper organization ("The rest of this paper is organized as...")

2. Related Work
   2.1 Classical Option Pricing (BS, binomial, FD, MC)
   2.2 Neural Networks for Option Pricing (vanilla NN, DGM)
   2.3 Physics-Informed Neural Networks (Raissi, Karniadakis, Wang)
   2.4 PINNs in Finance (Dhiman & Hu, Alonso et al.)
   2.5 Free Boundary Problems and American Options (NEW — penalty methods, LCP, variational)
   2.6 Explainability in Financial ML (SHAP, PDP)

3. Methodology
   3.1 Problem Setup and Notation
       3.1.1 European Call (Black-Scholes PDE)
       3.1.2 American Put (Linear Complementarity Problem) — NEW
   3.2 PINN Architecture
   3.3 Physics-Informed Loss Design
       3.3.1 European Loss (data + PDE + BC + terminal)
       3.3.2 American Loss (PDE inequality + exercise + complementarity) — NEW
   3.4 Synthetic Data Generation
   3.5 Training Procedure (specify ALL hyperparameters)
   3.6 Free Boundary Extraction Algorithm — NEW
   3.7 Greeks Computation via Autograd — NEW
   3.8 SHAP Explainability Framework

4. Experiments and Results
   4.1 European Call: Black-Scholes Validation (existing, cleaned up)
       - Loss convergence
       - Pricing surface accuracy
       - Hold-out metrics (Table 1)
       - Error by moneyness (Table 2)
   4.2 Ablation Study (NEW — Table with all experiments)
   4.3 Loss Weight Analysis (NEW)
   4.4 Comparison with Baselines — European (NEW — self-implemented, fair)
   4.5 American Put Option Pricing (NEW — THE MAIN CONTRIBUTION)
       4.5.1 Complementarity loss convergence
       4.5.2 American put price surface vs FD reference
       4.5.3 Error analysis (MAE, RMSE, R² vs three baselines)
       4.5.4 Free boundary extraction and accuracy
       4.5.5 Sensitivity to volatility (σ = 0.1, 0.2, 0.3, 0.4)
   4.6 Greeks Validation (NEW — Delta, Gamma, Theta plots + table)
   4.7 SHAP Analysis
   4.8 Computational Performance (NEW — speed table, European + American)

5. Discussion
   - European validation confirms PINNTO learns correct pricing dynamics
   - American extension demonstrates genuine utility where no closed-form exists
   - Ablation reveals contribution of each physics-informed loss component
   - Free boundary accuracy and its implications for early exercise decisions
   - Comparison with related work

6. Limitations
   - Single underlying asset (no multi-asset baskets)
   - Constant parameters (no regime switching or stochastic volatility)
   - Synthetic/numerical benchmarks only (no market calibration)
   - Penalty method sensitivity to hyperparameter tuning

7. Conclusion and Future Work
   - Summary of contributions
   - Future: multi-asset American options, stochastic volatility,
     real market data calibration, transfer learning across regimes

References (fix all typos)

Appendix A: Full Hyperparameter Table
Appendix B: Additional Figures
```

#### Step 2.2 — Write Contributions Statement
**File**: `docs/PINNTO_FINAL_JOURNAL.md` (end of Section 1)

```
Our contributions are:
(1) A physics-informed neural network framework (PINNTO) that prices
    European call options under the Black-Scholes model with R² > 0.999;
(2) Extension to American put options via a complementarity-constrained
    loss function, solving the free boundary problem where no closed-form
    solution exists, with accurate extraction of the optimal exercise boundary;
(3) Comprehensive ablation study quantifying the contribution of each
    physics-informed loss component to pricing accuracy;
(4) Direct Greeks validation via automatic differentiation, demonstrating
    that learned sensitivities match analytical/numerical derivatives;
(5) SHAP-based interpretability analysis confirming the model learns
    economically rational pricing behavior without explicit enforcement.
```

#### Step 2.3 — Hyperparameter Table
**File**: `docs/PINNTO_FINAL_JOURNAL.md` (Section 3.5)

| Parameter | Value |
|-----------|-------|
| Hidden layers | 4 |
| Neurons per layer | 64 |
| Activation | tanh |
| Output activation | linear |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Epochs | 5000 |
| n_data (supervised points) | 2000 |
| n_pde (collocation points) | 5000 |
| n_terminal | 1000 |
| n_boundary | 1000 |
| lambda_data | 1.0 |
| lambda_pde | 1.0 |
| lambda_bc | 1.0 |
| lambda_term | 1.0 |
| Random seed | 42 |
| Framework | PyTorch |

#### Step 2.4 — Fix All Writing Issues
- Fix equation numbering: start from (1)
- Remove ALL informal language ("This stops the model from going off course", "So, the Black-Scholes model is used to make all of...")
- Fix reference typos: [11] "Un certainty" -> "Uncertainty", [12] "Quan titative" -> "Quantitative"
- Fix repeated `--- | ---` in table formatting
- Remove claim "often normalized" (code doesn't normalize) OR add normalization to code
- Replace "we might also go through a fine-tuning phase" with what actually happens
- Replace "a set learning rate" with "a learning rate of 1e-3"

#### Step 2.5 — Fix SHAP Interpretation
- Remove claim that SHAP values directly correspond to Vega and Theta
- Instead: "The SHAP dependence profile for time-to-maturity exhibits a positive, increasing trend consistent with the theoretical behavior of time value in option pricing"
- Add caveat: "We note that SHAP attributions are not identical to the Greeks but reflect analogous sensitivities"
- Reference the new Greeks section for the rigorous validation

---

### Phase 3: Polish — LOWER PRIORITY

#### Step 3.1 — Add Publication-Quality Figures from Cell 17
- Parity plot (PINN vs BS prices)
- Residual distribution histogram
- MAE by moneyness bar chart
- Reference these as new figures in the paper

#### Step 3.2 — Add New Figures
- Greeks comparison plots (4 panels: Delta, Gamma, Theta, Vega)
- Ablation bar chart (MAE across experiments)
- Loss weight heatmap
- American put price surface + error heatmap
- Free boundary S_f(τ) comparison plot (PINN vs FD vs binomial)
- American put slices at selected maturities

#### Step 3.3 — Reproducibility Statement
Add to paper:
"All experiments were conducted using PyTorch [version] on [hardware]. Training PINNTO for 5000 epochs takes approximately [X] minutes on [GPU/CPU]. The random seed is fixed at 42 for reproducibility. Code is available at [repo URL]."

#### Step 3.4 — Add MAPE to Tables
The code computes MAPE but the paper doesn't report it. Add to Table 1.

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `major.ipynb` (cells 1-3) | Modify | Parametric 4D architecture |
| `major.ipynb` (cell 4) | Modify | Update loss functions for parametric inputs |
| `major.ipynb` (cell 5) | Modify | Update training loop for param sampling |
| `major.ipynb` (new cells) | Create | American put PINN — complementarity loss + training |
| `major.ipynb` (new cells) | Create | American baselines — binomial tree, implicit FD, LSM |
| `major.ipynb` (new cells) | Create | Free boundary extraction + visualization |
| `major.ipynb` (new cells) | Create | Greeks computation + validation |
| `major.ipynb` (new cells) | Create | Ablation study runner |
| `major.ipynb` (new cells) | Create | Loss weight sweep |
| `major.ipynb` (new cells) | Create | European baselines (vanilla NN, FD, MC) |
| `major.ipynb` (cell 17) | Modify | Add computational benchmark |
| `docs/PINNTO_FINAL_JOURNAL.md` | Rewrite | Full restructure per Phase 2 |

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| **American PINN doesn't converge** | The complementarity loss has competing objectives. Mitigation: (1) Start with large penalty weight on exercise constraint, anneal PDE weight up gradually. (2) Use curriculum learning — train on European first, then fine-tune with complementarity. (3) Fall back to pure penalty method (simpler) if full complementarity doesn't work. |
| **Free boundary extraction is noisy** | Neural networks are smooth — the exercise boundary (a discontinuity in the second derivative) may be smeared. Mitigation: (1) Use a fine grid search for the crossing point. (2) Apply smoothing/interpolation. (3) Report boundary accuracy honestly — even 1-2% error is publishable if clearly stated. |
| **Ablation shows PDE loss doesn't help for European** | Expected when training on analytical data. Mitigation: Frame as "PDE loss provides regularization, especially important when supervised data is scarce." Run experiment with n_data=200 to prove this. For American options, PDE loss is essential (no supervised data available), which strengthens the argument. |
| **Baselines (binomial/FD/LSM) disagree with each other** | Use extremely fine resolution for each (50k steps, 500x2000 grid, 1M paths). Verify agreement within 0.01 before comparing PINN. If they disagree, use FD as primary reference (most stable for American puts). |
| **Training 12+ ablation + American experiments takes time** | European ablation: ~1 hour total. American training: possibly 10-20k epochs needed. Budget 1 full day of GPU time. Run experiments in parallel if possible. |
| **Reviewers want real market data** | Acknowledge in Limitations. State: "calibration to market data is a natural extension; our framework validates the learning mechanism against high-fidelity numerical benchmarks before market deployment." |
| **Penalty weight tuning is ad-hoc** | Run a small sweep of λ_exercise ∈ {1, 10, 100, 1000} and report sensitivity. If adaptive weighting (Wang et al. [10]) helps, use it and cite. |

## Execution Order (Priority)

```
Week 1: Step 1.6 (American Put — THE MAIN CONTRIBUTION)
         — Implement complementarity loss, train, get initial results
         — This is the hardest and most important step — start here

Week 2: Step 1.3 (Ablation) + Step 1.5 (All Baselines) + Step 1.2 (Greeks)
         — European ablation + European/American baselines + Greeks validation
         — These provide the supporting evidence

Week 3: Step 1.4 (Loss weights) + Step 1.7 (Speed benchmark) + Step 2.1-2.5 (Full paper restructure)
         — Complete the experiments + rewrite the paper

Week 4: Phase 3 (Polish, figures, reproducibility) + Step 1.1 (Parametric, if time)
         — Final quality pass
         — Parametric 4D input is nice-to-have, not essential
```

## New References to Add

These should be cited in the Related Work section:

- Raissi et al. (2019) — foundational PINN paper [already ref 7]
- Wang et al. (2021) — gradient pathologies in PINNs [already ref 10]
- Longstaff & Schwartz (2001) — "Valuing American Options by Simulation: A Simple Least-Squares Approach"
- Cox, Ross, Rubinstein (1979) — binomial model [already ref 4]
- Recent PINN-for-American papers (2023-2025):
  - "Meshless methods for American option pricing through PINNs" (2023)
  - "Optimal approximations for free boundary problems of fractional BS equations using combined PINN" (2024, Scientific Reports)
  - "A fast and enhanced shallow learning framework for solving free boundary options pricing problems" (2024)
  - "Exact Terminal Condition Neural Network for American Option Pricing" (2024)

## SESSION_ID
- CODEX_SESSION: N/A (wrapper not available)
- GEMINI_SESSION: N/A (wrapper not available)
