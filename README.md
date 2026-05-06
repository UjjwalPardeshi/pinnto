# PINNTO — Physics-Informed Neural Networks for Trading Options

A physics-informed neural network framework that prices **European call** and **American put** options by embedding the Black–Scholes PDE (and its linear complementarity formulation) directly into the training loss. Mesh-free, differentiable, and validated against classical solvers.

> **Paper:** *PINNTO — Physics-Informed Neural Networks for Trading Options*
> Ujjwal Pardeshi · Mehul Ashra · Asha Abraham
> Dept. of Networking and Communications, SRM Institute of Science and Technology, Kattankulathur, Chennai, India

---

## Highlights

- **European call (Black–Scholes PDE):** MAE = 0.1041, R² = 0.999980 on out-of-sample grid.
- **American put (LCP):** Complementarity-constrained loss — no closed-form solution required. Benchmarked against binomial tree, implicit finite difference, and Longstaff–Schwartz Monte Carlo.
- **Free boundary:** Optimal exercise boundary extracted directly from the trained network.
- **Greeks:** Δ, Γ, Θ computed via PyTorch autograd, validated against analytical Black–Scholes derivatives.
- **Explainability:** SHAP analysis confirms economically rational pricing behaviour.
- **Robustness:** Multi-seed runs, ablation study, and loss-weight sweep.
- **Resumable training:** Checkpointed — close your laptop, rerun, and pick up where you left off.

---

## Repository Layout

```
pinnto/
├── run_experiments.py        # Single-file experiment runner (all 10 experiments + figures)
├── major.ipynb               # Original Colab notebook (exploratory)
├── docs/
│   ├── PINNTO_FINAL_JOURNAL.md     # Journal manuscript (Markdown)
│   ├── PINNTO_FINAL_JOURNAL.docx   # Journal manuscript (Word)
│   └── pinnto-paper-upgrade.md     # Revision notes
├── figures/                  # Generated figures (PNG + PDF)
│   ├── fig1_european_convergence.*
│   ├── fig2_european_surface.*
│   ├── fig4_ablation.*
│   ├── fig5_weight_sweep.*
│   ├── fig6_american_convergence.*
│   ├── fig7_american_surface.*
│   ├── fig9_free_boundary.*
│   ├── fig10_greeks_european.*
│   ├── fig10b_greeks_american.*
│   ├── fig11_shap.*
│   └── fig12_multi_seed.*
├── checkpoints/              # Auto-saved model & result caches (gitignored)
└── README.md
```

---

## Methodology Summary

### European Call

Solves the Black–Scholes PDE
```
∂C/∂τ = ½σ²S² ∂²C/∂S² + rS ∂C/∂S − rC
```
with composite loss:
```
L = λ_d · L_data + λ_p · L_PDE + λ_b · L_BC + λ_t · L_terminal
```

### American Put (Linear Complementarity Problem)

Solves the LCP
```
max[ ∂V/∂τ − L_BS[V],  V − max(K − S, 0) ] = 0
```
with curriculum-guided loss:
```
L = λ_ineq · L_PDE-ineq + λ_ex · L_exercise + λ_cp · L_compl + λ_b · L_BC + λ_t · L_terminal
```

Finite-difference supervision data is used early in training with a decaying weight; physics-based losses gradually take over.

### Architecture

| Parameter | European Call | American Put |
| --- | --- | --- |
| Hidden layers | 4 | 5 |
| Neurons / layer | 64 | 128 |
| Activation | tanh | tanh |
| Optimizer | Adam (lr=1e-3) | Adam (lr=1e-3) + Cosine LR |
| Input normalization | No | Yes |

---

## Installation

```bash
git clone <repo-url>
cd pinnto
python3 -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib seaborn shap scikit-learn
```

GPU is automatically detected via `torch.cuda.is_available()`. Falls back to CPU.

---

## Quick Start

Run the full experimental pipeline (10 experiments + 11 figures):

```bash
python3 run_experiments.py
```

Training is fully checkpointed. Interrupt at any time and rerun — it resumes from the last saved epoch and reuses cached results in `checkpoints/`.

Figures are written to `figures/` as both PNG and PDF.

---

## Experiments

| # | Experiment | Output |
|---|---|---|
| 1 | European PINN training | `fig1_european_convergence`, `fig2_european_surface` |
| 2 | Ablation study (loss components) | `fig4_ablation` |
| 3 | Loss-weight sweep | `fig5_weight_sweep` |
| 4 | American PINN training | `fig6_american_convergence`, `fig7_american_surface` |
| 5 | Free boundary extraction | `fig9_free_boundary` |
| 6 | European baselines (Vanilla NN, Monte Carlo) | console |
| 7 | Speed benchmark (PINN vs BS vs MC) | console |
| 8 | SHAP explainability | `fig11_shap` |
| 9 | Greeks validation (European + American) | `fig10_greeks_european`, `fig10b_greeks_american` |
| 10 | Multi-seed robustness (5 seeds) | `fig12_multi_seed` |

---

## Key Results

**European Call — out-of-sample evaluation grid**

| Method | MAE | RMSE |
|---|---|---|
| **PINNTO** | **0.1041** | low |
| Vanilla NN | higher | higher |
| Monte Carlo (50k paths) | higher | higher |

**Speed:** PINN inference is several orders of magnitude faster than Monte Carlo for batch pricing.

**American Put:** Free boundary recovered from the network closely tracks the finite-difference reference. Greeks (Δ, Γ, Θ) computed via autograd match FD baselines across τ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.

See `docs/PINNTO_FINAL_JOURNAL.md` for full numerical tables.

---

## Reproducibility

- All training data is **synthetic** (Black–Scholes for European, finite difference for American supervision).
- Multi-seed experiment uses seeds `[42, 123, 456, 789, 1337]`.
- Checkpoints (`checkpoints/*.pt`) and pickled results (`checkpoints/*.pkl`) make every figure regenerable without retraining.

---

## Citation

If you use PINNTO in your work, please cite:

```bibtex
@article{pardeshi2026pinnto,
  title   = {PINNTO: Physics-Informed Neural Networks for Trading Options},
  author  = {Pardeshi, Ujjwal and Ashra, Mehul and Abraham, Asha},
  year    = {2026},
  journal = {Manuscript under review}
}
```

---

## Authors

- **Ujjwal Pardeshi** — up6276@srmist.edu.in
- **Mehul Ashra** — ma8125@srmist.edu.in
- **Asha Abraham** — ashaa2@srmist.edu.in

Department of Networking and Communications,
SRM Institute of Science and Technology, Kattankulathur, Chennai, India.

---

## License

Research code released for academic use. Contact the authors for other use cases.
