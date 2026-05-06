#!/usr/bin/env python3
"""
PINNTO Experiment Runner — runs all experiments with checkpoints.
Close laptop anytime. Rerun this script and it resumes.

Usage: python3 run_experiments.py
"""
import math, os, sys, time, pickle
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for terminal
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Checkpoint utilities
# ============================================================
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(name, model, history, epoch, optimizer=None):
    state = {"model_state_dict": model.state_dict(), "history": history, "epoch": epoch}
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(state, os.path.join(CKPT_DIR, f"{name}.pt"))


def load_checkpoint(name, model, optimizer=None):
    path = os.path.join(CKPT_DIR, f"{name}.pt")
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    print(f"  Resumed '{name}' at epoch {state['epoch']}")
    return state["history"], state["epoch"]


def save_results(name, data):
    with open(os.path.join(CKPT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(data, f)


def load_results(name):
    path = os.path.join(CKPT_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# Data classes
# ============================================================
@dataclass
class BSParams:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2

@dataclass
class Domain:
    S_min: float = 1.0
    S_max: float = 200.0
    tau_min: float = 0.0
    tau_max: float = 1.0

@dataclass
class LossWeights:
    data: float = 1.0
    pde: float = 1.0
    bc: float = 1.0
    term: float = 1.0

@dataclass
class AmericanLossWeights:
    data: float = 5.0          # supervised FD data (curriculum)
    pde_ineq: float = 1.0
    exercise: float = 50.0     # strong early-exercise enforcement
    complementarity: float = 1.0
    bc: float = 5.0
    terminal: float = 10.0


# ============================================================
# Black-Scholes functions
# ============================================================
def black_scholes_call(S, K, tau, r, sigma):
    eps = 1e-12
    tau_c = torch.clamp(tau, min=eps)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * tau_c) / (sigma * torch.sqrt(tau_c))
    d2 = d1 - sigma * torch.sqrt(tau_c)
    N1 = 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))
    N2 = 0.5 * (1.0 + torch.erf(d2 / math.sqrt(2.0)))
    return S * N1 - K * torch.exp(-r * tau_c) * N2

def payoff_call(S, K):
    return torch.clamp(S - K, min=0.0)

def payoff_put(S, K):
    return torch.clamp(K - S, min=0.0)


# ============================================================
# Sampling
# ============================================================
def sample_uniform_2d(n, x_min, x_max, y_min, y_max):
    x = torch.rand(n, 1) * (x_max - x_min) + x_min
    y = torch.rand(n, 1) * (y_max - y_min) + y_min
    return x.to(device), y.to(device)

def make_data_points(n_data, domain, params):
    S_data, tau_data = sample_uniform_2d(n_data, domain.S_min, domain.S_max,
                                          domain.tau_min + 1e-3, domain.tau_max)
    K = torch.full_like(S_data, params.K)
    r = torch.full_like(S_data, params.r)
    sigma = torch.full_like(S_data, params.sigma)
    C_bs = black_scholes_call(S_data, K, tau_data, r, sigma).detach()
    return tau_data, S_data, C_bs

def monte_carlo_call_price(S0, K, tau, r, sigma, n_paths=100_000):
    Z = torch.randn(n_paths)
    ST = S0 * torch.exp(torch.tensor((r - 0.5 * sigma**2) * tau) + sigma * math.sqrt(tau) * Z)
    payoffs = torch.clamp(ST - K, min=0.0)
    return (math.exp(-r * tau) * payoffs).mean().item()


# ============================================================
# PINN model
# ============================================================
class PINN(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, n_hidden_layers=4,
                 normalize=False, S_scale=200.0, tau_scale=1.0):
        super().__init__()
        self.normalize = normalize
        self.S_scale = S_scale
        self.tau_scale = tau_scale
        layers = []
        dim_in = in_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(nn.Tanh())
            dim_in = hidden_dim
        layers.append(nn.Linear(dim_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, tau, S):
        if self.normalize:
            x = torch.cat([tau / self.tau_scale, S / self.S_scale], dim=1)
        else:
            x = torch.cat([tau, S], dim=1)
        return self.net(x)


# ============================================================
# European loss functions
# ============================================================
def pde_residual(model, tau, S, params):
    tau = tau.clone().detach().requires_grad_(True)
    S = S.clone().detach().requires_grad_(True)
    C = model(tau, S)
    dC_dtau = torch.autograd.grad(C, tau, torch.ones_like(C), retain_graph=True, create_graph=True)[0]
    dC_dS = torch.autograd.grad(C, S, torch.ones_like(C), retain_graph=True, create_graph=True)[0]
    d2C_dS2 = torch.autograd.grad(dC_dS, S, torch.ones_like(dC_dS), retain_graph=True, create_graph=True)[0]
    rhs = 0.5 * params.sigma**2 * S**2 * d2C_dS2 + params.r * S * dC_dS - params.r * C
    return dC_dtau - rhs

def loss_data(model, tau_data, S_data, C_bs):
    return torch.mean((model(tau_data, S_data) - C_bs) ** 2)

def loss_pde(model, n_pde, domain, params):
    S_pde, tau_pde = sample_uniform_2d(n_pde, domain.S_min, domain.S_max,
                                        domain.tau_min + 1e-3, domain.tau_max)
    return torch.mean(pde_residual(model, tau_pde, S_pde, params)**2)

def loss_terminal(model, n_term, domain, params):
    S_t, tau_t = sample_uniform_2d(n_term, domain.S_min, domain.S_max,
                                    domain.tau_min, domain.tau_min + 1e-3)
    K = torch.full_like(S_t, params.K)
    return torch.mean((model(tau_t, S_t) - payoff_call(S_t, K)) ** 2)

def loss_boundary(model, n_bc, domain, params):
    tau1 = torch.rand(n_bc, 1, device=device) * (domain.tau_max - domain.tau_min) + domain.tau_min
    S1 = torch.full_like(tau1, domain.S_min)
    loss1 = torch.mean(model(tau1, S1) ** 2)

    tau2 = torch.rand(n_bc, 1, device=device) * (domain.tau_max - domain.tau_min) + domain.tau_min
    S2 = torch.full_like(tau2, domain.S_max)
    K = torch.full_like(S2, params.K)
    target = S2 - K * torch.exp(-params.r * tau2)
    loss2 = torch.mean((model(tau2, S2) - target) ** 2)
    return loss1 + loss2


# ============================================================
# European training
# ============================================================
def train_pinn(n_epochs=5000, n_data=2000, n_pde=5000, n_term=1000, n_bc=1000,
               lr=1e-3, bs_params=None, domain=None, loss_weights=None,
               ckpt_name="european_pinn", ckpt_every=500):
    bs_params = bs_params or BSParams()
    domain = domain or Domain()
    loss_weights = loss_weights or LossWeights()

    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    tau_data, S_data, C_bs = make_data_points(n_data, domain, bs_params)

    history = {"total": [], "data": [], "pde": [], "bc": [], "term": []}
    start_epoch = 1

    ckpt = load_checkpoint(ckpt_name, model, optimizer)
    if ckpt is not None:
        history, start_epoch = ckpt
        start_epoch += 1
        if start_epoch > n_epochs:
            print(f"  Training already complete ({start_epoch-1}/{n_epochs})")
            return model, history

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        L_d = loss_data(model, tau_data, S_data, C_bs)
        L_p = loss_pde(model, n_pde, domain, bs_params)
        L_t = loss_terminal(model, n_term, domain, bs_params)
        L_b = loss_boundary(model, n_bc, domain, bs_params)
        total = loss_weights.data*L_d + loss_weights.pde*L_p + loss_weights.term*L_t + loss_weights.bc*L_b
        total.backward()
        optimizer.step()

        history["total"].append(total.item())
        history["data"].append(L_d.item())
        history["pde"].append(L_p.item())
        history["bc"].append(L_b.item())
        history["term"].append(L_t.item())

        if epoch % 500 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} | Loss={total.item():.4e} | data={L_d.item():.4e} | "
                  f"pde={L_p.item():.4e} | term={L_t.item():.4e} | bc={L_b.item():.4e}")
        if epoch % ckpt_every == 0:
            save_checkpoint(ckpt_name, model, history, epoch, optimizer)

    save_checkpoint(ckpt_name, model, history, n_epochs, optimizer)
    return model, history


def evaluate_on_grid(model, domain, params, n_S=80, n_tau=80):
    model.eval()
    S_lin = torch.linspace(domain.S_min, domain.S_max, n_S, device=device)
    tau_lin = torch.linspace(domain.tau_min + 1e-3, domain.tau_max, n_tau, device=device)
    tau_grid, S_grid = torch.meshgrid(tau_lin, S_lin, indexing="ij")
    with torch.no_grad():
        C_pinn = model(tau_grid.reshape(-1, 1), S_grid.reshape(-1, 1)).reshape(n_tau, n_S)
        K = torch.full_like(S_grid, params.K)
        r = torch.full_like(S_grid, params.r)
        sigma = torch.full_like(S_grid, params.sigma)
        C_bs = black_scholes_call(S_grid, K, tau_grid, r, sigma)
    return {
        "tau_grid": tau_grid.cpu(), "S_grid": S_grid.cpu(),
        "C_pinn": C_pinn.cpu(), "C_bs": C_bs.cpu(),
        "abs_error": (C_pinn - C_bs).abs().cpu(),
    }


# ============================================================
# American Put: baselines
# ============================================================
def binomial_american_put(S0, K, T, r, sigma, n_steps=50000):
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    S_T = S0 * d ** np.arange(n_steps, -1, -1) * u ** np.arange(0, n_steps + 1)
    V = np.maximum(K - S_T, 0.0)
    for i in range(n_steps - 1, -1, -1):
        S_i = S0 * d ** np.arange(i, -1, -1) * u ** np.arange(0, i + 1)
        V = np.maximum(disc * (p * V[1:i+2] + (1 - p) * V[0:i+1]), np.maximum(K - S_i, 0.0))
    return float(V[0])


def fd_american_put(S_max, K, T, r, sigma, n_S=500, n_t=2000):
    dS = S_max / n_S
    dt = T / n_t
    S_grid = np.linspace(0, S_max, n_S + 1)
    V = np.maximum(K - S_grid, 0.0)
    j = np.arange(0, n_S + 1)
    alpha = 0.5 * dt * (sigma**2 * j**2 - r * j)
    beta = -(1.0 + dt * (sigma**2 * j**2 + r))
    gamma = 0.5 * dt * (sigma**2 * j**2 + r * j)

    for n in range(n_t):
        rhs = -V.copy()
        rhs[0] = -np.maximum(K - S_grid[0], 0.0)
        rhs[-1] = 0.0

        a = alpha[1:-1].copy()
        b = beta[1:-1].copy()
        c = gamma[1:-1].copy()
        d_vec = rhs[1:-1].copy()

        n_inner = len(b)
        c_star = np.zeros(n_inner)
        d_star = np.zeros(n_inner)

        c_star[0] = c[0] / b[0]
        d_star[0] = d_vec[0] / b[0]
        for i in range(1, n_inner):
            denom = b[i] - a[i] * c_star[i-1]
            if abs(denom) < 1e-15:
                denom = 1e-15
            c_star[i] = c[i] / denom if i < len(c) else 0
            d_star[i] = (d_vec[i] - a[i] * d_star[i-1]) / denom

        V_new = np.zeros(n_S + 1)
        V_new[0] = np.maximum(K - S_grid[0], 0.0)
        V_new[-1] = 0.0
        V_new[-2] = max(d_star[-1], K - S_grid[-2])
        for i in range(n_inner - 2, -1, -1):
            V_new[i+1] = max(d_star[i] - c_star[i] * V_new[i+2], K - S_grid[i+1])
        V = V_new
    return S_grid, V


def lsm_american_put(S0, K, T, r, sigma, n_paths=500000, n_steps=200):
    dt = T / n_steps
    disc = np.exp(-r * dt)
    Z = np.random.randn(n_paths, n_steps)
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        S[:, t+1] = S[:, t] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t])

    payoff = np.maximum(K - S, 0.0)
    cashflows = payoff[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        itm = payoff[:, t] > 0
        if itm.sum() < 10:
            cashflows *= disc
            continue
        S_itm = S[itm, t]
        Y = cashflows[itm] * disc
        X = np.column_stack([np.ones_like(S_itm), S_itm, S_itm**2, S_itm**3])
        try:
            coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
            continuation = X @ coeffs
        except np.linalg.LinAlgError:
            cashflows *= disc
            continue
        exercise_value = payoff[itm, t]
        do_exercise = exercise_value > continuation
        idx_itm = np.where(itm)[0]
        cashflows[idx_itm[do_exercise]] = exercise_value[do_exercise]
        cashflows[idx_itm[~do_exercise]] *= disc
        cashflows[~itm] *= disc

    return float(np.mean(cashflows * disc))


# ============================================================
# American PINN loss functions
# ============================================================
def pde_residual_put(model, tau, S, params):
    tau = tau.clone().detach().requires_grad_(True)
    S = S.clone().detach().requires_grad_(True)
    V = model(tau, S)
    dV_dtau = torch.autograd.grad(V, tau, torch.ones_like(V), retain_graph=True, create_graph=True)[0]
    dV_dS = torch.autograd.grad(V, S, torch.ones_like(V), retain_graph=True, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S, torch.ones_like(dV_dS), retain_graph=True, create_graph=True)[0]
    rhs = 0.5 * params.sigma**2 * S**2 * d2V_dS2 + params.r * S * dV_dS - params.r * V
    return dV_dtau - rhs

def loss_american_pde_inequality(model, n_pde, domain, params):
    S_pde, tau_pde = sample_uniform_2d(n_pde, domain.S_min, domain.S_max, domain.tau_min + 1e-3, domain.tau_max)
    residual = pde_residual_put(model, tau_pde, S_pde, params)
    return torch.mean(torch.clamp(-residual, min=0.0) ** 2)

def loss_american_exercise(model, n_pts, domain, params):
    S_pts, tau_pts = sample_uniform_2d(n_pts, domain.S_min, domain.S_max, domain.tau_min, domain.tau_max)
    V_pred = model(tau_pts, S_pts)
    intrinsic = payoff_put(S_pts, torch.full_like(S_pts, params.K))
    return torch.mean(torch.clamp(intrinsic - V_pred, min=0.0) ** 2)

def loss_american_complementarity(model, n_pts, domain, params):
    S_pts, tau_pts = sample_uniform_2d(n_pts, domain.S_min, domain.S_max, domain.tau_min + 1e-3, domain.tau_max)
    residual = pde_residual_put(model, tau_pts, S_pts, params)
    V_pred = model(tau_pts, S_pts)
    intrinsic = payoff_put(S_pts, torch.full_like(S_pts, params.K))
    return torch.mean((residual * (V_pred - intrinsic)) ** 2)

def loss_american_terminal(model, n_term, domain, params):
    S_t, tau_t = sample_uniform_2d(n_term, domain.S_min, domain.S_max, domain.tau_min, domain.tau_min + 1e-3)
    return torch.mean((model(tau_t, S_t) - payoff_put(S_t, torch.full_like(S_t, params.K))) ** 2)

def loss_american_boundary(model, n_bc, domain, params):
    tau1 = torch.rand(n_bc, 1, device=device) * (domain.tau_max - domain.tau_min) + domain.tau_min
    S1 = torch.full_like(tau1, domain.S_min)
    target1 = torch.full_like(tau1, params.K) * torch.exp(-params.r * tau1)
    loss1 = torch.mean((model(tau1, S1) - target1) ** 2)

    tau2 = torch.rand(n_bc, 1, device=device) * (domain.tau_max - domain.tau_min) + domain.tau_min
    S2 = torch.full_like(tau2, domain.S_max)
    loss2 = torch.mean(model(tau2, S2) ** 2)
    return loss1 + loss2


# ============================================================
# American training
# ============================================================
def generate_fd_supervision_data(n_data, domain, params):
    """Generate supervised American put prices from FD solver."""
    S_pts, tau_pts = sample_uniform_2d(n_data, domain.S_min, domain.S_max,
                                        domain.tau_min + 1e-3, domain.tau_max)
    V_targets = []
    S_fd_cache = {}
    for i in range(n_data):
        tau_val = tau_pts[i].item()
        s_val = S_pts[i].item()
        # Quantize tau for caching (round to 0.01)
        tau_key = round(tau_val, 2)
        if tau_key not in S_fd_cache:
            S_grid, V_fd = fd_american_put(domain.S_max, params.K, max(tau_key, 0.01),
                                            params.r, params.sigma, n_S=500, n_t=2000)
            S_fd_cache[tau_key] = (S_grid, V_fd)
        S_grid, V_fd = S_fd_cache[tau_key]
        v = float(np.interp(s_val, S_grid, V_fd))
        V_targets.append(v)
    V_tensor = torch.tensor(V_targets, dtype=torch.float32, device=device).reshape(-1, 1)
    return tau_pts, S_pts, V_tensor


def train_american_pinn(n_epochs=20000, n_pde=3000, n_exercise=3000, n_compl=2000,
                        n_term=1000, n_bc=1000, n_data=1000, lr=1e-3,
                        bs_params=None, domain=None,
                        loss_weights=None, ckpt_name="american_pinn_v2", ckpt_every=500):
    bs_params = bs_params or BSParams()
    domain = domain or Domain()
    loss_weights = loss_weights or AmericanLossWeights()

    model = PINN(in_dim=2, hidden_dim=128, n_hidden_layers=5,
                 normalize=True, S_scale=domain.S_max, tau_scale=domain.tau_max).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    # Generate supervised FD data for curriculum learning
    print("  Generating FD supervision data...")
    tau_data, S_data, V_data = generate_fd_supervision_data(n_data, domain, bs_params)
    print(f"  Generated {n_data} FD supervision points.")

    history = {"total": [], "pde_ineq": [], "exercise": [], "complementarity": [],
               "bc": [], "terminal": [], "data": []}
    start_epoch = 1

    ckpt = load_checkpoint(ckpt_name, model, optimizer)
    if ckpt is not None:
        history, start_epoch = ckpt
        start_epoch += 1
        if start_epoch > n_epochs:
            print(f"  Training already complete ({start_epoch-1}/{n_epochs})")
            return model, history

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Supervised data loss (curriculum: higher weight early, lower later)
        V_pred_data = model(tau_data, S_data)
        L_data = torch.mean((V_pred_data - V_data) ** 2)

        L_pde = loss_american_pde_inequality(model, n_pde, domain, bs_params)
        L_ex = loss_american_exercise(model, n_exercise, domain, bs_params)
        L_cp = loss_american_complementarity(model, n_compl, domain, bs_params)
        L_tm = loss_american_terminal(model, n_term, domain, bs_params)
        L_bc = loss_american_boundary(model, n_bc, domain, bs_params)

        # Curriculum: data weight decays, physics weights grow
        progress = epoch / n_epochs
        w_data = loss_weights.data * max(1.0 - progress * 0.5, 0.3)
        w_pde = loss_weights.pde_ineq * min(1.0 + progress, 2.0)

        total = (w_data * L_data + w_pde * L_pde +
                 loss_weights.exercise * L_ex + loss_weights.complementarity * L_cp +
                 loss_weights.terminal * L_tm + loss_weights.bc * L_bc)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history["total"].append(total.item())
        history["data"].append(L_data.item())
        history["pde_ineq"].append(L_pde.item())
        history["exercise"].append(L_ex.item())
        history["complementarity"].append(L_cp.item())
        history["bc"].append(L_bc.item())
        history["terminal"].append(L_tm.item())

        if epoch % 2000 == 0 or epoch == 1:
            print(f"  Epoch {epoch:5d} | Loss={total.item():.4e} | Data={L_data.item():.4e} | "
                  f"PDE={L_pde.item():.4e} | Ex={L_ex.item():.4e} | Compl={L_cp.item():.4e}")
        if epoch % ckpt_every == 0:
            save_checkpoint(ckpt_name, model, history, epoch, optimizer)

    save_checkpoint(ckpt_name, model, history, n_epochs, optimizer)
    return model, history


# ============================================================
# Evaluation helpers
# ============================================================
def evaluate_american_on_grid(model, domain, params, n_S=80, n_tau=40):
    model.eval()
    S_lin = torch.linspace(domain.S_min, domain.S_max, n_S, device=device)
    tau_lin = torch.linspace(domain.tau_min + 1e-3, domain.tau_max, n_tau, device=device)
    tau_grid, S_grid = torch.meshgrid(tau_lin, S_lin, indexing="ij")
    with torch.no_grad():
        V_pinn = model(tau_grid.reshape(-1, 1), S_grid.reshape(-1, 1)).reshape(n_tau, n_S)
    V_fd_all = []
    for ti in range(n_tau):
        tau_val = tau_lin[ti].item()
        S_fd, V_fd = fd_american_put(domain.S_max, params.K, tau_val, params.r, params.sigma, n_S=500, n_t=2000)
        V_fd_all.append(np.interp(S_lin.cpu().numpy(), S_fd, V_fd))
    V_fd_tensor = torch.tensor(np.array(V_fd_all), dtype=torch.float32)
    return {
        "tau_grid": tau_grid.cpu(), "S_grid": S_grid.cpu(),
        "V_pinn": V_pinn.cpu(), "V_fd": V_fd_tensor,
        "abs_error": (V_pinn.cpu() - V_fd_tensor).abs(),
    }


def extract_free_boundary(model, domain, params, n_tau=50, n_S=1000):
    model.eval()
    tau_values = np.linspace(1e-3, domain.tau_max, n_tau)
    S_values = np.linspace(domain.S_min, domain.S_max, n_S)
    boundaries = []
    for tau_val in tau_values:
        tau_t = torch.full((n_S, 1), tau_val, device=device)
        S_t = torch.tensor(S_values.reshape(-1, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            V_pred = model(tau_t, S_t).cpu().numpy().flatten()
        intrinsic = np.maximum(params.K - S_values, 0.0)
        excess = V_pred - intrinsic
        exercise_mask = excess < 0.01 * params.K
        candidates = S_values[exercise_mask]
        candidates = candidates[candidates < params.K]
        boundaries.append(candidates.max() if len(candidates) > 0 else domain.S_min)
    return tau_values, np.array(boundaries)


def extract_fd_free_boundary(domain, params, n_tau=50):
    tau_values = np.linspace(1e-3, domain.tau_max, n_tau)
    boundaries = []
    for tau_val in tau_values:
        S_grid, V_fd = fd_american_put(domain.S_max, params.K, tau_val, params.r, params.sigma, n_S=500, n_t=2000)
        intrinsic = np.maximum(params.K - S_grid, 0.0)
        excess = V_fd - intrinsic
        exercise_mask = excess < 0.01 * params.K
        candidates = S_grid[exercise_mask]
        candidates = candidates[candidates < params.K]
        if len(candidates) > 0:
            boundaries.append(candidates.max())
        else:
            boundaries.append(domain.S_min)
    return tau_values, np.array(boundaries)


# ============================================================
# Greeks
# ============================================================
def compute_pinn_greeks(model, S_np, tau_val):
    model.eval()
    n = len(S_np)
    tau_t = torch.full((n, 1), tau_val, device=device, requires_grad=True)
    S_t = torch.tensor(S_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    C = model(tau_t, S_t)
    dC_dS = torch.autograd.grad(C, S_t, torch.ones_like(C), retain_graph=True, create_graph=True)[0]
    d2C_dS2 = torch.autograd.grad(dC_dS, S_t, torch.ones_like(dC_dS), retain_graph=True, create_graph=True)[0]
    dC_dtau = torch.autograd.grad(C, tau_t, torch.ones_like(C), retain_graph=True, create_graph=False)[0]
    return (dC_dS.detach().cpu().numpy().flatten(),
            d2C_dS2.detach().cpu().numpy().flatten(),
            dC_dtau.detach().cpu().numpy().flatten())


def bs_greeks_analytical(S, K, tau, r, sigma):
    """Returns Delta, Gamma, dC/dtau (not classical Theta).
    PINN computes dC/dtau. Classical Theta = -dC/dt = dC/dtau.
    So: dC/dtau = (S*N'(d1)*sigma)/(2*sqrt(tau)) + r*K*exp(-r*tau)*N(d2)
    """
    from scipy.stats import norm
    eps = 1e-12
    tau_c = np.maximum(tau, eps)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_c) / (sigma * np.sqrt(tau_c))
    d2 = d1 - sigma * np.sqrt(tau_c)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau_c))
    # dC/dtau = -Theta_classical (positive for calls: more time = more value)
    dC_dtau = (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(tau_c)) + r * K * np.exp(-r * tau_c) * norm.cdf(d2)
    return delta, gamma, dC_dtau


# ============================================================
# American Greeks (FD-based reference)
# ============================================================
def compute_american_greeks_fd(S_np, tau_val, params, domain, dS=0.5, dtau=0.001):
    """Compute American put Greeks via FD finite differences (bump and reprice)."""
    deltas, gammas, thetas = [], [], []
    for s in S_np:
        # Delta & Gamma via central differences in S
        S_grid_u, V_u = fd_american_put(domain.S_max, params.K, tau_val, params.r, params.sigma, n_S=500, n_t=2000)
        v_mid = float(np.interp(s, S_grid_u, V_u))
        v_up = float(np.interp(s + dS, S_grid_u, V_u))
        v_dn = float(np.interp(max(s - dS, 0.01), S_grid_u, V_u))
        d = (v_up - v_dn) / (2 * dS)
        g = (v_up - 2 * v_mid + v_dn) / (dS ** 2)
        deltas.append(d)
        gammas.append(g)
        # Theta via forward difference in tau
        tau_up = tau_val + dtau
        S_grid_t, V_t = fd_american_put(domain.S_max, params.K, tau_up, params.r, params.sigma, n_S=500, n_t=2000)
        v_tau_up = float(np.interp(s, S_grid_t, V_t))
        thetas.append((v_tau_up - v_mid) / dtau)
    return np.array(deltas), np.array(gammas), np.array(thetas)


def compute_pinn_greeks_put(model, S_np, tau_val):
    """Compute American put Greeks from PINN via autograd."""
    model.eval()
    n = len(S_np)
    tau_t = torch.full((n, 1), tau_val, device=device, requires_grad=True)
    S_t = torch.tensor(S_np.reshape(-1, 1), dtype=torch.float32, device=device, requires_grad=True)
    V = model(tau_t, S_t)
    dV_dS = torch.autograd.grad(V, S_t, torch.ones_like(V), retain_graph=True, create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S_t, torch.ones_like(dV_dS), retain_graph=True, create_graph=True)[0]
    dV_dtau = torch.autograd.grad(V, tau_t, torch.ones_like(V), retain_graph=True, create_graph=False)[0]
    return (dV_dS.detach().cpu().numpy().flatten(),
            d2V_dS2.detach().cpu().numpy().flatten(),
            dV_dtau.detach().cpu().numpy().flatten())


# ============================================================
# SHAP Explainability
# ============================================================
def run_shap_analysis(model, domain, params, n_background=200, n_explain=100):
    """Run SHAP KernelExplainer on the trained PINN."""
    import shap

    class PINNWrapper:
        def __init__(self, m):
            self.model = m
            self.model.eval()
        def __call__(self, X_np):
            tau = torch.tensor(X_np[:, 0:1], dtype=torch.float32, device=device)
            S = torch.tensor(X_np[:, 1:2], dtype=torch.float32, device=device)
            with torch.no_grad():
                return self.model(tau, S).cpu().numpy().flatten()

    wrapper = PINNWrapper(model)
    np.random.seed(42)
    bg_tau = np.random.uniform(1e-3, domain.tau_max, n_background)
    bg_S = np.random.uniform(domain.S_min, domain.S_max, n_background)
    X_bg = np.column_stack([bg_tau, bg_S])

    ex_tau = np.random.uniform(1e-3, domain.tau_max, n_explain)
    ex_S = np.random.uniform(domain.S_min, domain.S_max, n_explain)
    X_ex = np.column_stack([ex_tau, ex_S])

    explainer = shap.KernelExplainer(wrapper, X_bg)
    shap_values = explainer.shap_values(X_ex, nsamples=200)
    return shap_values, X_ex


# ============================================================
# Multi-seed Robustness
# ============================================================
def run_multi_seed(seeds, n_epochs=15000):
    """Train European PINN with multiple seeds and report statistics."""
    results = []
    _domain = Domain()
    _params = BSParams()
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        _model = PINN(in_dim=2, hidden_dim=64, n_hidden_layers=4).to(device)
        _opt = optim.Adam(_model.parameters(), lr=1e-3)
        tau_d, S_d, C_d = make_data_points(2000, _domain, _params)
        for ep in range(1, n_epochs + 1):
            _model.train()
            _opt.zero_grad()
            L_d = loss_data(_model, tau_d, S_d, C_d)
            L_p = loss_pde(_model, 5000, _domain, _params)
            L_t = loss_terminal(_model, 1000, _domain, _params)
            L_b = loss_boundary(_model, 1000, _domain, _params)
            total = L_d + L_p + L_t + L_b
            total.backward()
            _opt.step()
        ev = evaluate_on_grid(_model, _domain, _params, n_S=80, n_tau=80)
        mae = ev["abs_error"].mean().item()
        rmse = torch.sqrt(torch.mean(ev["abs_error"]**2)).item()
        C_p = ev["C_pinn"].flatten().numpy()
        C_b = ev["C_bs"].flatten().numpy()
        ss_res = np.sum((C_p - C_b)**2)
        ss_tot = np.sum((C_b - C_b.mean())**2)
        r2 = 1 - ss_res / ss_tot
        results.append({"seed": seed, "mae": mae, "rmse": rmse, "r2": r2})
        print(f"    Seed {seed}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.6f}")
    return results


# ============================================================
# Figure Generation
# ============================================================
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_european_convergence(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["total", "data", "pde", "bc", "term"]:
        ax.plot(history[key], label=key, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("European Call PINN — Loss Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_european_convergence.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig1_european_convergence.png"), dpi=300)
    plt.close(fig)


def plot_european_surface(eu_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tau_g = eu_results["tau_grid"].numpy()
    S_g = eu_results["S_grid"].numpy()

    im0 = axes[0].pcolormesh(S_g, tau_g, eu_results["C_pinn"].numpy(), shading="auto", cmap="viridis")
    axes[0].set_title("PINNTO Price Surface")
    axes[0].set_xlabel("S"); axes[0].set_ylabel("τ")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(S_g, tau_g, eu_results["C_bs"].numpy(), shading="auto", cmap="viridis")
    axes[1].set_title("Black–Scholes Analytical")
    axes[1].set_xlabel("S"); axes[1].set_ylabel("τ")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(S_g, tau_g, eu_results["abs_error"].numpy(), shading="auto", cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("S"); axes[2].set_ylabel("τ")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("European Call: PINNTO vs Black–Scholes", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_european_surface.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig2_european_surface.png"), dpi=300)
    plt.close(fig)


def plot_american_convergence(history):
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["total", "data", "pde_ineq", "exercise", "complementarity"]:
        if key in history:
            ax.plot(history[key], label=key, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("American Put PINN — Loss Convergence (Curriculum Learning)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_american_convergence.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig6_american_convergence.png"), dpi=300)
    plt.close(fig)


def plot_american_surface(am_model, domain, params):
    n_S, n_tau = 100, 50
    S_lin = torch.linspace(domain.S_min, domain.S_max, n_S, device=device)
    tau_lin = torch.linspace(1e-3, domain.tau_max, n_tau, device=device)
    tau_g, S_g = torch.meshgrid(tau_lin, S_lin, indexing="ij")
    am_model.eval()
    with torch.no_grad():
        V_pinn = am_model(tau_g.reshape(-1, 1), S_g.reshape(-1, 1)).reshape(n_tau, n_S).cpu().numpy()
    # FD reference
    V_fd_all = []
    for ti in range(n_tau):
        S_fd, V_fd = fd_american_put(domain.S_max, params.K, tau_lin[ti].item(), params.r, params.sigma, n_S=500, n_t=2000)
        V_fd_all.append(np.interp(S_lin.cpu().numpy(), S_fd, V_fd))
    V_fd_arr = np.array(V_fd_all)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    tau_np = tau_g.cpu().numpy()
    S_np = S_g.cpu().numpy()

    im0 = axes[0].pcolormesh(S_np, tau_np, V_pinn, shading="auto", cmap="viridis")
    axes[0].set_title("PINNTO American Put")
    axes[0].set_xlabel("S"); axes[0].set_ylabel("τ")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(S_np, tau_np, V_fd_arr, shading="auto", cmap="viridis")
    axes[1].set_title("Finite Difference Reference")
    axes[1].set_xlabel("S"); axes[1].set_ylabel("τ")
    fig.colorbar(im1, ax=axes[1])

    err = np.abs(V_pinn - V_fd_arr)
    im2 = axes[2].pcolormesh(S_np, tau_np, err, shading="auto", cmap="hot")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("S"); axes[2].set_ylabel("τ")
    fig.colorbar(im2, ax=axes[2])

    fig.suptitle("American Put: PINNTO vs Finite Difference", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig7_american_surface.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig7_american_surface.png"), dpi=300)
    plt.close(fig)


def plot_free_boundary(tau_pinn, bf_pinn, tau_fd, bf_fd):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau_pinn, bf_pinn, 'b-', linewidth=2, label="PINNTO")
    ax.plot(tau_fd, bf_fd, 'r--', linewidth=2, label="Finite Difference")
    ax.fill_between(tau_pinn, bf_pinn, 0, alpha=0.15, color='blue', label="Exercise region")
    ax.set_xlabel("Time to maturity (τ)")
    ax.set_ylabel("Free boundary S_f(τ)")
    ax.set_title("American Put: Optimal Exercise Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig9_free_boundary.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig9_free_boundary.png"), dpi=300)
    plt.close(fig)


def plot_greeks(eu_model, params, tau_val=0.5):
    S_test = np.linspace(20, 180, 300)
    d_pinn, g_pinn, t_pinn = compute_pinn_greeks(eu_model, S_test, tau_val)
    d_bs, g_bs, t_bs = bs_greeks_analytical(S_test, params.K, tau_val, params.r, params.sigma)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, pinn_v, bs_v, name in zip(
        axes, [d_pinn, g_pinn, t_pinn], [d_bs, g_bs, t_bs], ["Delta", "Gamma", "Theta (dC/dτ)"]):
        ax.plot(S_test, pinn_v, 'b-', linewidth=2, label="PINNTO")
        ax.plot(S_test, bs_v, 'r--', linewidth=2, label="Analytical")
        ax.set_xlabel("S")
        ax.set_ylabel(name)
        ax.set_title(f"{name} at τ={tau_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("European Call Greeks: PINNTO vs Black–Scholes", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig10_greeks_european.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig10_greeks_european.png"), dpi=300)
    plt.close(fig)


def plot_american_greeks(am_model, params, domain, tau_val=0.5):
    S_test = np.linspace(50, 150, 200)
    d_pinn, g_pinn, t_pinn = compute_pinn_greeks_put(am_model, S_test, tau_val)
    d_fd, g_fd, t_fd = compute_american_greeks_fd(S_test, tau_val, params, domain)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, pinn_v, fd_v, name in zip(
        axes, [d_pinn, g_pinn, t_pinn], [d_fd, g_fd, t_fd], ["Delta", "Gamma", "Theta (dV/dτ)"]):
        ax.plot(S_test, pinn_v, 'b-', linewidth=2, label="PINNTO")
        ax.plot(S_test, fd_v, 'r--', linewidth=2, label="FD Reference")
        ax.set_xlabel("S")
        ax.set_ylabel(name)
        ax.set_title(f"{name} at τ={tau_val}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("American Put Greeks: PINNTO vs Finite Difference", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig10b_greeks_american.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig10b_greeks_american.png"), dpi=300)
    plt.close(fig)


def plot_ablation_bar(ablation):
    names = list(ablation.keys())
    maes = [ablation[n]["mae"] for n in names]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2196F3' if n == 'Baseline (4x64)' else '#FF9800' for n in names]
    bars = ax.barh(names, maes, color=colors)
    ax.set_xlabel("MAE")
    ax.set_title("Ablation Study: MAE by Configuration")
    ax.axvline(x=ablation["Baseline (4x64)"]["mae"], color='red', linestyle='--', alpha=0.5, label="Baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_ablation.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig4_ablation.png"), dpi=300)
    plt.close(fig)


def plot_weight_sweep(sweep_results, weight_values):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(sweep_results, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"{w:.1f}" for w in weight_values],
                yticklabels=[f"{w:.1f}" for w in weight_values], ax=ax)
    ax.set_xlabel("λ_PDE")
    ax.set_ylabel("λ_data")
    ax.set_title("Loss Weight Sweep: MAE")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_weight_sweep.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig5_weight_sweep.png"), dpi=300)
    plt.close(fig)


def plot_shap(shap_values, X_ex):
    import shap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Summary plot
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_ex, feature_names=["τ", "S"], show=False, plot_size=None)
    axes[0].set_title("SHAP Summary")

    # Dependence on S
    plt.sca(axes[1])
    shap.dependence_plot(1, shap_values, X_ex, feature_names=["τ", "S"], show=False, ax=axes[1])
    axes[1].set_title("SHAP Dependence: S")

    # Dependence on tau
    plt.sca(axes[2])
    shap.dependence_plot(0, shap_values, X_ex, feature_names=["τ", "S"], show=False, ax=axes[2])
    axes[2].set_title("SHAP Dependence: τ")

    fig.suptitle("SHAP Explainability Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig11_shap.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig11_shap.png"), dpi=300)
    plt.close(fig)


def plot_multi_seed(seed_results):
    seeds = [r["seed"] for r in seed_results]
    maes = [r["mae"] for r in seed_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(s) for s in seeds], maes, color='#2196F3', alpha=0.8)
    ax.axhline(y=np.mean(maes), color='red', linestyle='--', label=f"Mean={np.mean(maes):.4f}")
    ax.fill_between(range(len(seeds)),
                     np.mean(maes) - np.std(maes), np.mean(maes) + np.std(maes),
                     alpha=0.2, color='red', label=f"±1σ={np.std(maes):.4f}")
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("MAE")
    ax.set_title("Multi-Seed Robustness: European PINN")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig12_multi_seed.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG_DIR, "fig12_multi_seed.png"), dpi=300)
    plt.close(fig)


# ============================================================
# Ablation runner
# ============================================================
def run_single_experiment(config, seed=42):
    torch.manual_seed(seed)
    _bs = BSParams(); _domain = Domain()
    _model = PINN(in_dim=2, hidden_dim=config.get("hidden_dim", 64),
                  n_hidden_layers=config.get("n_hidden_layers", 4)).to(device)
    _opt = optim.Adam(_model.parameters(), lr=config.get("lr", 1e-3))
    _n_epochs = config.get("n_epochs", 5000)
    _lw = LossWeights(data=config.get("w_data", 1.0), pde=config.get("w_pde", 1.0),
                       bc=config.get("w_bc", 1.0), term=config.get("w_term", 1.0))
    tau_d, S_d, C_d = make_data_points(config.get("n_data", 2000), _domain, _bs)

    t0 = time.time()
    for ep in range(1, _n_epochs + 1):
        _model.train(); _opt.zero_grad()
        L_d = loss_data(_model, tau_d, S_d, C_d) if _lw.data > 0 else torch.tensor(0.0)
        L_p = loss_pde(_model, config.get("n_pde", 5000), _domain, _bs) if _lw.pde > 0 else torch.tensor(0.0)
        L_t = loss_terminal(_model, config.get("n_term", 1000), _domain, _bs) if _lw.term > 0 else torch.tensor(0.0)
        L_b = loss_boundary(_model, config.get("n_bc", 1000), _domain, _bs) if _lw.bc > 0 else torch.tensor(0.0)
        total = _lw.data*L_d + _lw.pde*L_p + _lw.term*L_t + _lw.bc*L_b
        total.backward(); _opt.step()
    train_time = time.time() - t0

    results = evaluate_on_grid(_model, _domain, _bs, n_S=80, n_tau=80)
    mae = results["abs_error"].mean().item()
    rmse = torch.sqrt(torch.mean(results["abs_error"]**2)).item()
    max_err = results["abs_error"].max().item()
    C_p = results["C_pinn"].flatten().numpy()
    C_b = results["C_bs"].flatten().numpy()
    ss_res = np.sum((C_p - C_b)**2)
    ss_tot = np.sum((C_b - C_b.mean())**2)
    return {"mae": mae, "rmse": rmse, "max_err": max_err, "r2": 1 - ss_res/ss_tot, "train_time": train_time}


ABLATION_CONFIGS = {
    "Baseline (4x64)": {},
    "Shallow (2x64)": {"n_hidden_layers": 2},
    "Deep (6x64)": {"n_hidden_layers": 6},
    "Narrow (4x32)": {"hidden_dim": 32},
    "Wide (4x128)": {"hidden_dim": 128},
    "No PDE loss": {"w_pde": 0.0},
    "No Data loss": {"w_data": 0.0},
    "No BC loss": {"w_bc": 0.0},
    "No Terminal loss": {"w_term": 0.0},
    "Few PDE pts (500)": {"n_pde": 500},
    "Many PDE pts (20k)": {"n_pde": 20000},
    "High PDE weight (10)": {"w_pde": 10.0},
}


# ============================================================
# Vanilla NN baseline
# ============================================================
def train_vanilla_nn(n_epochs=5000, n_data=2000, lr=1e-3):
    torch.manual_seed(42)
    _model = PINN(in_dim=2, hidden_dim=64, n_hidden_layers=4).to(device)
    _opt = optim.Adam(_model.parameters(), lr=lr)
    tau_d, S_d, C_d = make_data_points(n_data, Domain(), BSParams())
    for ep in range(1, n_epochs + 1):
        _model.train(); _opt.zero_grad()
        loss = torch.mean((_model(tau_d, S_d) - C_d)**2)
        loss.backward(); _opt.step()
    return _model


# ############################################################
#                    MAIN EXPERIMENT RUNNER
# ############################################################
if __name__ == "__main__":
    PARAMS = BSParams(K=100.0, r=0.05, sigma=0.2)
    DOMAIN = Domain(S_min=1.0, S_max=200.0, tau_min=0.0, tau_max=1.0)

    # ── EXPERIMENT 1: European PINN ──
    print("\n" + "="*60)
    print("EXPERIMENT 1: European Call PINN (15000 epochs)")
    print("="*60)
    torch.manual_seed(42)
    eu_model, eu_history = train_pinn(
        n_epochs=15000, n_data=2000, n_pde=5000, n_term=1000, n_bc=1000,
        lr=1e-3, bs_params=PARAMS, domain=DOMAIN,
        ckpt_name="european_pinn", ckpt_every=500,
    )
    eu_results = evaluate_on_grid(eu_model, DOMAIN, PARAMS, n_S=80, n_tau=80)
    eu_mae = eu_results["abs_error"].mean().item()
    eu_rmse = torch.sqrt(torch.mean(eu_results["abs_error"]**2)).item()
    print(f"\n  European MAE:  {eu_mae:.4f}")
    print(f"  European RMSE: {eu_rmse:.4f}")
    save_results("european_eval", {"mae": eu_mae, "rmse": eu_rmse})

    # ── EXPERIMENT 2: American Put PINN ──
    print("\n" + "="*60)
    print("EXPERIMENT 2: American Put PINN (20000 epochs, curriculum learning)")
    print("="*60)
    torch.manual_seed(42)
    np.random.seed(42)
    am_model, am_history = train_american_pinn(
        n_epochs=20000, n_pde=3000, n_exercise=3000, n_compl=2000,
        n_term=1000, n_bc=1000, n_data=1000, lr=1e-3,
        bs_params=PARAMS, domain=DOMAIN,
        loss_weights=AmericanLossWeights(
            data=5.0, pde_ineq=1.0, exercise=50.0,
            complementarity=1.0, bc=5.0, terminal=10.0
        ),
        ckpt_name="american_pinn_v2", ckpt_every=500,
    )

    am_eval = load_results("american_eval_v2")
    if am_eval is None:
        print("\n  Evaluating American PINN vs FD (this takes a while)...")
        am_results = evaluate_american_on_grid(am_model, DOMAIN, PARAMS, n_S=80, n_tau=40)
        am_mae = am_results["abs_error"].mean().item()
        am_rmse = torch.sqrt(torch.mean(am_results["abs_error"]**2)).item()
        am_max = am_results["abs_error"].max().item()
        save_results("american_eval_v2", {"mae": am_mae, "rmse": am_rmse, "max": am_max})
    else:
        am_mae, am_rmse, am_max = am_eval["mae"], am_eval["rmse"], am_eval["max"]
    print(f"\n  American MAE:  {am_mae:.4f}")
    print(f"  American RMSE: {am_rmse:.4f}")
    print(f"  American Max:  {am_max:.4f}")

    # Free boundary
    bf = load_results("free_boundary_v2")
    if bf is None:
        print("\n  Extracting free boundaries...")
        tau_pinn, bf_pinn = extract_free_boundary(am_model, DOMAIN, PARAMS, n_tau=50)
        tau_fd, bf_fd = extract_fd_free_boundary(DOMAIN, PARAMS, n_tau=50)
        save_results("free_boundary_v2", {"tau_pinn": tau_pinn, "bf_pinn": bf_pinn, "tau_fd": tau_fd, "bf_fd": bf_fd})
    else:
        tau_pinn, bf_pinn, tau_fd, bf_fd = bf["tau_pinn"], bf["bf_pinn"], bf["tau_fd"], bf["bf_fd"]
    print(f"  Free Boundary MAE: {np.mean(np.abs(bf_pinn - bf_fd)):.4f}")

    # American baselines
    bl = load_results("american_baselines_v2")
    if bl is None:
        print("\n  Running American baselines (binomial, FD, LSM)...")
        test_spots = [80, 90, 95, 100, 105, 110, 120]
        rows = []
        for S0 in test_spots:
            tau_t = torch.tensor([[0.5]], device=device)
            S_t = torch.tensor([[float(S0)]], device=device)
            with torch.no_grad():
                v_pinn = am_model(tau_t, S_t).item()
            v_bin = binomial_american_put(S0, 100, 0.5, 0.05, 0.2, n_steps=10000)
            S_fd, V_fd = fd_american_put(200, 100, 0.5, 0.05, 0.2, n_S=500, n_t=2000)
            v_fd = float(np.interp(S0, S_fd, V_fd))
            v_lsm = lsm_american_put(S0, 100, 0.5, 0.05, 0.2, n_paths=200000)
            rows.append({"S": S0, "pinn": v_pinn, "binom": v_bin, "fd": v_fd, "lsm": v_lsm})
            print(f"    S={S0:>3}: PINN={v_pinn:.4f} Binom={v_bin:.4f} FD={v_fd:.4f} LSM={v_lsm:.4f}")
        save_results("american_baselines_v2", rows)
    else:
        for r in bl:
            print(f"    S={r['S']:>3}: PINN={r['pinn']:.4f} Binom={r['binom']:.4f} FD={r['fd']:.4f} LSM={r['lsm']:.4f}")

    # ── EXPERIMENT 3: Greeks ──
    print("\n" + "="*60)
    print("EXPERIMENT 3: Greeks Validation")
    print("="*60)
    greeks_cached = load_results("greeks")
    if greeks_cached is None:
        S_test = np.linspace(20, 180, 300)
        greeks_data = {}
        for tau_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            d_pinn, g_pinn, t_pinn = compute_pinn_greeks(eu_model, S_test, tau_val)
            d_bs, g_bs, t_bs = bs_greeks_analytical(S_test, PARAMS.K, tau_val, PARAMS.r, PARAMS.sigma)
            greeks_data[tau_val] = {
                "delta_mae": float(np.mean(np.abs(d_pinn - d_bs))),
                "gamma_mae": float(np.mean(np.abs(g_pinn - g_bs))),
                "theta_mae": float(np.mean(np.abs(t_pinn - t_bs))),
            }
            print(f"  tau={tau_val:.1f} | Delta MAE={greeks_data[tau_val]['delta_mae']:.6f} | "
                  f"Gamma MAE={greeks_data[tau_val]['gamma_mae']:.6f} | "
                  f"Theta MAE={greeks_data[tau_val]['theta_mae']:.6f}")
        save_results("greeks", greeks_data)
    else:
        for tau_val, g in greeks_cached.items():
            print(f"  tau={tau_val:.1f} | Delta MAE={g['delta_mae']:.6f} | "
                  f"Gamma MAE={g['gamma_mae']:.6f} | Theta MAE={g['theta_mae']:.6f}")

    # ── EXPERIMENT 4: Ablation Study ──
    print("\n" + "="*60)
    print("EXPERIMENT 4: Ablation Study (12 experiments)")
    print("="*60)
    ablation = load_results("ablation_results") or {}
    print(f"  {'Experiment':<25} | {'MAE':>8} | {'RMSE':>8} | {'R2':>10} | {'Time':>7}")
    print("  " + "-" * 72)
    for name, config in ABLATION_CONFIGS.items():
        if name in ablation:
            r = ablation[name]
            print(f"  {name:<25} | {r['mae']:>8.4f} | {r['rmse']:>8.4f} | {r['r2']:>10.6f} | {r['train_time']:>6.0f}s  [cached]")
            continue
        print(f"  Running: {name}...", end=" ", flush=True)
        r = run_single_experiment(config)
        ablation[name] = r
        save_results("ablation_results", ablation)
        print(f"MAE={r['mae']:.4f} | RMSE={r['rmse']:.4f} | R2={r['r2']:.6f} | {r['train_time']:.0f}s")

    # ── EXPERIMENT 5: Loss Weight Sweep ──
    print("\n" + "="*60)
    print("EXPERIMENT 5: Loss Weight Sweep (9 experiments)")
    print("="*60)
    sweep = load_results("weight_sweep")
    weight_values = [0.1, 1.0, 10.0]
    if sweep is not None:
        sweep_results = sweep["sweep_results"]
        print("  Loaded from cache.")
    else:
        sweep_results = np.full((3, 3), np.nan)

    for i, w_data in enumerate(weight_values):
        for j, w_pde in enumerate(weight_values):
            if not np.isnan(sweep_results[i, j]):
                print(f"  w_data={w_data:.1f}, w_pde={w_pde:.1f} => MAE={sweep_results[i,j]:.4f} [cached]")
                continue
            print(f"  Running w_data={w_data:.1f}, w_pde={w_pde:.1f}...", end=" ", flush=True)
            r = run_single_experiment({"w_data": w_data, "w_pde": w_pde})
            sweep_results[i, j] = r["mae"]
            save_results("weight_sweep", {"sweep_results": sweep_results})
            print(f"MAE={r['mae']:.4f}")

    # ── EXPERIMENT 6: European Baselines ──
    print("\n" + "="*60)
    print("EXPERIMENT 6: European Baselines")
    print("="*60)
    eu_baselines = load_results("european_baselines")
    if eu_baselines is None:
        print("  Training Vanilla NN...")
        vanilla_model = train_vanilla_nn()
        van_results = evaluate_on_grid(vanilla_model, DOMAIN, PARAMS, n_S=80, n_tau=80)
        van_mae = van_results["abs_error"].mean().item()
        van_rmse = torch.sqrt(torch.mean(van_results["abs_error"]**2)).item()
        print(f"  Vanilla NN: MAE={van_mae:.4f}, RMSE={van_rmse:.4f}")

        print("  Running MC baseline (500 points)...")
        torch.manual_seed(42)
        S_ho, tau_ho = sample_uniform_2d(500, 1, 200, 1e-3, 1.0)
        K_t = torch.full_like(S_ho, 100.0)
        r_t = torch.full_like(S_ho, 0.05)
        sig_t = torch.full_like(S_ho, 0.2)
        C_bs_ho = black_scholes_call(S_ho, K_t, tau_ho, r_t, sig_t)
        mc_prices = np.array([monte_carlo_call_price(S_ho[i].item(), 100, tau_ho[i].item(), 0.05, 0.2, 50000)
                              for i in range(500)])
        mc_mae = float(np.mean(np.abs(mc_prices - C_bs_ho[:500].cpu().numpy().flatten())))
        mc_rmse = float(np.sqrt(np.mean((mc_prices - C_bs_ho[:500].cpu().numpy().flatten())**2)))
        print(f"  Monte Carlo: MAE={mc_mae:.4f}, RMSE={mc_rmse:.4f}")

        eu_baselines = {"vanilla_mae": van_mae, "vanilla_rmse": van_rmse,
                        "mc_mae": mc_mae, "mc_rmse": mc_rmse,
                        "pinn_mae": eu_mae, "pinn_rmse": eu_rmse}
        save_results("european_baselines", eu_baselines)
    else:
        print(f"  PINNTO:     MAE={eu_baselines['pinn_mae']:.4f}, RMSE={eu_baselines['pinn_rmse']:.4f}")
        print(f"  Vanilla NN: MAE={eu_baselines['vanilla_mae']:.4f}, RMSE={eu_baselines['vanilla_rmse']:.4f}")
        print(f"  MC (50k):   MAE={eu_baselines['mc_mae']:.4f}, RMSE={eu_baselines['mc_rmse']:.4f}")

    # ── EXPERIMENT 7: Speed Benchmark ──
    print("\n" + "="*60)
    print("EXPERIMENT 7: Speed Benchmark")
    print("="*60)
    speed = load_results("speed_benchmark")
    if speed is None:
        n_q = 5000
        torch.manual_seed(99)
        S_q, tau_q = sample_uniform_2d(n_q, 1, 200, 1e-3, 1.0)

        eu_model.eval()
        t0 = time.time()
        for _ in range(10):
            with torch.no_grad():
                _ = eu_model(tau_q, S_q)
        pinn_t = (time.time() - t0) / 10

        K_q = torch.full_like(S_q, 100.0); r_q = torch.full_like(S_q, 0.05); sig_q = torch.full_like(S_q, 0.2)
        t0 = time.time()
        for _ in range(10):
            _ = black_scholes_call(S_q, K_q, tau_q, r_q, sig_q)
        bs_t = (time.time() - t0) / 10

        t0 = time.time()
        for i in range(20):
            monte_carlo_call_price(S_q[i].item(), 100, tau_q[i].item(), 0.05, 0.2, 50000)
        mc_per = (time.time() - t0) / 20
        mc_t = mc_per * n_q

        speed = {"pinn": pinn_t, "bs": bs_t, "mc_total": mc_t}
        save_results("speed_benchmark", speed)

    print(f"  PINNTO (5000 pts):  {speed['pinn']:.4f}s")
    print(f"  BS Analytical:      {speed['bs']:.4f}s")
    print(f"  MC (50k, est):      {speed['mc_total']:.1f}s")
    print(f"  PINN speedup vs MC: {speed['mc_total']/speed['pinn']:.0f}x")

    # ── EXPERIMENT 8: SHAP Explainability ──
    print("\n" + "="*60)
    print("EXPERIMENT 8: SHAP Explainability")
    print("="*60)
    shap_cached = load_results("shap_analysis")
    if shap_cached is None:
        print("  Running SHAP KernelExplainer (this takes a few minutes)...")
        shap_values, X_ex = run_shap_analysis(eu_model, DOMAIN, PARAMS)
        save_results("shap_analysis", {"shap_values": shap_values, "X_ex": X_ex})
        print(f"  SHAP complete. Feature importance: τ={np.mean(np.abs(shap_values[:, 0])):.4f}, "
              f"S={np.mean(np.abs(shap_values[:, 1])):.4f}")
    else:
        shap_values, X_ex = shap_cached["shap_values"], shap_cached["X_ex"]
        print(f"  SHAP (cached). Feature importance: τ={np.mean(np.abs(shap_values[:, 0])):.4f}, "
              f"S={np.mean(np.abs(shap_values[:, 1])):.4f}")

    # ── EXPERIMENT 9: American Greeks ──
    print("\n" + "="*60)
    print("EXPERIMENT 9: American Greeks Validation")
    print("="*60)
    am_greeks_cached = load_results("american_greeks")
    if am_greeks_cached is None:
        S_test_am = np.linspace(50, 150, 100)
        am_greeks_data = {}
        for tau_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
            print(f"  Computing American Greeks at τ={tau_val}...")
            d_pinn, g_pinn, t_pinn = compute_pinn_greeks_put(am_model, S_test_am, tau_val)
            d_fd, g_fd, t_fd = compute_american_greeks_fd(S_test_am, tau_val, PARAMS, DOMAIN)
            am_greeks_data[tau_val] = {
                "delta_mae": float(np.mean(np.abs(d_pinn - d_fd))),
                "gamma_mae": float(np.mean(np.abs(g_pinn - g_fd))),
                "theta_mae": float(np.mean(np.abs(t_pinn - t_fd))),
                "delta_pinn": d_pinn, "gamma_pinn": g_pinn, "theta_pinn": t_pinn,
                "delta_fd": d_fd, "gamma_fd": g_fd, "theta_fd": t_fd,
            }
            print(f"    Delta MAE={am_greeks_data[tau_val]['delta_mae']:.6f} | "
                  f"Gamma MAE={am_greeks_data[tau_val]['gamma_mae']:.6f} | "
                  f"Theta MAE={am_greeks_data[tau_val]['theta_mae']:.6f}")
        save_results("american_greeks", am_greeks_data)
    else:
        am_greeks_data = am_greeks_cached
        for tau_val, g in am_greeks_data.items():
            print(f"  tau={tau_val:.1f} | Delta MAE={g['delta_mae']:.6f} | "
                  f"Gamma MAE={g['gamma_mae']:.6f} | Theta MAE={g['theta_mae']:.6f}")

    # ── EXPERIMENT 10: Multi-Seed Robustness ──
    print("\n" + "="*60)
    print("EXPERIMENT 10: Multi-Seed Robustness (5 seeds)")
    print("="*60)
    seed_cached = load_results("multi_seed")
    if seed_cached is None:
        seeds = [42, 123, 456, 789, 1337]
        print(f"  Training European PINN with seeds: {seeds}")
        seed_results = run_multi_seed(seeds)
        save_results("multi_seed", seed_results)
    else:
        seed_results = seed_cached
        for r in seed_results:
            print(f"    Seed {r['seed']}: MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} R2={r['r2']:.6f}")
    maes = [r["mae"] for r in seed_results]
    print(f"\n  Mean MAE: {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"  Mean R²:  {np.mean([r['r2'] for r in seed_results]):.6f}")

    # ── FIGURE GENERATION ──
    print("\n" + "="*60)
    print("GENERATING ALL FIGURES")
    print("="*60)

    print("  Fig 1: European convergence...")
    plot_european_convergence(eu_history)

    print("  Fig 2: European pricing surface & error...")
    plot_european_surface(eu_results)

    print("  Fig 4: Ablation bar chart...")
    plot_ablation_bar(ablation)

    print("  Fig 5: Weight sweep heatmap...")
    plot_weight_sweep(sweep_results, weight_values)

    print("  Fig 6: American convergence...")
    plot_american_convergence(am_history)

    print("  Fig 7: American pricing surface...")
    plot_american_surface(am_model, DOMAIN, PARAMS)

    print("  Fig 9: Free boundary...")
    plot_free_boundary(tau_pinn, bf_pinn, tau_fd, bf_fd)

    print("  Fig 10: European Greeks...")
    plot_greeks(eu_model, PARAMS, tau_val=0.5)

    print("  Fig 10b: American Greeks...")
    plot_american_greeks(am_model, PARAMS, DOMAIN, tau_val=0.5)

    print("  Fig 11: SHAP analysis...")
    plot_shap(shap_values, X_ex)

    print("  Fig 12: Multi-seed robustness...")
    plot_multi_seed(seed_results)

    print(f"\n  All figures saved to: {os.path.abspath(FIG_DIR)}/")

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved in: {os.path.abspath(CKPT_DIR)}/")
    print(f"Figures saved in: {os.path.abspath(FIG_DIR)}/")
